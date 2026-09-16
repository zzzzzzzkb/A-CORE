// g++ -O3 -std=c++17 -march=native -I../include search_mrng_cluster_skip_small.cpp -o search_mrng_cluster_skip_small
// 先对 query 做 projected-greedy 聚类，只搜索 size > batch_gate 的簇；
// 小簇跳过，不计入 recall/QPS。
// 运行示例：
// ./search_mrng \
//   ../data/clip-webvid-2.5M/base.2.5M.fbin \
//   ../results/clip-webvid-2.5M.mrng \
//   ./outputs_webvid_new_train2_test2/s1.xq.fbin \
//   ./outputs_webvid_new_train2_test2/s1.gt.ibin \
//   --K=1,10,100 \
//   --L=500\
//   --S=8 \
//   --runs=1 \
//   --warmup=1 \
//   --seed=42 \
//   --csv=webvid_mrng_eval2.csv
//
// ./search_mrng \
// ../data/laion-10M/base.10M.fbin \
// ../results/laion-10M.mrng \
// ./outputs_laion_new_train2_test2/s1.xq.fbin \
// ./outputs_laion_new_train2_test2/s1.gt.ibin \
// --K=1,10,100 \
// --L=1000\
// --S=8 \
// --runs=1 \
// --warmup=0 \
// --seed=42 \
// --csv=laion_mrng_eval.csv
// ./search_mrng \
// ../data/t2i-10M/base.10M.fbin \
// ../results/t2i-10M.mrng \
//  ./outputs_t2i_new_train_test/s1.xq.fbin \
//  ./outputs_t2i_new_train_test/s1.gt.ibin \
// --K=1,10,100 \
// --L=3000:10000:200\
// --S=8 \
// --runs=1 \
// --warmup=0 \
// --seed=42 \
// --csv=t2i_mrng_eval.csv

// 功能：在 MRNG 上做标准 NSG 风格的 best-first 搜索（候选池大小 L），
//       统计 QPS 与 Recall@K，并将不同 (K,L) 的网格结果写入 CSV。
// 说明：单线程实现（不使用 OpenMP），与给定的 search_nsg.cpp 行为/接口一致。

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>

#include <hnswlib/hnswlib.h>

using u32 = uint32_t;

// --------- data io ----------
struct Data { u32 n=0, dim=0; std::vector<float> x; };
static void die(const std::string& m){ std::cerr<<"[ERR] "<<m<<"\n"; std::exit(1); }

static Data read_fbin(const std::string& path){
  FILE* fp = std::fopen(path.c_str(), "rb");
  if(!fp){ perror(path.c_str()); die("open fbin failed: "+path); }
  int n=0,d=0;
  if(std::fread(&n,4,1,fp)!=1 || std::fread(&d,4,1,fp)!=1){ std::fclose(fp); die("bad fbin header"); }
  if(n<=0||d<=0){ std::fclose(fp); die("invalid fbin header values"); }
  Data R; R.n=(u32)n; R.dim=(u32)d; R.x.resize((size_t)n*(size_t)d);
  size_t need=(size_t)n*(size_t)d;
  if(std::fread(R.x.data(), sizeof(float), need, fp)!=need){ std::fclose(fp); die("read fbin body failed"); }
  std::fclose(fp); return R;
}

static std::vector<std::vector<u32>> read_mrng(const std::string& path){
  std::ifstream ifs(path, std::ios::binary);
  if(!ifs) die("open mrng failed: "+path);
  u32 n=0; ifs.read((char*)&n, 4); if(!ifs) die("read n failed");
  std::vector<std::vector<u32>> G(n);
  for(u32 i=0;i<n;++i){
    u32 deg=0; ifs.read((char*)&deg, 4); if(!ifs) die("read deg failed at node "+std::to_string(i));
    G[i].resize(deg);
    if(deg){ ifs.read((char*)G[i].data(), (std::streamsize)deg*4); if(!ifs) die("read neighbors failed at node "+std::to_string(i)); }
  }
  return G;
}

// --------- GT (.ibin) ----------
struct GT {
  u32 nq=0, k=0;
  std::vector<int32_t> idx; // size = nq*k
};

static long long file_size(const std::string& p){
  std::ifstream f(p, std::ios::binary|std::ios::ate);
  if(!f) return -1;
  return (long long)f.tellg();
}

static GT read_gt_ibin(const std::string& path){
  GT gt; long long sz = file_size(path);
  if(sz < 0) die("gt file not found: "+path);
  std::ifstream ifs(path, std::ios::binary);
  if(!ifs) die("open gt failed");
  int a=0,b=0;
  ifs.read((char*)&a,4);
  if(ifs.read((char*)&b,4)){
    long long expect = 8ll + (long long)a*(long long)b*4ll;
    if(expect == sz){
      gt.nq=a; gt.k=b; gt.idx.resize((size_t)gt.nq*(size_t)gt.k);
      ifs.read((char*)gt.idx.data(), (std::streamsize)gt.idx.size()*4);
      return gt;
    }
  }
  // no header
  ifs.clear(); ifs.seekg(0, std::ios::beg);
  long long cnt = sz/4;
  gt.idx.resize((size_t)cnt);
  ifs.read((char*)gt.idx.data(), (std::streamsize)sz);
  int tryK[5] = {100, 50, 20, 10, 1};
  for(int tk: tryK){ if(cnt % tk == 0){ gt.k=tk; gt.nq = (u32)(cnt/tk); break; } }
  if(gt.k==0) die("cannot infer (nq,k) from gt without header");
  return gt;
}

// --------- math ----------
static inline float l2sqr(const float* a, const float* b, u32 d){
  float s=0.f;
  for(u32 i=0;i<d;++i){ float t=a[i]-b[i]; s+=t*t; }
  return s;
}

// --------- pivots & entry ----------
struct Pivots { std::vector<u32> ids; };

static Pivots make_pivots(u32 n, int S, uint64_t seed){
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<u32> dist(0, n-1);
  Pivots P; P.ids.resize(std::max(1,S));
  for(int i=0;i<S;++i) P.ids[i] = dist(rng);
  return P;
}

static u32 choose_entry(const Data& base, const Pivots& P, const float* q){
  float best=1e38f; u32 be=0;
  for(u32 id: P.ids){
    const float* xb = &base.x[(size_t)id*base.dim];
    float d = l2sqr(q, xb, base.dim);
    if(d<best){ best=d; be=id; }
  }
  return be;
}

// 与 NSG search 相同的辅助结构与比较器
struct MinCand { u32 id; float dist; };  // C：按距离升序取最近
struct MaxRes  { float dist; u32 id; };  // W：按距离降序取最远

struct CmpMinCand {
  bool operator()(const MinCand& a, const MinCand& b) const {
    // priority_queue 默认是最大堆，这里把距离小的放“上面”
    return a.dist > b.dist;
  }
};

struct CmpMaxRes {
  bool operator()(const MaxRes& a, const MaxRes& b) const {
    // 距离大的优先（最远的在堆顶）
    return a.dist < b.dist;
  }
};

// --------- MRNG search（NSG风格 best-first，候选池 L） ----------
// 仅负责搜索阶段，返回未排序的前K个结果（通过一个最大堆维护）
static void mrng_search_one(
  const Data& base,
  const std::vector<std::vector<u32>>& G,
  const float* q, u32 K, u32 L,
  const Pivots& piv,
  std::vector<u32>& out_ids,
  std::vector<float>& out_dists
){
  const u32 n = (u32)G.size();
  if (n == 0) {
    out_ids.clear();
    out_dists.clear();
    return;
  }

  // ---------- V：visited ----------
  std::vector<char> vis(n, 0);
  auto seen = [&](u32 x)->bool {
    if (vis[x]) return true;
    vis[x] = 1;
    return false;
  };

  // ---------- 入口点 ----------
  u32 ep = choose_entry(base, piv, q);
  if (ep >= n) {
    // 防御性判断（按需处理）
    ep = n - 1;
  }

  float d_ep = l2sqr(q, &base.x[(size_t)ep * base.dim], base.dim);
  seen(ep);

  // ---------- C：候选池（按距离升序） ----------
  using CandPQ = std::priority_queue<MinCand, std::vector<MinCand>, CmpMinCand>;
  CandPQ C;
  C.push(MinCand{ep, d_ep});

  // ---------- W：结果池（按距离降序，最多 L 个） ----------
  using ResPQ = std::priority_queue<MaxRes, std::vector<MaxRes>, CmpMaxRes>;
  ResPQ W;
  W.push(MaxRes{d_ep, ep});

  // ---------- 搜索主循环（完全沿用 NSG 的逻辑和终止条件） ----------
  while (!C.empty()) {
    MinCand c = C.top(); C.pop();
    MaxRes f = W.top();

    // 终止条件：当前最优候选已经比结果集中最差的还“差”
    if (c.dist > f.dist) break;

    const auto& nbrs = G[c.id];
    for (u32 e : nbrs) {
      if (e >= n) continue;
      if (seen(e)) continue;

      float de = l2sqr(q, &base.x[(size_t)e * base.dim], base.dim);
      f = W.top();
      // 与 NSG 一致：只有在 W 未满 L 或更优于当前最差结果时才加入
      if (W.size() < (size_t)L || de < f.dist) {
        C.push(MinCand{e, de});
        W.push(MaxRes{de, e});
        if (W.size() > (size_t)L) {
          W.pop();  // 保持 W 大小不超过 L
        }
      }
    }
  }

  // ---------- 从 W 中取出前 K 个结果 ----------
  // 保持“未排序、长度最多 K”的语义：只做截断，不保证顺序
  while (W.size() > (size_t)K) {
    W.pop();
  }

  out_ids.clear();
  out_dists.clear();
  while (!W.empty()) {
    out_ids.push_back(W.top().id);
    out_dists.push_back(W.top().dist);
    W.pop();
  }
}

// ---------- Projected Greedy Clustering ----------
struct PGClusterOut {
  std::vector<std::vector<u32>> clusters;
  size_t non_empty = 0;
  size_t min_size = 0;
  size_t max_size = 0;
};

static std::vector<float> random_project_pg(const Data& queries, int proj_dim, uint64_t seed) {
  std::vector<float> proj((size_t)queries.n * (size_t)proj_dim, 0.0f);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f / std::sqrt((float)proj_dim));

  std::vector<float> R((size_t)proj_dim * (size_t)queries.dim);
  for (float& v : R) v = gauss(rng);

  for (u32 i = 0; i < queries.n; ++i) {
    const float* x = &queries.x[(size_t)i * queries.dim];
    float* z = proj.data() + (size_t)i * (size_t)proj_dim;
    for (int r = 0; r < proj_dim; ++r) {
      const float* row = R.data() + (size_t)r * queries.dim;
      float acc = 0.0f;
      for (u32 c = 0; c < queries.dim; ++c) acc += row[c] * x[c];
      z[r] = acc;
    }
  }
  return proj;
}

static std::vector<std::vector<u32>> build_lowdim_topm_hnsw_pg(const std::vector<float>& z,
                                                               u32 n,
                                                               int proj_dim,
                                                               int m,
                                                               int hnsw_m,
                                                               int hnsw_efc,
                                                               int hnsw_ef) {
  std::vector<std::vector<u32>> nbrs(n);
  hnswlib::L2Space space((size_t)proj_dim);
  hnswlib::HierarchicalNSW<float> hnsw(&space, (size_t)n, (size_t)hnsw_m, (size_t)hnsw_efc);
  for (u32 i = 0; i < n; ++i) {
    hnsw.addPoint((const void*)(z.data() + (size_t)i * (size_t)proj_dim), (size_t)i);
  }
  hnsw.setEf((size_t)std::max(hnsw_ef, m + 1));

  for (u32 i = 0; i < n; ++i) {
    auto pq = hnsw.searchKnn((const void*)(z.data() + (size_t)i * (size_t)proj_dim), (size_t)(m + 1));
    auto& ids = nbrs[i];
    ids.reserve((size_t)m);
    while (!pq.empty() && (int)ids.size() < m) {
      u32 id = (u32)pq.top().second;
      pq.pop();
      if (id == i) continue;
      if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
    }
  }
  return nbrs;
}

static std::vector<std::vector<u32>> build_compat_graph_pg(const Data& queries,
                                                           const std::vector<std::vector<u32>>& cand,
                                                           float tau_edge) {
  const float tau_edge_sq = tau_edge * tau_edge;
  std::vector<std::vector<u32>> g(queries.n);
  for (u32 i = 0; i < queries.n; ++i) {
    const float* xi = &queries.x[(size_t)i * queries.dim];
    for (u32 j : cand[i]) {
      if (j <= i) continue;
      const float* xj = &queries.x[(size_t)j * queries.dim];
      if (l2sqr(xi, xj, queries.dim) <= tau_edge_sq) {
        g[i].push_back(j);
        g[j].push_back(i);
      }
    }
  }
  return g;
}

static bool can_add_under_radius_pg(const Data& queries,
                                    const std::vector<u32>& cluster,
                                    const std::vector<float>& sum_vec,
                                    u32 cand,
                                    float tau_cluster_sq) {
  const float* cand_vec = &queries.x[(size_t)cand * queries.dim];
  std::vector<float> center(queries.dim, 0.0f);
  float inv = 1.0f / (float)(cluster.size() + 1);
  for (u32 i = 0; i < queries.dim; ++i) center[i] = (sum_vec[i] + cand_vec[i]) * inv;

  for (u32 id : cluster) {
    const float* x = &queries.x[(size_t)id * queries.dim];
    if (l2sqr(x, center.data(), queries.dim) > tau_cluster_sq) return false;
  }
  return l2sqr(cand_vec, center.data(), queries.dim) <= tau_cluster_sq;
}

static PGClusterOut build_projected_greedy_clusters(const Data& queries,
                                                    int proj_dim,
                                                    int m,
                                                    int cl_hnsw_m,
                                                    int cl_hnsw_efc,
                                                    int cl_hnsw_ef,
                                                    float tau_edge,
                                                    float tau_cluster,
                                                    uint64_t seed) {
  PGClusterOut out;
  if (queries.n == 0) return out;

  auto z = random_project_pg(queries, proj_dim, seed);
  auto cand = build_lowdim_topm_hnsw_pg(z, queries.n, proj_dim, m, cl_hnsw_m, cl_hnsw_efc, cl_hnsw_ef);
  auto g = build_compat_graph_pg(queries, cand, tau_edge);

  std::vector<int> degree(queries.n, 0);
  for (u32 i = 0; i < queries.n; ++i) degree[i] = (int)g[i].size();

  std::vector<u32> order(queries.n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](u32 a, u32 b) {
    if (degree[a] != degree[b]) return degree[a] > degree[b];
    return a < b;
  });

  std::vector<uint8_t> assigned(queries.n, 0), in_cluster(queries.n, 0);
  const float tau_cluster_sq = tau_cluster * tau_cluster;

  for (u32 seed_id : order) {
    if (assigned[seed_id]) continue;
    std::vector<u32> cluster;
    cluster.reserve(64);
    cluster.push_back(seed_id);
    in_cluster[seed_id] = 1;

    std::vector<float> sum_vec(queries.dim, 0.0f);
    const float* seed_vec = &queries.x[(size_t)seed_id * queries.dim];
    for (u32 i = 0; i < queries.dim; ++i) sum_vec[i] = seed_vec[i];

    std::vector<u32> frontier{seed_id};
    size_t head = 0;
    while (head < frontier.size()) {
      u32 u = frontier[head++];
      for (u32 v : g[u]) {
        if (assigned[v] || in_cluster[v]) continue;
        if (!can_add_under_radius_pg(queries, cluster, sum_vec, v, tau_cluster_sq)) continue;
        cluster.push_back(v);
        in_cluster[v] = 1;
        frontier.push_back(v);
        const float* vv = &queries.x[(size_t)v * queries.dim];
        for (u32 i = 0; i < queries.dim; ++i) sum_vec[i] += vv[i];
      }
    }

    for (u32 id : cluster) {
      assigned[id] = 1;
      in_cluster[id] = 0;
    }
    out.clusters.push_back(std::move(cluster));
  }

  out.non_empty = out.clusters.size();
  out.min_size = std::numeric_limits<size_t>::max();
  out.max_size = 0;
  for (const auto& c : out.clusters) {
    out.min_size = std::min(out.min_size, c.size());
    out.max_size = std::max(out.max_size, c.size());
  }
  if (out.non_empty == 0) out.min_size = 0;
  return out;
}



// --------- args ----------
static std::unordered_map<std::string,std::string> parse_args(int argc, char** argv){
  std::unordered_map<std::string,std::string> m;
  for(int i=1;i<argc;++i){
    std::string s(argv[i]);
    if(s.rfind("--",0)==0){
      auto p = s.find('=');
      if(p==std::string::npos) m[s.substr(2)] = "1";
      else m[s.substr(2,p-2)] = s.substr(p+1);
    }else if(s.rfind("-",0)==0){
      std::string key = s.substr(1);
      auto p = key.find('=');
      if(p != std::string::npos){
        m[key.substr(0,p)] = key.substr(p+1);
      }else if(i + 1 < argc && std::string(argv[i+1]).rfind("-",0) != 0){
        m[key] = argv[++i];
      }else{
        m[key] = "1";
      }
    }else{
      m[std::to_string(i)] = s; // positional
    }
  }
  return m;
}

static std::vector<int> parse_int_list(const std::string& s){
  std::vector<int> v; std::string buf;
  for(char c: s){
    if(c==',' || c==' ') { if(!buf.empty()){ v.push_back(std::stoi(buf)); buf.clear(); } }
    else buf.push_back(c);
  }
  if(!buf.empty()) v.push_back(std::stoi(buf));
  return v;
}

// 支持三种写法：
//  1) 逗号列表:  "50,100,200"
//  2) 区间含步长: "50:200:50" (start:end:step, end 包含)
//  3) 简单区间:   "50-200" (步长 = 1 或默认推断为 50? 这里取 1，更灵活)
// 若包含 ':' 按 start[:end[:step]] 解析；若包含 '-' 且不含 ':' 按 start-end 解析；否则回退到逗号列表。
static std::vector<int> parse_range_or_list(const std::string& s){
  std::vector<int> out;
  if(s.find(':') != std::string::npos){
    // start:end(:step)
    std::vector<long long> nums; std::string buf;
    for(char c: s){
      if(c==':'){ if(!buf.empty()){ nums.push_back(std::stoll(buf)); buf.clear(); } }
      else buf.push_back(c);
    }
    if(!buf.empty()) nums.push_back(std::stoll(buf));
    if(nums.size() < 2){ return parse_int_list(s); }
    // 兼容两种三段式写法：
    //  1) start:end:step  （旧实现）
    //  2) start:step:end  （用户期望的“网格”语法，如 3000:200:10000）
    // 判定策略：出现 3 个数字时，如果中间值位于首尾之间（严格介于 start 与 end 之间），视为 start:step:end；否则视为 start:end:step。
    if(nums.size() == 3){
      long long a = nums[0], b = nums[1], c = nums[2];
      bool middle_is_step_form = ( (a < b && b < c) || (a > b && b > c) );
      if(middle_is_step_form){
        long long start = a;
        long long step  = b;
        long long end   = c;
        if(step == 0) step = 1;
        if(start <= end){
          if(step < 0) step = -step;
          for(long long v = start; v <= end; v += step) out.push_back((int)v);
        }else{
          if(step > 0) step = -step;
          for(long long v = start; v >= end; v -= step) out.push_back((int)v);
        }
        return out;
      }
      // 否则按旧语义 start:end:step
      long long start = a;
      long long end   = b;
      long long step  = c;
      if(step == 0) step = (start <= end ? 1 : -1);
      if(start <= end){
        if(step < 0) step = -step;
        for(long long v = start; v <= end; v += step) out.push_back((int)v);
      }else{
        if(step > 0) step = -step;
        for(long long v = start; v >= end; v += step) out.push_back((int)v);
      }
      return out;
    }
    // 只有两个数字：按 start:end 解析，步长自适应为 1 或 -1
    long long start = nums[0];
    long long end   = nums[1];
    long long step  = (start <= end ? 1 : -1);
    for(long long v = start; (step>0? v <= end : v >= end); v += step) out.push_back((int)v);
    return out;
  }else if(s.find('-') != std::string::npos){
    size_t p = s.find('-');
    std::string a = s.substr(0,p);
    std::string b = s.substr(p+1);
    if(a.empty() || b.empty()) return parse_int_list(s);
    long long start = std::stoll(a);
    long long end   = std::stoll(b);
    if(start <= end){
      for(long long v = start; v <= end; ++v) out.push_back((int)v);
    }else{
      for(long long v = start; v >= end; --v) out.push_back((int)v);
    }
    return out;
  }else{
    return parse_int_list(s);
  }
}

// --------- main ----------
int main(int argc, char** argv){
  if(argc < 5){
    std::cerr <<
    "Usage:\n  "<<argv[0]<<" <base.fbin> <mrng_path> <xq.fbin> <gt.ibin>\n"
    "  or\n  "<<argv[0]<<" -data base.fbin -graph graph.mrng -qfile xq.fbin -gt gt.ibin\n"
    "Options:\n"
    "  --K=10,20        (Recall@K list)\n"
    "  --L=50,100,200   或 --L=50:200:50 或 --L=50-200  (candidate pool size list / range)\n"
    "  --S=8            (pivot count)\n"
    "  --runs=1         (repeat runs for avg)\n"
    "  --warmup=0       (warmup runs not counted)\n"
    "  --seed=42        (random seed)\n"
    "  --csv=out.csv    (csv output)\n"
    "  --batch_gate=8   (clusters with size <= gate are skipped)\n"
    "  --proj_dim=32 --proj_m=16 --cl_hnsw_m=16 --cl_hnsw_efc=100 --cl_hnsw_ef=64\n"
    "  --tau_edge=0.4 --tau_cluster=0.5\n";
    return 1;
  }
  auto a = parse_args(argc, argv);
  std::string base_path = a.count("data") ? a["data"] : a["1"];
  std::string mrng_path = a.count("graph") ? a["graph"] : a["2"];
  std::string xq_path   = a.count("qfile") ? a["qfile"] : a["3"];
  std::string gt_path   = a.count("gt") ? a["gt"] : a["4"];
  if(base_path.empty() || mrng_path.empty() || xq_path.empty() || gt_path.empty()) die("missing -data/-graph/-qfile/-gt");

  std::vector<int> Ks = a.count("K")? parse_int_list(a["K"]) : std::vector<int>{10};
  std::vector<int> Ls = a.count("L")? parse_range_or_list(a["L"]) : std::vector<int>{50,100,200};
  int S    = a.count("S")     ? std::stoi(a["S"])     : 8;
  int runs = a.count("runs")  ? std::stoi(a["runs"])  : 1;
  int warm = a.count("warmup")? std::stoi(a["warmup"]): 0;
  uint64_t seed = a.count("seed")? (uint64_t)std::stoull(a["seed"]) : 42ull;
  std::string csv = a.count("csv")? a["csv"] : "mrng_search_results.csv";
  int batch_gate = a.count("batch_gate") ? std::stoi(a["batch_gate"]) : 8;
  int proj_dim = a.count("proj_dim") ? std::stoi(a["proj_dim"]) : 32;
  int proj_m = a.count("proj_m") ? std::stoi(a["proj_m"]) : 16;
  int cl_hnsw_m = a.count("cl_hnsw_m") ? std::stoi(a["cl_hnsw_m"]) : 16;
  int cl_hnsw_efc = a.count("cl_hnsw_efc") ? std::stoi(a["cl_hnsw_efc"]) : 100;
  int cl_hnsw_ef = a.count("cl_hnsw_ef") ? std::stoi(a["cl_hnsw_ef"]) : 64;
  float tau_edge = a.count("tau_edge") ? std::stof(a["tau_edge"]) : 0.4f;
  float tau_cluster = a.count("tau_cluster") ? std::stof(a["tau_cluster"]) : 0.5f;

  std::cout<<"[INFO] loading base...\n";
  Data base = read_fbin(base_path);
  std::cout<<"[INFO] base: n="<<base.n<<" dim="<<base.dim<<"\n";

  std::cout<<"[INFO] loading mrng...\n";
  auto G = read_mrng(mrng_path);
  if(G.size()!=base.n) std::cerr<<"[WARN] MRNG n="<<G.size()<<" != base n="<<base.n<<"\n";

  std::cout<<"[INFO] loading queries & gt...\n";
  Data xq = read_fbin(xq_path);
  GT gt = read_gt_ibin(gt_path);
  const u32 nq = std::min<u32>(xq.n, gt.nq);
  std::cout<<"[INFO] nq="<<nq<<" gt.k="<<gt.k<<"\n";

  Data xq_eval = xq;
  xq_eval.n = nq;
  std::cout<<"[ProjectedGreedy] start clustering nq="<<nq
           <<" proj_dim="<<proj_dim
           <<" M="<<proj_m
           <<" tau_edge="<<tau_edge
           <<" tau_cluster="<<tau_cluster
           <<" batch_gate="<<batch_gate<<"\n";
  auto t_cluster0 = std::chrono::high_resolution_clock::now();
  PGClusterOut pg = build_projected_greedy_clusters(
      xq_eval, proj_dim, proj_m, cl_hnsw_m, cl_hnsw_efc, cl_hnsw_ef, tau_edge, tau_cluster, seed);
  auto t_cluster1 = std::chrono::high_resolution_clock::now();
  double cluster_build_ms = std::chrono::duration<double,std::milli>(t_cluster1 - t_cluster0).count();

  std::vector<u32> active_qids;
  std::vector<u32> skipped_qids;
  size_t large_clusters = 0, small_clusters = 0;
  for(const auto& C : pg.clusters){
    if(C.empty()) continue;
    if((int)C.size() <= batch_gate){
      ++small_clusters;
      skipped_qids.insert(skipped_qids.end(), C.begin(), C.end());
    }else{
      ++large_clusters;
      active_qids.insert(active_qids.end(), C.begin(), C.end());
    }
  }
  std::cout<<"[ProjectedGreedy] total_clusters="<<pg.clusters.size()
           <<" non_empty="<<pg.non_empty
           <<" min_size="<<pg.min_size
           <<" max_size="<<pg.max_size
           <<" large(>gate)="<<large_clusters
           <<" small(<=gate, skipped)="<<small_clusters
           <<" active_q="<<active_qids.size()
           <<" skipped_q="<<skipped_qids.size()<<"\n";

  // CSV header
  {
    std::ofstream ofs(csv, std::ios::app);
    if(ofs.tellp()==0){
      ofs<<"base_path,mrng_path,xq_path,gt_path,K,L,S,runs,threads,nq,effective_q,skipped_q,total_clusters,large_clusters,small_clusters,batch_gate,avg_qps,avg_ms_per_query,recall\n";
    }
  }

  Pivots piv = make_pivots(base.n, std::max(1,S), seed);
  std::cout<<"[INFO] threads = 1 (no OpenMP)\n";

  // 为了“一次跑出多K的qps+recall”：
  // 1) 先确定 Kmax = max(Ks)
  // 2) 对每个 L，仅计时一次：为全部查询构造 Top-Kmax（未排序）
  // 3) 计时结束后再做排序和按不同K的前缀裁剪，并统计 recall；
  //    QPS 与 ms/q 对同一 L 的所有 K 复用。
  int Kmax = 1;
  for(int k: Ks) Kmax = std::max(Kmax, k);
  for(int L: Ls){
    // warmup（不计时）
    for(int r=0;r<warm;++r){
      for(u32 iq: active_qids){
        const float* q = &xq.x[(size_t)iq*xq.dim];
        std::vector<u32> tmp_ids; std::vector<float> tmp_dists;
        mrng_search_one(base, G, q, (u32)Kmax, (u32)L, piv, tmp_ids, tmp_dists);
      }
    }

    double sum_ms = 0.0;
    // 为每次 run 保留一份未排序的 top-Kmax 结果，用于计时外的排序与 recall
    // 由于我们只需要平均 qps/ms 与平均 recall，这里每次 run 都独立计算 recall 再求平均
    double sum_recall_dummy = 0.0; // 不再用于单个K，保留变量避免误删

    // 对于每个 K 的累计 recall（跨 runs 平均），单独记录
    std::unordered_map<int,double> recall_accum; // K -> sum_recall_over_runs
    for(int k: Ks) recall_accum[k] = 0.0;

    for(int r=0;r<runs;++r){
      auto t0 = std::chrono::high_resolution_clock::now();

      // 搜索阶段（计时范围）：仅为 active queries 构造 Top-Kmax（未排序）
      std::vector<u32> all_ids_unsorted; all_ids_unsorted.resize((size_t)nq*(size_t)Kmax, UINT32_MAX);
      std::vector<float> all_dists_unsorted; all_dists_unsorted.resize((size_t)nq*(size_t)Kmax, std::numeric_limits<float>::infinity());
      for(u32 iq: active_qids){
        const float* q = &xq.x[(size_t)iq*xq.dim];
        std::vector<u32> out_ids; std::vector<float> out_dists;
        mrng_search_one(base, G, q, (u32)Kmax, (u32)L, piv, out_ids, out_dists);
        // 拷贝到平铺数组
        for(int j=0;j<Kmax;++j){
          if(j < (int)out_ids.size()){
            all_ids_unsorted[(size_t)iq*(size_t)Kmax + j] = out_ids[(size_t)j];
            all_dists_unsorted[(size_t)iq*(size_t)Kmax + j] = out_dists[(size_t)j];
          }
        }
      }

      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
      sum_ms += ms;

      // 计时之外：对每个查询按距离升序排序，并计算不同K的 recall
      std::vector<u32> all_ids_sorted; all_ids_sorted.resize((size_t)nq*(size_t)Kmax, UINT32_MAX);
      for(u32 iq: active_qids){
        std::vector<int> ord(Kmax); std::iota(ord.begin(), ord.end(), 0);
        size_t off = (size_t)iq*(size_t)Kmax;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b){
          float da = all_dists_unsorted[off + (size_t)a];
          float db = all_dists_unsorted[off + (size_t)b];
          return da < db;
        });
        for(int j=0;j<Kmax;++j){
          all_ids_sorted[off + (size_t)j] = all_ids_unsorted[off + (size_t)ord[j]];
        }
      }

      // 针对每个K计算 recall（共享同一轮搜索时间）
      for(int K : Ks){
        if(K > (int)gt.k) std::cerr<<"[WARN] K="<<K<<" > gt.k="<<gt.k<<", recall@K 用 gt 前 "<<gt.k<<" 评估。\n";
        const int Kg = std::min(K, (int)gt.k);
        double hit=0.0, denom = (double)Kg * (double)active_qids.size();
        for(u32 iq: active_qids){
          const int32_t* row = &gt.idx[(size_t)iq*(size_t)gt.k];
          // 为了简单起见，直接线性扫描前K个预测结果
          for(int j=0;j<Kg;++j){
            int32_t g = row[j];
            for(int t=0;t<K;++t){
              if((int32_t)all_ids_sorted[(size_t)iq*(size_t)Kmax + (size_t)t] == g){ hit += 1.0; break; }
            }
          }
        }
        double recall = (denom>0? hit/denom : 0.0);
        recall_accum[K] += recall;
      }
    }

    double avg_ms = sum_ms / runs;
    double qps = active_qids.empty() ? 0.0 : ((double)active_qids.size() * 1000.0) / avg_ms;

    // 输出：对同一 L 的所有 K 复用同一个 qps 与 ms/q
    for(int K : Ks){
      double avg_recall = recall_accum[K] / runs;
      std::cout<<"K="<<K<<" L="<<L<<" | effective_q="<<active_qids.size()<<" skipped_q="<<skipped_qids.size()
               <<" | QPS="<<qps
               <<" | ms/q="<<(active_qids.empty()?0.0:avg_ms/(double)active_qids.size())
               <<" | Recall="<<avg_recall<<"\n";
      std::ofstream ofs(csv, std::ios::app);
      ofs<<base_path<<","<<mrng_path<<","<<xq_path<<","<<gt_path<<"," 
         <<K<<","<<L<<","<<S<<","<<runs<<"," 
         <<1 /* threads */ <<","<<nq<<","<<active_qids.size()<<","<<skipped_qids.size()<<","
         <<pg.clusters.size()<<","<<large_clusters<<","<<small_clusters<<","<<batch_gate<<","
         <<qps<<","<<(active_qids.empty()?0.0:avg_ms/(double)active_qids.size())<<","<<avg_recall<<"\n";
    }
  }

  std::cout<<"Done. Results -> "<<csv<<"\n";
  return 0;
}
