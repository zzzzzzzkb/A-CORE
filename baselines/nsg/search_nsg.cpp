// g++ -O3 -std=c++17 -march=native search_nsg.cpp -o search_nsg
// 运行示例：
// ./search_nsg \
//   ../data/clip-webvid-2.5M/base.2.5M.fbin \
//   ../results/clip-webvid.nsg \
//   ./outputs_webvid_new_train2_test2/s1.xq.fbin \
//   ./outputs_webvid_new_train2_test2/s1.gt.ibin \
//   --K=1,10,100 \
//   --L=1000 \
//   --runs=1 \
//   --warmup=0 \
//   --csv=webvid_nsg_eval.csv
  // ./search_nsg \
  // ../data/laion-10M/base.10M.fbin \
  // ../results/laion.nsg \
  // ./outputs_laion_new_train2_test2/s1.xq.fbin \
  // ./outputs_laion_new_train2_test2/s1.gt.ibin \
  // --K=1,10,100 \
  // --L=3000:500:8000 \
  // --runs=1 \
  // --warmup=0 \
  // --csv=laion_nsg_eval_final_new2.csv
// ./search_nsg \
//   ../data/t2i-10M/base.10M.fbin \
//   ../results/t2i.nsg \
//   ./outputs_t2i_new_train_test/s1.xq.fbin \
//   ./outputs_t2i_new_train_test/s1.gt.ibin \
//   --K=1,10,100 \
//   --L=1000 \
//   --runs=1 \
//   --warmup=0 \
//   --csv=t2i_nsg_eval33.csv
// 新增：支持 --entry_id=<u32> 手动指定入口；否则默认用“质心最近样本”作为 navigating node。
// 说明：固定入口模式下会忽略 --S 与 --seed（沿用旧 CSV 列名以兼容）。
//
// 索引/数据格式：
//   .fbin : [int32 n][int32 dim][n*dim float]
//   .nsg  : [uint32 n][ per-node: uint32 deg; deg*uint32 neighbors ]
//   .ibin : A)[int32 nq][int32 k][nq*k int32] 或 B)[nq*k int32]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <queue>

using u32 = uint32_t;

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

static std::vector<std::vector<u32>> read_nsg(const std::string& path){
  std::ifstream ifs(path, std::ios::binary);
  if(!ifs) die("open nsg failed: "+path);
  u32 n=0; ifs.read((char*)&n, 4); if(!ifs) die("read n failed");
  std::vector<std::vector<u32>> G(n);
  for(u32 i=0;i<n;++i){
    u32 deg=0; ifs.read((char*)&deg, 4); if(!ifs) die("read deg failed");
    G[i].resize(deg);
    if(deg){ ifs.read((char*)G[i].data(), (std::streamsize)deg*4); if(!ifs) die("read neighbors failed"); }
  }
  return G;
}

// --------- GT (.ibin) ----------
struct GT { u32 nq=0, k=0; std::vector<int32_t> idx; };
static long long file_size(const std::string& p){ std::ifstream f(p, std::ios::binary|std::ios::ate); return (long long)f.tellg(); }

static GT read_gt_ibin(const std::string& path){
  GT gt; long long sz = file_size(path);
  if(sz < 0) die("gt file not found: "+path);
  std::ifstream ifs(path, std::ios::binary);
  if(!ifs) die("open gt failed");
  int a=0,b=0; ifs.read((char*)&a,4);
  if(ifs.read((char*)&b,4)){
    long long expect = 8ll + (long long)a*(long long)b*4ll;
    if(expect == sz){
      gt.nq=a; gt.k=b; gt.idx.resize((size_t)gt.nq*(size_t)gt.k);
      ifs.read((char*)gt.idx.data(), (std::streamsize)gt.idx.size()*4);
      return gt;
    }
  }
  ifs.clear(); ifs.seekg(0, std::ios::beg);
  long long cnt = sz/4; gt.idx.resize((size_t)cnt);
  ifs.read((char*)gt.idx.data(), (std::streamsize)sz);
  int tryK[5] = {100, 50, 20, 10, 1};
  for(int tk: tryK){ if(cnt % tk == 0){ gt.k=tk; gt.nq = (u32)(cnt/tk); break; } }
  if(gt.k==0) die("cannot infer (nq,k) from gt without header");
  return gt;
}

// --------- math ----------
static inline float l2sqr(const float* a, const float* b, u32 d){
  float s=0.f; for(u32 i=0;i<d;++i){ float t=a[i]-b[i]; s+=t*t; } return s;
}

// --------- fixed navigating node (centroid-nearest) ----------
static u32 compute_entry_centroid_nearest(const Data& base){
  if(base.n==0) die("empty base");
  std::vector<float> mu(base.dim, 0.f);
  // mean
  const float invN = 1.0f / (float)base.n;
  for(u32 i=0;i<base.n;++i){
    const float* xi = &base.x[(size_t)i*base.dim];
    for(u32 j=0;j<base.dim;++j) mu[j] += xi[j]*invN;
  }
  // nearest to mean
  u32 best_id = 0; float best = 1e38f;
  for(u32 i=0;i<base.n;++i){
    const float* xi = &base.x[(size_t)i*base.dim];
    float d=0.f; for(u32 j=0;j<base.dim;++j){ float t=xi[j]-mu[j]; d+=t*t; }
    if(d<best){ best=d; best_id=i; }
  }
  return best_id;
}


// --------- NSG search (best-first with pool L = ef) ----------
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

// 进行 NSG 搜索并返回按距离升序的 (dist,id) 前 K 结果
static void nsg_search_one_fixed_entry(
  const Data& base,
  const std::vector<std::vector<u32>>& G,
  const float* q, u32 K, u32 L,         // 这里把 L 当作图中算法的 ef
  u32 entry_id,
  std::vector<std::pair<float,u32>>& out_sorted
){
  const u32 n = (u32)G.size();
  if (entry_id >= n) die("entry_id out of range");

  // ---------- V：visited ----------
  std::vector<char> vis(n, 0);
  auto seen = [&](u32 x)->bool {
    if (vis[x]) return true;
    vis[x] = 1;
    return false;
  };

  // ---------- 初始化 ----------
  float d_ep = l2sqr(q, &base.x[(size_t)entry_id * base.dim], base.dim);
  seen(entry_id);
  using CandPQ = std::priority_queue<MinCand, std::vector<MinCand>, CmpMinCand>;
  CandPQ C;
  C.push(MinCand{entry_id, d_ep});
  using ResPQ = std::priority_queue<MaxRes, std::vector<MaxRes>, CmpMaxRes>;
  ResPQ W;
  W.push(MaxRes{d_ep, entry_id});
  while (!C.empty()) {
    MinCand c = C.top(); C.pop();
    MaxRes f = W.top();
    if (c.dist > f.dist) break;  
    const auto& nbrs = G[c.id];
    for (u32 e : nbrs) {
      if (e >= n) continue;
      if (seen(e)) continue;
      float de = l2sqr(q, &base.x[(size_t)e * base.dim], base.dim);
      f = W.top();
      if (W.size() < (size_t)L || de < f.dist) {
        C.push(MinCand{e, de});
        W.push(MaxRes{de, e});
        if (W.size() > (size_t)L) {
          W.pop();  
        }
      }
    }
  }

  // ---------- 从 W 中取出结果并按距离升序输出 ----------
while (W.size() > static_cast<std::size_t>(K)) {
    W.pop();  
}
out_sorted.clear();
out_sorted.reserve(W.size());
while (!W.empty()) {
    const auto &t = W.top();
    out_sorted.emplace_back(t.dist, t.id);
    W.pop();
}
std::reverse(out_sorted.begin(), out_sorted.end());
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
    }else{
      m[std::to_string(i)] = s; // positional
    }
  }
  return m;
}
static std::vector<int> parse_int_list(const std::string& s){
  std::vector<int> v; std::string buf;
  for(char c: s){
    if(c==',' || c==' '){ if(!buf.empty()){ v.push_back(std::stoi(buf)); buf.clear(); } }
    else buf.push_back(c);
  }
  if(!buf.empty()) v.push_back(std::stoi(buf));
  return v;
}

// 支持网格语法："a:b"（含端点，步长±1），"a:s:b"（含端点，步长s），以及原有的以逗号/空格分隔的列表
static void append_grid_token(const std::string& tok, std::vector<int>& out){
  if(tok.empty()) return;
  size_t c1 = tok.find(':');
  if(c1 == std::string::npos){
    out.push_back(std::stoi(tok));
    return;
  }
  size_t c2 = tok.find(':', c1+1);
  auto to_i = [](const std::string& s){ return std::stoi(s); };
  if(c2 == std::string::npos){
    int a = to_i(tok.substr(0, c1));
    int b = to_i(tok.substr(c1+1));
    if(a <= b){ for(int x=a; x<=b; ++x) out.push_back(x); }
    else       { for(int x=a; x>=b; --x) out.push_back(x); }
    return;
  }
  int a = to_i(tok.substr(0, c1));
  int s = to_i(tok.substr(c1+1, c2-(c1+1)));
  int b = to_i(tok.substr(c2+1));
  if(s == 0) return; // 忽略步长为0的非法输入
  if((long long)(b - a) * (long long)s < 0) return; // 方向不一致，忽略
  if(s > 0){ for(int x=a; x<=b; x+=s) out.push_back(x); }
  else      { for(int x=a; x>=b; x+=s) out.push_back(x); }
}

static std::vector<int> parse_int_list_or_grid(const std::string& s){
  std::vector<int> v; std::string buf;
  for(char c: s){
    if(c==',' || c==' '){ if(!buf.empty()){ append_grid_token(buf, v); buf.clear(); } }
    else buf.push_back(c);
  }
  if(!buf.empty()) append_grid_token(buf, v);
  return v;
}

// --------- main ----------
int main(int argc, char** argv){
  if(argc < 5){
    std::cerr <<
    "Usage:\n  "<<argv[0]<<" <base.fbin> <nsg_path> <xq.fbin> <gt.ibin>\n"
    "Options:\n"
    "  --K=10,20        (Recall@K list)\n"
    "  --L=50,100,200   (candidate pool size list)\n"
    "                   也支持网格语法: a:b 含端点步长1; a:s:b 为步长s (例如 50:50:200)\n"
    "  --runs=1         (repeat runs for avg)\n"
    "  --warmup=0       (warmup runs not counted)\n"
    "  --csv=out.csv    (csv output)\n"
    "  --entry_id=ID    (optional, fixed navigating node id; default uses centroid-nearest)\n"
    "  --S / --seed     (accepted for compatibility; ignored in fixed-entry mode)\n";
    return 1;
  }
  auto a = parse_args(argc, argv);
  std::string base_path = a["1"], nsg_path=a["2"], xq_path=a["3"], gt_path=a["4"];

  std::vector<int> Ks = a.count("K")? parse_int_list(a["K"]) : std::vector<int>{10};
  // L 支持网格语法（a:b 或 a:s:b），兼容原来的逗号分隔
  std::vector<int> Ls = a.count("L")? parse_int_list_or_grid(a["L"]) : std::vector<int>{50,100,200};
  int runs = a.count("runs")  ? std::stoi(a["runs"])  : 1;
  int warm = a.count("warmup")? std::stoi(a["warmup"]): 0;
  std::string csv = a.count("csv")? a["csv"] : "nsg_search_results.csv";

  std::cout<<"[INFO] loading base...\n";
  Data base = read_fbin(base_path);
  std::cout<<"[INFO] base: n="<<base.n<<" dim="<<base.dim<<"\n";

  std::cout<<"[INFO] loading nsg...\n";
  auto G = read_nsg(nsg_path);
  if(G.size()!=base.n) std::cerr<<"[WARN] NSG n="<<G.size()<<" != base n="<<base.n<<"\n";

  std::cout<<"[INFO] loading queries & gt...\n";
  Data xq = read_fbin(xq_path);
  GT gt = read_gt_ibin(gt_path);
  const u32 nq = std::min<u32>(xq.n, gt.nq);
  std::cout<<"[INFO] nq="<<nq<<" gt.k="<<gt.k<<"\n";

  // 决定固定入口
  u32 entry_id = 0;
  if(a.count("entry_id")){
    long long v = std::stoll(a["entry_id"]);
    if(v < 0 || (unsigned long long)v >= (unsigned long long)base.n)
      die("--entry_id out of range");
    entry_id = (u32)v;
    std::cout<<"[INFO] using user-specified entry_id="<<entry_id<<"\n";
  }else{
    auto t0 = std::chrono::high_resolution_clock::now();
    entry_id = compute_entry_centroid_nearest(base);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout<<"[INFO] using centroid-nearest entry_id="<<entry_id
             <<" (built in "<<std::chrono::duration<double,std::milli>(t1-t0).count()<<" ms)\n";
  }
  if(a.count("S") || a.count("seed")){
    std::cout<<"[INFO] fixed-entry mode: --S / --seed provided but ignored.\n";
  }

  // CSV header（为兼容保留 S 列；此处写 0 表示固定入口）
  {
    std::ofstream ofs(csv, std::ios::app);
    if(ofs.tellp()==0){
      ofs<<"base_path,nsg_path,xq_path,gt_path,K,L,S,runs,threads,nq,avg_qps,avg_ms_per_query,recall,entry_id\n";
    }
  }

  std::cout<<"[INFO] threads = 1 (no OpenMP)\n";

  // 统一以 Kmax 构造一次结果，再对各 K 复用（排序后取前缀），计时仅覆盖“搜索阶段”
  int Kmax = 0; for(int v : Ks) Kmax = std::max(Kmax, v);
  if(Kmax <= 0) Kmax = 1;

  for(int L: Ls){
    if((int)gt.k < Kmax) std::cerr<<"[WARN] Kmax="<<Kmax<<" > gt.k="<<gt.k<<", recall@K 用 gt 前 "<<gt.k<<" 评估。\n";

    // warmup：用 Kmax
    for(int r=0;r<warm;++r){
      for(u32 iq=0;iq<nq;++iq){
        const float* q = &xq.x[(size_t)iq*xq.dim];
        std::vector<std::pair<float,u32>> outp;
        nsg_search_one_fixed_entry(base, G, q, (u32)Kmax, (u32)L, entry_id, outp);
      }
    }

    double sum_ms = 0.0;
    std::vector<double> sum_recall(Ks.size(), 0.0);

    for(int r=0;r<runs;++r){
      auto t0 = std::chrono::high_resolution_clock::now();

      // 搜索阶段：构造每个查询的前 Kmax（已按距离升序），并仅保留 ID
      std::vector<u32> all_ids; all_ids.resize((size_t)nq*(size_t)Kmax);
      for(u32 iq=0;iq<nq;++iq){
        const float* q = &xq.x[(size_t)iq*xq.dim];
        std::vector<std::pair<float,u32>> outp;
        nsg_search_one_fixed_entry(base, G, q, (u32)Kmax, (u32)L, entry_id, outp);
        for(int j=0;j<Kmax; ++j){
          all_ids[(size_t)iq*(size_t)Kmax + j] = outp[j].second;
        }
      }

      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
      sum_ms += ms;

      // 后处理阶段：统计多个 K 的 recall（不计时）
      for(size_t ki=0; ki<Ks.size(); ++ki){
        int K = Ks[ki];
        int Kg = std::min(K, (int)gt.k);
        double hit=0.0, denom = (double)Kg * (double)nq;
        for(u32 iq=0;iq<nq;++iq){
          const int32_t* row = &gt.idx[(size_t)iq*(size_t)gt.k];
          for(int j=0;j<Kg;++j){
            int32_t g = row[j];
            // 在按距离排序后的 all_ids 前缀 K 中查找命中
            for(int t=0;t<K;++t){
              if((int32_t)all_ids[(size_t)iq*(size_t)Kmax + t] == g){ hit += 1.0; break; }
            }
          }
        }
        sum_recall[ki] += (denom>0? hit/denom : 0.0);
      }
    }

    double avg_ms = sum_ms / runs;
    double qps = (nq * 1000.0) / avg_ms;

    for(size_t ki=0; ki<Ks.size(); ++ki){
      int K = Ks[ki];
      double avg_recall = sum_recall[ki] / runs;

      std::cout<<"K="<<K<<" L="<<L<<" | nq="<<nq
               <<" | QPS="<<qps<<" | ms/q="<<(avg_ms/nq)
               <<" | Recall="<<avg_recall
               <<" | entry_id="<<entry_id<<"\n";

      std::ofstream ofs(csv, std::ios::app);
      ofs<<base_path<<","<<nsg_path<<","<<xq_path<<","<<gt_path<<"," 
         <<K<<","<<L<<","<<0/*S ignored*/<<","<<runs<<"," 
         <<1/*threads*/<<","<<nq<<","<<qps<<","<<(avg_ms/nq)<<","<<avg_recall<<","<<entry_id<<"\n";
    }
  }

  std::cout<<"Done. Results -> "<<csv<<"\n";
  return 0;
}
