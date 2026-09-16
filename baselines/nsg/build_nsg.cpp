// baselinensg.cpp
// Build NSG from an external kNN graph (baseline, single-file, OpenMP-accelerated).
// CLI:
//   ./baselinensg <data_dir> <out_nsg> <knn_graph_path> [R=64] [C=500] [L=500]
//
// Output .nsg format:
//   uint32 n;
//   repeat n times: uint32 deg; uint32 neighbors[deg];
//
// kNN .graph supported formats (auto-detect):
//   A) [uint32 n][uint32 K][n*K * uint32 neighbors]  // fixed K
//   B) [uint32 n][ per-node: uint32 deg; deg * uint32 neighbors ] // variable deg
//
// Data .fbin format: [int32 n][int32 dim][n*dim * float]
//
// Build steps (per node u):
//   1) Gather candidates up to L via: kNN[u] ∪ kNN[neighbors of u] (去重)；
//   2) 若候选 < C，继续二跳扩展或随机补齐；
//   3) Robust prune：按 dist(u,·) 升序，迭代选入，若存在已选 p 使 dist(p,v) < dist(u,v) 则丢弃；直到 R；
////////////////////////////////////////////////////////////////////////////////
// g++ -O3 -std=c++17 -march=native -fopenmp build_nsg.cpp -o baseline_nsg
// # 目录结构示例：
// #   data/
// #     base.fbin   (或 base.2.5M.fbin / base.10M.fbin)
// #   knn/
// #     base.graph  (固定K或逐点度的任意一种格式)
// ./baseline_nsg /home/zhangkai/RoarGraph-main/data/clip-webvid-2.5M/base.2.5M.fbin ../results/toy1k.nsg ../results/knng/clip-webvid-2.5M/base.100NN.graph  64 200 200 --K=50

// baselinensg.cpp
// Build NSG from an external kNN graph (baseline, single-file, OpenMP-accelerated).
// CLI:
//   ./baselinensg <data_dir_or_fbin> <out_nsg> <knn_graph_or_dir> [R=64] [C=500] [L=500]
//
// Output .nsg format:
//   uint32 n;
//   repeat n times: uint32 deg; uint32 neighbors[deg];
//
// kNN .graph supported formats (auto-detect):
//   A) [uint32 n][uint32 K][n*K * uint32 neighbors]  // fixed K
//   B) [uint32 n][ per-node: uint32 deg; deg * uint32 neighbors ] // variable deg
//
// Data .fbin format: [int32 n][int32 dim][n*dim * float]
//
// Build steps (per node u):
//   1) Gather candidates up to L via: kNN[u] ∪ kNN[neighbors of u] (去重)；
//   2) 若候选 < C，继续二跳扩展或邻居扩展；
//   3) Robust prune：按 dist(u,·) 升序，迭代选入，若存在已选 p 使 dist(p,v) < dist(u,v) 则丢弃；直到 R。
////////////////////////////////////////////////////////////////////////////////

// baselinensg.cpp (robust header detection + OMP + bound checks)
// CLI:
//   ./baselinensg <data_dir_or_fbin> <out_nsg> <knn_graph_or_dir> [R=64] [C=500] [L=500]

// baselinensg.cpp
// Build NSG from an external kNN graph. Robust .graph parser + OpenMP.
// CLI:
//   ./baselinensg <data_dir_or_fbin> <out_nsg> <knn_graph_or_dir> [R=64] [C=500] [L=500] [--K=50] [--layout=auto|flat|pernode] [--header=auto|NK|KN]

// baselinensg.cpp
// Build NSG from an external kNN graph. Robust .graph parser + OpenMP.
// CLI:
//   ./baselinensg <data_dir_or_fbin> <out_nsg> <knn_graph_or_dir> [R=64] [C=500] [L=500]
//                  [--K=50|--k=50] [--layout=auto|flat|pernode] [--header=auto|NK|KN]
//
// 支持 .graph 二进制布局（均为小端 u32）：
// F1: [n][K] + 扁平邻接（紧随 n*K 个 u32）               => 总大小 8 + n*K*4
// F2: [K][n] + 扁平邻接                                 => 同上
// F3: [K]    + 扁平邻接（仅 K 头，常见导出）            => 总大小 4 + n*K*4  ★新增
// G1: [n][K] + 逐点固定 K：每点 [K][K 个邻居]           => 总大小 8 + n*(1+K)*4
// G2: [K][n] + 逐点固定 K：每点 [K][K 个邻居]           => 同上
// V : [n]    + 逐点可变度：每点 [deg][deg 个邻居]       => 大小不定
//
// 解析优先级：若指定 --K，则优先用 (expected_n, K) 与文件大小匹配上述布局；
// 否则根据文件头与大小自动判定；均失败则回退至可变度格式。

// baselinensg.cpp
// Build NSG from an external kNN graph. Robust parser + filename K detection + self-loop trim + OpenMP.
//
// 用法：
//   ./baseline_nsg <data_dir_or_fbin> <out_nsg> <knn_graph_path_or_dir> [R=64] [C=500] [L=500]
// 支持额外可选：--K=50 / --k=50（显式覆盖文件名 K）
//
// 解析支持（小端 u32）：
//  F1: [n][K] + flat  ：总大小 = 8 + n*K*4
//  F2: [K][n] + flat  ：同上
//  F3: [K]    + flat  ：总大小 = 4 + n*K*4   （K-only，很多导出器使用）
//  G1: [n][K] + per-node：8 + n*(1+K)*4
//  G2: [K][n] + per-node：同上
//   V: [n]    + variable degree（兜底）
//
// 额外增强：若检测到实际 K == K_hint+1（常见于含自环），会按行优先丢掉自环（v==i），否则截断到 K_hint。

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using u32 = uint32_t;
using u64 = uint64_t;

struct Data { u32 n=0, dim=0; std::vector<float> x; };

static void die(const std::string& m){ std::cerr<<"[ERR] "<<m<<"\n"; std::exit(1); }
static bool exists_file(const std::string& p){
  std::error_code ec; return std::filesystem::exists(p,ec)&&std::filesystem::is_regular_file(p,ec);
}

// ---------------- 路径解析 ----------------
static std::string resolve_base_path(const std::string& arg){
  std::error_code ec;
  if (std::filesystem::exists(arg,ec)&&std::filesystem::is_regular_file(arg,ec)) return arg;
  for (auto n: { "base.fbin","base.2.5M.fbin","base.10M.fbin" }){
    std::string p = arg + "/" + n;
    if (std::filesystem::exists(p,ec)&&std::filesystem::is_regular_file(p,ec)) return p;
  }
  return arg + "/base.fbin";
}
static std::string resolve_graph_path(const std::string& arg){
  std::error_code ec;
  if (std::filesystem::exists(arg,ec)&&std::filesystem::is_regular_file(arg,ec)) return arg;
  if (std::filesystem::exists(arg,ec)&&std::filesystem::is_directory(arg,ec)){
    for (auto& e: std::filesystem::directory_iterator(arg)){
      if (e.is_regular_file() && e.path().extension()==".graph") return e.path().string();
    }
  }
  return arg;
}

// ---------------- base.fbin ----------------
static Data read_fbin(const std::string& path){
  FILE* fp = std::fopen(path.c_str(),"rb");
  if(!fp){ perror(path.c_str()); die("open fbin failed: "+path); }
  int n=0,d=0;
  if (std::fread(&n,4,1,fp)!=1 || std::fread(&d,4,1,fp)!=1){ std::fclose(fp); die("bad fbin header"); }
  if (n<=0||d<=0){ std::fclose(fp); die("invalid fbin header values"); }
  Data data; data.n=(u32)n; data.dim=(u32)d;
  data.x.resize((size_t)n*(size_t)d);
  size_t need=(size_t)n*(size_t)d;
  if (std::fread(data.x.data(), sizeof(float), need, fp)!=need){ std::fclose(fp); die("read fbin body failed"); }
  std::fclose(fp); return data;
}

// ---------------- 从文件名提取 K ----------------
// 规则示例：xxx/BASE.50NN.graph、BASE.100Knn.graph、foo/bar.128knn.graph
static int K_from_filename(const std::string& path){
  std::string fname = std::filesystem::path(path).filename().string();
  std::string s = fname;
  for (auto& c: s) c = char(std::tolower((unsigned char)c));
  // 寻找 "nn" 的位置；优先解析紧挨着 "nn" 前的连续数字；允许 'knn'
  size_t pos = s.find("nn");
  if (pos != std::string::npos){
    // 允许 k 在 nn 前
    size_t j = pos;
    if (j>=1 && s[j-1]=='k') j--;
    // 向左收集连续数字
    size_t end = j;
    size_t start = end;
    while (start>0 && std::isdigit((unsigned char)s[start-1])) --start;
    if (start < end){
      int K = std::stoi(s.substr(start, end-start));
      if (K>0) return K;
    }
  }
  // 兜底：取最后一段“纯数字”
  int last_num = -1;
  size_t i = s.size();
  while (i>0){
    // 跳过非数字
    while (i>0 && !std::isdigit((unsigned char)s[i-1])) --i;
    if (i==0) break;
    size_t e = i;
    while (i>0 && std::isdigit((unsigned char)s[i-1])) --i;
    int v = std::stoi(s.substr(i, e-i));
    if (v>0){ last_num = v; break; }
  }
  return last_num; // 可能是 -1
}

// ---------------- 解析提示 ----------------
enum class Layout { AUTO, FLAT, PERNODE };
struct GraphHints {
  int  K_hint = -1;          // 文件名/命令行的 K
  Layout layout = Layout::AUTO;
};

// ---------------- .graph 读取器 ----------------
static std::vector<std::vector<u32>>
read_knn_graph(const std::string& path, u32 expected_n, const GraphHints& hints){
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) die("open knn_graph failed: "+path);
  const u64 fsz = std::filesystem::file_size(path);
  auto fail = [&](const std::string& m){ die("read_knn_graph: "+m+" ("+path+")"); };

  auto fits_flat_sz     = [&](u32 n,u32 K,u64 hdr)->bool{ return n>0&&K>0 && fsz == hdr + (u64)n*(u64)K*4; };
  auto fits_pernode_sz  = [&](u32 n,u32 K,u64 hdr)->bool{ return n>0&&K>0 && fsz == hdr + (u64)n*(u64)(1+K)*4; };

  // 支持“自环修剪”：若检测到实际K == K_hint+1，则每行优先丢掉 self（v==i），否则截断。
  auto build_from_flat_with_optional_trim = [&](u64 offset, u32 n, u32 K, int K_hint)
      -> std::vector<std::vector<u32>> {
    ifs.clear(); ifs.seekg((std::streamoff)offset, std::ios::beg);
    std::vector<u32> buf((size_t)n*(size_t)K);
    ifs.read(reinterpret_cast<char*>(buf.data()), buf.size()*sizeof(u32));
    if (!ifs) fail("flat body read failed");

    std::vector<std::vector<u32>> G(n);
    bool need_trim = (K_hint>0 && K == (u32)(K_hint+1));
    if (!need_trim){
      for (u32 i=0;i<n;++i){
        G[i].reserve(K);
        for (u32 j=0;j<K;++j) G[i].push_back(buf[(size_t)i*K + j]);
      }
      std::cerr<<"[INFO] Parsed fixed-K(flat) graph n="<<n<<" K="<<K<<" offset="<<offset
               <<" (expected_n="<<expected_n<<")\n";
      return G;
    }

    // 修剪到 K_hint
    u32 Kh = (u32)K_hint;
    for (u32 i=0;i<n;++i){
      const u32* row = &buf[(size_t)i*K];
      std::vector<u32> tmp; tmp.reserve(K);
      bool dropped = false;
      for (u32 j=0;j<K;++j){
        u32 v = row[j];
        if (!dropped && v == i){ dropped = true; continue; } // 丢掉自环
        tmp.push_back(v);
      }
      if (tmp.size() > Kh) tmp.resize(Kh); // 如果没有自环，简单截断
      G[i] = std::move(tmp);
    }
    std::cerr<<"[INFO] Parsed fixed-K(flat) graph n="<<n<<" K="<<K<<" offset="<<offset
             <<"; trimmed to K_hint="<<Kh<<" by dropping self/truncation.\n";
    return G;
  };

  auto parse_pernode = [&](u64 offset, u32 n, u32 K, int K_hint)->std::vector<std::vector<u32>>{
    ifs.clear(); ifs.seekg((std::streamoff)offset, std::ios::beg);
    std::vector<std::vector<u32>> G(n);
    bool need_trim = (K_hint>0 && K == (u32)(K_hint+1));
    for(u32 i=0;i<n;++i){
      u32 ki=0; ifs.read(reinterpret_cast<char*>(&ki), sizeof(u32));
      if(!ifs) fail("pernode K read failed at node "+std::to_string(i));
      u32 take = ki;
      // 若需要修剪，则优先丢掉自环，再保证最多 K_hint 个
      if (need_trim && ki >= (u32)K_hint){
        std::vector<u32> row(ki);
        ifs.read(reinterpret_cast<char*>(row.data()), ki*sizeof(u32));
        if(!ifs) fail("pernode neighbors read failed at node "+std::to_string(i));
        std::vector<u32> tmp; tmp.reserve(ki);
        bool dropped=false;
        for (u32 v: row){ if (!dropped && v==i){ dropped=true; continue; } tmp.push_back(v); }
        if ((int)tmp.size() > K_hint) tmp.resize((size_t)K_hint);
        G[i] = std::move(tmp);
        continue; // 下一点
      }
      // 无需修剪：常规读取
      take = std::min(ki, K);
      G[i].resize(take);
      if (take){
        ifs.read(reinterpret_cast<char*>(G[i].data()), take*sizeof(u32));
        if(!ifs) fail("pernode neighbors read failed at node "+std::to_string(i));
      }
      if (ki>take){ ifs.seekg((std::streamoff)((ki-take)*sizeof(u32)), std::ios::cur);
        if(!ifs) fail("skip extra neighbors failed");
      }
    }
    std::cerr<<"[INFO] Parsed fixed-K(per-node) graph n="<<n<<" K="<<K<<" offset="<<offset
             << ( (K_hint>0 && K==(u32)(K_hint+1)) ? "; trimmed to K_hint" : "" )
             <<"\n";
    return G;
  };

  auto parse_vardeg = [&]()->std::vector<std::vector<u32>>{
    ifs.clear(); ifs.seekg(0, std::ios::beg);
    u32 n=0; ifs.read(reinterpret_cast<char*>(&n), sizeof(u32));
    if(!ifs) fail("vardeg read n failed");
    std::vector<std::vector<u32>> G(n);
    for(u32 i=0;i<n;++i){
      u32 deg=0; ifs.read(reinterpret_cast<char*>(&deg), sizeof(u32));
      if(!ifs) fail("vardeg read deg failed at node "+std::to_string(i));
      G[i].resize(deg);
      if (deg){
        ifs.read(reinterpret_cast<char*>(G[i].data()), deg*sizeof(u32));
        if(!ifs) fail("vardeg read neighbors failed at node "+std::to_string(i));
      }
    }
    std::cerr<<"[INFO] Parsed variable-degree graph n="<<n<<" (expected_n="<<expected_n<<")\n";
    return G;
  };

  // 读取前两个 u32（用于 NK/KN 判断；K-only 情况下 b 可能不是 n）
  u32 a=0,b=0;
  ifs.read(reinterpret_cast<char*>(&a), sizeof(u32));
  if (!ifs) fail("cannot read first u32");
  ifs.read(reinterpret_cast<char*>(&b), sizeof(u32));
  if (!ifs) { b=0; ifs.clear(); }

  // 1) 优先按“文件名/命令行 K_hint + expected_n”匹配
  if (hints.K_hint > 0){
    u32 n = expected_n, K = (u32)hints.K_hint;
    // K-only flat
    if (fits_flat_sz(n, K, 4)) return build_from_flat_with_optional_trim(4, n, K, hints.K_hint);
    // 常规 flat
    if (fits_flat_sz(n, K, 8)) return build_from_flat_with_optional_trim(8, n, K, hints.K_hint);
    // per-node
    if (fits_pernode_sz(n, K, 8)) return parse_pernode(8, n, K, hints.K_hint);
    // 检测 K+1 情况（K-only flat）
    if (fits_flat_sz(n, K+1, 4)) return build_from_flat_with_optional_trim(4, n, K+1, hints.K_hint);
    if (fits_flat_sz(n, K+1, 8)) return build_from_flat_with_optional_trim(8, n, K+1, hints.K_hint);
    if (fits_pernode_sz(n, K+1, 8)) return parse_pernode(8, n, K+1, hints.K_hint);
    std::cerr << "[WARN] Filename/flag K="<<K<<" 与 expected_n="<<n<<" 未匹配已知布局（文件大小="<<fsz<<"），尝试自动识别。\n";
  }

  // 2) 自动识别（NK/KN + flat/pernode + K-only）
  struct Cand { u32 n,K; u64 hdr; int kind; int score; const char* tag; }; // kind: 0=flat,1=pernode
  std::vector<Cand> cs;
  auto push_if = [&](u32 n,u32 K,u64 hdr,int kind,const char* tag){
    bool ok = (kind==0? fits_flat_sz(n,K,hdr): fits_pernode_sz(n,K,hdr));
    if (!ok) return;
    int sc=0;
    if ((int)n==(int)expected_n) sc+=1000;
    if (n>=K) sc+=10;
    if (K<=65536) sc+=5;
    cs.push_back({n,K,hdr,kind,sc,tag});
  };

  // NK/KN
  push_if(a,b,8,0,"flat NK");     push_if(a,b,8,1,"pernode NK");
  push_if(b,a,8,0,"flat KN");     push_if(b,a,8,1,"pernode KN");
  // K-only flat（优先用 a、b 分别当 K 试一遍）
  if (expected_n>0){
    push_if(expected_n, a, 4, 0, "flat K-only(a)");
    if (b>0) push_if(expected_n, b, 4, 0, "flat K-only(b)");
  }

  if (!cs.empty()){
    std::sort(cs.begin(), cs.end(), [](const Cand& x,const Cand& y){ return x.score>y.score; });
    const auto c = cs.front();
    if (c.kind==0) return build_from_flat_with_optional_trim(c.hdr, c.n, c.K, hints.K_hint);
    else           return parse_pernode(c.hdr, c.n, c.K, hints.K_hint);
  }

  // 3) 按文件大小反推（K-only / flat / pernode）
  if (expected_n>0){
    // K-only flat：4 + n*K*4
    if (fsz > 4){
      u64 rem = fsz - 4;
      if (rem % (4ull*expected_n) == 0){
        u32 K = (u32)( rem / (4ull*expected_n) );
        return build_from_flat_with_optional_trim(4, expected_n, K, hints.K_hint);
      }
    }
    // flat：8 + n*K*4
    if (fsz > 8){
      u64 rem = fsz - 8;
      if (rem % (4ull*expected_n) == 0){
        u32 K = (u32)( rem / (4ull*expected_n) );
        return build_from_flat_with_optional_trim(8, expected_n, K, hints.K_hint);
      }
      // pernode：8 + n*(1+K)*4
      u64 q = (fsz - 8) / 4 / expected_n;
      if (8 + (u64)expected_n * q * 4 == fsz && q>=1){
        u32 K = (u32)(q-1);
        return parse_pernode(8, expected_n, K, hints.K_hint);
      }
    }
  }

  // 4) 兜底：variable degree
  return parse_vardeg();
}

// ---------------- 距离 & 构图 ----------------
static inline float l2sqr(const float* a, const float* b, u32 dim){
  float s=0.0f;
  #pragma omp simd reduction(+:s)
  for(u32 i=0;i<dim;++i){ float d=a[i]-b[i]; s+=d*d; }
  return s;
}
struct CandNode { u32 id; float dist; };

static void collect_candidates(u32 u, const std::vector<std::vector<u32>>& knn, u32 target_L, u32 data_n, std::vector<u32>& out){
  out.clear(); out.reserve(target_L*2);
  std::unordered_set<u32> seen; seen.reserve(target_L*4);
  auto add = [&](u32 v){ if(v==u) return; if(v>=data_n) return; if(seen.insert(v).second) out.push_back(v); };

  if (u<knn.size()){
    for(u32 v: knn[u]){ add(v); if(out.size()>=target_L) break; }
  }
  if (out.size()<target_L && u<knn.size()){
    for(u32 v: knn[u]){
      if(v<knn.size()){
        for(u32 w: knn[v]){ add(w); if(out.size()>=target_L) break; }
      }
      if(out.size()>=target_L) break;
    }
  }
  if (out.size()<target_L){
    for(u32 v: out){ if(v<knn.size()){ for(u32 w: knn[v]){ add(w); if(out.size()>=target_L) break; } } if(out.size()>=target_L) break; }
  }
  if (out.size()<target_L){ for(u32 v=0; v<data_n && out.size()<target_L; ++v) add(v); }
}

static void robust_prune_one(u32 u, const Data& data, const std::vector<u32>& cands, u32 R, u32 C,
                             std::vector<u32>& out_neighbors, std::vector<CandNode>& buf){
  out_neighbors.clear();
  if (cands.empty()) return;
  const float* pu = &data.x[(size_t)u*data.dim];

  buf.clear(); buf.reserve(cands.size());
  for(u32 v: cands){ if(v>=data.n) continue; const float* pv=&data.x[(size_t)v*data.dim]; buf.push_back({v, l2sqr(pu,pv,data.dim)}); }
  if (buf.empty()) return;

  std::sort(buf.begin(), buf.end(), [](const CandNode& a, const CandNode& b){ return a.dist < b.dist; });
  if (buf.size()>C) buf.resize(C);

  out_neighbors.reserve(std::min<u32>(R,(u32)buf.size()));
  for(auto &cn: buf){
    u32 v=cn.id; const float* pv=&data.x[(size_t)v*data.dim]; bool occ=false;
    for(u32 p: out_neighbors){ const float* pp=&data.x[(size_t)p*data.dim]; if(l2sqr(pp,pv,data.dim) < cn.dist){ occ=true; break; } }
    if(!occ){ out_neighbors.push_back(v); if(out_neighbors.size()>=R) break; }
  }
}

// 使对称（限度 R）
static void symmetrize_with_cap(std::vector<std::vector<u32>>& G, u32 R){
  const u32 n=(u32)G.size();
  std::vector<std::vector<u32>> rev(n);
  #pragma omp parallel
  {
    std::vector<std::pair<u32,u32>> local; local.reserve(1024);
    #pragma omp for schedule(dynamic,1024) nowait
    for(int u=0; u<(int)n; ++u){ for(u32 v: G[u]) if(v<n) local.emplace_back(v,(u32)u); }
    #pragma omp critical
    { for(auto &e: local) rev[e.first].push_back(e.second); }
  }
  #pragma omp parallel for schedule(dynamic,1024)
  for(int v=0; v<(int)n; ++v){
    auto& dst=G[v]; std::unordered_set<u32> has(dst.begin(), dst.end());
    for(u32 u: rev[v]){ if(dst.size()>=R) break; if(has.insert(u).second) dst.push_back(u); }
    if (dst.size()>R) dst.resize(R);
  }
}

// 简单连通性修复
static void connectivity_repair(const Data& data, const std::vector<std::vector<u32>>& knn, u32 R, std::vector<std::vector<u32>>& G){
  const u32 n=(u32)G.size(); if(!n) return;
  std::vector<char> vis(n,0); std::vector<u32> q; q.reserve(n);
  auto bfs=[&](u32 s, std::vector<u32>& comp){
    comp.clear(); q.clear(); q.push_back(s); vis[s]=1; comp.push_back(s);
    for(size_t qi=0; qi<q.size(); ++qi){ u32 u=q[qi]; for(u32 v: G[u]) if(v<n && !vis[v]){ vis[v]=1; q.push_back(v); comp.push_back(v);} }
  };
  std::vector<std::vector<u32>> comps; comps.reserve(16); std::vector<u32> comp;
  for(u32 i=0;i<n;++i) if(!vis[i]){ bfs(i,comp); comps.push_back(comp); }
  if (comps.size()<=1) return;

  std::unordered_set<u32> mainset(comps[0].begin(), comps[0].end());
  for(size_t ci=1; ci<comps.size(); ++ci){
    float best=std::numeric_limits<float>::infinity(); u32 bu=comps[ci][0], bv=comps[0][0];
    for(u32 u: comps[ci]){
      const float* pu=&data.x[(size_t)u*data.dim];
      if(u<knn.size()){
        for(u32 v: knn[u]) if(v<n && mainset.count(v)){
          const float* pv=&data.x[(size_t)v*data.dim]; float d=l2sqr(pu,pv,data.dim);
          if(d<best){ best=d; bu=u; bv=v; }
        }
      }
    }
    if (std::find(G[bu].begin(), G[bu].end(), bv)==G[bu].end()){ if(G[bu].size()<R) G[bu].push_back(bv); else G[bu][rand()%R]=bv; }
    if (std::find(G[bv].begin(), G[bv].end(), bu)==G[bv].end()){ if(G[bv].size()<R) G[bv].push_back(bu); else G[bv][rand()%R]=bu; }
    for(u32 u: comps[ci]) mainset.insert(u);
  }
}

static void save_nsg(const std::string& path, const std::vector<std::vector<u32>>& G){
  FILE* fp=std::fopen(path.c_str(),"wb"); if(!fp){ perror(path.c_str()); die("open out_nsg failed"); }
  u32 n=(u32)G.size(); if(std::fwrite(&n,sizeof(u32),1,fp)!=1) die("write n failed");
  for(u32 i=0;i<n;++i){
    u32 deg=(u32)G[i].size(); if(std::fwrite(&deg,sizeof(u32),1,fp)!=1) die("write deg failed");
    if (deg && std::fwrite(G[i].data(), sizeof(u32), deg, fp)!=deg) die("write neighbors failed");
  }
  std::fclose(fp);
}

// ---------------- 主程序 ----------------
int main(int argc, char** argv){
  if (argc < 4){
    std::cerr<<"Usage:\n  "<<argv[0]<<" <data_dir_or_fbin> <out_nsg> <knn_graph_or_dir> [R=64] [C=500] [L=500] [--K=50|--k=50]\n";
    return 1;
  }
  std::string data_arg=argv[1], out_nsg=argv[2], knn_arg=argv[3];
  u32 R = (argc>4)? (u32)std::stoul(argv[4]):64;
  u32 C = (argc>5)? (u32)std::stoul(argv[5]):500;
  u32 L = (argc>6)? (u32)std::stoul(argv[6]):500;

  GraphHints hints;
  for(int i=7;i<argc;++i){
    std::string s=argv[i];
    if (s.rfind("--K=",0)==0 || s.rfind("--k=",0)==0) hints.K_hint = std::stoi(s.substr(4));
  }

  std::string base_path = resolve_base_path(data_arg);
  if (!exists_file(base_path)){
    std::cerr<<"[ERR] base fbin not found.\n  You passed: "<<data_arg<<"\n  I tried: "<<base_path<<"\n"; return 2;
  }
  std::cout<<"[INFO] base_path: "<<base_path<<"\n";
  auto data = read_fbin(base_path);
  std::cout<<"[INFO] base loaded: n="<<data.n<<" dim="<<data.dim<<"\n";

  std::string knn_path = resolve_graph_path(knn_arg);
  if (!exists_file(knn_path)){ std::cerr<<"[ERR] knn_graph not found.\n  You passed: "<<knn_arg<<"\n  I tried: "<<knn_path<<"\n"; return 3; }
  std::cout<<"[INFO] knn_graph: "<<knn_path<<"\n";

  // 从文件名提取 K（若命令行未给）
  if (hints.K_hint <= 0){
    hints.K_hint = K_from_filename(knn_path);
    if (hints.K_hint > 0) {
      std::cout << "[INFO] K inferred from filename: " << hints.K_hint << "\n";
    } else {
      std::cout << "[WARN] Cannot infer K from filename. Will auto-detect by size.\n";
    }
  } else {
    std::cout << "[INFO] K overridden by flag: " << hints.K_hint << "\n";
  }

  auto knn = read_knn_graph(knn_path, data.n, hints);
  if (knn.size()!=data.n){
    std::cerr<<"[WARN] kNN graph n="<<knn.size()<<" != data n="<<data.n<<". Build will proceed safely, but quality may be impacted.\n";
  }
  const u32 n = data.n;
  std::vector<std::vector<u32>> G(n);

  auto t0 = std::chrono::high_resolution_clock::now();
  #pragma omp parallel
  {
    std::vector<u32> cands; std::vector<CandNode> buf; std::vector<u32> nei; nei.reserve(R);
    #pragma omp for schedule(dynamic,64)
    for(int iu=0; iu<(int)n; ++iu){
      u32 u=(u32)iu;
      collect_candidates(u, knn, L, data.n, cands);
      robust_prune_one(u, data, cands, R, C, nei, buf);
      G[u]=nei;
    }
  }
  symmetrize_with_cap(G, R);
  connectivity_repair(data, knn, R, G);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout<<"[INFO] NSG built in "<<std::chrono::duration<double>(t1-t0).count()<<" s\n";

  std::error_code ec; std::filesystem::create_directories(std::filesystem::path(out_nsg).parent_path(), ec);
  if (ec) std::cerr<<"[WARN] create_directories: "<<ec.message()<<"\n";
  std::cout<<"[INFO] Saving to: "<<out_nsg<<"\n";
  save_nsg(out_nsg, G);
  std::cout<<"Done.\n";
  return 0;
}




