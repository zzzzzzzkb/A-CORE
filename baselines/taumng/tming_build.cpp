// tm_mrng.cpp
// Build t-ming MRNG graph on Linux from an existing kNN graph.
// - Robust .graph parser (K-only / [n][K] flat / [K][n] flat / per-node / variable-degree)
// - K inferred from filename like base.50NN.graph (overridable via --K=50/--k=50)
// - If detected K == K_hint+1, auto drop self-loop or truncate back to K_hint
// - MRNG pruning with tau: RNG threshold uses thr = (1+tau)^2 * d(u,v)  (squared L2)
// - OpenMP parallel
//
// CLI:
//   tm_mrng <data_dir_or_fbin> <out_mrng> <knn_graph_path_or_dir> [R=64] [L=200] [two_hop=1] [--K=50|--k=50] [--tau=0.01]
//
// Data .fbin (little-endian):
//   [int32 n][int32 dim][n*dim floats]
//
// Output .mrng (little-endian):
//   [uint32 n] then per-node: [uint32 deg][deg * uint32 neighbors]
//
// Example:
//   ./tm_mrng /data/toy-1k/base.fbin \
//             /data/out/toy1k.mrng \
//             /data/toy-1k/base.50NN.graph \
//             64 200 1 --tau=0.01 --K=50
//# 可显式指定 K（推荐和你的 50NN 对齐）
// ./tm_mrng ../data/clip-webvid-2.5M/base.2.5Mfbin ../data/out/clip-webvid-2.5M.mrng ../results/knng/clip-webvid-2.5M/base.100NN.graph 64 200 1 --tau=0.01 

// Ported for Linux. Self-contained single translation unit.

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
#include <numeric>
#include <string>
#include <unordered_set>  
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _MSC_VER
#pragma warning(disable:4996) // fopen on MSVC
#endif

using u32 = uint32_t;
using u64 = uint64_t;

struct Data { u32 n=0, dim=0; std::vector<float> x; };

static void die(const std::string& m){ std::cerr<<"[ERR] "<<m<<"\n"; std::exit(1); }
static bool exists_file(const std::string& p){
  std::error_code ec; return std::filesystem::exists(p,ec)&&std::filesystem::is_regular_file(p,ec);
}

// ---------- path helpers ----------
static std::string resolve_base_path(const std::string& arg){
  std::error_code ec;
  if (std::filesystem::exists(arg,ec)&&std::filesystem::is_regular_file(arg,ec)) return arg;
  for (auto n: { "base.fbin","base.2.5M.fbin","base.10M.fbin" }){
    std::string p = (std::filesystem::path(arg) / n).string();
    if (std::filesystem::exists(p,ec)&&std::filesystem::is_regular_file(p,ec)) return p;
  }
  return (std::filesystem::path(arg) / "base.fbin").string();
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

// ---------- read base.fbin ----------
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

// ---------- infer K from filename: <name>.<K>NN.graph / <K>Knn.graph ----------
static int K_from_filename(const std::string& path){
  std::string fname = std::filesystem::path(path).filename().string();
  std::string s = fname; for (auto& c: s) c = char(std::tolower((unsigned char)c));
  size_t pos = s.find("nn");
  if (pos != std::string::npos){
    size_t j = pos; if (j>=1 && s[j-1]=='k') j--;
    size_t end=j, start=end;
    while (start>0 && std::isdigit((unsigned char)s[start-1])) --start;
    if (start<end){
      int K = std::stoi(s.substr(start, end-start));
      if (K>0) return K;
    }
  }
  // fallback: last numeric chunk
  int last_num = -1; size_t i = s.size();
  while (i>0){
    while (i>0 && !std::isdigit((unsigned char)s[i-1])) --i;
    if (i==0) break;
    size_t e=i; while (i>0 && std::isdigit((unsigned char)s[i-1])) --i;
    int v = std::stoi(s.substr(i, e-i));
    if (v>0){ last_num=v; break; }
  }
  return last_num;
}

enum class Layout { AUTO, FLAT, PERNODE };
struct GraphHints { int K_hint=-1; Layout layout=Layout::AUTO; };

// ---------- robust .graph reader ----------
static std::vector<std::vector<u32>>
read_knn_graph(const std::string& path, u32 expected_n, const GraphHints& hints){
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) die("open knn_graph failed: "+path);
  std::error_code ec;
  const u64 fsz = std::filesystem::file_size(path, ec);
  if (ec) die("file_size failed for "+path);

  auto fail = [&](const std::string& m){ die("read_knn_graph: "+m+" ("+path+")"); };

  auto fits_flat_sz     = [&](u32 n,u32 K,u64 hdr)->bool{ return n>0&&K>0 && fsz == hdr + (u64)n*(u64)K*4; };
  auto fits_pernode_sz  = [&](u32 n,u32 K,u64 hdr)->bool{ return n>0&&K>0 && fsz == hdr + (u64)n*(u64)(1+K)*4; };

  auto build_flat_with_optional_trim = [&](u64 offset, u32 n, u32 K, int K_hint)
      -> std::vector<std::vector<u32>> {
    ifs.clear(); ifs.seekg((std::streamoff)offset, std::ios::beg);
    std::vector<u32> buf((size_t)n*(size_t)K);
    ifs.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)(buf.size()*sizeof(u32)));
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
    // trim to K_hint: drop self if present, else truncate
    u32 Kh = (u32)K_hint;
    for (u32 i=0;i<n;++i){
      const u32* row = &buf[(size_t)i*K];
      std::vector<u32> tmp; tmp.reserve(K);
      bool dropped=false;
      for (u32 j=0;j<K;++j){
        u32 v = row[j];
        if (!dropped && v==i){ dropped=true; continue; }
        tmp.push_back(v);
      }
      if (tmp.size() > Kh) tmp.resize(Kh);
      G[i] = std::move(tmp);
    }
    std::cerr<<"[INFO] Parsed fixed-K(flat) n="<<n<<" K="<<K<<" offset="<<offset
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
      if (need_trim && ki >= (u32)K_hint){
        std::vector<u32> row(ki);
        ifs.read(reinterpret_cast<char*>(row.data()), (std::streamsize)(ki*sizeof(u32)));
        if(!ifs) fail("pernode neighbors read failed at node "+std::to_string(i));
        std::vector<u32> tmp; tmp.reserve(ki);
        bool dropped=false; for (u32 v: row){ if(!dropped && v==i){ dropped=true; continue; } tmp.push_back(v); }
        if ((int)tmp.size() > K_hint) tmp.resize((size_t)K_hint);
        G[i] = std::move(tmp);
      } else {
        u32 take = std::min(ki, K);
        G[i].resize(take);
        if (take){
          ifs.read(reinterpret_cast<char*>(G[i].data()), (std::streamsize)(take*sizeof(u32)));
          if(!ifs) fail("pernode neighbors read failed at node "+std::to_string(i));
        }
        if (ki>take){
          ifs.seekg((std::streamoff)((ki-take)*sizeof(u32)), std::ios::cur);
          if(!ifs) fail("skip extra neighbors failed");
        }
      }
    }
    std::cerr<<"[INFO] Parsed fixed-K(per-node) n="<<n<<" K="<<K<<" offset="<<offset
             << (need_trim? "; trimmed to K_hint" : "") <<"\n";
    return G;
  };

  auto parse_vardeg = [&]()->std::vector<std::vector<u32>>{
    ifs.clear(); ifs.seekg(0, std::ios::beg);
    u32 n=0; ifs.read(reinterpret_cast<char*>(&n), sizeof(u32)); if(!ifs) fail("vardeg read n failed");
    std::vector<std::vector<u32>> G(n);
    for(u32 i=0;i<n;++i){
      u32 deg=0; ifs.read(reinterpret_cast<char*>(&deg), sizeof(u32)); if(!ifs) fail("vardeg read deg failed");
      G[i].resize(deg);
      if (deg){
        ifs.read(reinterpret_cast<char*>(G[i].data()), (std::streamsize)(deg*sizeof(u32)));
        if(!ifs) fail("vardeg read neighbors failed at node "+std::to_string(i));
      }
    }
    std::cerr<<"[INFO] Parsed variable-degree graph n="<<n<<" (expected_n="<<expected_n<<")\n";
    return G;
  };

  // read first two u32 (for NK/KN guess; in K-only case, 'b' may be arbitrary)
  u32 a=0,b=0; ifs.read(reinterpret_cast<char*>(&a), sizeof(u32));
  if (!ifs) fail("cannot read first u32");
  ifs.read(reinterpret_cast<char*>(&b), sizeof(u32)); if (!ifs) { b=0; ifs.clear(); }

  // 1) prefer filename/flag K + expected_n
  if (hints.K_hint>0){
    u32 n = expected_n, K=(u32)hints.K_hint;
    if (fits_flat_sz(n,K,4)) return build_flat_with_optional_trim(4,n,K,hints.K_hint);     // K-only flat
    if (fits_flat_sz(n,K,8)) return build_flat_with_optional_trim(8,n,K,hints.K_hint);     // NK/KN flat
    if (fits_pernode_sz(n,K,8)) return parse_pernode(8,n,K,hints.K_hint);                  // per-node
    // try K+1 (self included)
    if (fits_flat_sz(n,K+1,4)) return build_flat_with_optional_trim(4,n,K+1,hints.K_hint);
    if (fits_flat_sz(n,K+1,8)) return build_flat_with_optional_trim(8,n,K+1,hints.K_hint);
    if (fits_pernode_sz(n,K+1,8)) return parse_pernode(8,n,K+1,hints.K_hint);
    std::cerr<<"[WARN] K="<<K<<" with expected_n="<<n<<" not matching known layouts (size="<<fsz<<"), fallback auto.\n";
  }

  // 2) auto detect (NK/KN + flat/pernode + K-only)
  struct Cand { u32 n,K; u64 hdr; int kind; int score; const char* tag; }; // kind:0 flat,1 pernode
  std::vector<Cand> cs;
  auto push_if = [&](u32 n,u32 K,u64 hdr,int kind,const char* tag){
    bool ok = (kind==0? fits_flat_sz(n,K,hdr): fits_pernode_sz(n,K,hdr));
    if (!ok) return;
    int sc=0; if ((int)n==(int)expected_n) sc+=1000; if (n>=K) sc+=10; if (K<=65536) sc+=5;
    cs.push_back({n,K,hdr,kind,sc,tag});
  };
  // NK/KN
  push_if(a,b,8,0,"flat NK");     push_if(a,b,8,1,"pernode NK");
  push_if(b,a,8,0,"flat KN");     push_if(b,a,8,1,"pernode KN");
  // K-only
  if (expected_n>0){ push_if(expected_n,a,4,0,"flat K-only(a)"); if (b>0) push_if(expected_n,b,4,0,"flat K-only(b)"); }

  if (!cs.empty()){
    std::sort(cs.begin(), cs.end(), [](const Cand& x,const Cand& y){ return x.score>y.score; });
    const auto c = cs.front();
    if (c.kind==0) return build_flat_with_optional_trim(c.hdr, c.n, c.K, hints.K_hint);
    else           return parse_pernode(c.hdr, c.n, c.K, hints.K_hint);
  }

  // 3) size-only inference (K-only / flat / pernode)
  if (expected_n>0){
    if (fsz > 4){
      u64 rem = fsz - 4;
      if (rem % (4ull*expected_n) == 0){
        u32 K = (u32)( rem / (4ull*expected_n) );
        return build_flat_with_optional_trim(4, expected_n, K, hints.K_hint);
      }
    }
    if (fsz > 8){
      u64 rem = fsz - 8;
      if (rem % (4ull*expected_n) == 0){
        u32 K = (u32)( rem / (4ull*expected_n) );
        return build_flat_with_optional_trim(8, expected_n, K, hints.K_hint);
      }
      u64 q = (fsz - 8) / 4 / expected_n;
      if (8 + (u64)expected_n * q * 4 == fsz && q>=1){
        u32 K = (u32)(q-1);
        return parse_pernode(8, expected_n, K, hints.K_hint);
      }
    }
  }

  // 4) fallback: variable-degree
  return parse_vardeg();
}

// ---------- distances ----------
static inline float l2sqr(const float* a, const float* b, u32 dim){
  float s=0.0f;
  #pragma omp simd reduction(+:s)
  for(u32 i=0;i<dim;++i){ float d=a[i]-b[i]; s+=d*d; }
  return s;
}

struct Candidate { u32 id; float dist; };

// ---------- gather candidates (1-hop + optional 2-hop up to L) ----------
static void collect_candidates(
  u32 u,
  const std::vector<std::vector<u32>>& knn,
  u32 target_L,
  bool use_two_hop,
  u32 data_n,
  std::vector<u32>& out
){
  out.clear(); out.reserve(target_L*2);
  std::unordered_set<u32> seen; seen.reserve(target_L*4);
  auto add = [&](u32 v){ if(v==u) return; if(v>=data_n) return; if(seen.insert(v).second) out.push_back(v); };

  if (u < knn.size()){
    for (u32 v: knn[u]){ add(v); if (out.size()>=target_L) break; }
    if (use_two_hop && out.size()<target_L){
      for (u32 v: knn[u]){
        if (v<knn.size()){
          for (u32 w: knn[v]){ add(w); if (out.size()>=target_L) break; }
        }
        if (out.size()>=target_L) break;
      }
    }
  }

  if (out.size()<target_L){
    for (u32 v=0; v<data_n && out.size()<target_L; ++v) add(v);
  }
}

// ---------- MRNG prune (local RNG rule with tau) ----------
static void mrng_prune_one(
  u32 u,
  const Data& data,
  const std::vector<u32>& cands,
  u32 R,
  double tau,
  std::vector<u32>& out_neighbors,
  std::vector<Candidate>& buf
){
  out_neighbors.clear();
  if (cands.empty()) return;

  const float* pu = &data.x[(size_t)u * data.dim];

  buf.clear(); buf.reserve(cands.size());
  for (u32 v: cands){
    if (v >= data.n) continue;
    const float* pv = &data.x[(size_t)v * data.dim];
    float d = l2sqr(pu, pv, data.dim);
    buf.push_back({v, d});
  }
  if (buf.empty()) return;

  std::sort(buf.begin(), buf.end(), [](const Candidate& a, const Candidate& b){ return a.dist < b.dist; });

  const float tau2 = float((1.0 + tau) * (1.0 + tau)); // squared factor
  out_neighbors.reserve(std::min<u32>(R, (u32)buf.size()));
  for (auto &cand : buf){
    u32 v = cand.id;
    const float* pv = &data.x[(size_t)v * data.dim];
    const float duv = cand.dist;           // ||u - v||^2
    const float thr = tau2 * duv;          // (1+tau)^2 * ||u-v||^2

    bool forbidden = false;
    for (u32 p : out_neighbors){
      const float* pp = &data.x[(size_t)p * data.dim];
      float dup = l2sqr(pu, pp, data.dim);
      if (dup < thr){
        float dvp = l2sqr(pv, pp, data.dim);
        if (dvp < thr){ forbidden = true; break; }
      }
    }
    if (!forbidden){
      out_neighbors.push_back(v);
      if (out_neighbors.size() >= R) break;
    }
  }
}

// ---------- post: symmetrize + connectivity repair ----------
static void symmetrize_with_cap(std::vector<std::vector<u32>>& G, u32 R_cap){
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
    for(u32 u: rev[v]){ if(dst.size()>=R_cap) break; if(has.insert(u).second) dst.push_back(u); }
    if (dst.size()>R_cap) dst.resize(R_cap);
  }
}

static void connectivity_repair(
  const Data& data,
  const std::vector<std::vector<u32>>& knn,
  u32 R_cap,
  std::vector<std::vector<u32>>& G
){
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
    // add undirected bridge (bounded by R_cap)
    auto &Gu = G[bu]; auto &Gv = G[bv];
    if (std::find(Gu.begin(), Gu.end(), bv)==Gu.end()){ if(Gu.size()<R_cap) Gu.push_back(bv); else Gu[bu % R_cap]=bv; }
    if (std::find(Gv.begin(), Gv.end(), bu)==Gv.end()){ if(Gv.size()<R_cap) Gv.push_back(bu); else Gv[bv % R_cap]=bu; }
    for(u32 u: comps[ci]) mainset.insert(u);
  }
}

// ---------- save ----------
static void save_mrng(const std::string& path, const std::vector<std::vector<u32>>& G){
  std::error_code ec; std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  FILE* fp=std::fopen(path.c_str(),"wb"); if(!fp){ perror(path.c_str()); die("open out_mrng failed"); }
  u32 n=(u32)G.size(); if(std::fwrite(&n,sizeof(u32),1,fp)!=1) die("write n failed");
  for(u32 i=0;i<n;++i){
    u32 deg=(u32)G[i].size(); if(std::fwrite(&deg,sizeof(u32),1,fp)!=1) die("write deg failed");
    if (deg && std::fwrite(G[i].data(), sizeof(u32), deg, fp)!=deg) die("write neighbors failed");
  }
  std::fclose(fp);
}

// ---------- main ----------
int main(int argc, char** argv){
  if (argc < 4){
    std::cerr<<"Usage:\n  "<<argv[0]<<" <data_dir_or_fbin> <out_mrng> <knn_graph_path_or_dir> [R=64] [L=200] [two_hop=1] [--K=50|--k=50] [--tau=0.01]\n";
  }
  std::string data_arg      = (argc>=2)? argv[1]: "";
  std::string out_mrng      = (argc>=3)? argv[2]: "out.mrng";
  std::string knn_arg       = (argc>=4)? argv[3]: "";
  u32 R                     = (argc>=5)? (u32)std::stoul(argv[4]):64;
  u32 L                     = (argc>=6)? (u32)std::stoul(argv[5]):200;
  int two_hop               = (argc>=7)? std::stoi(argv[6]) : 1; // 1: use two-hop expansion
  double tau                = 0.01; // default tau

  GraphHints hints;
  for(int i=7;i<argc;++i){
    std::string s=argv[i];
    if (s.rfind("--K=",0)==0 || s.rfind("--k=",0)==0) hints.K_hint = std::stoi(s.substr(4));
    if (s.rfind("--tau=",0)==0) tau = std::stod(s.substr(6));
  }

  if (data_arg.empty() || knn_arg.empty()){
    std::cerr<<"[ERR] missing required args. See usage above.\n";
    return 1;
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

  if (hints.K_hint <= 0){
    hints.K_hint = K_from_filename(knn_path);
    if (hints.K_hint > 0) std::cout<<"[INFO] K inferred from filename: "<<hints.K_hint<<"\n";
    else std::cout<<"[WARN] Cannot infer K from filename. Will auto-detect by size.\n";
  } else {
    std::cout<<"[INFO] K overridden by flag: "<<hints.K_hint<<"\n";
  }
  std::cout<<"[INFO] tau = "<<tau<<"  -> RNG threshold factor = (1+tau)^2\n";

  auto knn = read_knn_graph(knn_path, data.n, hints);
  if (knn.size()!=data.n){
    std::cerr<<"[WARN] kNN graph n="<<knn.size()<<" != data n="<<data.n<<". Build proceeds safely; verify coverage.\n";
  }

  const u32 n = data.n;
  std::vector<std::vector<u32>> G(n);

  auto t0 = std::chrono::high_resolution_clock::now();

  #pragma omp parallel
  {
    std::vector<u32> cands;
    std::vector<Candidate> buf;
    std::vector<u32> nei; nei.reserve(R);

    #pragma omp for schedule(dynamic,64)
    for (int iu=0; iu<(int)n; ++iu){
      u32 u = (u32)iu;
      collect_candidates(u, knn, L, two_hop!=0, data.n, cands);
      mrng_prune_one(u, data, cands, R, tau, nei, buf);
      G[u] = nei;
    }
  }

  symmetrize_with_cap(G, R);
  connectivity_repair(data, knn, R, G);

  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout<<"[INFO] MRNG built in "<<std::chrono::duration<double>(t1-t0).count()<<" s\n";

  std::cout<<"[INFO] Saving to: "<<out_mrng<<"\n";
  save_mrng(out_mrng, G);
  std::cout<<"Done.\n";
  return 0;
}
