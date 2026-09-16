#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <hnswlib/hnswlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using u32 = uint32_t;

struct Data {
  u32 n = 0;
  u32 dim = 0;
  std::vector<float> x;
};

struct GT {
  u32 nq = 0;
  u32 k = 0;
  std::vector<int32_t> idx;
};

struct PGClusterOut {
  std::vector<std::vector<u32>> clusters;
  size_t non_empty = 0;
  size_t min_size = 0;
  size_t max_size = 0;
};

static void die(const std::string &m) {
  std::cerr << "[ERR] " << m << "\n";
  std::exit(1);
}

static long long file_size(const std::string &p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) {
    return -1;
  }
  return (long long)f.tellg();
}

static Data read_fbin(const std::string &path) {
  FILE *fp = std::fopen(path.c_str(), "rb");
  if (!fp) {
    perror(path.c_str());
    die("open fbin failed: " + path);
  }
  int n = 0, d = 0;
  if (std::fread(&n, 4, 1, fp) != 1 || std::fread(&d, 4, 1, fp) != 1) {
    std::fclose(fp);
    die("bad fbin header");
  }
  if (n <= 0 || d <= 0) {
    std::fclose(fp);
    die("invalid fbin header values");
  }
  Data R;
  R.n = (u32)n;
  R.dim = (u32)d;
  R.x.resize((size_t)n * (size_t)d);
  size_t need = (size_t)n * (size_t)d;
  if (std::fread(R.x.data(), sizeof(float), need, fp) != need) {
    std::fclose(fp);
    die("read fbin body failed");
  }
  std::fclose(fp);
  return R;
}

static GT read_gt_ibin(const std::string &path) {
  GT gt;
  long long sz = file_size(path);
  if (sz < 0) {
    die("gt file not found: " + path);
  }
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    die("open gt failed: " + path);
  }

  int a = 0, b = 0;
  ifs.read((char *)&a, 4);
  if (ifs.read((char *)&b, 4)) {
    long long expect = 8ll + (long long)a * (long long)b * 4ll;
    if (expect == sz) {
      gt.nq = (u32)a;
      gt.k = (u32)b;
      gt.idx.resize((size_t)gt.nq * (size_t)gt.k);
      ifs.read((char *)gt.idx.data(), (std::streamsize)gt.idx.size() * 4);
      return gt;
    }
  }

  ifs.clear();
  ifs.seekg(0, std::ios::beg);
  long long cnt = sz / 4;
  gt.idx.resize((size_t)cnt);
  ifs.read((char *)gt.idx.data(), (std::streamsize)sz);

  int tryK[5] = {100, 50, 20, 10, 1};
  for (int tk : tryK) {
    if (cnt % tk == 0) {
      gt.k = (u32)tk;
      gt.nq = (u32)(cnt / tk);
      break;
    }
  }
  if (gt.k == 0) {
    die("cannot infer (nq,k) from gt without header");
  }
  return gt;
}

static std::vector<std::vector<u32>> read_nsg(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    die("open nsg failed: " + path);
  }
  u32 n = 0;
  ifs.read((char *)&n, 4);
  if (!ifs) {
    die("read n failed");
  }

  std::vector<std::vector<u32>> G(n);
  for (u32 i = 0; i < n; ++i) {
    u32 deg = 0;
    ifs.read((char *)&deg, 4);
    if (!ifs) {
      die("read deg failed");
    }
    G[i].resize(deg);
    if (deg) {
      ifs.read((char *)G[i].data(), (std::streamsize)deg * 4);
      if (!ifs) {
        die("read neighbors failed");
      }
    }
  }
  return G;
}

static inline float l2sqr(const float *a, const float *b, u32 d) {
  float s = 0.f;
  for (u32 i = 0; i < d; ++i) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

static u32 compute_entry_centroid_nearest(const Data &base) {
  if (base.n == 0) {
    die("empty base");
  }
  std::vector<float> mu(base.dim, 0.f);
  const float invN = 1.0f / (float)base.n;
  for (u32 i = 0; i < base.n; ++i) {
    const float *xi = &base.x[(size_t)i * base.dim];
    for (u32 j = 0; j < base.dim; ++j) {
      mu[j] += xi[j] * invN;
    }
  }

  u32 best_id = 0;
  float best = std::numeric_limits<float>::max();
  for (u32 i = 0; i < base.n; ++i) {
    const float *xi = &base.x[(size_t)i * base.dim];
    float d = 0.f;
    for (u32 j = 0; j < base.dim; ++j) {
      float t = xi[j] - mu[j];
      d += t * t;
    }
    if (d < best) {
      best = d;
      best_id = i;
    }
  }
  return best_id;
}

struct MinCand {
  u32 id;
  float dist;
};

struct MaxRes {
  float dist;
  u32 id;
};

struct CmpMinCand {
  bool operator()(const MinCand &a, const MinCand &b) const { return a.dist > b.dist; }
};

struct CmpMaxRes {
  bool operator()(const MaxRes &a, const MaxRes &b) const { return a.dist < b.dist; }
};

static void nsg_search_one_fixed_entry(const Data &base,
                                       const std::vector<std::vector<u32>> &G,
                                       const float *q,
                                       u32 K,
                                       u32 L,
                                       u32 entry_id,
                                       std::vector<u32> &out_ids_sorted) {
  const u32 n = (u32)G.size();
  if (entry_id >= n) {
    die("entry_id out of range");
  }

  std::vector<char> vis(n, 0);
  auto seen = [&](u32 x) -> bool {
    if (vis[x]) {
      return true;
    }
    vis[x] = 1;
    return false;
  };

  float d_ep = l2sqr(q, &base.x[(size_t)entry_id * base.dim], base.dim);
  seen(entry_id);

  using CandPQ = std::priority_queue<MinCand, std::vector<MinCand>, CmpMinCand>;
  using ResPQ = std::priority_queue<MaxRes, std::vector<MaxRes>, CmpMaxRes>;
  CandPQ C;
  ResPQ W;
  C.push(MinCand{entry_id, d_ep});
  W.push(MaxRes{d_ep, entry_id});

  while (!C.empty()) {
    MinCand c = C.top();
    C.pop();
    MaxRes f = W.top();
    if (c.dist > f.dist) {
      break;
    }

    const auto &nbrs = G[c.id];
    for (u32 e : nbrs) {
      if (e >= n) {
        continue;
      }
      if (seen(e)) {
        continue;
      }
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

  while (W.size() > (size_t)K) {
    W.pop();
  }

  out_ids_sorted.clear();
  out_ids_sorted.reserve(W.size());
  while (!W.empty()) {
    out_ids_sorted.push_back(W.top().id);
    W.pop();
  }
  std::reverse(out_ids_sorted.begin(), out_ids_sorted.end());
}

static std::vector<float> random_project_pg(const std::vector<float> &xq, u32 n, int d, int proj_dim, unsigned seed) {
  std::vector<float> proj((size_t)n * (size_t)proj_dim, 0.0f);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f / std::sqrt((float)proj_dim));

  std::vector<float> R((size_t)proj_dim * (size_t)d);
  for (size_t i = 0; i < R.size(); ++i) {
    R[i] = gauss(rng);
  }

  for (u32 i = 0; i < n; ++i) {
    const float *x = xq.data() + (size_t)i * (size_t)d;
    float *z = proj.data() + (size_t)i * (size_t)proj_dim;
    for (int r = 0; r < proj_dim; ++r) {
      const float *row = R.data() + (size_t)r * (size_t)d;
      float acc = 0.0f;
      for (int c = 0; c < d; ++c) {
        acc += row[c] * x[c];
      }
      z[r] = acc;
    }
  }
  return proj;
}

static std::vector<std::vector<int>> build_lowdim_topm_hnsw_pg(const std::vector<float> &z,
                                                               u32 n,
                                                               int proj_dim,
                                                               int m,
                                                               int hnsw_m,
                                                               int hnsw_efc,
                                                               int hnsw_ef) {
  std::vector<std::vector<int>> nbrs((size_t)n);
  hnswlib::L2Space space((size_t)proj_dim);
  hnswlib::HierarchicalNSW<float> hnsw(&space, (size_t)n, (size_t)hnsw_m, (size_t)hnsw_efc);

  for (u32 i = 0; i < n; ++i) {
    const float *zi = z.data() + (size_t)i * (size_t)proj_dim;
    hnsw.addPoint((const void *)zi, (size_t)i);
  }
  hnsw.setEf((size_t)std::max(hnsw_ef, m + 1));

  for (u32 i = 0; i < n; ++i) {
    const float *zi = z.data() + (size_t)i * (size_t)proj_dim;
    auto pq = hnsw.searchKnn((const void *)zi, (size_t)(m + 1));
    std::vector<int> ids;
    ids.reserve((size_t)m);
    while (!pq.empty() && (int)ids.size() < m) {
      int id = (int)pq.top().second;
      pq.pop();
      if (id == (int)i) {
        continue;
      }
      bool dup = false;
      for (int x : ids) {
        if (x == id) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        ids.push_back(id);
      }
    }
    nbrs[(size_t)i] = std::move(ids);
  }
  return nbrs;
}

static std::vector<std::vector<int>> build_compat_graph_pg(const std::vector<float> &xq,
                                                           u32 n,
                                                           int d,
                                                           const std::vector<std::vector<int>> &cand,
                                                           float tau_edge) {
  const float tau_edge_sq = tau_edge * tau_edge;
  std::vector<std::vector<int>> g((size_t)n);
  for (u32 i = 0; i < n; ++i) {
    const float *xi = xq.data() + (size_t)i * (size_t)d;
    for (int j : cand[(size_t)i]) {
      if (j <= (int)i) {
        continue;
      }
      const float *xj = xq.data() + (size_t)j * (size_t)d;
      float dij = l2sqr(xi, xj, (u32)d);
      if (dij <= tau_edge_sq) {
        g[(size_t)i].push_back(j);
        g[(size_t)j].push_back((int)i);
      }
    }
  }
  return g;
}

static bool can_add_under_radius_pg(const std::vector<float> &xq,
                                    int d,
                                    const std::vector<int> &cluster,
                                    const std::vector<float> &sum_vec,
                                    int cand,
                                    float tau_cluster_sq) {
  const float *cand_vec = xq.data() + (size_t)cand * (size_t)d;
  std::vector<float> center((size_t)d, 0.0f);
  float inv = 1.0f / (float)(cluster.size() + 1);
  for (int i = 0; i < d; ++i) {
    center[(size_t)i] = (sum_vec[(size_t)i] + cand_vec[i]) * inv;
  }

  for (int id : cluster) {
    const float *x = xq.data() + (size_t)id * (size_t)d;
    if (l2sqr(x, center.data(), (u32)d) > tau_cluster_sq) {
      return false;
    }
  }
  if (l2sqr(cand_vec, center.data(), (u32)d) > tau_cluster_sq) {
    return false;
  }
  return true;
}

static PGClusterOut build_projected_greedy_clusters(const std::vector<float> &xq,
                                                    u32 n,
                                                    int d,
                                                    int proj_dim,
                                                    int m,
                                                    int cl_hnsw_m,
                                                    int cl_hnsw_efc,
                                                    int cl_hnsw_ef,
                                                    float tau_edge,
                                                    float tau_cluster,
                                                    unsigned seed) {
  PGClusterOut out;
  if (n == 0) {
    return out;
  }

  std::vector<float> z = random_project_pg(xq, n, d, proj_dim, seed);
  auto cand = build_lowdim_topm_hnsw_pg(z, n, proj_dim, m, cl_hnsw_m, cl_hnsw_efc, cl_hnsw_ef);
  auto g = build_compat_graph_pg(xq, n, d, cand, tau_edge);

  std::vector<int> degree((size_t)n, 0);
  for (u32 i = 0; i < n; ++i) {
    degree[(size_t)i] = (int)g[(size_t)i].size();
  }

  std::vector<int> order((size_t)n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (degree[(size_t)a] != degree[(size_t)b]) {
      return degree[(size_t)a] > degree[(size_t)b];
    }
    return a < b;
  });

  std::vector<uint8_t> assigned((size_t)n, 0);
  std::vector<uint8_t> in_cluster((size_t)n, 0);
  const float tau_cluster_sq = tau_cluster * tau_cluster;

  for (int seed_id : order) {
    if (assigned[(size_t)seed_id]) {
      continue;
    }

    std::vector<int> cluster;
    cluster.reserve(64);
    cluster.push_back(seed_id);
    in_cluster[(size_t)seed_id] = 1;

    std::vector<float> sum_vec((size_t)d, 0.0f);
    const float *seed_vec = xq.data() + (size_t)seed_id * (size_t)d;
    for (int i = 0; i < d; ++i) {
      sum_vec[(size_t)i] = seed_vec[i];
    }

    std::vector<int> frontier;
    frontier.push_back(seed_id);
    size_t head = 0;
    while (head < frontier.size()) {
      int u = frontier[head++];
      for (int v : g[(size_t)u]) {
        if (assigned[(size_t)v] || in_cluster[(size_t)v]) {
          continue;
        }
        if (!can_add_under_radius_pg(xq, d, cluster, sum_vec, v, tau_cluster_sq)) {
          continue;
        }

        cluster.push_back(v);
        in_cluster[(size_t)v] = 1;
        frontier.push_back(v);
        const float *v_vec = xq.data() + (size_t)v * (size_t)d;
        for (int i = 0; i < d; ++i) {
          sum_vec[(size_t)i] += v_vec[i];
        }
      }
    }

    std::vector<u32> one;
    one.reserve(cluster.size());
    for (int id : cluster) {
      assigned[(size_t)id] = 1;
      in_cluster[(size_t)id] = 0;
      one.push_back((u32)id);
    }
    out.clusters.push_back(std::move(one));
  }

  for (u32 i = 0; i < n; ++i) {
    if (assigned[(size_t)i]) {
      continue;
    }
    assigned[(size_t)i] = 1;
    out.clusters.push_back(std::vector<u32>{i});
  }

  out.non_empty = out.clusters.size();
  out.min_size = std::numeric_limits<size_t>::max();
  out.max_size = 0;
  for (const auto &c : out.clusters) {
    out.min_size = std::min(out.min_size, c.size());
    out.max_size = std::max(out.max_size, c.size());
  }
  if (out.non_empty == 0) {
    out.min_size = 0;
  }
  return out;
}

static std::vector<int> parse_int_list(const std::string &s) {
  std::vector<int> v;
  std::string buf;
  for (char c : s) {
    if (c == ',' || c == ' ') {
      if (!buf.empty()) {
        v.push_back(std::stoi(buf));
        buf.clear();
      }
    } else {
      buf.push_back(c);
    }
  }
  if (!buf.empty()) {
    v.push_back(std::stoi(buf));
  }
  return v;
}

static void append_grid_token(const std::string &tok, std::vector<int> &out) {
  if (tok.empty()) {
    return;
  }
  size_t c1 = tok.find(':');
  if (c1 == std::string::npos) {
    out.push_back(std::stoi(tok));
    return;
  }
  size_t c2 = tok.find(':', c1 + 1);
  auto to_i = [](const std::string &s) { return std::stoi(s); };
  if (c2 == std::string::npos) {
    int a = to_i(tok.substr(0, c1));
    int b = to_i(tok.substr(c1 + 1));
    if (a <= b) {
      for (int x = a; x <= b; ++x) {
        out.push_back(x);
      }
    } else {
      for (int x = a; x >= b; --x) {
        out.push_back(x);
      }
    }
    return;
  }

  int a = to_i(tok.substr(0, c1));
  int s = to_i(tok.substr(c1 + 1, c2 - (c1 + 1)));
  int b = to_i(tok.substr(c2 + 1));
  if (s == 0) {
    return;
  }
  if ((long long)(b - a) * (long long)s < 0) {
    return;
  }
  if (s > 0) {
    for (int x = a; x <= b; x += s) {
      out.push_back(x);
    }
  } else {
    for (int x = a; x >= b; x += s) {
      out.push_back(x);
    }
  }
}

static std::vector<int> parse_int_list_or_grid(const std::string &s) {
  std::vector<int> v;
  std::string buf;
  for (char c : s) {
    if (c == ',' || c == ' ') {
      if (!buf.empty()) {
        append_grid_token(buf, v);
        buf.clear();
      }
    } else {
      buf.push_back(c);
    }
  }
  if (!buf.empty()) {
    append_grid_token(buf, v);
  }
  return v;
}

static std::string get_str(int argc, char **argv, const std::string &key, const std::string &defv = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      return std::string(argv[i + 1]);
    }
  }
  return defv;
}

static int get_int(int argc, char **argv, const std::string &key, int defv) {
  std::string s = get_str(argc, argv, key, "");
  return s.empty() ? defv : std::stoi(s);
}

static float get_float(int argc, char **argv, const std::string &key, float defv) {
  std::string s = get_str(argc, argv, key, "");
  return s.empty() ? defv : std::stof(s);
}

static bool has_flag(int argc, char **argv, const std::string &key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i]) {
      return true;
    }
  }
  return false;
}

static float recall_at_k_active(const std::vector<u32> &all_ids,
                                int Kmax,
                                const GT &gt,
                                int K,
                                const std::vector<u32> &active_qids) {
  int Kg = std::min(K, (int)gt.k);
  if (Kg <= 0 || active_qids.empty()) {
    return 0.0f;
  }
  long long hit = 0;
  long long need = 0;
  for (u32 qi : active_qids) {
    const int32_t *row = &gt.idx[(size_t)qi * (size_t)gt.k];
    for (int j = 0; j < Kg; ++j) {
      int32_t g = row[j];
      if (g < 0) {
        break;
      }
      for (int t = 0; t < K; ++t) {
        if ((int32_t)all_ids[(size_t)qi * (size_t)Kmax + (size_t)t] == g) {
          ++hit;
          break;
        }
      }
      ++need;
    }
  }
  return need > 0 ? (float)hit / (float)need : 0.0f;
}

int main(int argc, char **argv) {
  if (argc < 3 || has_flag(argc, argv, "-h") || has_flag(argc, argv, "--help")) {
    std::cout
        << "Usage:\n"
        << "  " << argv[0] << " -data base.fbin -graph graph.nsg -qfile query.fbin -gt gt.ibin \\\n"
        << "       -K 1,10,100 -L 100:100:1500 -threads 1 -csv out.csv [options]\n\n"
        << "Options:\n"
        << "  -runs <int>             repeat runs (default 1)\n"
        << "  -warmup <int>           warmup runs (default 0)\n"
        << "  -entry_id <u32>         fixed entry id (default centroid-nearest)\n"
        << "  -batch_gate <int>       only clusters with size > gate are searched (default 8)\n"
        << "  -seed <int>             projected-greedy seed (default 42)\n"
        << "  -proj_dim <int>         projection dim (default 32)\n"
        << "  -proj_m <int>           projected top-m neighbors (default 16)\n"
        << "  -cl_hnsw_m <int>        HNSW M for projected graph (default 16)\n"
        << "  -cl_hnsw_efc <int>      HNSW efConstruction (default 100)\n"
        << "  -cl_hnsw_ef <int>       HNSW efSearch in projected graph (default 64)\n"
        << "  -tau_edge <float>       edge compatibility threshold (default 0.4)\n"
        << "  -tau_cluster <float>    cluster radius threshold (default 0.5)\n";
    return 0;
  }

  std::string base_path = get_str(argc, argv, "-data", "");
  std::string nsg_path = get_str(argc, argv, "-graph", "");
  std::string qfile_path = get_str(argc, argv, "-qfile", "");
  std::string gt_path = get_str(argc, argv, "-gt", "");
  if (base_path.empty() || nsg_path.empty() || qfile_path.empty() || gt_path.empty()) {
    die("missing required args: -data -graph -qfile -gt");
  }

  std::vector<int> Ks = parse_int_list(get_str(argc, argv, "-K", "1,10,100"));
  std::vector<int> Ls = parse_int_list_or_grid(get_str(argc, argv, "-L", "100:100:1500"));
  if (Ks.empty() || Ls.empty()) {
    die("-K or -L parsed empty");
  }

  int runs = get_int(argc, argv, "-runs", 1);
  int warmup = get_int(argc, argv, "-warmup", 0);
  int threads = std::max(1, get_int(argc, argv, "-threads", 1));
  std::string csv = get_str(argc, argv, "-csv", "nsg_cluster_skip_small.csv");

  int batch_gate = get_int(argc, argv, "-batch_gate", 8);
  unsigned seed = (unsigned)get_int(argc, argv, "-seed", 42);
  int proj_dim = get_int(argc, argv, "-proj_dim", 32);
  int proj_m = get_int(argc, argv, "-proj_m", 16);
  int cl_hnsw_m = get_int(argc, argv, "-cl_hnsw_m", 16);
  int cl_hnsw_efc = get_int(argc, argv, "-cl_hnsw_efc", 100);
  int cl_hnsw_ef = get_int(argc, argv, "-cl_hnsw_ef", 64);
  float tau_edge = get_float(argc, argv, "-tau_edge", 0.4f);
  float tau_cluster = get_float(argc, argv, "-tau_cluster", 0.5f);

  std::cout << "[INFO] loading base...\n";
  Data base = read_fbin(base_path);
  std::cout << "[INFO] base: n=" << base.n << " dim=" << base.dim << "\n";

  std::cout << "[INFO] loading nsg...\n";
  auto G = read_nsg(nsg_path);
  if (G.size() != base.n) {
    std::cerr << "[WARN] nsg n=" << G.size() << " != base n=" << base.n << "\n";
  }

  std::cout << "[INFO] loading query & gt...\n";
  Data xq = read_fbin(qfile_path);
  GT gt = read_gt_ibin(gt_path);
  u32 nq = std::min<u32>(xq.n, gt.nq);
  if (xq.dim != base.dim) {
    die("query dim != base dim");
  }
  std::cout << "[INFO] nq=" << nq << " gt.k=" << gt.k << "\n";

  u32 entry_id = 0;
  std::string entry_s = get_str(argc, argv, "-entry_id", "");
  if (!entry_s.empty()) {
    long long v = std::stoll(entry_s);
    if (v < 0 || (unsigned long long)v >= (unsigned long long)base.n) {
      die("-entry_id out of range");
    }
    entry_id = (u32)v;
    std::cout << "[INFO] using user entry_id=" << entry_id << "\n";
  } else {
    auto t0 = std::chrono::high_resolution_clock::now();
    entry_id = compute_entry_centroid_nearest(base);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] using centroid-nearest entry_id=" << entry_id << " ("
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms)\n";
  }

#ifdef _OPENMP
  omp_set_num_threads(threads);
#endif
  std::cout << "[INFO] threads=" << threads << "\n";

  std::cout << "[ProjectedGreedy] start clustering nq=" << nq
            << " proj_dim=" << proj_dim
            << " M=" << proj_m
            << " tau_edge=" << tau_edge
            << " tau_cluster=" << tau_cluster
            << " gate=" << batch_gate << "\n";
  auto t_cluster0 = std::chrono::high_resolution_clock::now();
  PGClusterOut pg = build_projected_greedy_clusters(xq.x,
                                                    nq,
                                                    (int)xq.dim,
                                                    proj_dim,
                                                    proj_m,
                                                    cl_hnsw_m,
                                                    cl_hnsw_efc,
                                                    cl_hnsw_ef,
                                                    tau_edge,
                                                    tau_cluster,
                                                    seed);
  auto t_cluster1 = std::chrono::high_resolution_clock::now();
  double cluster_build_ms = std::chrono::duration<double, std::milli>(t_cluster1 - t_cluster0).count();

  std::vector<u32> active_qids;
  std::vector<u32> skipped_qids;
  size_t large_clusters = 0;
  size_t small_clusters = 0;
  for (const auto &C : pg.clusters) {
    if (C.empty()) {
      continue;
    }
    if ((int)C.size() <= batch_gate) {
      ++small_clusters;
      skipped_qids.insert(skipped_qids.end(), C.begin(), C.end());
    } else {
      ++large_clusters;
      active_qids.insert(active_qids.end(), C.begin(), C.end());
    }
  }

  std::cout << "[ProjectedGreedy] total_clusters=" << pg.clusters.size()
            << " non_empty=" << pg.non_empty
            << " min_size=" << pg.min_size
            << " max_size=" << pg.max_size << "\n";
  std::cout << "[Gate] large_clusters=" << large_clusters
            << " small_clusters=" << small_clusters
            << " active_q=" << active_qids.size()
            << " skipped_q=" << skipped_qids.size() << "\n";

  int Kmax = 1;
  for (int v : Ks) {
    Kmax = std::max(Kmax, v);
  }

  std::ofstream ofs(csv);
  if (!ofs) {
    die("cannot open csv for write: " + csv);
  }
  ofs << "data,graph,qfile,gt,K,L,threads,runs,effective_q,skipped_q,large_clusters,small_clusters,qps,ms_per_query,recall,entry_id,batch_gate,proj_dim,proj_m,tau_edge,tau_cluster\n";

  for (int L : Ls) {
    if ((int)gt.k < Kmax) {
      std::cerr << "[WARN] Kmax=" << Kmax << " > gt.k=" << gt.k << ", recall uses gt@" << gt.k << "\n";
    }

    for (int w = 0; w < warmup; ++w) {
#pragma omp parallel for schedule(dynamic, 32) if(_OPENMP)
      for (long long ii = 0; ii < (long long)active_qids.size(); ++ii) {
        u32 qi = active_qids[(size_t)ii];
        const float *q = &xq.x[(size_t)qi * xq.dim];
        std::vector<u32> tmp;
        nsg_search_one_fixed_entry(base, G, q, (u32)Kmax, (u32)L, entry_id, tmp);
      }
    }

    double sum_ms = 0.0;
    std::vector<double> sum_recall((size_t)Ks.size(), 0.0);

    for (int r = 0; r < runs; ++r) {
      std::vector<u32> all_ids((size_t)nq * (size_t)Kmax, std::numeric_limits<u32>::max());
      auto t0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 32) if(_OPENMP)
      for (long long ii = 0; ii < (long long)active_qids.size(); ++ii) {
        u32 qi = active_qids[(size_t)ii];
        const float *q = &xq.x[(size_t)qi * xq.dim];
        std::vector<u32> out;
        nsg_search_one_fixed_entry(base, G, q, (u32)Kmax, (u32)L, entry_id, out);
        for (int j = 0; j < Kmax && j < (int)out.size(); ++j) {
          all_ids[(size_t)qi * (size_t)Kmax + (size_t)j] = out[(size_t)j];
        }
      }

      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      sum_ms += ms;

      for (size_t ki = 0; ki < Ks.size(); ++ki) {
        int K = Ks[ki];
        sum_recall[ki] += recall_at_k_active(all_ids, Kmax, gt, K, active_qids);
      }
    }

    double avg_ms = sum_ms / std::max(1, runs);
    double qps = active_qids.empty() ? 0.0 : ((double)active_qids.size() * 1000.0 / std::max(1e-12, avg_ms));

    for (size_t ki = 0; ki < Ks.size(); ++ki) {
      int K = Ks[ki];
      double avg_recall = sum_recall[ki] / std::max(1, runs);
      double ms_per_query = active_qids.empty() ? 0.0 : (avg_ms / (double)active_qids.size());

      std::cout << "K=" << K << " L=" << L
                << " | effective_q=" << active_qids.size()
                << " skipped_q=" << skipped_qids.size()
                << " | QPS=" << qps
                << " | ms/q=" << ms_per_query
                << " | Recall=" << avg_recall << "\n";

      ofs << base_path << "," << nsg_path << "," << qfile_path << "," << gt_path << ","
          << K << "," << L << "," << threads << "," << runs << ","
          << active_qids.size() << "," << skipped_qids.size() << ","
          << large_clusters << "," << small_clusters << ","
          << qps << "," << ms_per_query << "," << avg_recall << ","
          << entry_id << "," << batch_gate << ","
          << proj_dim << "," << proj_m << ","
          << tau_edge << "," << tau_cluster << "\n";
    }
  }

  ofs.close();
  std::cout << "Done. Results -> " << csv << "\n";
  return 0;
}
