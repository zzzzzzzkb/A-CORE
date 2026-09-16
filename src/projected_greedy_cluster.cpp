#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <hnswlib/hnswlib.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Args {
  std::string xq_path;
  std::string knn_mode = "hnsw";
  std::string cluster_mode = "greedy";
  int32_t proj_dim = 32;
  int32_t m = 16;
  int32_t hnsw_m = 16;
  int32_t hnsw_efc = 100;
  int32_t hnsw_ef = 64;
  float tau_edge = 0.4f;
  float tau_cluster = 0.5f;
  uint64_t seed = 42;
};

struct FBinData {
  int32_t rows = 0;
  int32_t dim = 0;
  std::vector<float> data;
};

static float l2_sq(const float* a, const float* b, int32_t dim) {
  float acc = 0.0f;
  for (int32_t d = 0; d < dim; ++d) {
    const float diff = a[d] - b[d];
    acc += diff * diff;
  }
  return acc;
}

FBinData read_fbin(const std::string& path) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    throw std::runtime_error("Failed to open fbin: " + path);
  }
  int32_t rows = 0;
  int32_t dim = 0;
  fin.read(reinterpret_cast<char*>(&rows), sizeof(int32_t));
  fin.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
  if (!fin || rows <= 0 || dim <= 0) {
    throw std::runtime_error("Invalid fbin header in: " + path);
  }

  std::vector<float> data(static_cast<size_t>(rows) * static_cast<size_t>(dim));
  fin.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!fin) {
    throw std::runtime_error("Invalid fbin payload in: " + path);
  }
  return FBinData{rows, dim, std::move(data)};
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](const std::string& k) {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for arg: " + k);
      }
      return std::string(argv[++i]);
    };

    if (key == "--xq") {
      args.xq_path = need_value(key);
    } else if (key == "--knn-mode") {
      args.knn_mode = need_value(key);
    } else if (key == "--cluster-mode") {
      args.cluster_mode = need_value(key);
    } else if (key == "--proj-dim") {
      args.proj_dim = std::stoi(need_value(key));
    } else if (key == "--M") {
      args.m = std::stoi(need_value(key));
    } else if (key == "--hnsw-M") {
      args.hnsw_m = std::stoi(need_value(key));
    } else if (key == "--hnsw-efc") {
      args.hnsw_efc = std::stoi(need_value(key));
    } else if (key == "--hnsw-ef") {
      args.hnsw_ef = std::stoi(need_value(key));
    } else if (key == "--tau-edge") {
      args.tau_edge = std::stof(need_value(key));
    } else if (key == "--tau-cluster") {
      args.tau_cluster = std::stof(need_value(key));
    } else if (key == "--seed") {
      args.seed = static_cast<uint64_t>(std::stoull(need_value(key)));
    } else {
      throw std::runtime_error("Unknown arg: " + key);
    }
  }

  if (args.xq_path.empty()) {
    throw std::runtime_error("--xq is required");
  }
  if (args.knn_mode != "hnsw" && args.knn_mode != "brute") {
    throw std::runtime_error("--knn-mode must be hnsw or brute");
  }
  if (args.cluster_mode != "greedy" && args.cluster_mode != "seed-radius") {
    throw std::runtime_error("--cluster-mode must be greedy or seed-radius");
  }
  if (args.proj_dim <= 0 || args.m <= 0) {
    throw std::runtime_error("--proj-dim and --M must be positive");
  }
  if (args.hnsw_m <= 0 || args.hnsw_efc <= 0 || args.hnsw_ef <= 0) {
    throw std::runtime_error("--hnsw-M/--hnsw-efc/--hnsw-ef must be positive");
  }
  if (args.tau_edge <= 0.0f || args.tau_cluster <= 0.0f) {
    throw std::runtime_error("tau thresholds must be positive");
  }
  return args;
}

std::vector<float> random_project(const FBinData& xq, int32_t proj_dim, uint64_t seed) {
  const int32_t n = xq.rows;
  const int32_t d = xq.dim;
  std::vector<float> proj(static_cast<size_t>(n) * static_cast<size_t>(proj_dim), 0.0f);

  std::mt19937_64 rng(seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f / std::sqrt(static_cast<float>(proj_dim)));

  std::vector<float> R(static_cast<size_t>(proj_dim) * static_cast<size_t>(d));
  for (size_t i = 0; i < R.size(); ++i) {
    R[i] = gauss(rng);
  }

  for (int32_t i = 0; i < n; ++i) {
    const float* x = xq.data.data() + static_cast<size_t>(i) * static_cast<size_t>(d);
    float* z = proj.data() + static_cast<size_t>(i) * static_cast<size_t>(proj_dim);
    for (int32_t r = 0; r < proj_dim; ++r) {
      const float* row = R.data() + static_cast<size_t>(r) * static_cast<size_t>(d);
      float acc = 0.0f;
      for (int32_t c = 0; c < d; ++c) {
        acc += row[c] * x[c];
      }
      z[r] = acc;
    }
  }
  return proj;
}

std::vector<std::vector<int32_t>> build_lowdim_topm_hnsw(const std::vector<float>& z,
                                                          int32_t n,
                                                          int32_t proj_dim,
                                                          int32_t m,
                                                          int32_t hnsw_m,
                                                          int32_t hnsw_efc,
                                                          int32_t hnsw_ef) {
  std::vector<std::vector<int32_t>> nbrs(static_cast<size_t>(n));

  hnswlib::L2Space space(static_cast<size_t>(proj_dim));
  hnswlib::HierarchicalNSW<float> hnsw(&space, static_cast<size_t>(n), static_cast<size_t>(hnsw_m),
                                       static_cast<size_t>(hnsw_efc));
  for (int32_t i = 0; i < n; ++i) {
    const float* zi = z.data() + static_cast<size_t>(i) * static_cast<size_t>(proj_dim);
    hnsw.addPoint(static_cast<const void*>(zi), static_cast<size_t>(i));
  }
  hnsw.setEf(std::max(hnsw_ef, m + 1));

  for (int32_t i = 0; i < n; ++i) {
    const float* zi = z.data() + static_cast<size_t>(i) * static_cast<size_t>(proj_dim);
    auto pq = hnsw.searchKnn(static_cast<const void*>(zi), static_cast<size_t>(m + 1));

    std::vector<int32_t> ids;
    ids.reserve(static_cast<size_t>(m));
    while (!pq.empty() && static_cast<int32_t>(ids.size()) < m) {
      const int32_t id = static_cast<int32_t>(pq.top().second);
      pq.pop();
      if (id == i) {
        continue;
      }
      if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
        ids.push_back(id);
      }
    }
    nbrs[static_cast<size_t>(i)] = std::move(ids);
  }
  return nbrs;
}

std::vector<std::vector<int32_t>> build_lowdim_topm_brute(const std::vector<float>& z,
                                                           int32_t n,
                                                           int32_t proj_dim,
                                                           int32_t m) {
  std::vector<std::vector<int32_t>> nbrs(static_cast<size_t>(n));
  if (n <= 1) {
    return nbrs;
  }

  const int32_t keep = std::min(m, n - 1);
  std::vector<std::pair<float, int32_t>> dist_ids;
  dist_ids.reserve(static_cast<size_t>(n - 1));

  auto by_dist_then_id = [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    return a.second < b.second;
  };

  for (int32_t i = 0; i < n; ++i) {
    dist_ids.clear();
    const float* zi = z.data() + static_cast<size_t>(i) * static_cast<size_t>(proj_dim);

    for (int32_t j = 0; j < n; ++j) {
      if (j == i) {
        continue;
      }
      const float* zj = z.data() + static_cast<size_t>(j) * static_cast<size_t>(proj_dim);
      dist_ids.emplace_back(l2_sq(zi, zj, proj_dim), j);
    }

    if (static_cast<int32_t>(dist_ids.size()) > keep) {
      std::nth_element(dist_ids.begin(), dist_ids.begin() + keep, dist_ids.end(), by_dist_then_id);
      dist_ids.resize(static_cast<size_t>(keep));
    }
    std::sort(dist_ids.begin(), dist_ids.end(), by_dist_then_id);

    std::vector<int32_t> ids;
    ids.reserve(static_cast<size_t>(keep));
    for (const auto& item : dist_ids) {
      ids.push_back(item.second);
    }
    nbrs[static_cast<size_t>(i)] = std::move(ids);
  }

  return nbrs;
}

std::vector<std::vector<int32_t>> build_compat_graph(const FBinData& xq,
                                                     const std::vector<std::vector<int32_t>>& cand,
                                                     float tau_edge) {
  const int32_t n = xq.rows;
  const int32_t d = xq.dim;
  const float tau_edge_sq = tau_edge * tau_edge;

  std::vector<std::vector<int32_t>> g(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    const float* xi = xq.data.data() + static_cast<size_t>(i) * static_cast<size_t>(d);
    for (int32_t j : cand[static_cast<size_t>(i)]) {
      if (j <= i) {
        continue;
      }
      const float* xj = xq.data.data() + static_cast<size_t>(j) * static_cast<size_t>(d);
      const float dij = l2_sq(xi, xj, d);
      if (dij <= tau_edge_sq) {
        g[static_cast<size_t>(i)].push_back(j);
        g[static_cast<size_t>(j)].push_back(i);
      }
    }
  }
  return g;
}

static bool can_add_under_radius(const FBinData& xq,
                                 const std::vector<int32_t>& cluster,
                                 const std::vector<float>& sum_vec,
                                 int32_t cand,
                                 float tau_cluster_sq) {
  const int32_t d = xq.dim;
  const float* cand_vec = xq.data.data() + static_cast<size_t>(cand) * static_cast<size_t>(d);

  std::vector<float> center(static_cast<size_t>(d), 0.0f);
  const float inv = 1.0f / static_cast<float>(cluster.size() + 1);
  for (int32_t i = 0; i < d; ++i) {
    center[static_cast<size_t>(i)] = (sum_vec[static_cast<size_t>(i)] + cand_vec[i]) * inv;
  }

  for (int32_t id : cluster) {
    const float* x = xq.data.data() + static_cast<size_t>(id) * static_cast<size_t>(d);
    if (l2_sq(x, center.data(), d) > tau_cluster_sq) {
      return false;
    }
  }
  if (l2_sq(cand_vec, center.data(), d) > tau_cluster_sq) {
    return false;
  }
  return true;
}

struct Stats {
  int32_t num_clusters = 0;
  int32_t min_size = 0;
  int32_t max_size = 0;
  int64_t size_gt_10 = 0;
  double size_gt_10_ratio = 0.0;
};

Stats cluster_greedy(const FBinData& xq, std::vector<std::vector<int32_t>>& g, float tau_cluster) {
  const int32_t n = xq.rows;
  const int32_t d = xq.dim;
  const float tau_cluster_sq = tau_cluster * tau_cluster;

  std::vector<int32_t> degree(static_cast<size_t>(n), 0);
  for (int32_t i = 0; i < n; ++i) {
    degree[static_cast<size_t>(i)] = static_cast<int32_t>(g[static_cast<size_t>(i)].size());
  }

  std::vector<int32_t> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    if (degree[static_cast<size_t>(a)] != degree[static_cast<size_t>(b)]) {
      return degree[static_cast<size_t>(a)] > degree[static_cast<size_t>(b)];
    }
    return a < b;
  });

  std::vector<uint8_t> assigned(static_cast<size_t>(n), 0);
  std::vector<uint8_t> in_cluster(static_cast<size_t>(n), 0);
  std::vector<int32_t> cluster_sizes;
  cluster_sizes.reserve(static_cast<size_t>(n));

  for (int32_t seed : order) {
    if (assigned[static_cast<size_t>(seed)]) {
      continue;
    }

    std::vector<int32_t> cluster;
    cluster.reserve(64);
    cluster.push_back(seed);
    in_cluster[static_cast<size_t>(seed)] = 1;

    std::vector<float> sum_vec(static_cast<size_t>(d), 0.0f);
    const float* seed_vec = xq.data.data() + static_cast<size_t>(seed) * static_cast<size_t>(d);
    for (int32_t i = 0; i < d; ++i) {
      sum_vec[static_cast<size_t>(i)] = seed_vec[i];
    }

    std::vector<int32_t> frontier;
    frontier.push_back(seed);
    size_t head = 0;

    while (head < frontier.size()) {
      const int32_t u = frontier[head++];
      for (int32_t v : g[static_cast<size_t>(u)]) {
        if (assigned[static_cast<size_t>(v)] || in_cluster[static_cast<size_t>(v)]) {
          continue;
        }

        if (can_add_under_radius(xq, cluster, sum_vec, v, tau_cluster_sq)) {
          cluster.push_back(v);
          in_cluster[static_cast<size_t>(v)] = 1;
          frontier.push_back(v);

          const float* v_vec = xq.data.data() + static_cast<size_t>(v) * static_cast<size_t>(d);
          for (int32_t i = 0; i < d; ++i) {
            sum_vec[static_cast<size_t>(i)] += v_vec[i];
          }
        }
      }
    }

    for (int32_t id : cluster) {
      assigned[static_cast<size_t>(id)] = 1;
      in_cluster[static_cast<size_t>(id)] = 0;
    }
    cluster_sizes.push_back(static_cast<int32_t>(cluster.size()));
  }

  for (int32_t i = 0; i < n; ++i) {
    if (!assigned[static_cast<size_t>(i)]) {
      cluster_sizes.push_back(1);
      assigned[static_cast<size_t>(i)] = 1;
    }
  }

  Stats st;
  st.num_clusters = static_cast<int32_t>(cluster_sizes.size());
  st.min_size = *std::min_element(cluster_sizes.begin(), cluster_sizes.end());
  st.max_size = *std::max_element(cluster_sizes.begin(), cluster_sizes.end());
  for (int32_t s : cluster_sizes) {
    if (s > 10) {
      st.size_gt_10 += s;
    }
  }
  st.size_gt_10_ratio = static_cast<double>(st.size_gt_10) / static_cast<double>(n);
  return st;
}

Stats cluster_seed_radius_component(const FBinData& xq,
                                    const std::vector<std::vector<int32_t>>& g,
                                    float tau_cluster) {
  const int32_t n = xq.rows;
  const int32_t d = xq.dim;
  const float tau_cluster_sq = tau_cluster * tau_cluster;

  std::vector<int32_t> degree(static_cast<size_t>(n), 0);
  for (int32_t i = 0; i < n; ++i) {
    degree[static_cast<size_t>(i)] = static_cast<int32_t>(g[static_cast<size_t>(i)].size());
  }

  std::vector<int32_t> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    if (degree[static_cast<size_t>(a)] != degree[static_cast<size_t>(b)]) {
      return degree[static_cast<size_t>(a)] > degree[static_cast<size_t>(b)];
    }
    return a < b;
  });

  std::vector<uint8_t> assigned(static_cast<size_t>(n), 0);
  std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
  std::vector<int32_t> touched;
  std::vector<int32_t> component;
  std::vector<int32_t> cluster_sizes;
  touched.reserve(static_cast<size_t>(n));
  component.reserve(static_cast<size_t>(n));
  cluster_sizes.reserve(static_cast<size_t>(n));

  for (int32_t seed : order) {
    if (assigned[static_cast<size_t>(seed)]) {
      continue;
    }

    component.clear();
    touched.clear();
    component.push_back(seed);
    touched.push_back(seed);
    seen[static_cast<size_t>(seed)] = 1;

    size_t head = 0;
    while (head < component.size()) {
      const int32_t u = component[head++];
      for (int32_t v : g[static_cast<size_t>(u)]) {
        if (assigned[static_cast<size_t>(v)] || seen[static_cast<size_t>(v)]) {
          continue;
        }
        seen[static_cast<size_t>(v)] = 1;
        touched.push_back(v);
        component.push_back(v);
      }
    }

    const float* seed_vec = xq.data.data() + static_cast<size_t>(seed) * static_cast<size_t>(d);
    int32_t cluster_size = 0;
    for (int32_t v : component) {
      const float* v_vec = xq.data.data() + static_cast<size_t>(v) * static_cast<size_t>(d);
      if (l2_sq(seed_vec, v_vec, d) <= tau_cluster_sq) {
        assigned[static_cast<size_t>(v)] = 1;
        ++cluster_size;
      }
    }
    cluster_sizes.push_back(cluster_size);

    for (int32_t v : touched) {
      seen[static_cast<size_t>(v)] = 0;
    }
  }

  for (int32_t i = 0; i < n; ++i) {
    if (!assigned[static_cast<size_t>(i)]) {
      cluster_sizes.push_back(1);
      assigned[static_cast<size_t>(i)] = 1;
    }
  }

  Stats st;
  st.num_clusters = static_cast<int32_t>(cluster_sizes.size());
  st.min_size = *std::min_element(cluster_sizes.begin(), cluster_sizes.end());
  st.max_size = *std::max_element(cluster_sizes.begin(), cluster_sizes.end());
  for (int32_t s : cluster_sizes) {
    if (s > 10) {
      st.size_gt_10 += s;
    }
  }
  st.size_gt_10_ratio = static_cast<double>(st.size_gt_10) / static_cast<double>(n);
  return st;
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const FBinData xq = read_fbin(args.xq_path);

    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> z = random_project(xq, args.proj_dim, args.seed);
    const auto t1 = std::chrono::steady_clock::now();

    std::vector<std::vector<int32_t>> cand;
    if (args.knn_mode == "brute") {
      cand = build_lowdim_topm_brute(z, xq.rows, args.proj_dim, args.m);
    } else {
      cand = build_lowdim_topm_hnsw(z, xq.rows, args.proj_dim, args.m, args.hnsw_m, args.hnsw_efc, args.hnsw_ef);
    }
    const auto t2 = std::chrono::steady_clock::now();

    std::vector<std::vector<int32_t>> g = build_compat_graph(xq, cand, args.tau_edge);
    const auto t3 = std::chrono::steady_clock::now();

    const Stats st = (args.cluster_mode == "seed-radius")
      ? cluster_seed_radius_component(xq, g, args.tau_cluster)
      : cluster_greedy(xq, g, args.tau_cluster);
    const auto t4 = std::chrono::steady_clock::now();

    const double proj_time = std::chrono::duration<double>(t1 - t0).count();
    const double topm_time = std::chrono::duration<double>(t2 - t1).count();
    const double edge_time = std::chrono::duration<double>(t3 - t2).count();
    const double cluster_time = std::chrono::duration<double>(t4 - t3).count();
    const double total_time = std::chrono::duration<double>(t4 - t0).count();

    std::cout << "===== projected-greedy-cluster-cpp =====\n";
    std::cout << "xq: " << args.xq_path << "\n";
    std::cout << "knn_mode=" << args.knn_mode << ", cluster_mode=" << args.cluster_mode
              << ", proj_dim=" << args.proj_dim << ", M=" << args.m << ", tau_edge=" << args.tau_edge
              << ", tau_cluster=" << args.tau_cluster << ", seed=" << args.seed << "\n";
    std::cout << "hnsw_params: M=" << args.hnsw_m << ", efc=" << args.hnsw_efc << ", ef=" << args.hnsw_ef
          << "\n";
    std::cout << "num_vectors: " << xq.rows << ", dim: " << xq.dim << "\n";
    std::cout << "num_clusters: " << st.num_clusters << "\n";
    std::cout << "cluster_size_range: [" << st.min_size << ", " << st.max_size << "]\n";
    std::cout << "queries_in_size_gt_10_clusters: " << st.size_gt_10 << "\n";
    std::cout << "queries_in_size_gt_10_ratio: " << st.size_gt_10_ratio << "\n";
    std::cout << "queries_in_size_gt_10_percent: " << (st.size_gt_10_ratio * 100.0) << "%\n";
    std::cout << "time_projection_sec: " << proj_time << "\n";
    std::cout << "time_topm_sec: " << topm_time << "\n";
    std::cout << "time_edge_build_sec: " << edge_time << "\n";
    std::cout << "time_cluster_sec: " << cluster_time << "\n";
    std::cout << "time_total_sec: " << total_time << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
