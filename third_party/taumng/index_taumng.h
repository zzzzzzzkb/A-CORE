#pragma once
// Tau-MNG index header (compatible with efanna2e / NSG codebase)

#include <cassert>
#include <cstddef>
#include <mutex>
#include <stack>
#include <set>
#include <map>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include <efanna2e/index.h>
#include <efanna2e/parameters.h>
#include <efanna2e/neighbor.h>
#include <efanna2e/util.h>

namespace efanna2e {

// forward declarations come from efanna2e/neighbor.h; we include that above.
// struct SimpleNeighbor;
// struct nhood;

class IndexTauMNG : public Index {
 public:
  const std::vector<std::vector<unsigned>>& graph() const { return final_graph_; }
  unsigned entry_point() const { return ep_; }
  explicit IndexTauMNG(size_t dimension, size_t n,
                       Metric m = L2, Index* initializer = nullptr);
  ~IndexTauMNG() override;
// === Add safe getters in public section of IndexTauMNG ===
  // 入口点（ep_）
  unsigned EntryPoint() const { return ep_; }

  // 点的数量
  size_t NumPoints() const { return nd_; }

  // 图的邻接表（只读引用）
  const std::vector<std::vector<unsigned>>& Graph() const { return final_graph_; }

  // 第 i 个点的出度
  unsigned OutDegree(size_t i) const {
    return (i < final_graph_.size()) ? static_cast<unsigned>(final_graph_[i].size()) : 0u;
  }

  // Persist / load (same binary layout as NSG .nsg)
  void Save(const char* filename) override;
  void Load(const char* filename) override;

  // Build tau-MNG graph given an external KNN graph (path via Parameters)
  void Build(size_t n, const float* data, const Parameters& parameters) override;

  // Search variants kept from the original implementation
  void Search(const float* query, const float* x, size_t K,
              const Parameters& parameters, unsigned* indices,
              std::vector<std::vector<int>>& perm_list,
              int gtNN, float* trans_q, float* trans_data);

  void Search_QEO(const float* query, const float* x, size_t K,
                  const Parameters& parameters, unsigned* indices,
                  std::vector<std::vector<int>>& perm_list,
                  int gtNN, float* trans_q, float* trans_data);

  // Declared (some repos forgot this in the header)
  void Search_QEO_PDP(const float* query, const float* x, size_t K,
                      const Parameters& parameters, unsigned* indices,
                      std::vector<std::vector<int>>& perm_list,
                      int gtNN, float* trans_q, float* trans_data);

  void Search_QEO_PDP_PII(const float* query, const float* x, size_t K,
                          const Parameters& parameters, unsigned* indices,
                          std::vector<std::vector<float>>& uT2,
                          float* trans_q, float* trans_data,
                          float* data_step_sum,
                          std::vector<float>& trans_q_steps);

  float eval_recall(std::vector<std::vector<unsigned>> query_res,
                    std::vector<std::vector<int>> gts, int K);

 protected:
  using CompactGraph = std::vector<std::vector<unsigned>>;
  using LockGraph    = std::vector<SimpleNeighbor>; // NOTE: singular "SimpleNeighbor"
  using KNNGraph     = std::vector<nhood>;

  // Build helpers (NSG-style pipeline)
  void init_graph(const Parameters& parameters);

  void get_neighbors(const float* query, const Parameters& parameter,
                     std::vector<Neighbor>& retset,
                     std::vector<Neighbor>& fullset);

  void get_neighbors(const float* query, const Parameters& parameter,
                     boost::dynamic_bitset<>& flags,
                     std::vector<Neighbor>& retset,
                     std::vector<Neighbor>& fullset);

  void InterInsert(unsigned n, unsigned range,
                   std::vector<std::mutex>& locks, SimpleNeighbor* cut_graph_);

  void sync_prune(unsigned q, std::vector<Neighbor>& pool,
                  const Parameters& parameter,
                  boost::dynamic_bitset<>& flags,
                  SimpleNeighbor* cut_graph_);

  void Link(const Parameters& parameters, SimpleNeighbor* cut_graph_);
  void Load_nn_graph(const char* filename);
  void tree_grow(const Parameters& parameter);
  void DFS(boost::dynamic_bitset<>& flag, unsigned root, unsigned& cnt);
  void findroot(boost::dynamic_bitset<>& flag, unsigned& root,
                const Parameters& parameter);
virtual void Search(const float* query, const float* x, size_t K,
                    const Parameters& parameters, unsigned* indices) override;

 protected:
  // Graph
  CompactGraph final_graph_{};
  Index*       initializer_{nullptr};

  // Build/search state
  unsigned              width{0};
  unsigned              ep_{0};
  std::vector<std::mutex> locks;

  // Optional optimized layout (kept for compatibility with NSG-style optimize)
  char*  opt_graph_{nullptr};
  size_t node_size{0};
  size_t data_len{0};
  size_t neighbor_len{0};
  KNNGraph nnd_graph;

 public:
  // Stats / controls (kept from upstream code)
  std::vector<std::vector<float>> final_graph_edge_length_{};
  double NDC{0.0};
  int    hops{0};
  float  ang{0.0f};
  float  avg_tau{0.0f};
  double comp_amount{0.0};
  float  tau{0.0f};

  // Some variants use these; safe defaults provided
  double not_break_comp_NDC{0.0};
  double not_break_comp_amount{0.0};
};

} // namespace efanna2e
