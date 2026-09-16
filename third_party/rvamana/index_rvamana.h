#pragma once
#include <efanna2e/index.h>
#include <efanna2e/parameters.h>
#include <efanna2e/neighbor.h>
#include <boost/dynamic_bitset.hpp>
#include <mutex>
#include <vector>
#include <string>

namespace efanna2e {

/* RobustVamana: 与 NSG 类似的图搜索框架，
 * 差异点在 prune：采用 alpha-robust 修剪策略（更保守，提升稳健性）
 * 存储格式与 NSG 一致：width, ep, [deg, neighbors...]*N
 */
class IndexRVa : public Index {
 public:
  explicit IndexRVa(size_t dim, size_t n, Metric m = L2, Index* initializer = nullptr);
  ~IndexRVa() override;

  void Save(const char* filename) override;
  void Load(const char* filename) override;

  void Build(size_t n, const float* data, const Parameters& parameters) override;

  // 可选：和 NSG 一样的图上搜索（给 check 用）
  void Search(const float* query, const float* x, size_t K,
              const Parameters& parameters, unsigned* indices) override;

  // 供主程序设置的公开参数
  float alpha = 1.2f;
  bool  ensure_connectivity = true;

  // 读取候选 KNN 图（.graph/.knng）
  void LoadKNNGraph(const char* filename);

  // 只读访问（给 checker）
  inline const std::vector<std::vector<unsigned>>& graph() const { return final_graph_; }
  inline unsigned ep() const { return ep_; }
  inline unsigned width_param() const { return width; }

 private:
  using CompactGraph = std::vector<std::vector<unsigned>>;

  void init_entry(const Parameters& parameters);
  void get_neighbors(const float* query, const Parameters& params,
                     boost::dynamic_bitset<>& flags,
                     std::vector<Neighbor>& retset,
                     std::vector<Neighbor>& fullset);

  // Robust prune（Vamana 风格）：alpha 控制“冗余”判定的松紧
  void robust_prune(unsigned q, std::vector<Neighbor>& pool,
                    const Parameters& params,
                    boost::dynamic_bitset<>& flags,
                    SimpleNeighbor* cut_graph);

  void inter_insert(unsigned n, unsigned R, std::vector<std::mutex>& locks,
                    SimpleNeighbor* cut_graph);

  void link_phase(const Parameters& params, SimpleNeighbor* cut_graph);

  void ensure_connectivity_bfs(const Parameters& params);

 private:
  Index* initializer_ = nullptr;
  CompactGraph final_graph_;
  unsigned width = 0;
  unsigned ep_ = 0;
};

} // namespace efanna2e
