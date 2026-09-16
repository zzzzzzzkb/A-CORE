// HNSW + A-CORE with projected-greedy clustering and batch_gate skip_small.
//
// Based on run_hnsw_cluster_consensus_real_test_light_newfeat_new2_ablation_final.cpp
// but replaces oracle-label clustering with projected-greedy clustering
// (random projection → low-dim HNSW → compatibility graph → seed-radius/greedy clustering),
// and skips clusters with |C| <= batch_gate from recall/QPS computation.
//
// Mirrors the NGFix skip_small variant
// (run_ngfix_cluster_consensus_real_test_light_newfeat_new2_ablation_final_projected_greedy_batch_gate_skip_small.cpp)
// but adapted for the HNSW (hnswlib) index.

#include <hnswlib/hnswlib.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;
using idx_t = size_t;

// ================================================
// Minimal LightGBM TXT inference (GBDT)
// ================================================
struct GBDT {
    struct Tree {
        std::vector<int> left_child, right_child, split_feature;
        std::vector<float> threshold, leaf_value;
        int num_leaves = 0;
        int root = 0;
        float predict(const std::vector<float>& feat) const {
            int node = 0;
            while (true) {
                if (node < 0) {
                    int leaf_idx = -1 - node;
                    if (leaf_idx >= 0 && leaf_idx < (int)leaf_value.size()) return leaf_value[leaf_idx];
                    return 0.f;
                }
                if (node >= (int)split_feature.size()) {
                    if (!leaf_value.empty()) return leaf_value[0];
                    return 0.f;
                }
                int f = split_feature[node];
                float thr = threshold[node];
                float x = (f>=0 && f<(int)feat.size()) ? feat[f] : 0.f;
                int nxt = (x <= thr) ? left_child[node] : right_child[node];
                node = nxt;
            }
        }
    };

    std::vector<Tree> trees;

    static std::string trim(const std::string& s){
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        if (a==std::string::npos) return "";
        return s.substr(a, b-a+1);
    }

    bool load_from_txt(const std::string& path){
        std::ifstream fin(path);
        if (!fin) { std::cerr<<"[GBDT] cannot open "<<path<<"\n"; return false; }
        trees.clear();
        std::string line; Tree cur;
        auto flush_tree=[&](){
            if (!cur.split_feature.empty() || !cur.leaf_value.empty()){
                trees.push_back(std::move(cur));
                cur = Tree();
            }
        };
        while (std::getline(fin, line)){
            line = trim(line);
            if (line.rfind("Tree=",0)==0){ flush_tree(); continue; }
            if (line.rfind("num_leaves=",0)==0){ cur.num_leaves = std::stoi(line.substr(11)); continue; }
            if (line.rfind("split_feature=",0)==0){
                std::string s = line.substr(14);
                cur.split_feature.clear();
                std::stringstream ss(s);
                std::string tok;
                while (std::getline(ss, tok, ' ')){
                    tok = trim(tok);
                    if (tok.empty()) continue;
                    cur.split_feature.push_back(std::stoi(tok));
                }
                continue;
            }
            if (line.rfind("threshold=",0)==0){
                std::string s = line.substr(10);
                cur.threshold.clear();
                std::stringstream ss(s);
                std::string tok;
                while (std::getline(ss, tok, ' ')){
                    tok = trim(tok);
                    if (tok.empty()) continue;
                    cur.threshold.push_back(std::stof(tok));
                }
                continue;
            }
            if (line.rfind("left_child=",0)==0){
                std::string s = line.substr(11);
                cur.left_child.clear();
                std::stringstream ss(s);
                std::string tok;
                while (std::getline(ss, tok, ' ')){
                    tok = trim(tok);
                    if (tok.empty()) continue;
                    cur.left_child.push_back(std::stoi(tok));
                }
                continue;
            }
            if (line.rfind("right_child=",0)==0){
                std::string s = line.substr(12);
                cur.right_child.clear();
                std::stringstream ss(s);
                std::string tok;
                while (std::getline(ss, tok, ' ')){
                    tok = trim(tok);
                    if (tok.empty()) continue;
                    cur.right_child.push_back(std::stoi(tok));
                }
                continue;
            }
            if (line.rfind("leaf_value=",0)==0){
                std::string s = line.substr(11);
                cur.leaf_value.clear();
                std::stringstream ss(s);
                std::string tok;
                while (std::getline(ss, tok, ' ')){
                    tok = trim(tok);
                    if (tok.empty()) continue;
                    cur.leaf_value.push_back(std::stof(tok));
                }
                continue;
            }
        }
        flush_tree();
        if (trees.empty()){
            std::cerr<<"[GBDT] no trees parsed from "<<path<<"\n";
            return false;
        }
        return true;
    }

    float predict_sum(const std::vector<float>& feat) const {
        double s = 0.0;
        for (const auto& t: trees) s += (double)t.predict(feat);
        return (float)s;
    }
};

struct FeatureBank {
    std::unordered_map<std::string, float> kv;

    float get(const std::string& name, float default_val = 0.0f) const {
        auto it = kv.find(name);
        return (it == kv.end()) ? default_val : it->second;
    }
};

static std::vector<float> vector_from_names(
    const std::vector<std::string>& names,
    const FeatureBank& F
){
    std::vector<float> x; x.reserve(names.size());
    for (auto& n: names) x.push_back(F.get(n, 0.0f));
    return x;
}

// ---------- 简易 RAII 计时器 ----------
struct ScopeTimer {
    std::chrono::high_resolution_clock::time_point t0;
    double* acc;
    explicit ScopeTimer(double& x): t0(std::chrono::high_resolution_clock::now()), acc(&x) {}
    ~ScopeTimer(){
        auto t1 = std::chrono::high_resolution_clock::now();
        *acc += std::chrono::duration<double>(t1 - t0).count();
    }
};

// ---------- I/O ----------
static idx_t read_fbin(const char* f, std::vector<float>& out, int& dim) {
    FILE* fp = fopen(f, "rb"); if (!fp) { perror(f); std::exit(1); }
    int n=0,d=0; fread(&n,4,1,fp); fread(&d,4,1,fp);
    out.resize((idx_t)n*(idx_t)d);
    size_t tot=(size_t)n*(size_t)d;
    if (fread(out.data(), sizeof(float), tot, fp)!=tot) { std::cerr<<"read "<<f<<" fail\n"; std::exit(1); }
    fclose(fp); dim=d; return (idx_t)n;
}
static idx_t read_ibin(const char* f, std::vector<int>& out, int& dim) {
    FILE* fp = fopen(f, "rb"); if (!fp) { perror(f); std::exit(1); }
    int n=0,d=0; fread(&n,4,1,fp); fread(&d,4,1,fp);
    out.resize((idx_t)n*(idx_t)d);
    size_t tot=(size_t)n*(size_t)d;
    if (fread(out.data(), sizeof(int), tot, fp)!=tot) { std::cerr<<"read "<<f<<" fail\n"; std::exit(1); }
    fclose(fp); dim=d; return (idx_t)n;
}
static bool read_fbin_meta(const std::string& path, uint32_t& n, uint32_t& d) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    uint32_t nn=0, dd=0;
    bool ok = (fread(&nn, sizeof(uint32_t), 1, fp)==1) && (fread(&dd, sizeof(uint32_t), 1, fp)==1);
    fclose(fp);
    if (!ok) return false;
    n = nn; d = dd;
    return true;
}
static inline float l2sq(const float* a, const float* b, int d) {
    float s=0.f; for (int i=0;i<d;++i){ float df=a[i]-b[i]; s+=df*df; } return s;
}
static std::string first_existing(const std::string& dir, const std::vector<std::string>& names) {
    for (auto& n: names){ auto p=dir+"/"+n; if (std::filesystem::exists(p)) return p; }
    return dir + "/" + names.front();
}

static double topk_overlap_ratio_limited(const std::vector<hnswlib::tableint>& A,
                                         const std::vector<hnswlib::tableint>& B,
                                         int K){
    size_t Ke = (size_t)std::max(0, K);
    Ke = std::min<size_t>(Ke, std::min(A.size(), B.size()));
    if (Ke == 0) return 0.0;
    std::unordered_set<hnswlib::tableint> sa; sa.reserve(Ke*2);
    std::unordered_set<hnswlib::tableint> sb; sb.reserve(Ke*2);
    for(size_t i=0;i<Ke;++i) sa.insert(A[i]);
    for(size_t i=0;i<Ke;++i) sb.insert(B[i]);
    size_t inter=0;
    if(sa.size()<sb.size()){ for(auto v: sa) if(sb.count(v)) ++inter; }
    else { for(auto v: sb) if(sa.count(v)) ++inter; }
    return (double)inter / (double)Ke;
}
static double topk_jaccard_limited(const std::vector<hnswlib::tableint>& A,
                                   const std::vector<hnswlib::tableint>& B,
                                   int K){
    size_t Ke = (size_t)std::max(0, K);
    Ke = std::min<size_t>(Ke, std::min(A.size(), B.size()));
    if (Ke == 0) return 0.0;
    std::unordered_set<hnswlib::tableint> s; s.reserve(Ke*3);
    size_t inter=0;
    for(size_t i=0;i<Ke;++i) s.insert(A[i]);
    for(size_t i=0;i<Ke;++i) inter += s.count(B[i]) ? 1 : 0;
    for(size_t i=0;i<Ke;++i) s.insert(B[i]);
    size_t uni = s.size();
    return (uni>0)? (double)inter/(double)uni : 0.0;
}

// --- compute LID/RC/Expansion from a sorted top-id list ---
static void compute_lid_rc_expansion_from_top(
    const hnswlib::HierarchicalNSW<float>& hnsw,
    const float* qvec, int dim,
    const std::vector<hnswlib::tableint>& top_ids,
    int k,
    float& out_lid, float& out_rc, float& out_expand2k_over_k)
{
    out_lid = 0.f; out_rc = 0.f; out_expand2k_over_k = 1.f;
    if (top_ids.empty() || k <= 0) return;

    std::vector<float> d; d.reserve(top_ids.size());
    for (auto id: top_ids){
        const float* xv = reinterpret_cast<const float*>(hnsw.getDataByInternalId(id));
        d.push_back(std::sqrt(l2sq(qvec, xv, dim)));
    }
    std::sort(d.begin(), d.end());
    int kk = std::min(k, (int)d.size());
    if (kk <= 0) return;

    const float eps = 1e-12f;
    float r_k = d[kk-1];
    float s = 0.f;
    for (int i=0;i<kk;++i){
        float ratio = (r_k > eps) ? (d[i]/(r_k+eps)) : 1.f;
        ratio = std::max(ratio, eps);
        s += std::log(ratio);
    }
    float mean_log = s / std::max(1, kk);
    out_lid = (std::abs(mean_log) > 1e-12f) ? (-1.0f/mean_log) : 0.f;

    double mean = 0.0; for (float v: d) mean += v; mean /= (double)std::max<size_t>(1, d.size());
    out_rc = (r_k > eps) ? (float)(mean / r_k) : 0.f;

    int k2 = std::min((int)d.size(), kk*2);
    float r_k2 = d[k2-1];
    out_expand2k_over_k = (r_k > eps) ? (r_k2 / r_k) : 1.f;
}

static float compute_recall_subset(const std::vector<std::vector<idx_t>>& results,
                                   const std::vector<int>& gt,
                                   int k,
                                   const std::vector<idx_t>& qids) {
    int nq = (int)results.size();
    if (nq == 0) return 0.f;
    int per_row = (int)gt.size() / nq;
    int expect  = std::min(k, per_row);

    long long tot_hit = 0, tot_need = 0;
    for (auto qi : qids){
        int hit = 0;
        const auto& R = results[qi];
        for (int j=0; j<expect; ++j){
            int g = gt[(size_t)qi * per_row + j];
            if (g < 0) break;
            for (int t=0; t<std::min((int)R.size(), k); ++t){
                if ((int)R[t] == g){ ++hit; break; }
            }
        }
        tot_hit  += hit;
        tot_need += expect;
    }
    return tot_need>0 ? (float)tot_hit/(float)tot_need : 0.f;
}

// ---------- 扩展 HNSW：暴露 L0 邻接 & 从"当时工具"继续 ----------
struct WarmHierarchicalNSW : public hnswlib::HierarchicalNSW<float> {
    using Base=hnswlib::HierarchicalNSW<float>;
    using tableint=hnswlib::tableint; using labeltype=hnswlib::labeltype;
    explicit WarmHierarchicalNSW(hnswlib::SpaceInterface<float>* s, size_t max_elements, size_t M, size_t efc)
        : Base(s, max_elements, M, efc) {}
    const hnswlib::linklistsizeint* linklist0(tableint id) const { return this->get_linklist0(id); }
    tableint entrypoint() const { return this->enterpoint_node_; }

    tableint entryPointL0ForQuery(const void* q) const {
        if (this->enterpoint_node_ == (tableint)-1 || this->cur_element_count == 0) return (tableint)-1;
        tableint curr = this->enterpoint_node_;
        float currDist = this->fstdistfunc_(q, this->getDataByInternalId(curr), this->dist_func_param_);
        int currLevel = this->maxlevel_;
        for (int level = currLevel; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                const auto* ll = this->get_linklist(curr, level);
                if (!ll) break;
                unsigned sz = *ll;
                const tableint* nb = reinterpret_cast<const tableint*>(ll + 1);
                for (unsigned i = 0; i < sz; ++i) {
                    tableint v = nb[i];
                    float d = this->fstdistfunc_(q, this->getDataByInternalId(v), this->dist_func_param_);
                    if (d < currDist) { currDist = d; curr = v; changed = true; }
                }
            }
        }
        return curr;
    }

    void searchL0Heaps(const void* q, size_t ef,
                       std::vector<tableint>& out_cand_ids,
                       std::vector<tableint>& out_top_ids,
                       std::vector<tableint>& out_visited_ids) const
    {
        out_cand_ids.clear(); out_top_ids.clear(); out_visited_ids.clear();
        if (this->cur_element_count == 0) return;

        hnswlib::VisitedList* vl = this->visited_list_pool_->getFreeVisitedList();
        vl->reset();
        auto* vis = vl->mass;

        struct MinCmp { bool operator()(const std::pair<float,tableint>&a,const std::pair<float,tableint>&b) const {return a.first>b.first;} };
        std::priority_queue<std::pair<float,tableint>, std::vector<std::pair<float,tableint>>, MinCmp> cand;
        std::priority_queue<std::pair<float,tableint>> top;

        auto mark = [&](tableint id){
            if (id>=this->cur_element_count) return false;
            if (vis[id]==vl->curV) return false;
            vis[id]=vl->curV; out_visited_ids.push_back(id); return true;
        };

        auto push_both = [&](tableint id, float d){
            cand.emplace(d,id);
            top.emplace(d,id);
            if (top.size()>ef) top.pop();
        };

        tableint ep = entryPointL0ForQuery(q);
        if (ep == (tableint)-1) ep = this->enterpoint_node_;
        if (ep != (tableint)-1) {
            mark(ep);
            float d0 = this->fstdistfunc_(q, this->getDataByInternalId(ep), this->dist_func_param_);
            push_both(ep, d0);
        }

        float lb = top.empty()? std::numeric_limits<float>::infinity() : top.top().first;

        while (!cand.empty()){
            auto cu = cand.top(); if (cu.first>lb) break; cand.pop();
            const auto* l0 = this->get_linklist0(cu.second);
            unsigned sz=*l0; const auto* nb=reinterpret_cast<const tableint*>(l0+1);
            for (unsigned i=0;i<sz;++i){
                tableint v=nb[i];
                if (!mark(v)) continue;
                float d=this->fstdistfunc_(q,this->getDataByInternalId(v), this->dist_func_param_);
                if (top.size()<ef || d<lb){ cand.emplace(d,v); top.emplace(d,v); if (top.size()>ef) top.pop(); lb=top.top().first; }
            }
        }

        {
            auto cand_copy = cand;
            std::vector<std::pair<float,tableint>> tmp;
            tmp.reserve(ef);
            for (size_t i=0; i<ef && !cand_copy.empty(); ++i){
                tmp.push_back(cand_copy.top());
                cand_copy.pop();
            }
            std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
            out_cand_ids.reserve(tmp.size());
            for (auto &p : tmp) out_cand_ids.push_back(p.second);
        }
        {
            auto top_copy = top;
            std::vector<std::pair<float,tableint>> tmp;
            tmp.reserve(ef);
            for (size_t i=0; i<ef && !top_copy.empty(); ++i){
                tmp.push_back(top_copy.top());
                top_copy.pop();
            }
            std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
            out_top_ids.reserve(tmp.size());
            for (auto &p : tmp) out_top_ids.push_back(p.second);
        }

        this->visited_list_pool_->releaseVisitedList(vl);
    }

    // Continue L0 search from a snapshot (seed cand/top), with ef and m (seed count).
    // Returns k results as priority_queue<dist, label>.
    std::priority_queue<std::pair<float,labeltype>>
    continueFromSnapshotL0(const void* q, size_t k,
                           const std::vector<tableint>& init_cand_ids,
                           const std::vector<tableint>& init_top_ids,
                           size_t ef, float m) const {
        hnswlib::VisitedList* vl = this->visited_list_pool_->getFreeVisitedList();
        vl->reset();
        auto* vis = vl->mass;

        struct MinCmp { bool operator()(const std::pair<float,tableint>&a,const std::pair<float,tableint>&b) const {return a.first>b.first;} };
        std::priority_queue<std::pair<float,tableint>, std::vector<std::pair<float,tableint>>, MinCmp> cand;
        std::priority_queue<std::pair<float,tableint>> top;

        auto try_emplace = [&](tableint id){
            if (id>=this->cur_element_count) return;
            if (vis[id]!=vl->curV) vis[id]=vl->curV;
            float d = this->fstdistfunc_(q,this->getDataByInternalId(id), this->dist_func_param_);
            cand.emplace(d,id);
            top.emplace(d,id);
        };

        int count=0;
        int max_count = (int)m;
        for (auto id: init_top_ids) {
            if(count>=max_count) break;
            try_emplace(id);
            ++count;
        }

        while (top.size()>ef) top.pop();

        float lb = top.empty()? std::numeric_limits<float>::infinity() : top.top().first;

        while (!cand.empty()){
            auto cu = cand.top(); if (cu.first>lb) break; cand.pop();
            auto* l0 = this->get_linklist0(cu.second);
            unsigned sz=*l0; auto* nb=reinterpret_cast<tableint*>(l0+1);
            for (unsigned i=0;i<sz;++i){
                tableint v=nb[i];
                if (vis[v]==vl->curV) continue;
                vis[v]=vl->curV;
                float d=this->fstdistfunc_(q,this->getDataByInternalId(v), this->dist_func_param_);
                if (top.size()<ef || d<lb){ cand.emplace(d,v); top.emplace(d,v); if (top.size()>ef) top.pop(); lb=top.top().first; }
            }
        }

        this->visited_list_pool_->releaseVisitedList(vl);
        while (top.size()>k) top.pop();
        std::priority_queue<std::pair<float,labeltype>> res;
        while(!top.empty()){
            res.emplace(top.top().first, (labeltype)this->getExternalLabel(top.top().second));
            top.pop();
        }
        return res;
    }
};

// ================================================
// Projected-Greedy Clustering (for skip_small)
// ================================================
struct PGClusterOut {
    std::vector<std::vector<idx_t>> clusters;
    size_t non_empty = 0;
    size_t min_size = 0;
    size_t max_size = 0;
};

static std::vector<float> random_project_pg(const std::vector<float>& xq, idx_t n, int d, int proj_dim, unsigned seed) {
    std::vector<float> proj((size_t)n * (size_t)proj_dim, 0.0f);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f / std::sqrt((float)proj_dim));

    std::vector<float> R((size_t)proj_dim * (size_t)d);
    for (size_t i = 0; i < R.size(); ++i) {
        R[i] = gauss(rng);
    }

    for (idx_t i = 0; i < n; ++i) {
        const float* x = xq.data() + i * (idx_t)d;
        float* z = proj.data() + i * (idx_t)proj_dim;
        for (int r = 0; r < proj_dim; ++r) {
            const float* row = R.data() + (size_t)r * (size_t)d;
            float acc = 0.0f;
            for (int c = 0; c < d; ++c) {
                acc += row[c] * x[c];
            }
            z[r] = acc;
        }
    }
    return proj;
}

static std::vector<std::vector<int>> build_lowdim_topm_hnsw_pg(const std::vector<float>& z,
                                                               idx_t n,
                                                               int proj_dim,
                                                               int m,
                                                               int hnsw_m,
                                                               int hnsw_efc,
                                                               int hnsw_ef) {
    std::vector<std::vector<int>> nbrs((size_t)n);
    hnswlib::L2Space space((size_t)proj_dim);
    hnswlib::HierarchicalNSW<float> hnsw(&space, (size_t)n, (size_t)hnsw_m, (size_t)hnsw_efc);

    for (idx_t i = 0; i < n; ++i) {
        const float* zi = z.data() + i * (idx_t)proj_dim;
        hnsw.addPoint((const void*)zi, (size_t)i);
    }
    hnsw.setEf((size_t)std::max(hnsw_ef, m + 1));

    for (idx_t i = 0; i < n; ++i) {
        const float* zi = z.data() + i * (idx_t)proj_dim;
        auto pq = hnsw.searchKnn((const void*)zi, (size_t)(m + 1));
        std::vector<int> ids;
        ids.reserve((size_t)m);
        while (!pq.empty() && (int)ids.size() < m) {
            int id = (int)pq.top().second;
            pq.pop();
            if (id == (int)i) continue;
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
                ids.push_back(id);
            }
        }
        nbrs[(size_t)i] = std::move(ids);
    }
    return nbrs;
}

static std::vector<std::vector<int>> build_compat_graph_pg(const std::vector<float>& xq,
                                                           idx_t n,
                                                           int d,
                                                           const std::vector<std::vector<int>>& cand,
                                                           float tau_edge) {
    const float tau_edge_sq = tau_edge * tau_edge;
    std::vector<std::vector<int>> g((size_t)n);
    for (idx_t i = 0; i < n; ++i) {
        const float* xi = xq.data() + i * (idx_t)d;
        for (int j : cand[(size_t)i]) {
            if (j <= (int)i) continue;
            const float* xj = xq.data() + (idx_t)j * (idx_t)d;
            float dij = l2sq(xi, xj, d);
            if (dij <= tau_edge_sq) {
                g[(size_t)i].push_back(j);
                g[(size_t)j].push_back((int)i);
            }
        }
    }
    return g;
}

static bool can_add_under_radius_pg(const std::vector<float>& xq,
                                    int d,
                                    const std::vector<int>& cluster,
                                    const std::vector<float>& sum_vec,
                                    int cand,
                                    float tau_cluster_sq) {
    const float* cand_vec = xq.data() + (idx_t)cand * (idx_t)d;
    std::vector<float> center((size_t)d, 0.0f);
    float inv = 1.0f / (float)(cluster.size() + 1);
    for (int i = 0; i < d; ++i) {
        center[(size_t)i] = (sum_vec[(size_t)i] + cand_vec[i]) * inv;
    }

    for (int id : cluster) {
        const float* x = xq.data() + (idx_t)id * (idx_t)d;
        if (l2sq(x, center.data(), d) > tau_cluster_sq) {
            return false;
        }
    }
    if (l2sq(cand_vec, center.data(), d) > tau_cluster_sq) {
        return false;
    }
    return true;
}

static PGClusterOut build_projected_greedy_clusters(const std::vector<float>& xq,
                                                    idx_t n,
                                                    int d,
                                                    int proj_dim,
                                                    int m,
                                                    int cl_hnsw_m,
                                                    int cl_hnsw_efc,
                                                    int cl_hnsw_ef,
                                                    float tau_edge,
                                                    float tau_cluster,
                                                    unsigned seed,
                                                    const std::string& cluster_mode) {
    PGClusterOut out;
    if (n == 0) return out;

    std::vector<float> z = random_project_pg(xq, n, d, proj_dim, seed);
    auto cand = build_lowdim_topm_hnsw_pg(z, n, proj_dim, m, cl_hnsw_m, cl_hnsw_efc, cl_hnsw_ef);
    auto g = build_compat_graph_pg(xq, n, d, cand, tau_edge);

    std::vector<int> degree((size_t)n, 0);
    for (idx_t i = 0; i < n; ++i) {
        degree[(size_t)i] = (int)g[(size_t)i].size();
    }

    std::vector<int> order((size_t)n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (degree[(size_t)a] != degree[(size_t)b]) return degree[(size_t)a] > degree[(size_t)b];
        return a < b;
    });

    std::vector<uint8_t> assigned((size_t)n, 0);
    const float tau_cluster_sq = tau_cluster * tau_cluster;

    if (cluster_mode == "seed-radius") {
        std::vector<uint8_t> seen((size_t)n, 0);
        std::vector<int> touched;
        std::vector<int> component;
        touched.reserve((size_t)n);
        component.reserve((size_t)n);

        for (int seed_id : order) {
            if (assigned[(size_t)seed_id]) continue;

            touched.clear();
            component.clear();
            touched.push_back(seed_id);
            component.push_back(seed_id);
            seen[(size_t)seed_id] = 1;

            size_t head = 0;
            while (head < component.size()) {
                int u = component[head++];
                for (int v : g[(size_t)u]) {
                    if (assigned[(size_t)v] || seen[(size_t)v]) continue;
                    seen[(size_t)v] = 1;
                    touched.push_back(v);
                    component.push_back(v);
                }
            }

            const float* seed_vec = xq.data() + (idx_t)seed_id * (idx_t)d;
            std::vector<idx_t> one;
            one.reserve(component.size());
            for (int id : component) {
                const float* x = xq.data() + (idx_t)id * (idx_t)d;
                if (l2sq(seed_vec, x, d) <= tau_cluster_sq) {
                    assigned[(size_t)id] = 1;
                    one.push_back((idx_t)id);
                }
            }
            out.clusters.push_back(std::move(one));

            for (int id : touched) seen[(size_t)id] = 0;
        }
    } else {
        // greedy
        std::vector<uint8_t> in_cluster((size_t)n, 0);
        for (int seed_id : order) {
            if (assigned[(size_t)seed_id]) continue;

            std::vector<int> cluster;
            cluster.reserve(64);
            cluster.push_back(seed_id);
            in_cluster[(size_t)seed_id] = 1;

            std::vector<float> sum_vec((size_t)d, 0.0f);
            const float* seed_vec = xq.data() + (idx_t)seed_id * (idx_t)d;
            for (int i = 0; i < d; ++i) sum_vec[(size_t)i] = seed_vec[i];

            std::vector<int> frontier;
            frontier.push_back(seed_id);
            size_t head = 0;
            while (head < frontier.size()) {
                int u = frontier[head++];
                for (int v : g[(size_t)u]) {
                    if (assigned[(size_t)v] || in_cluster[(size_t)v]) continue;
                    if (!can_add_under_radius_pg(xq, d, cluster, sum_vec, v, tau_cluster_sq)) continue;

                    cluster.push_back(v);
                    in_cluster[(size_t)v] = 1;
                    frontier.push_back(v);
                    const float* v_vec = xq.data() + (idx_t)v * (idx_t)d;
                    for (int i = 0; i < d; ++i) sum_vec[(size_t)i] += v_vec[i];
                }
            }

            std::vector<idx_t> one;
            one.reserve(cluster.size());
            for (int id : cluster) {
                assigned[(size_t)id] = 1;
                in_cluster[(size_t)id] = 0;
                one.push_back((idx_t)id);
            }
            out.clusters.push_back(std::move(one));
        }
    }

    for (idx_t i = 0; i < n; ++i) {
        if (assigned[(size_t)i]) continue;
        assigned[(size_t)i] = 1;
        out.clusters.push_back(std::vector<idx_t>{i});
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

// ---------- 参数与主程序 ----------
static std::vector<float> parse_R_targets(const std::string& raw){
    std::string s = raw;
    std::vector<float> out;
    if (s.find(':') != std::string::npos){
        std::stringstream ss(s);
        std::string a_str, st_str, b_str;
        if (std::getline(ss, a_str, ':') && std::getline(ss, st_str, ':') && std::getline(ss, b_str, ':')){
            double a = std::stod(a_str);
            double st= std::stod(st_str);
            double b = std::stod(b_str);
            if (std::abs(st) < 1e-12) { out.push_back((float)a); return out; }
            if ((st > 0 && a > b) || (st < 0 && a < b)) std::swap(a, b);
            if (st > 0){
                for (double x=a; x<=b+1e-12; x+=st) out.push_back((float)x);
            }else{
                for (double x=a; x>=b-1e-12; x+=st) out.push_back((float)x);
            }
            if (out.empty()) out.push_back((float)a);
            return out;
        }
    }
    try { out.push_back(std::stof(s)); } catch (...) {}
    return out;
}

struct Args {
    std::string dir, index_path;
    int ef=200, k=10, ef_warm=120;
    int k_collect=100;
    int batch_gate=8;
    unsigned seed=42;
    int proj_dim=32;
    int proj_m=96;
    int cl_hnsw_m=16;
    int cl_hnsw_efc=100;
    int cl_hnsw_ef=128;
    float tau_edge=0.5f;
    float tau_cluster=0.5f;
    std::string cluster_mode="seed-radius";
    std::string load_synth_prefix;

    float R_target = 0.90f;
    std::vector<float> R_targets;
    float L_floor = 1.f;
    float L_step  = 0.05f;
    float L_safety = 0.0f;

    std::string model_dir = "model_out_txt";
    std::string model_A_txt = "model_A_efc_mono.txt";
    std::string model_B_txt = "model_B_efw_mono.txt";
    std::string model_rank_txt = "model_rank_m_mono.txt";

    std::string csv_out;
    std::string per_query_csv_out;
};

static void usage(){
    std::cout
        << "Usage:\n  run_hnsw_cluster_consensus_real_test_light_newfeat_new2_ablation_final_projected_greedy_batch_gate_skip_small <data_dir> [index_path]\n"
        << "    --k <int> --k_collect <int> --ef <int> --ef_warm <int>\n"
        << "    --batch_gate <int> --seed <uint>\n"
        << "    --proj_dim <int> --proj_m <int> --cl_hnsw_m <int> --cl_hnsw_efc <int> --cl_hnsw_ef <int>\n"
        << "    --tau_edge <float> --tau_cluster <float>\n"
        << "    --cluster_mode <seed-radius|greedy>\n"
        << "    --load_synth_prefix <path>\n"
        << "    --R_target <float|a:s:b> --L_floor <float> --L_step <float> --L_safety <float>\n"
        << "    --model_dir <dir> --model_A <txt> --model_B <txt> --model_rank <txt>\n"
        << "    --csv_out <path> --per_query_csv <path>\n";
}

static Args parse_args(int argc, char** argv){
    if (argc<2){ usage(); std::exit(1); }
    Args a; a.dir=argv[1];
    a.index_path = (argc>2 && argv[2][0]!='-') ? argv[2] : a.dir + "/base.hnsw";
    a.csv_out = a.dir + "/grid_results.csv";
    int i = (argc>2 && argv[2][0]!='-') ? 3 : 2;
    for (; i<argc; ++i){
        std::string s=argv[i]; auto next=[&](){ if(i+1>=argc){std::cerr<<"missing after "<<s<<"\n"; std::exit(1);} return std::string(argv[++i]); };
        if      (s=="--k")            a.k=std::stoi(next());
        else if (s=="--k_collect")    a.k_collect=std::stoi(next());
        else if (s=="--ef")           a.ef=std::stoi(next());
        else if (s=="--ef_warm")      a.ef_warm=std::stoi(next());
        else if (s=="--batch_gate")   a.batch_gate=std::stoi(next());
        else if (s=="--seed")         a.seed=(unsigned)std::stoul(next());
        else if (s=="--proj_dim")     a.proj_dim=std::stoi(next());
        else if (s=="--proj_m")       a.proj_m=std::stoi(next());
        else if (s=="--cl_hnsw_m")    a.cl_hnsw_m=std::stoi(next());
        else if (s=="--cl_hnsw_efc")  a.cl_hnsw_efc=std::stoi(next());
        else if (s=="--cl_hnsw_ef")   a.cl_hnsw_ef=std::stoi(next());
        else if (s=="--tau_edge")     a.tau_edge=std::stof(next());
        else if (s=="--tau_cluster")  a.tau_cluster=std::stof(next());
        else if (s=="--cluster_mode") a.cluster_mode=next();
        else if (s=="--load_synth_prefix") a.load_synth_prefix=next();
        else if (s=="--R_target"){
            a.R_targets = parse_R_targets(next());
            if (!a.R_targets.empty()) a.R_target = a.R_targets.front();
        }
        else if (s=="--L_floor")  a.L_floor  = std::stof(next());
        else if (s=="--L_step")   a.L_step   = std::stof(next());
        else if (s=="--L_safety") a.L_safety = std::stof(next());
        else if (s=="--model_dir") a.model_dir = next();
        else if (s=="--model_A")  a.model_A_txt = next();
        else if (s=="--model_B")  a.model_B_txt = next();
        else if (s=="--model_rank")  a.model_rank_txt = next();
        else if (s=="--csv_out") a.csv_out = next();
        else if (s=="--per_query_csv") a.per_query_csv_out = next();
        else { std::cerr<<"Unknown arg "<<s<<"\n"; usage(); std::exit(1); }
    }
    if (a.load_synth_prefix.empty()) {
        std::cerr << "ERROR: --load_synth_prefix is required\n";
        std::exit(1);
    }
    if (a.cluster_mode != "seed-radius" && a.cluster_mode != "greedy") {
        std::cerr << "ERROR: --cluster_mode must be seed-radius or greedy\n";
        std::exit(1);
    }
    return a;
}

int main(int argc, char** argv){
    Args args = parse_args(argc, argv);
    auto path_join = [](const std::string& a, const std::string& b){
        if (a.empty()) return b;
        if (a.back()=='/'||a.back()=='\\') return a+b;
        return a + "/" + b;
    };

    std::vector<std::string> featsA_names, featsB_names, featsM_names;
    {
        std::string meta_path = path_join(args.model_dir, "meta.json");
        std::ifstream fin(meta_path);
        if (fin) {
            json j; fin >> j;
            auto try_get = [&](const char* key, std::vector<std::string>& out){
                if (j.contains(key) && j[key].is_array()) {
                    out.clear();
                    for (auto& v : j[key]) if (v.is_string()) out.push_back(v.get<std::string>());
                }
            };
            try_get("features_A_used",        featsA_names);
            try_get("features_B_used",        featsB_names);
            try_get("features_rank_m_used",   featsM_names);
            std::cout << "[meta] loaded feature orders: "
                    << "A=" << featsA_names.size() << ", "
                    << "B=" << featsB_names.size() << ", "
                    << "M=" << featsM_names.size() << "\n";
        } else {
            std::cout << "[meta] " << meta_path << " not found. Fallback to legacy fixed feature order.\n";
        }
    }

    // base
    std::string base_path  = first_existing(args.dir, {"base.fbin","base.2.5M.fbin","base.10M.fbin"});
    std::vector<float> xb; int dim=0; idx_t nb = read_fbin(base_path.c_str(), xb, dim);
    hnswlib::L2Space space(dim);

    // HNSW
    const size_t M=32, efc=200;
    WarmHierarchicalNSW hnsw(&space, nb, M, efc);
    if (std::filesystem::exists(args.index_path)){
        hnsw.loadIndex(args.index_path, &space, nb);
    }else{
        for (idx_t i=0;i<nb;++i) hnsw.addPoint(xb.data()+i*dim, i);
        hnsw.saveIndex(args.index_path);
    }
    hnsw.setEf(args.ef);

    // Load models
    GBDT modelA, modelB, modelRank;
    if (!modelA.load_from_txt(path_join(args.model_dir, args.model_A_txt))) { std::cerr<<"FATAL: cannot load A\n"; return 1; }
    if (!modelB.load_from_txt(path_join(args.model_dir, args.model_B_txt))) { std::cerr<<"FATAL: cannot load B\n"; return 1; }
    if (!modelRank.load_from_txt(path_join(args.model_dir, args.model_rank_txt))) { std::cerr<<"FATAL: cannot load Rank\n"; return 1; }

    std::cout<<"[Models] loaded A(efc), B(efw), Rank-M\n";

    // queries/gt
    std::string xqf = args.load_synth_prefix + ".xq.fbin";
    std::string gtf = args.load_synth_prefix + ".gt.ibin";
    if (!std::filesystem::exists(xqf) || !std::filesystem::exists(gtf)) {
        std::cerr << "ERROR: Missing synth files under prefix: " << args.load_synth_prefix << "\n"
                  << "  expect: .xq.fbin, .gt.ibin\n";
        return 1;
    }
    std::vector<float> xq; int dq=0; idx_t nq = read_fbin(xqf.c_str(), xq, dq);
    if (dq!=dim){ std::cerr<<"Dim mismatch in xq: "<<dq<<" vs base "<<dim<<"\n"; return 1; }

    std::vector<int> gt_full; int d_gtk=0; idx_t ngt_rows = read_ibin(gtf.c_str(), gt_full, d_gtk);
    if (ngt_rows < nq){ std::cerr << "Loaded gt rows < nq\n"; return 1; }
    std::vector<int> gt((size_t)nq * args.k, -1);
    for (idx_t i=0;i<nq;++i){
        std::copy(
            gt_full.data() + (size_t)i * d_gtk,
            gt_full.data() + (size_t)i * d_gtk + std::min(args.k, d_gtk),
            gt.begin() + (size_t)i * args.k
        );
    }

    // ---- Projected-Greedy Clustering ----
    std::vector<std::vector<idx_t>> clusters;
    size_t non_empty = 0;
    size_t min_size = 0;
    size_t max_size = 0;
    double cluster_build_s = 0.0;
    {
        auto t_cluster0 = std::chrono::high_resolution_clock::now();
        std::cout << "[ProjectedGreedy] start clustering nq=" << nq
                  << " cluster_mode=" << args.cluster_mode
                  << " proj_dim=" << args.proj_dim
                  << " M=" << args.proj_m
                  << " tau_edge=" << args.tau_edge
                  << " tau_cluster=" << args.tau_cluster
                  << " ...\n";
        PGClusterOut pg = build_projected_greedy_clusters(
            xq, nq, dq,
            args.proj_dim, args.proj_m,
            args.cl_hnsw_m, args.cl_hnsw_efc, args.cl_hnsw_ef,
            args.tau_edge, args.tau_cluster,
            args.seed, args.cluster_mode);
        clusters = std::move(pg.clusters);
        non_empty = pg.non_empty;
        min_size = pg.min_size;
        max_size = pg.max_size;
        std::cout << "[ProjectedGreedy] total_clusters=" << clusters.size()
                  << " non_empty=" << non_empty
                  << " min_size=" << min_size
                  << " max_size=" << max_size
                  << " batch_gate=" << args.batch_gate << "\n";
        auto t_cluster1 = std::chrono::high_resolution_clock::now();
        cluster_build_s = std::chrono::duration<double>(t_cluster1 - t_cluster0).count();
    }

    // ---- Gate: skip small clusters ----
    std::vector<idx_t> active_qids;
    std::vector<idx_t> skipped_qids;
    size_t large_clusters = 0;
    size_t small_clusters = 0;
    for (const auto& C : clusters) {
        if (C.empty()) continue;
        if ((int)C.size() <= args.batch_gate) {
            ++small_clusters;
            skipped_qids.insert(skipped_qids.end(), C.begin(), C.end());
        } else {
            ++large_clusters;
            active_qids.insert(active_qids.end(), C.begin(), C.end());
        }
    }
    std::cout << "[Gate] large_clusters=" << large_clusters
              << " small_clusters=" << small_clusters
              << " active_q=" << active_qids.size()
              << " skipped_q=" << skipped_qids.size() << "\n";

    auto centroid = [&](const std::vector<idx_t>& ids){
        std::vector<float> c(dim, 0.f);
        if (ids.empty()) return c;
        for (auto qi: ids){
            const float* qv = xq.data()+qi*dim;
            for (int j=0;j<dim;++j) c[j]+=qv[j];
        }
        for (int j=0;j<dim;++j) c[j]/=(float)ids.size();
        return c;
    };

    struct ClusterFeatBundle {
        std::vector<float> base;
        float lid_probe256 = 0.f;
        float rc_probe256  = 0.f;
        float exp2k_over_k_probe256 = 1.f;
    };

    auto cluster_features = [&](const std::vector<idx_t>& ids, const std::vector<float>& c_vec){
        struct Feat {
            float cluster_size = 0.f;
            float cluster_size_log = 0.f;
            float log1p_cluster_size = 0.f;
            float cluster_density = 0.f;
            float cluster_radius_p50 = 0.f;
            float cluster_radius_p90 = 0.f;
            float dist_c_to_entryL0 = 0.f;
            float dist_c_top1_smallEF = 0.f;
        } F;

        F.cluster_size       = (float)ids.size();
        F.cluster_size_log   = (F.cluster_size > 0.f) ? std::log(F.cluster_size) : 0.f;
        F.log1p_cluster_size = std::log1p(F.cluster_size);

        std::vector<float> dists; dists.reserve(ids.size());
        for (auto qi: ids){
            const float* qv = xq.data()+qi*dim;
            dists.push_back(std::sqrt(l2sq(qv, c_vec.data(), dim)));
        }
        if (!dists.empty()){
            std::vector<float> tmp = dists;
            std::nth_element(tmp.begin(), tmp.begin()+tmp.size()/2, tmp.end());
            F.cluster_radius_p50 = tmp[tmp.size()/2];

            std::vector<float> tmp2 = dists;
            std::sort(tmp2.begin(), tmp2.end());
            size_t idx90 = (size_t)std::floor(0.9 * (double)(tmp2.size()-1));
            F.cluster_radius_p90 = tmp2[idx90];

            double mean = 0.0; for (float v: dists) mean += v;
            mean /= (double)dists.size();
            F.cluster_density = (float)(1.0 / (mean + 1e-6));
        } else {
            F.cluster_radius_p50 = 0.f;
            F.cluster_radius_p90 = 0.f;
            F.cluster_density    = 0.f;
        }

        hnswlib::tableint epL0 = hnsw.entryPointL0ForQuery((const void*)c_vec.data());
        if (epL0 == (hnswlib::tableint)-1) epL0 = hnsw.entrypoint();
        if (epL0 != (hnswlib::tableint)-1){
            const float* epv = xb.data() + (size_t)epL0 * dim;
            F.dist_c_to_entryL0 = std::sqrt(l2sq(c_vec.data(), epv, dim));
        } else {
            F.dist_c_to_entryL0 = 0.f;
        }

        const size_t smallEF    = 128;
        const size_t smallEF256 = 256;
        std::vector<hnswlib::tableint> cand128, top128, vis128;
        hnsw.searchL0Heaps((const void*)c_vec.data(), (size_t)smallEF, cand128, top128, vis128);
        if (!top128.empty()){
            auto id0 = top128.front();
            const float* xv = xb.data() + (size_t)id0 * dim;
            F.dist_c_top1_smallEF = std::sqrt(l2sq(c_vec.data(), xv, dim));
        } else {
            F.dist_c_top1_smallEF = 0.f;
        }

        std::vector<hnswlib::tableint> cand256, top256, vis256;
        hnsw.searchL0Heaps((const void*)c_vec.data(), (size_t)smallEF256, cand256, top256, vis256);

        float lid, rc, exp2;
        compute_lid_rc_expansion_from_top(hnsw, c_vec.data(), dim, top256, 10, lid, rc, exp2);

        float entry_dist_norm = (F.cluster_radius_p50 > 0.f) ? (F.dist_c_to_entryL0 / F.cluster_radius_p50) : 0.f;
        float radius_skew     = (F.cluster_radius_p50 > 0.f) ? (F.cluster_radius_p90 / F.cluster_radius_p50) : 0.f;

        ClusterFeatBundle B;
        B.base = {
            F.cluster_size,
            F.cluster_size_log,
            F.log1p_cluster_size,
            F.cluster_density,
            F.cluster_radius_p50,
            F.cluster_radius_p90,
            radius_skew,
            F.dist_c_to_entryL0,
            entry_dist_norm,
            F.dist_c_top1_smallEF,
            (float)topk_overlap_ratio_limited(top128, top256, (int)std::min<size_t>(args.k, std::min(top128.size(), top256.size()))),
            (float)topk_jaccard_limited      (top128, top256, (int)std::min<size_t>(args.k, std::min(top128.size(), top256.size())))
        };
        B.lid_probe256 = lid;
        B.rc_probe256  = rc;
        B.exp2k_over_k_probe256 = exp2;
        return B;
    };

    // ---- Precompute static cluster features for large clusters ----
    struct ClusterStaticCache {
        std::vector<float> centroid;
        ClusterFeatBundle feat_bundle;
        bool valid = false;
    };

    std::vector<ClusterStaticCache> cluster_cache(clusters.size());
    size_t cached_clusters = 0;
    for (size_t ci = 0; ci < clusters.size(); ++ci) {
        const auto& C = clusters[ci];
        if (C.empty() || (int)C.size() <= args.batch_gate) continue;
        cluster_cache[ci].centroid = centroid(C);
        cluster_cache[ci].feat_bundle = cluster_features(C, cluster_cache[ci].centroid);
        cluster_cache[ci].valid = true;
        ++cached_clusters;
    }
    std::cout << "[Cache] precomputed static cluster features for " << cached_clusters << " large clusters\n";

    const std::vector<std::string> base_names = {
        "cluster_size", "cluster_size_log", "log1p_cluster_size", "cluster_density", "cluster_radius_p50", "cluster_radius_p90",
        "radius_skew", "dist_centroid_to_entryL0", "entry_dist_norm", "dist_centroid_top1_smallEF", "overlap128_vs_256", "jaccard128_vs_256"};

    std::vector<float> target_list = args.R_targets.empty() ? std::vector<float>{args.R_target} : args.R_targets;

    struct Row {
        float target;
        double qps;
        float recall;
        double avg_visited;
        size_t effective_q;
        size_t skipped_q;
    };
    std::vector<Row> rows;

    for (size_t ridx = 0; ridx < target_list.size(); ++ridx) {
        args.R_target = target_list[ridx];
        std::cout << "\n=== RUN for target_recall=" << std::fixed << std::setprecision(4) << args.R_target << " ===\n";

        std::vector<std::vector<idx_t>> out_topk(nq);
        std::vector<double> per_query_time(nq, 0.0);
        std::vector<float>  per_query_recall(nq, 0.0f);
        std::vector<size_t> per_query_visited(nq, 0);

        auto t0 = std::chrono::high_resolution_clock::now();

        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            const auto& C = clusters[ci];
            if (C.empty() || (int)C.size() <= args.batch_gate || !cluster_cache[ci].valid) continue;

            const std::vector<float>& c = cluster_cache[ci].centroid;
            const ClusterFeatBundle& FB = cluster_cache[ci].feat_bundle;

            FeatureBank F;
            for (size_t i = 0; i < base_names.size(); ++i) F.kv[base_names[i]] = FB.base[i];
            F.kv["lid_probe256_k10"]                = FB.lid_probe256;
            F.kv["rc_probe256_k10"]                 = FB.rc_probe256;
            F.kv["expansion2k_over_k_probe256_k10"] = FB.exp2k_over_k_probe256;
            F.kv["recall_at_k"]                     = args.R_target;

            // ---- A: efc ----
            std::vector<float> xA = featsA_names.empty()
                ? std::vector<float>{
                    F.get("cluster_size"), F.get("cluster_size_log"), F.get("log1p_cluster_size"),
                    F.get("cluster_density"), F.get("cluster_radius_p50"), F.get("cluster_radius_p90"), F.get("radius_skew"),
                    F.get("dist_centroid_to_entryL0"), F.get("entry_dist_norm"),
                    F.get("dist_centroid_top1_smallEF"),
                    F.get("overlap128_vs_256"), F.get("jaccard128_vs_256"),
                    F.get("lid_probe256_k10"), F.get("rc_probe256_k10"), F.get("expansion2k_over_k_probe256_k10"),
                    F.get("recall_at_k"),
                }
                : vector_from_names(featsA_names, F);

            int efc_hat = std::max(1, (int)std::round(std::expm1((double)modelA.predict_sum(xA))));
            F.kv["efc"]     = (float)efc_hat;
            F.kv["log_efc"] = std::log1p((float)efc_hat);

            // ---- centroid search with efc_hat ----
            std::vector<hnswlib::tableint> candE, topE, visE;
            hnsw.searchL0Heaps((const void*)c.data(), (size_t)efc_hat, candE, topE, visE);

            float lid_efc=0.f, rc_efc=0.f, exp_efc=1.f;
            compute_lid_rc_expansion_from_top(hnsw, c.data(), dim, topE, 10, lid_efc, rc_efc, exp_efc);
            F.kv["lid_k_efc"]           = lid_efc;
            F.kv["rc_k_efc"]            = rc_efc;
            F.kv["expand2k_over_k_efc"] = exp_efc;

            // ---- B: ef_warm ----
            std::vector<float> xB = featsB_names.empty()
                ? std::vector<float>{
                    F.get("cluster_size"), F.get("cluster_size_log"), F.get("log1p_cluster_size"),
                    F.get("cluster_density"), F.get("cluster_radius_p50"), F.get("cluster_radius_p90"), F.get("radius_skew"),
                    F.get("dist_centroid_to_entryL0"), F.get("entry_dist_norm"),
                    F.get("dist_centroid_top1_smallEF"),
                    F.get("overlap128_vs_256"), F.get("jaccard128_vs_256"),
                    F.get("lid_k_efc"), F.get("rc_k_efc"), F.get("expand2k_over_k_efc"),
                    F.get("efc"), F.get("log_efc"),
                    F.get("recall_at_k"),
                }
                : vector_from_names(featsB_names, F);

            int efw_hat = std::max(1, (int)std::round(std::expm1((double)modelB.predict_sum(xB))));
            F.kv["ef_warm"]      = (float)efw_hat;
            F.kv["log_efw"]      = std::log1p((float)efw_hat);
            F.kv["efw_over_efc"] = (float)efw_hat / std::max(1.0f, (float)efc_hat);

            // ---- Rank-M: m* ----
            std::vector<float> xR = featsM_names.empty()
                ? std::vector<float>{
                    F.get("cluster_size"), F.get("cluster_size_log"), F.get("log1p_cluster_size"),
                    F.get("cluster_density"), F.get("cluster_radius_p50"), F.get("cluster_radius_p90"), F.get("radius_skew"),
                    F.get("dist_centroid_to_entryL0"), F.get("entry_dist_norm"),
                    F.get("dist_centroid_top1_smallEF"),
                    F.get("overlap128_vs_256"), F.get("jaccard128_vs_256"),
                    F.get("lid_k_efc"), F.get("rc_k_efc"), F.get("expand2k_over_k_efc"),
                    F.get("efc"), F.get("ef_warm"), F.get("log_efc"), F.get("log_efw"), F.get("efw_over_efc"),
                    F.get("recall_at_k"),
                }
                : vector_from_names(featsM_names, F);

            double m_hat = std::expm1((double)modelRank.predict_sum(xR));
            if (!std::isfinite(m_hat)) m_hat = 0.0;

            float Lpred = (float)(m_hat / std::max(1.0f, (float)efw_hat));
            Lpred = std::max(Lpred, args.L_floor);
            float L_cap = std::max(1.0f, (float)efc_hat / std::max(1.0f, (float)efw_hat));
            Lpred = std::min(Lpred + args.L_safety, L_cap);
            if (args.L_step > 1e-9f) Lpred = std::ceil(Lpred / args.L_step) * args.L_step;

            size_t ef_query = (size_t)std::max(1, (int)std::lround((double)efw_hat * std::max(1.0f, Lpred)));
            ef_query = std::min<size_t>(ef_query, (size_t)std::max(efc_hat, efw_hat));
            const size_t seed_cap = std::min<size_t>(topE.size(), std::max<size_t>(1, (size_t)std::lround((double)Lpred * (double)efw_hat)));

            #pragma omp parallel for schedule(dynamic, 8) if(C.size() > 1)
            for (long long i = 0; i < (long long)C.size(); ++i) {
                idx_t qi = C[(size_t)i];
                auto qs = std::chrono::high_resolution_clock::now();
                auto res = hnsw.continueFromSnapshotL0(
                    (const void*)(xq.data() + (size_t)qi * dim),
                    (size_t)args.k_collect,
                    candE, topE,
                    ef_query,
                    (float)seed_cap
                );
                auto qe = std::chrono::high_resolution_clock::now();
                per_query_time[qi] += std::chrono::duration<double>(qe - qs).count();

                std::vector<idx_t> pred;
                while (!res.empty()) { pred.push_back((idx_t)res.top().second); res.pop(); }
                std::reverse(pred.begin(), pred.end());
                out_topk[qi] = std::move(pred);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double total_time_seconds = std::chrono::duration<double>(t1 - t0).count();
        double total_time_with_cluster_seconds = total_time_seconds + cluster_build_s;
        double qps = active_qids.empty() ? 0.0 : (double)active_qids.size() / std::max(1e-12, total_time_with_cluster_seconds);

        float recall = compute_recall_subset(out_topk, gt, args.k, active_qids);

        // per-query recall
        int per_row = (int)gt.size() / (int)nq;
        int expect = std::min(args.k, per_row);
        for (idx_t qi : active_qids) {
            int hit = 0;
            const auto& R = out_topk[qi];
            for (int j = 0; j < expect; ++j) {
                int g = gt[(size_t)qi * per_row + j];
                if (g < 0) break;
                for (int t = 0; t < std::min((int)R.size(), args.k); ++t) {
                    if ((int)R[t] == g) { ++hit; break; }
                }
            }
            per_query_recall[qi] = (expect > 0) ? ((float)hit / (float)expect) : 0.0f;
        }

        std::cout << "\n=== HNSW Cluster Consensus (projected-greedy; skip_small) ===\n";
        std::cout << "TargetRecall=" << std::fixed << std::setprecision(4) << args.R_target
                  << " | Base: nb=" << nb << " dim=" << dim
                  << "  Queries: nq=" << active_qids.size() << "\n";
        std::cout << "ef=" << args.ef << " k=" << args.k << " batch_gate=" << args.batch_gate << "\n";
        std::cout << "clusters: total=" << clusters.size() << " non_empty=" << non_empty
                  << " large(>gate)=" << large_clusters
                  << " small(<=gate, skipped)=" << small_clusters << "\n";
        std::cout << "effective_q=" << active_qids.size() << " skipped_q=" << skipped_qids.size() << "\n";
        std::cout << "Total time(search): " << total_time_seconds << " s"
                  << "  |  cluster: " << cluster_build_s << " s"
                  << "  |  total(search+cluster): " << total_time_with_cluster_seconds << " s"
                  << "  |  QPS: " << qps << "\n";
        std::cout << "Recall@" << args.k << ": " << recall << "\n";

        rows.push_back(Row{args.R_target, qps, recall, 0.0, active_qids.size(), skipped_qids.size()});

        if (!args.per_query_csv_out.empty()) {
            std::ios_base::openmode mode = std::ios::out;
            if (ridx > 0) mode |= std::ios::app;
            std::ofstream qout(args.per_query_csv_out, mode);
            if (qout) {
                if (ridx == 0) qout << "target_recall,qid,search_time_s,recall\n";
                qout << std::fixed << std::setprecision(6);
                for (idx_t qi : active_qids) {
                    qout << args.R_target << "," << qi << "," << per_query_time[qi] << "," << per_query_recall[qi] << "\n";
                }
                std::cout << "[Per-Query CSV] wrote nq=" << active_qids.size() << " rows to " << args.per_query_csv_out << "\n";
            }
        }
    }

    {
        std::ofstream fout(args.csv_out);
        if (!fout) {
            std::cerr << "[WARN] cannot open CSV for write: " << args.csv_out << "\n";
        } else {
            fout << "target_recall,qps,recall,avg_visited,effective_q,skipped_q\n";
            fout << std::fixed << std::setprecision(6);
            for (const auto& r : rows) {
                fout << r.target << "," << r.qps << "," << r.recall << "," << r.avg_visited << "," << r.effective_q
                     << "," << r.skipped_q << "\n";
            }
            std::cout << "\n[CSV] wrote " << rows.size() << " rows to " << args.csv_out << "\n";
        }
    }

    return 0;
}
