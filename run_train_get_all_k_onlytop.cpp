// A-CORE training feature collector
//
// This executable runs HNSW searches over clustered queries and
// exports per-cluster summary statistics and per-query recall
// signals used as training data for the conditional models in
// train_full_conditional_and_recall_newfeat.py.
//
// The resulting CSV is consumed directly by the Python training
// script. For the exact experimental configuration and feature
// definitions, please refer to the accompanying paper and README.

#include <hnswlib/hnswlib.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <vector>
#include <cstring>
#include <cmath>
#include <cctype>

#ifdef _OPENMP
#include <omp.h>
#endif

using idx_t = size_t;

// ---------- 常量：overlap/jaccard 评估的三种 K ----------
constexpr int K_OV1   = 1;
constexpr int K_OV10  = 10;
constexpr int K_OV100 = 100;
constexpr size_t K_MAX_FEAT = (size_t)K_OV100; // smallEF top 列表中至少保留这么多

// ---------------- util ----------------
static inline float l2sq(const float* a, const float* b, int d){
    float s=0.f; for(int i=0;i<d;++i){ float df=a[i]-b[i]; s+=df*df; } return s;
}
static idx_t read_fbin(const char* f, std::vector<float>& out, int& dim){
    FILE* fp=fopen(f,"rb"); if(!fp){ perror(f); std::exit(1); }
    int n=0,d=0; fread(&n,4,1,fp); fread(&d,4,1,fp); out.resize((idx_t)n*(idx_t)d);
    size_t tot=(size_t)n*(size_t)d; if(fread(out.data(),sizeof(float),tot,fp)!=tot){ std::cerr<<"read "<<f<<" fail\n"; std::exit(1); }
    fclose(fp); dim=d; return (idx_t)n;
}
static idx_t read_ibin(const char* f, std::vector<int>& out, int& dim){
    FILE* fp=fopen(f,"rb"); if(!fp){ perror(f); std::exit(1); }
    int n=0,d=0; fread(&n,4,1,fp); fread(&d,4,1,fp); out.resize((idx_t)n*(idx_t)d);
    size_t tot=(size_t)n*(size_t)d; if(fread(out.data(),sizeof(int),tot,fp)!=tot){ std::cerr<<"read "<<f<<" fail\n"; std::exit(1); }
    fclose(fp); dim=d; return (idx_t)n;
}
static std::string first_existing(const std::string& dir, const std::vector<std::string>& names){
    for (auto& n: names){ auto p=dir+"/"+n; if(std::filesystem::exists(p)) return p; }
    return dir + "/" + names.front();
}
template<typename T>
static T median(std::vector<T> v){
    if(v.empty()) return T{};
    size_t n=v.size(); size_t m=n/2;
    std::nth_element(v.begin(), v.begin()+m, v.end());
    if(n%2) return v[m];
    auto a=*std::max_element(v.begin(), v.begin()+m);
    return (a+v[m])/(T)2;
}
template<typename T>
static T percentile(std::vector<T> v, double p){ // p in [0,100]
    if(v.empty()) return T{};
    std::sort(v.begin(), v.end());
    double idx = (p/100.0) * (v.size()-1);
    size_t i=(size_t)std::floor(idx), j=(size_t)std::ceil(idx);
    if(i==j) return v[i];
    double w=idx-i; return (T)((1.0-w)*v[i] + w*v[j]);
}
static std::vector<float> parse_float_list(const std::string& s){
    std::vector<float> out; size_t i=0; while(i<s.size()){
        size_t j=s.find(',',i); if(j==std::string::npos) j=s.size();
        out.push_back(std::stof(s.substr(i,j-i))); i=j+1;
    }
    return out;
}
static std::vector<int> parse_int_list(const std::string& s){
    std::vector<int> out; size_t i=0; while(i<s.size()){
        while(i<s.size() && (s[i]==','||isspace((unsigned char)s[i]))) ++i;
        if(i>=s.size()) break; size_t j=i; while(j<s.size() && s[j]!=',') ++j;
        out.push_back(std::stoi(s.substr(i,j-i))); i=j+1;
    } return out;
}

// ---- TopK 对比度量（overlap/jaccard） ----
static double topk_overlap_ratio_limited(const std::vector<hnswlib::tableint>& A,
                                         const std::vector<hnswlib::tableint>& B,
                                         int K){
    size_t K_eff = (size_t)std::max(0, K);
    K_eff = std::min<size_t>(K_eff, std::min(A.size(), B.size()));
    if (K_eff == 0) return 0.0;
    std::unordered_set<hnswlib::tableint> sa; sa.reserve(K_eff*2);
    std::unordered_set<hnswlib::tableint> sb; sb.reserve(K_eff*2);
    for(size_t i=0;i<K_eff;++i) sa.insert(A[i]);
    for(size_t i=0;i<K_eff;++i) sb.insert(B[i]);
    size_t inter=0;
    if(sa.size()<sb.size()){ for(auto v: sa) if(sb.count(v)) ++inter; }
    else { for(auto v: sb) if(sa.count(v)) ++inter; }
    return (double)inter / (double)K_eff;
}
static double topk_jaccard_limited(const std::vector<hnswlib::tableint>& A,
                                   const std::vector<hnswlib::tableint>& B,
                                   int K){
    size_t K_eff = (size_t)std::max(0, K);
    K_eff = std::min<size_t>(K_eff, std::min(A.size(), B.size()));
    if (K_eff == 0) return 0.0;
    std::unordered_set<hnswlib::tableint> s; s.reserve(K_eff*3);
    size_t inter=0;
    for(size_t i=0;i<K_eff;++i) s.insert(A[i]);
    for(size_t i=0;i<K_eff;++i) inter += s.count(B[i]) ? 1 : 0;
    for(size_t i=0;i<K_eff;++i) s.insert(B[i]);
    size_t uni = s.size();
    return (uni>0)? (double)inter/(double)uni : 0.0;
}

// ---------- LID/RC/Expansion 三元组（按 top 距离序列计算；k=10） ----------
static inline double safe_log_ratio(double ri, double rk){
    const double eps=1e-12;
    double den = rk + eps;
    double num = std::max(ri, eps);
    return std::log(num/den);
}
struct Trio { double lid=std::numeric_limits<double>::quiet_NaN();
              double rc =std::numeric_limits<double>::quiet_NaN();
              double exp2k_over_k=std::numeric_limits<double>::quiet_NaN(); };

static Trio compute_trio_from_top_pairs(const std::vector<std::pair<float,hnswlib::tableint>>& top_sorted_by_dist2, int k){
    Trio t;
    if ((int)top_sorted_by_dist2.size() < k) return t;
    std::vector<double> r; r.reserve(top_sorted_by_dist2.size());
    for (auto& p: top_sorted_by_dist2) r.push_back(std::sqrt(std::max(0.0,(double)p.first))); // p.first 是 L2^2
    double rk = r[(size_t)k-1];

    // LID
    double s = 0.0;
    for (int i=0;i<k;++i) s += safe_log_ratio(r[i], rk);
    double denom = (s / (double)k);
    if (std::fabs(denom) > 1e-18) t.lid = -1.0 / denom;

    // RC: 顶列表所有项的均距 / r_k
    double mean_all = 0.0;
    for (double v : r) mean_all += v;
    mean_all /= (double)r.size();
    if (rk > 0) t.rc = mean_all / rk;

    // Expansion: r_{2k}/r_k
    if ((int)r.size() >= 2*k){
        double r2k = r[(size_t)2*k - 1];
        if (rk > 0) t.exp2k_over_k = r2k / rk;
    }
    return t;
}

// ---------------- Warm HNSW (导出 cand/top/visited) ----------------
struct WarmHierarchicalNSW : public hnswlib::HierarchicalNSW<float> {
    using Base=hnswlib::HierarchicalNSW<float>;
    using tableint=hnswlib::tableint;
    explicit WarmHierarchicalNSW(hnswlib::SpaceInterface<float>* s, size_t max_elements, size_t M, size_t efc)
        : Base(s, max_elements, M, efc) {}
    tableint entrypoint() const { return this->enterpoint_node_; }
    const hnswlib::linklistsizeint* linklist0(tableint id) const { return this->get_linklist0(id); }

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
                       std::vector<tableint>& out_visited_ids,
                       int& step_count) const
    {
        out_cand_ids.clear(); out_top_ids.clear(); out_visited_ids.clear();
        step_count = 0;
        if (this->cur_element_count == 0) return;

        auto* vl = this->visited_list_pool_->getFreeVisitedList(); vl->reset();
        auto* vis = vl->mass;

        struct MinCmp { bool operator()(const std::pair<float,tableint>&a,const std::pair<float,tableint>&b) const {return a.first>b.first;} };
        std::priority_queue<std::pair<float,tableint>, std::vector<std::pair<float,tableint>>, MinCmp> cand;
        std::priority_queue<std::pair<float,tableint>> top;

        auto mark = [&](tableint id){
            if (id>=this->cur_element_count) return false;
            if (vis[id]==vl->curV) return false;
            step_count++;
            vis[id]=vl->curV; out_visited_ids.push_back(id); return true;
        };
        auto push_both = [&](tableint id, float d){
            cand.emplace(d,id); top.emplace(d,id); if(top.size()>ef) top.pop();
        };

        tableint ep = entryPointL0ForQuery(q);
        if (ep == (tableint)-1) ep = this->enterpoint_node_;
        if (ep != (tableint)-1){
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
                if (top.size()<ef || d<lb){ cand.emplace(d,v); top.emplace(d,v); if(top.size()>ef) top.pop(); lb=top.top().first; }
            }
        }

        // 导出 cand/top（升序距离）
        // {
        //     auto cc=cand;
        //     std::vector<std::pair<float,tableint>> buf;
        //     while(!cc.empty()){ buf.push_back(cc.top()); cc.pop(); }
        //     std::sort(buf.begin(), buf.end(), [](auto&a,auto&b){return a.first<b.first;});
        //     for(size_t i=0;i<buf.size();++i) out_cand_ids.push_back(buf[i].second);
        // }
        {
            auto tc=top;
            std::vector<std::pair<float,tableint>> buf;
            while(!tc.empty()){ buf.push_back(tc.top()); tc.pop(); }
            std::sort(buf.begin(), buf.end(), [](auto&a,auto&b){return a.first<b.first;});
            for(size_t i=0;i<buf.size();++i) out_top_ids.push_back(buf[i].second);
        }

        this->visited_list_pool_->releaseVisitedList(vl);
    }

    // 从 cand/top 快照继续 L0，返回升序 top-K_collect 结果（K_collect>=max需要的k）
    void continueFromSnapshotL0(
        const void* q, size_t K_collect,
        const std::vector<tableint>& init_cand_ids,
        const std::vector<tableint>& init_top_ids,
        size_t ef, float L,
        std::vector<idx_t>& out_res_ids,
        size_t& out_visited_count) const
    {
        out_res_ids.clear(); out_visited_count=0;
        auto* vl = this->visited_list_pool_->getFreeVisitedList(); vl->reset();
        auto* vis = vl->mass;

        struct MinCmp { bool operator()(const std::pair<float,tableint>&a,const std::pair<float,tableint>&b) const {return a.first>b.first;} };
        std::priority_queue<std::pair<float,tableint>, std::vector<std::pair<float,tableint>>, MinCmp> cand;
        std::priority_queue<std::pair<float,tableint>> top;

        auto mark = [&](tableint id){
            if (id>=this->cur_element_count) return false;
            if (vis[id]==vl->curV) return false;
            vis[id]=vl->curV; ++out_visited_count; return true;
        };
        // auto push_cand = [&](tableint id){
        //     if(!mark(id)) return;
        //     float d=this->fstdistfunc_(q,this->getDataByInternalId(id), this->dist_func_param_);
        //     cand.emplace(d,id);
        // };
        auto push_top_cand = [&](tableint id){
            if(!mark(id)) return;
            float d=this->fstdistfunc_(q,this->getDataByInternalId(id), this->dist_func_param_);
            top.emplace(d,id); 
            cand.emplace(d,id);
            if(top.size()>ef) top.pop();
        };

        // 受 L*ef 截断
        size_t max_init;
        // for(size_t i=0;i<max_init && i<init_cand_ids.size();++i) push_cand(init_cand_ids[i]);

        max_init = (size_t)std::min<double>((double)init_top_ids.size(), std::max<double>(0.0, (double)L) * (double)ef);
        for(size_t i=0;i<max_init && i<init_top_ids.size();++i) push_top_cand(init_top_ids[i]);

        float lb = top.empty()? std::numeric_limits<float>::infinity() : top.top().first;
        while(!cand.empty()){
            auto cu = cand.top(); if(cu.first>lb) break; cand.pop();
            const auto* l0 = this->get_linklist0(cu.second);
            unsigned sz=*l0; const auto* nb=reinterpret_cast<const tableint*>(l0+1);
            for(unsigned i=0;i<sz;++i){
                auto v=nb[i];
                if(!mark(v)) continue;
                float d=this->fstdistfunc_(q,this->getDataByInternalId(v), this->dist_func_param_);
                if(top.size()<ef || d<lb){ cand.emplace(d,v); top.emplace(d,v); if(top.size()>ef) top.pop(); lb=top.top().first; }
            }
        }

        // 仅在输出阶段裁到 K_collect
        while(top.size()>K_collect) top.pop();
        std::vector<std::pair<float,tableint>> buf;
        while(!top.empty()){ buf.push_back(top.top()); top.pop(); }
        std::sort(buf.begin(), buf.end(), [](auto&a,auto&b){return a.first<b.first;});
        out_res_ids.reserve(buf.size());
        for(auto& p: buf) out_res_ids.push_back((idx_t)this->getExternalLabel(p.second));

        this->visited_list_pool_->releaseVisitedList(vl);
    }
};

// ---------------- 簇与质心特征（仅保留需要的, OpenMP版） ----------------
static std::vector<float> compute_centroid(const std::vector<float>& X, const std::vector<idx_t>& ids, int d){
    std::vector<float> c(d,0.f); if(ids.empty()) return c;
    for(auto i: ids){ const float* x = X.data() + (size_t)i*d; for(int j=0;j<d;++j) c[j]+=x[j]; }
    for(int j=0;j<d;++j) c[j]/=(float)ids.size(); return c;
}
static float cluster_radius_p50(const std::vector<float>& X, const std::vector<idx_t>& ids, const std::vector<float>& c, int d){
    if(ids.empty()) return 0.f;
    std::vector<float> dist(ids.size());
    #pragma omp parallel for if(ids.size()>256) schedule(static)
    for(size_t t=0;t<ids.size();++t){
        const float* x = X.data() + (size_t)ids[t]*d;
        dist[t] = std::sqrt(l2sq(x,c.data(),d));
    }
    return median(dist);
}
static float cluster_radius_p90(const std::vector<float>& X, const std::vector<idx_t>& ids, const std::vector<float>& c, int d){
    if(ids.empty()) return 0.f;
    std::vector<float> dist(ids.size());
    #pragma omp parallel for if(ids.size()>256) schedule(static)
    for(size_t t=0;t<ids.size();++t){
        const float* x = X.data() + (size_t)ids[t]*d;
        dist[t] = std::sqrt(l2sq(x,c.data(),d));
    }
    return percentile(dist, 90.0);
}
// density proxy: 簇内第10近邻距离的中位数的倒数（OpenMP）
static float cluster_density_proxy_p10(const std::vector<float>& X, const std::vector<idx_t>& ids, int d){
    if(ids.size()<11) return 0.f;
    std::vector<float> d10(ids.size(), 0.f);
    #pragma omp parallel for if(ids.size()>32) schedule(dynamic,4)
    for(long long a=0; a<(long long)ids.size(); ++a){
        const float* xa = X.data() + (size_t)ids[a]*d;
        std::vector<float> ds; ds.reserve(ids.size()-1);
        for(size_t b=0;b<ids.size();++b){
            if((size_t)a==b) continue;
            const float* xb = X.data() + (size_t)ids[b]*d;
            ds.push_back(std::sqrt(l2sq(xa,xb,d)));
        }
        std::nth_element(ds.begin(), ds.begin()+9, ds.end());
        d10[(size_t)a] = ds[9];
    }
    float m = median(d10); return (m>0.f)? (1.0f/m) : 0.f;
}

// ---------------- Grid Runner & CSV ----------------
struct Args {
    std::string dir, index_path, synth_prefix, csv_out="training_runs.csv";
    int k=10;                 // 仅用于旧口径 overlap/jaccard 两列（保留兼容，不影响 recall）
    int k_collect=100;        // 新增：检索输出保留的结果数上限（>=max需要的k）
    std::vector<int> efc_list, efw_list;
    std::vector<float> L_list;
    float Rstar=0.0f, delta=0.0f; // 未使用，但保留参数解析兼容
    unsigned seed=42;
    std::string feat_profile="full"; // 未使用，但保留参数解析兼容
};
static void usage(){
    std::cout<<"Usage:\n  run_hnsw_collect_training <data_dir> [index_path]\n"
             <<"    --load_synth_prefix <path>           # <path>.xq.fbin, .gt.ibin, .labels.ibin\n"
             <<"    --k <int>                            # for legacy overlap/jaccard columns only (default 10)\n"
             <<"    --k_collect <int>                    # results kept per query (default 100)\n"
             <<"    --efc_list  \"128,200,256\"\n"
             <<"    --efw_list  \"80,120,160\"\n"
             <<"    --L_list    \"1,1.2,1.4\"\n"
             <<"    --csv_out <file>\n";
}
static Args parse_args(int argc, char** argv){
    if(argc<2){ usage(); std::exit(1); }
    Args a; a.dir=argv[1];
    a.index_path = (argc>2 && argv[2][0]!='-') ? argv[2] : a.dir + "/base.hnsw";
    int i = (argc>2 && argv[2][0]!='-') ? 3 : 2;
    for(; i<argc; ++i){
        std::string s=argv[i]; auto next=[&](){ if(i+1>=argc){ std::cerr<<"missing after "<<s<<"\n"; std::exit(1);} return std::string(argv[++i]); };
        if(s=="--load_synth_prefix") a.synth_prefix=next();
        else if(s=="--k") a.k=std::stoi(next());
        else if(s=="--k_collect") a.k_collect=std::stoi(next());
        else if(s=="--efc_list") a.efc_list=parse_int_list(next());
        else if(s=="--efw_list") a.efw_list=parse_int_list(next());
        else if(s=="--L_list") a.L_list=parse_float_list(next());
        else if(s=="--csv_out") a.csv_out=next();
        else if(s=="--seed") a.seed=(unsigned)std::stoul(next());
        else if(s=="--feat_profile") a.feat_profile=next();
        else if(s=="--Rstar") a.Rstar=std::stof(next());
        else if(s=="--delta") a.delta=std::stof(next());
        else { std::cerr<<"Unknown arg "<<s<<"\n"; usage(); std::exit(1); }
    }
    if(a.synth_prefix.empty()){ std::cerr<<"--load_synth_prefix required\n"; std::exit(1); }
    if(a.efc_list.empty()||a.efw_list.empty()||a.L_list.empty()){
        std::cerr<<"efc_list / efw_list / L_list must be non-empty\n"; std::exit(1);
    }
    if(a.k_collect < 1) a.k_collect = 1;
    return a;
}
static void ensure_csv_header(const std::string& path){
    bool need_header = !std::filesystem::exists(path) || std::filesystem::file_size(path)==0;
    if(!need_header) return;
    std::ofstream ofs(path, std::ios::app);
    ofs
    << "cluster_id,cluster_size,cluster_density,cluster_radius_p50,cluster_radius_p90,radius_skew,"
    << "dist_centroid_to_entryL0,entry_dist_norm,dist_centroid_top1_smallEF,"
    << "overlap128_vs_256,jaccard128_vs_256,"
    << "overlap128_vs_256_k1,overlap128_vs_256_k10,overlap128_vs_256_k100,"
    << "jaccard128_vs_256_k1,jaccard128_vs_256_k10,jaccard128_vs_256_k100,"
    << "lid_probe256_k10,rc_probe256_k10,expansion2k_over_k_probe256_k10,"
    << "efc,ef_warm,L_aligned,recall_at_1,recall_at_10,recall_at_100,qps_run,"
    << "centroid_steps,visited_mean"
    << "\n";
}

int main(int argc, char** argv){
    Args args = parse_args(argc, argv);

    // base & index
    std::string base_path = first_existing(args.dir, {"base.fbin","base.2.5M.fbin","base.10M.fbin"});
    std::vector<float> xb; int dim=0; idx_t nb = read_fbin(base_path.c_str(), xb, dim);
    hnswlib::L2Space space(dim);

    WarmHierarchicalNSW hnsw(&space, nb, /*M=*/32, /*efc=*/200);
    if (std::filesystem::exists(args.index_path)) hnsw.loadIndex(args.index_path, &space, nb);
    else {
        for (idx_t i=0;i<nb;++i) hnsw.addPoint(xb.data()+i*dim, i);
        hnsw.saveIndex(args.index_path);
    }

    // queries / gt / labels
    std::string xqf = args.synth_prefix + ".xq.fbin";
    std::string gtf = args.synth_prefix + ".gt.ibin";
    std::string labf= args.synth_prefix + ".labels.ibin";
    if(!std::filesystem::exists(xqf)||!std::filesystem::exists(gtf)||!std::filesystem::exists(labf)){
        std::cerr<<"synth files missing under prefix "<<args.synth_prefix<<"\n"; return 1;
    }
    std::vector<float> xq; int dq=0; idx_t nq = read_fbin(xqf.c_str(), xq, dq);
    if(dq!=dim){ std::cerr<<"Dim mismatch xq "<<dq<<" vs base "<<dim<<"\n"; return 1; }

    std::vector<int> gt_full; int dgtk=0; idx_t ngt = read_ibin(gtf.c_str(), gt_full, dgtk); (void)ngt;

    std::vector<int> labels; int dlab=0; idx_t nlab=read_ibin(labf.c_str(), labels, dlab);
    if((idx_t)nlab!=nq || dlab!=1){ std::cerr<<"labels size mismatch\n"; return 1; }

    // clusters (oracle)
    std::vector<std::vector<idx_t>> clusters;
    {
        int maxlab=-1; for(int v: labels) if(v>maxlab) maxlab=v;
        clusters.assign((size_t)std::max(1, maxlab+1), {});
        for(idx_t i=0;i<nq;++i){ int c=labels[(size_t)i]; if(c>=0) clusters[(size_t)c].push_back(i); }
        std::cout<<"[Oracle] clusters="<<clusters.size()<<"\n";
    }

    // CSV header
    ensure_csv_header(args.csv_out);
    std::ofstream csv(args.csv_out, std::ios::app);

    // 预先计算每簇的静态/质心特征（仅保留所需） —— 并行
    struct ClusterFeat {
        std::vector<idx_t> ids;
        std::vector<float> centroid;
        float radius_p50=0, radius_p90=0, density_proxy=0;
        float dist_centroid_to_entryL0=0, dist_centroid_top1_smallEF=0;
        std::vector<hnswlib::tableint> smallEF_topk_ids;     // ef=128, 前 K_MAX_FEAT
        std::vector<hnswlib::tableint> smallEF256_topk_ids;  // ef=256, 前 K_MAX_FEAT
        // trio@k=10 on smallEF=256 top
        double lid_k10=std::numeric_limits<double>::quiet_NaN();
        double rc_k10 =std::numeric_limits<double>::quiet_NaN();
        double exp_k10=std::numeric_limits<double>::quiet_NaN();
    };
    std::vector<ClusterFeat> CF(clusters.size());

    const size_t smallEF = 128;
    const size_t smallEF256 = 256;

    #pragma omp parallel for schedule(dynamic,1) if(clusters.size()>1)
    for(long long ci_ll=0; ci_ll<(long long)clusters.size(); ++ci_ll){
        size_t ci = (size_t)ci_ll;
        auto& ids = (std::vector<idx_t>&)clusters[ci];
        ClusterFeat cf; cf.ids = ids;
        if(ids.empty()){ CF[ci]=std::move(cf); continue; }

        cf.centroid = compute_centroid(xq, ids, dim);
        cf.radius_p50 = cluster_radius_p50(xq, ids, cf.centroid, dim);
        cf.radius_p90 = cluster_radius_p90(xq, ids, cf.centroid, dim);
        cf.density_proxy = cluster_density_proxy_p10(xq, ids, dim);

        // dist_centroid_to_entryL0
        auto ep = hnsw.entryPointL0ForQuery(cf.centroid.data());
        if (ep!=(hnswlib::tableint)-1){
            cf.dist_centroid_to_entryL0 = std::sqrt(l2sq(cf.centroid.data(), xb.data()+ (size_t)ep*dim, dim));
        }

        // smallEF = 128
        {
            std::vector<hnswlib::tableint> cand_ids_se, top_ids_se, vis_ids_se;
            int small_steps = 0;
            hnsw.searchL0Heaps(cf.centroid.data(), smallEF, cand_ids_se, top_ids_se, vis_ids_se, small_steps);
            size_t tk = std::min(K_MAX_FEAT, top_ids_se.size());
            cf.smallEF_topk_ids.assign(top_ids_se.begin(), top_ids_se.begin()+tk);
            if(!top_ids_se.empty()){
                auto id = top_ids_se[0];
                cf.dist_centroid_top1_smallEF =
                    std::sqrt(l2sq(cf.centroid.data(), xb.data()+ (size_t)id*dim, dim));
            }
        }
        // smallEF = 256
        {
            std::vector<hnswlib::tableint> cand_ids_256, top_ids_256, vis_ids_256;
            int steps_256 = 0;
            hnsw.searchL0Heaps(cf.centroid.data(), smallEF256, cand_ids_256, top_ids_256, vis_ids_256, steps_256);
            size_t tk = std::min(K_MAX_FEAT, top_ids_256.size());
            cf.smallEF256_topk_ids.assign(top_ids_256.begin(), top_ids_256.begin()+tk);

            // trio@k=10 on smallEF=256 top —— 构造 (dist2, id) 升序并计算
            std::vector<std::pair<float,hnswlib::tableint>> top_pairs;
            top_pairs.reserve(tk);
            for(size_t i=0;i<tk;++i){
                auto id = top_ids_256[i];
                float d2 = l2sq(cf.centroid.data(), xb.data() + (size_t)id*dim, dim);
                top_pairs.emplace_back(d2, id);
            }
            std::sort(top_pairs.begin(), top_pairs.end(), [](auto& a, auto& b){ return a.first < b.first; });
            Trio t = compute_trio_from_top_pairs(top_pairs, /*k=*/K_OV10);
            cf.lid_k10 = t.lid;
            cf.rc_k10  = t.rc;
            cf.exp_k10 = t.exp2k_over_k;
        }

        CF[ci]=std::move(cf);
    }

    // 主循环：参数网格 × 簇 —— 按簇并行
    for(int efc: args.efc_list){
        for(int efw: args.efw_list){
            if(efw>efc){
                std::cerr<<"Warning: efw "<<efw<<" > efc "<<efc<<", skipping this pair\n";
                continue;
            }
            for(float L_in: args.L_list){
                if(L_in*efw>efc+1e-5){
                    std::cerr<<"Warning: L*efw "<<(L_in*efw)<<" > efc "<<efc<<", skipping this triple\n";
                    continue;
                }
                float L = L_in;

                
                double run_total_time=0.0;
                size_t run_total_queries=0;

                #pragma omp parallel for schedule(dynamic,1) reduction(+:run_total_time,run_total_queries) if(clusters.size()>1)
                for(long long ci_ll=0; ci_ll<(long long)clusters.size(); ++ci_ll){
                    size_t ci = (size_t)ci_ll;
                    const auto& ids = CF[ci].ids;
                    if(ids.empty()) continue;

                    // 1) 质心一次纯 HNSW（ef=efc）→ cand/top
                    std::vector<hnswlib::tableint> cand_ids, top_ids, vis_ids;
                    int centroid_steps = 0;
                    auto t0 = std::chrono::high_resolution_clock::now();
                    hnsw.searchL0Heaps(CF[ci].centroid.data(), (size_t)efc, cand_ids, top_ids, vis_ids, centroid_steps);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double centroid_time = std::chrono::duration<double>(t1-t0).count();

                    // 2) 逐 query 用共享工具继续 L0（ef=efw, L 原样） —— 并行
                    double visited_sum = 0.0;
                    long long hit1_sum = 0, need1_sum = 0;
                    long long hit10_sum = 0, need10_sum = 0;
                    long long hit100_sum = 0, need100_sum = 0;

                    auto cstart = std::chrono::high_resolution_clock::now();

                    #pragma omp parallel for schedule(dynamic,8) reduction(+:visited_sum,hit1_sum,need1_sum,hit10_sum,need10_sum,hit100_sum,need100_sum) if(ids.size()>1)
                    for(long long jj=0; jj<(long long)ids.size(); ++jj){
                        idx_t qi = ids[(size_t)jj];
                        std::vector<idx_t> res;  // 升序 top-K_collect
                        size_t visited=0;
                        hnsw.continueFromSnapshotL0(
                            (const void*)(xq.data() + (size_t)qi*dim),
                            (size_t)args.k_collect,
                            cand_ids, top_ids,
                            (size_t)efw, (float)L,
                            res, visited
                        );
                        visited_sum += (double)visited;

                        // 计算 recall@1/10/100（按各自的 expect=min(k,dgtk)）
                        auto count_hits = [&](int K)->std::pair<int,int>{
                            int expect = std::min(K, dgtk);
                            int local_hit = 0;
                            for(int j=0;j<expect;++j){
                                int g = gt_full[(size_t)qi * dgtk + j]; if(g<0) break;
                                for(int t=0; t<std::min((int)res.size(), K); ++t){ if((int)res[t]==g){ ++local_hit; break; } }
                            }
                            return {local_hit, expect};
                        };
                        { auto [h,e]=count_hits(1);   hit1_sum+=h;   need1_sum+=e; }
                        { auto [h,e]=count_hits(10);  hit10_sum+=h;  need10_sum+=e; }
                        { auto [h,e]=count_hits(100); hit100_sum+=h; need100_sum+=e; }
                    }
                    auto cend = std::chrono::high_resolution_clock::now();

                    // 3) 端到端时延（质心一次 + 簇内全部 query） & QPS
                    double queries_time = std::chrono::duration<double>(cend-cstart).count();
                    double cluster_total_time = centroid_time + queries_time;
                    double qps_run = (double)ids.size() / std::max(1e-12, cluster_total_time);

                    // 4) recalls
                    float recall1   = (need1_sum>0)?   (float)hit1_sum   /(float)need1_sum   : 0.f;
                    float recall10  = (need10_sum>0)?  (float)hit10_sum  /(float)need10_sum  : 0.f;
                    float recall100 = (need100_sum>0)? (float)hit100_sum /(float)need100_sum : 0.f;

                    // 5) 访问均值
                    double visited_mean = ids.empty()? 0.0 : (visited_sum / (double)ids.size());

                    // 6) 派生量：radius_skew、entry_dist_norm
                    double radius_skew = (CF[ci].radius_p50>0.f) ? (double)CF[ci].radius_p90 / (double)CF[ci].radius_p50 : 0.0;
                    double entry_dist_norm = (CF[ci].radius_p50>0.f)? (double)CF[ci].dist_centroid_to_entryL0 / (double)CF[ci].radius_p50 : 0.0;

                    // 7) 128 vs 256 的多尺度一致性（簇级；三种 K）
                    double ov_k1   = topk_overlap_ratio_limited(CF[ci].smallEF_topk_ids, CF[ci].smallEF256_topk_ids, K_OV1);
                    double ov_k10  = topk_overlap_ratio_limited(CF[ci].smallEF_topk_ids, CF[ci].smallEF256_topk_ids, K_OV10);
                    double ov_k100 = topk_overlap_ratio_limited(CF[ci].smallEF_topk_ids, CF[ci].smallEF256_topk_ids, K_OV100);
                    double jc_k1   = topk_jaccard_limited      (CF[ci].smallEF_topk_ids, CF[ci].smallEF256_topk_ids, K_OV1);
                    double jc_k10  = topk_jaccard_limited      (CF[ci].smallEF_topk_ids, CF[ci].smallEF256_topk_ids, K_OV10);
                    double jc_k100 = topk_jaccard_limited      (CF[ci].smallEF_topk_ids, CF[ci].smallEF256_topk_ids, K_OV100);

                    // 原两列沿用 K=10 口径（向后兼容）
                    double overlap128_vs_256  = ov_k10;
                    double jaccard128_vs_256  = jc_k10;

                    // 8) 写 CSV（严格按表头顺序） —— 串行写
                    #pragma omp critical
                    {
                        csv  << ci
                             << "," << ids.size()
                             << "," << CF[ci].density_proxy
                             << "," << CF[ci].radius_p50
                             << "," << CF[ci].radius_p90
                             << "," << radius_skew
                             << "," << CF[ci].dist_centroid_to_entryL0
                             << "," << entry_dist_norm
                             << "," << CF[ci].dist_centroid_top1_smallEF
                             << "," << overlap128_vs_256
                             << "," << jaccard128_vs_256
                             << "," << ov_k1
                             << "," << ov_k10
                             << "," << ov_k100
                             << "," << jc_k1
                             << "," << jc_k10
                             << "," << jc_k100
                             << "," << CF[ci].lid_k10
                             << "," << CF[ci].rc_k10
                             << "," << CF[ci].exp_k10
                             << "," << efc
                             << "," << efw
                             << "," << L              // L_aligned
                             << "," << recall1
                             << "," << recall10
                             << "," << recall100
                             << "," << qps_run
                             << "," << centroid_steps
                             << "," << visited_mean
                             << "\n";
                    }

                    run_total_time += cluster_total_time;
                    run_total_queries += ids.size();
                } // end parallel-for clusters

                std::cout<<"[grid] efc="<<efc<<" efw="<<efw<<" L="<<L
                         <<"  avg_lat(ms)="<< (run_total_queries? (run_total_time/run_total_queries*1000.0):0.0)
                         <<"  overall_qps="<< (run_total_time>0? (run_total_queries/run_total_time):0.0) <<"\n";
            }
        }
    }

    std::cout<<"Done. CSV written to "<<args.csv_out<<"\n";
    return 0;
}
