// augment_features.cpp
// 给已跑完的簇级 CSV 补充特征：
//   仅追加：质心按该簇 efc 搜索(结果集 top) 的 LID_k / RC_k / Expansion_{2k|k}
//   （假设 CSV 中已经有 *probe256 相关列，不再重复计算 ef=256 预探索指标）
//
// 用法：
//   ./augment_features <data_dir> <csv_in> [csv_out]
//      --load_synth_prefix <path>   # 用来读取 .xq.fbin / .labels.ibin
//      --index <index_path>         # 默认 data_dir/base.hnsw
//      --k <int>                    # 计算指标的 k（默认与检索 k 一致，默认为10）
//      （已不再需要 --ef_probe，因为只补充 efc 视图的指标）
//
// 说明：
// - CSV 必须至少有列：cluster_id, efc
// - 若 csv_out 省略，则在 csv_in 同目录生成 <name>.with_new_feats.csv
//
// 注意：RC 的 d_mean 取自搜索“top 列表”的平均距离；如需改成 visited 均值，可见 TODO 标记。

#include <hnswlib/hnswlib.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>

using idx_t = size_t;

// ---------- I/O helpers ----------
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
static inline float l2sq(const float* a, const float* b, int d) {
    float s=0.f; for (int i=0;i<d;++i){ float df=a[i]-b[i]; s+=df*df; } return s;
}

// ---------- Minimal L0 search to extract "top" list (ids+dist) ----------
struct L0Top {
    std::vector<std::pair<float,hnswlib::tableint>> top_sorted; // asc by distance
};
static L0Top searchL0_top_only(hnswlib::HierarchicalNSW<float>& H,
                               const void* q, size_t ef,
                               hnswlib::L2Space& space,
                               int dim)
{
    using tableint = hnswlib::tableint;
    // We follow similar logic as your searchL0Heaps; here we only care about "top"
    struct MinCmp { bool operator()(const std::pair<float,tableint>&a,const std::pair<float,tableint>&b) const {return a.first>b.first;} };
    std::priority_queue<std::pair<float,tableint>, std::vector<std::pair<float,tableint>>, MinCmp> cand;
    std::priority_queue<std::pair<float,tableint>> top;

    auto* vl = H.visited_list_pool_->getFreeVisitedList();
    vl->reset();
    auto* vis = vl->mass;

    auto mark = [&](tableint id){
        if (id>=H.cur_element_count) return false;
        if (vis[id]==vl->curV) return false;
        vis[id]=vl->curV; return true;
    };
    auto push_both = [&](tableint id, float d){
        cand.emplace(d,id);
        top.emplace(d,id);
        if (top.size()>ef) top.pop();
    };

    // entrypoint down to L0:
    tableint curr = H.enterpoint_node_;
    if (curr != (tableint)-1 && H.cur_element_count>0){
        float currDist = space.get_dist_func()(q, H.getDataByInternalId(curr), space.get_dist_func_param());
        for (int level = H.maxlevel_; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                const auto* ll = H.get_linklist(curr, level);
                if (!ll) break;
                unsigned sz = *ll;
                const tableint* nb = reinterpret_cast<const tableint*>(ll + 1);
                for (unsigned i = 0; i < sz; ++i) {
                    tableint v = nb[i];
                    float d = space.get_dist_func()(q, H.getDataByInternalId(v), space.get_dist_func_param());
                    if (d < currDist) { currDist = d; curr = v; changed = true; }
                }
            }
        }
        // L0 init
        mark(curr);
        float d0 = space.get_dist_func()(q, H.getDataByInternalId(curr), space.get_dist_func_param());
        push_both(curr, d0);
    }

    float lb = top.empty()? std::numeric_limits<float>::infinity() : top.top().first;
    while (!cand.empty()){
        auto cu = cand.top(); if (cu.first>lb) break; cand.pop();
        const auto* l0 = H.get_linklist0(cu.second);
        unsigned sz=*l0; const auto* nb=reinterpret_cast<const tableint*>(l0+1);
        for (unsigned i=0;i<sz;++i){
            tableint v=nb[i];
            if (!mark(v)) continue;
            float d=space.get_dist_func()(q, H.getDataByInternalId(v), space.get_dist_func_param());
            if (top.size()<ef || d<lb){ cand.emplace(d,v); top.emplace(d,v); if (top.size()>ef) top.pop(); lb=top.top().first; }
        }
    }

    // Export top (asc)
    std::vector<std::pair<float,tableint>> tmp;
    tmp.reserve(ef);
    while(!top.empty()){ tmp.push_back(top.top()); top.pop(); }
    std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b){return a.first < b.first;});
    H.visited_list_pool_->releaseVisitedList(vl);
    return L0Top{std::move(tmp)};
}

// ---------- stats: LID, RC, Expansion ----------
static inline double safe_log_ratio(double ri, double rk){
    const double eps=1e-12;
    double den = rk + eps;
    double num = std::max(ri, eps);
    return std::log(num/den);
}
struct Trio { double lid=std::numeric_limits<double>::quiet_NaN();
              double rc =std::numeric_limits<double>::quiet_NaN();
              double exp2k_over_k=std::numeric_limits<double>::quiet_NaN(); };

static Trio compute_trio_from_top(const std::vector<std::pair<float,hnswlib::tableint>>& top, int k){
    Trio t;
    if ((int)top.size() < k) return t;
    std::vector<double> r; r.reserve(top.size());
    for (auto& p: top) r.push_back(std::sqrt(std::max(0.0,(double)p.first))); // HNSW 存的是 L2^2
    double rk = r[(size_t)k-1];

    // LID
    double s = 0.0;
    for (int i=0;i<k;++i) s += safe_log_ratio(r[i], rk);
    double denom = (s / (double)k);
    if (std::fabs(denom) > 1e-18) t.lid = -1.0 / denom;

    // RC: d_mean / r_k  （d_mean 用 top 列表全部项的均值）
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

// ---------- CSV 简单读写 ----------
static std::vector<std::string> split_csv_line(const std::string& line){
    // 简易分割：假定无引号/逗号转义
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}
static std::string join_csv_line(const std::vector<std::string>& cells){
    std::ostringstream os;
    for (size_t i=0;i<cells.size();++i){
        if (i) os<<",";
        os<<cells[i];
    }
    return os.str();
}

int main(int argc, char** argv){
    if (argc < 3){
    std::cerr<<"Usage:\n  "<<argv[0]<<" <data_dir> <csv_in> [csv_out]\n"
         <<"    --load_synth_prefix <path>\n"
         <<"    --index <index_path>\n"
         <<"    --k <int>\n";
        return 1;
    }
    std::string data_dir = argv[1];
    std::string csv_in   = argv[2];
    std::string csv_out;
    std::string load_synth_prefix;
    std::string index_path;
    int K = 10;
    // 旧逻辑中的预探索 ef 已弃用

    // parse args
    for (int i=3;i<argc;++i){
        std::string s = argv[i];
        auto next=[&](){ if(i+1>=argc){std::cerr<<"missing after "<<s<<"\n"; std::exit(1);} return std::string(argv[++i]); };
    if (s=="--load_synth_prefix") load_synth_prefix = next();
    else if (s=="--index") index_path = next();
    else if (s=="--k") K = std::stoi(next());
    else if (s=="--ef_probe") { (void)next(); /* deprecated: ignore */ }
        else if (csv_out.empty()) csv_out = s; // 第三个位置参数（如果没用 flag）
        else { std::cerr<<"Unknown arg "<<s<<"\n"; return 1; }
    }
    if (csv_out.empty()){
        auto stem = std::filesystem::path(csv_in).stem().string();
        auto dir  = std::filesystem::path(csv_in).parent_path();
        csv_out = (dir / (stem + ".with_new_feats.csv")).string();
    }
    if (load_synth_prefix.empty()){
        std::cerr<<"ERROR: --load_synth_prefix is required (.xq.fbin, .labels.ibin)\n";
        return 1;
    }
    if (index_path.empty()) index_path = (std::filesystem::path(data_dir) / "base.hnsw").string();

    // load base / queries / labels
    std::string base_path  = (std::filesystem::exists((data_dir + "/base.10M.fbin")) ?
                              data_dir + "/base.10M.fbin" : data_dir + "/base.2.5M.fbin");
    std::vector<float> xb; int dim=0; idx_t nb = read_fbin(base_path.c_str(), xb, dim);
    std::vector<float> xq; int dq=0; idx_t nq = read_fbin((load_synth_prefix + ".xq.fbin").c_str(), xq, dq);
    if (dq!=dim){ std::cerr<<"Dim mismatch: xq "<<dq<<" vs base "<<dim<<"\n"; return 1; }
    std::vector<int> labels; int dlab=0; idx_t nlab = read_ibin((load_synth_prefix + ".labels.ibin").c_str(), labels, dlab);
    if (nlab!=nq || dlab!=1){ std::cerr<<"labels size mismatch\n"; return 1; }

    // build clusters -> list of qids
    int maxlab=-1; for (int v: labels) if (v>maxlab) maxlab=v;
    std::vector<std::vector<idx_t>> cluster_qids((size_t)std::max(1,maxlab+1));
    for (idx_t i=0;i<nq;++i){ int c=labels[(size_t)i]; if (c>=0) cluster_qids[(size_t)c].push_back(i); }

    // load index
    hnswlib::L2Space space(dim);
    const size_t M=32, efc_def=200; // efConstruction 不影响查询
    hnswlib::HierarchicalNSW<float> hnsw(&space, nb, M, efc_def);
    if (!std::filesystem::exists(index_path)){
        std::cerr<<"ERROR: index not found: "<<index_path<<"\n";
        return 1;
    }
    hnsw.loadIndex(index_path, &space, nb);

    auto centroid_of = [&](const std::vector<idx_t>& ids){
        std::vector<float> c(dim,0.f);
        if (ids.empty()) return c;
        for (auto qi: ids){
            const float* qv = xq.data()+qi*dim;
            for (int j=0;j<dim;++j) c[j]+=qv[j];
        }
        for (int j=0;j<dim;++j) c[j]/=(float)ids.size();
        return c;
    };

    // open CSV
    std::ifstream fin(csv_in);
    if (!fin){ std::cerr<<"Cannot open csv_in "<<csv_in<<"\n"; return 1; }
    std::string header_line; std::getline(fin, header_line);
    std::vector<std::string> headers = split_csv_line(header_line);

    // find essential columns
    auto find_col = [&](const std::string& name)->int{
        for (size_t i=0;i<headers.size();++i) if (headers[i]==name) return (int)i;
        return -1;
    };
    int col_cluster_id = find_col("cluster_id");
    int col_efc        = find_col("efc");
    if (col_cluster_id<0 || col_efc<0){
        std::cerr<<"CSV must contain columns: cluster_id, efc\n";
        return 1;
    }

    // new columns (append if not exists)
    auto ensure_col = [&](const std::string& name){
        int c = find_col(name);
        if (c<0){ headers.push_back(name); return (int)headers.size()-1; }
        return c;
    };
    // 仅新增 efc 相关列；probe256 列若已存在保持不动（不重算、不新增）
    // int c_lid_probe   = find_col("lid_k_probe256");   // 可能为 -1
    // int c_rc_probe    = find_col("rc_k_probe256");    // 可能为 -1
    // int c_exp_probe   = find_col("expand2k_over_k_probe256"); // 可能为 -1
    int c_lid_efc     = ensure_col("lid_k_efc");
    int c_rc_efc      = ensure_col("rc_k_efc");
    int c_exp_efc     = ensure_col("expand2k_over_k_efc");

    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(fin, line)){
        if (line.empty()) continue;
        rows.push_back(split_csv_line(line));
    }
    fin.close();

    // process each row (仅补充 *_efc 列；若存在则覆盖，若缺失则写入)
    for (auto& row : rows){
        if ((int)row.size() < (int)headers.size()) row.resize(headers.size(), "");
        int cid = std::stoi(row[col_cluster_id]);
        if (cid<0 || (size_t)cid >= cluster_qids.size()){
            // 无效簇 id -> 仅处理 efc 列
            row[c_lid_efc]   = row[c_rc_efc]   = row[c_exp_efc]   = "NaN";
            continue;
        }
        const auto& qids = cluster_qids[(size_t)cid];
        if (qids.empty()){
            row[c_lid_efc]   = row[c_rc_efc]   = row[c_exp_efc]   = "NaN";
            continue;
        }
        // centroid
        std::vector<float> c = centroid_of(qids);

        // efc run (per-row efc) —— 新增/覆盖
        int efc = 0;
        try { efc = std::max(1, std::stoi(row[col_efc])); }
        catch(...){ efc = 0; }
        if (efc <= 0){
            row[c_lid_efc] = row[c_rc_efc] = row[c_exp_efc] = "NaN";
        } else {
            auto top = searchL0_top_only(hnsw, (const void*)c.data(), (size_t)efc, space, dim).top_sorted;
            auto trio = compute_trio_from_top(top, K);
            auto to_s = [](double v){ std::ostringstream os; os<<std::setprecision(10)<<v; return os.str(); };
            row[c_lid_efc] = to_s(trio.lid);
            row[c_rc_efc]  = to_s(trio.rc);
            row[c_exp_efc] = to_s(trio.exp2k_over_k);
        }
    }

    // write out
    std::ofstream fout(csv_out);
    if (!fout){ std::cerr<<"Cannot open csv_out "<<csv_out<<"\n"; return 1; }
    fout<<join_csv_line(headers)<<"\n";
    for (auto& r : rows) fout<<join_csv_line(r)<<"\n";
    fout.close();

    std::cout<<"Done. Wrote "<<csv_out<<"\n";
    return 0;
}
