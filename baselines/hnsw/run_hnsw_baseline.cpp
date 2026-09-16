// run_hnsw_baseline.cpp
// Baseline: plain HNSW search per query (no cluster logic)
// Inputs kept consistent with run_hnsw_cluster_consensus_base.cpp:
//   - <data_dir> for base vectors and default index path unless overridden
//   - --load_synth_prefix <prefix> to read <prefix>.xq.fbin and <prefix>.gt.ibin
//   - optional explicit index path as argv[2]
// ./run_hnsw_baseline ../data/t2i-10M/base.10M.fbin ../data/t2i_base.hnsw \
//   --load_synth_prefix ./outputs_t2i_new_train_test/s1 \
//   --K=1,10,100 \
//   --EF=500:200:5000 \
//   --csv=t2i_hnsw_baseline_eval.csv
// ./run_hnsw_baseline ../data/laion-10M/base.10M.fbin ../data/laion_base.hnsw \
//   --load_synth_prefix ./outputs_laion_new_train_test/s1 \
//   --K=1,10,100 \
//   --EF=500:200:5000 \
//   --csv=laion_hnsw_baseline_eval.csv
// ./run_hnsw_baseline ../data/clip-webvid-2.5M/ ../data/webvid_base.hnsw \
//   --load_synth_prefix ./clip_wikianswers_vectors/s1 \
//   --K=10\
//   --EF=100:500:2000 \
//   --csv=webvid_hnsw_baseline_wiki.csv \
//   --per_query_csv=webvid_hnsw_baseline_per_query_wiki.csv
#include <hnswlib/hnswlib.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>

using idx_t = size_t;

// ----- Instrumented HNSW for visited counting -----
struct StatHNSW : public hnswlib::HierarchicalNSW<float> {
    using Base = hnswlib::HierarchicalNSW<float>;
    using tableint = hnswlib::tableint;
    using labeltype = hnswlib::labeltype;

    explicit StatHNSW(hnswlib::SpaceInterface<float>* s, size_t max_elements, size_t M, size_t efc)
        : Base(s, max_elements, M, efc) {}

    // Instrumented wrapper: return original search results to preserve recall,
    // and count exact L2 distance calls by temporarily wrapping the distance function.
    std::priority_queue<std::pair<float, labeltype>>
    searchKnnWithStats(const void* q, size_t k, size_t ef, size_t& out_visited_unique) const {
        // Save current distance function and parameter
        auto self = const_cast<StatHNSW*>(this);
        hnswlib::DISTFUNC<float> saved_func = self->fstdistfunc_;
        void* saved_param = self->dist_func_param_;
        struct Ctx { hnswlib::DISTFUNC<float> base; void* base_param; size_t* counter; } ctx{saved_func, saved_param, &out_visited_unique};
        out_visited_unique = 0; // here: counts L2 distance calls
        auto counting = +[](const void* a, const void* b, const void* param)->float {
            const Ctx* c = (const Ctx*)param;
            ++(*c->counter);
            return c->base(a, b, c->base_param);
        };
        self->fstdistfunc_ = counting;
        self->dist_func_param_ = &ctx;

        // Perform the actual HNSW search (uses current ef already set via setEf)
        auto original_results = this->searchKnn(q, k);

        // Restore
        self->fstdistfunc_ = saved_func;
        self->dist_func_param_ = saved_param;
        return original_results;
    }

    // expose get_linklist0
    const hnswlib::linklistsizeint* get_linklist0(tableint internal_id) const {
        return (const hnswlib::linklistsizeint*)this->linkLists_[internal_id];
    }
};

// ----- I/O helpers (aligned with your style) -----
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
static std::string first_existing(const std::string& dir, const std::vector<std::string>& names) {
    for (auto& n: names){ auto p=dir+"/"+n; if (std::filesystem::exists(p)) return p; }
    return dir + "/" + names.front();
}

static float compute_recall_at_k(const std::vector<std::vector<idx_t>>& pred,
                                 const std::vector<int>& gt, int gt_cols,
                                 int k, idx_t nq) {
    double sum = 0.0;
    for (idx_t i=0;i<nq;++i){
        int hit=0; int kk = std::min(k, (int)pred[i].size());
        for (int t=0;t<kk;++t){
            int id = (int)pred[i][t];
            const int* g = gt.data() + (size_t)i*gt_cols;
            for (int j=0;j<std::min(k, gt_cols); ++j){ if (g[j]==id){ ++hit; break; } }
        }
        sum += (double)hit / std::min((double)k, (double)gt_cols);
    }
    return (float)(sum / (double)nq);
}

// ---- list / grid parsing helpers (similar style to search_nsg) ----
static void append_grid_token(const std::string& tok, std::vector<int>& out){
    if(tok.empty()) return;
    size_t c1 = tok.find(':');
    if(c1 == std::string::npos){ out.push_back(std::stoi(tok)); return; }
    size_t c2 = tok.find(':', c1+1);
    auto to_i = [](const std::string& s){ return std::stoi(s); };
    if(c2 == std::string::npos){
        int a = to_i(tok.substr(0,c1));
        int b = to_i(tok.substr(c1+1));
        if(a<=b){ for(int x=a;x<=b;++x) out.push_back(x); }
        else     { for(int x=a;x>=b;--x) out.push_back(x); }
        return;
    }
    int a = to_i(tok.substr(0,c1));
    int s = to_i(tok.substr(c1+1, c2-(c1+1)));
    int b = to_i(tok.substr(c2+1));
    if(s==0) return;
    if( (long long)(b-a) * (long long)s < 0 ) return;
    if(s>0){ for(int x=a;x<=b;x+=s) out.push_back(x); }
    else    { for(int x=a;x>=b;x+=s) out.push_back(x); }
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
static std::vector<int> parse_int_list_simple(const std::string& s){
    std::vector<int> v; std::string buf;
    for(char c: s){
        if(c==',' || c==' '){ if(!buf.empty()){ v.push_back(std::stoi(buf)); buf.clear(); } }
        else buf.push_back(c);
    }
    if(!buf.empty()) v.push_back(std::stoi(buf));
    return v;
}

struct Args {
    std::string dir, index_path;
    std::vector<int> Ks;            // e.g. 1,10,100
    std::vector<int> EFs;           // ef search list/grid
    std::string load_synth_prefix;  // prefix for xq/gt
    std::string csv_path = "hnsw_baseline_results.csv";
    std::string per_query_csv_path; // optional: per-query detail CSV
    std::vector<int> cluster_ids;    // subset of clusters to evaluate
    int batch_size = 0;              // max queries per cluster (0 = all)
    std::string gt_row_map_path;     // optional ibin: maps query idx -> GT row idx
    bool use_internal_ids = false;   // compare internal node ids to GT (if labels mismatch)
};
static void usage(){
    std::cout << "Usage:\n  run_hnsw_baseline <data_dir> [index_path]\n"
              << "    --load_synth_prefix <path>       # reads <path>.xq.fbin / <path>.gt.ibin\n"
              << "                                      and <path>.labels.ibin for per-query cluster IDs\n"
              << "    --K=1,10,100                     # recall@K list (default 10)\n"
              << "    --EF=200,400 or 100:100:1000     # ef list or grid (default 200)\n"
              << "    --k <int> --ef <int>             # legacy single-value (appended)\n"
              << "    --csv <path>                     # output overall csv file\n"
              << "    --per_query_csv <path>           # output per-query csv (with headers)\n"
              << "    --clusters=1,2,3                 # only evaluate queries whose cluster id in list\n"
              << "    --batch_size <int>              # evaluate at most this many queries per cluster\n"
              << "    --gt_row_map <path>              # ibin mapping: query index -> GT row index\n"
              << "    --use_internal_ids               # use internal ids for recall instead of labels\n";
}
static Args parse_args(int argc, char** argv){
    if(argc < 2){ usage(); std::exit(1); }
    Args a; a.dir = argv[1];
    a.index_path = (argc>2 && argv[2][0] != '-') ? argv[2] : a.dir ;
    int i = (argc>2 && argv[2][0] != '-') ? 3 : 2;
    for(; i<argc; ++i){
        std::string s = argv[i]; auto need_next=[&](){ if(i+1>=argc){ std::cerr<<"missing arg after "<<s<<"\n"; std::exit(1);} return std::string(argv[++i]); };
        if(s.rfind("--",0)==0){
            // forms --X=value or legacy --x <value>
            size_t eq = s.find('=');
            std::string key = s.substr(2, (eq==std::string::npos? s.size()-2 : eq-2));
            std::string val = (eq==std::string::npos? "" : s.substr(eq+1));
            auto ensure_val=[&](){ if(val.empty()){ val = need_next(); } };
            if(key=="load_synth_prefix"){ ensure_val(); a.load_synth_prefix = val; }
            else if(key=="K"){ ensure_val(); a.Ks = parse_int_list_simple(val); }
            else if(key=="EF"){ ensure_val(); a.EFs = parse_int_list_or_grid(val); }
            else if(key=="k"){ ensure_val(); a.Ks.push_back(std::stoi(val)); }
            else if(key=="ef"){ ensure_val(); a.EFs.push_back(std::stoi(val)); }
            else if(key=="csv"){ ensure_val(); a.csv_path = val; }
            else if(key=="per_query_csv"){ ensure_val(); a.per_query_csv_path = val; }
            else if(key=="clusters"){ ensure_val(); a.cluster_ids = parse_int_list_simple(val); }
            else if(key=="batch_size"){ ensure_val(); a.batch_size = std::stoi(val); }
            else if(key=="gt_row_map"){ ensure_val(); a.gt_row_map_path = val; }
            else if(key=="use_internal_ids"){ a.use_internal_ids = true; }
            else { std::cerr<<"Unknown option: "<<s<<"\n"; usage(); std::exit(1); }
        }else{
            std::cerr<<"Unexpected positional arg after data/index paths: "<<s<<"\n"; usage(); std::exit(1);
        }
    }
    if(a.load_synth_prefix.empty()){ std::cerr<<"ERROR: --load_synth_prefix is required\n"; std::exit(1); }
    if(a.Ks.empty()) a.Ks = {10};
    if(a.EFs.empty()) a.EFs = {200};
    // de-duplicate & sort
    std::sort(a.Ks.begin(), a.Ks.end()); a.Ks.erase(std::unique(a.Ks.begin(), a.Ks.end()), a.Ks.end());
    std::sort(a.EFs.begin(), a.EFs.end()); a.EFs.erase(std::unique(a.EFs.begin(), a.EFs.end()), a.EFs.end());
    return a;
}

int main(int argc, char** argv){
    Args args = parse_args(argc, argv);

    // Load base & index
    std::string base_path = first_existing(args.dir, {"base.2.5M.fbin","base.10M.fbin"});
    std::vector<float> xb; int dim=0; idx_t nb = read_fbin(base_path.c_str(), xb, dim);
    hnswlib::L2Space space(dim);
    const size_t M=32, efc=200;
    StatHNSW hnsw(&space, nb, M, efc);
    if(std::filesystem::exists(args.index_path)){
        hnsw.loadIndex(args.index_path, &space, nb);
    }else{
        for(idx_t i=0;i<nb;++i) hnsw.addPoint(xb.data()+i*dim, i);
        hnsw.saveIndex(args.index_path);
    }

    // Queries & GT
    std::string xqf = args.load_synth_prefix + ".xq.fbin";
    std::string gtf = args.load_synth_prefix + ".gt.ibin";
    if(!std::filesystem::exists(xqf) || !std::filesystem::exists(gtf)){
        std::cerr<<"ERROR: missing prefix files: "<<args.load_synth_prefix<<"\n"; return 1; }
    std::vector<float> xq; int dq=0; idx_t nq = read_fbin(xqf.c_str(), xq, dq);
    if(dq!=dim){ std::cerr<<"Dim mismatch xq="<<dq<<" base="<<dim<<"\n"; return 1; }
    std::vector<int> gt; int dgt=0; idx_t ngt_rows = read_ibin(gtf.c_str(), gt, dgt);
    if(ngt_rows < nq){ std::cerr<<"gt rows < nq\n"; return 1; }

    // Cluster labels from <prefix>.labels.ibin & subset filtering
    std::vector<int> query_clusters; // size nq, if labels file exists
    std::vector<idx_t> subset_ids;   // indices of queries to evaluate
    {
        std::string labelsf = args.load_synth_prefix + ".labels.ibin";
        if(std::filesystem::exists(labelsf)){
            int cdim=0; std::vector<int> raw; idx_t rows = read_ibin(labelsf.c_str(), raw, cdim);
            if(rows < nq){ std::cerr<<"labels rows < nq\n"; return 1; }
            query_clusters.resize(nq);
            for(idx_t i=0;i<nq;++i){ query_clusters[i] = raw[(size_t)i*cdim]; }
        } else {
            if(!args.cluster_ids.empty() || args.batch_size > 0){
                std::cerr << "ERROR: --clusters/--batch_size provided but labels file not found: " << labelsf << "\n";
                return 1;
            }
        }
        if(!query_clusters.empty() && (!args.cluster_ids.empty() || args.batch_size > 0)){
            std::set<int> keep(args.cluster_ids.begin(), args.cluster_ids.end());
            std::unordered_map<int, int> kept_per_cluster;
            for(idx_t i=0;i<nq;++i){
                int cid = query_clusters[i];
                if(!keep.empty() && !keep.count(cid)) continue;
                if(args.batch_size > 0 && kept_per_cluster[cid] >= args.batch_size) continue;
                subset_ids.push_back(i);
                ++kept_per_cluster[cid];
            }
        }
    }
    bool use_subset = !subset_ids.empty();
    idx_t eval_nq = use_subset ? (idx_t)subset_ids.size() : nq;

    // Optional GT row remapping (handles query reordering between xq and gt)
    std::vector<int> gt_row_map; // size nq, value = row index in gt for query i
    if(!args.gt_row_map_path.empty()){
        int dmap=0; idx_t rmap_rows=0; std::vector<int> rawmap; rmap_rows = read_ibin(args.gt_row_map_path.c_str(), rawmap, dmap);
        if((idx_t)rmap_rows < nq){ std::cerr<<"gt_row_map rows < nq\n"; return 1; }
        gt_row_map.resize(nq);
        for(idx_t i=0;i<nq;++i){ gt_row_map[i] = rawmap[(size_t)i*dmap]; }
    }

    // Prepare CSV header
    bool new_csv = !std::filesystem::exists(args.csv_path) || std::filesystem::file_size(args.csv_path)==0;
    {
        std::ofstream ofs(args.csv_path, std::ios::app);
        if(!ofs){ std::cerr<<"WARNING: cannot open csv: "<<args.csv_path<<"\n"; }
        else if(new_csv){
            ofs << "data_dir,index_path,base_path,prefix,ef,K,nb,nq,subset_nq,dim,QPS,ms_per_query,l2_calls_avg,recall,clusters\n";
        }
    }

    int Kmax = 0; for(int k : args.Ks) Kmax = std::max(Kmax, k);
    if(Kmax <= 0) Kmax = 1;

    std::cout << "\n===== HNSW Baseline Multi-EF =====\n";
    std::cout << "Base nb="<<nb<<" dim="<<dim<<" Queries nq="<<nq;
    if(use_subset) std::cout << " (subset="<<eval_nq<<")";
    std::cout << "\n";
    if(use_subset){
        std::cout << "Selected clusters:";
        for(int cid: args.cluster_ids) std::cout << ' ' << cid;
        std::cout << "\n";
    }
    if(args.batch_size > 0) std::cout << "Batch size per cluster: " << args.batch_size << "\n";
    std::cout << "K list:"; for(int k: args.Ks) std::cout << ' ' << k; std::cout << " (Kmax="<<Kmax<<")\n";
    std::cout << "EF list:"; for(int ef: args.EFs) std::cout << ' ' << ef; std::cout << "\n";

    // For each EF, run search once with Kmax, then compute recall@K for each K
    for(int ef : args.EFs){
        hnsw.setEf(ef);
        std::vector<std::vector<idx_t>> pred(eval_nq);
        std::vector<double> per_query_time_s(eval_nq, 0.0);
        std::vector<size_t> per_query_visited(eval_nq, 0);
        auto t0 = std::chrono::high_resolution_clock::now();
        for(idx_t qi=0; qi<eval_nq; ++qi){
            idx_t orig_i = use_subset ? subset_ids[qi] : qi;
            const float* qv = xq.data() + (size_t)orig_i*dim;
            size_t visited_nodes = 0;
            auto t_q0 = std::chrono::high_resolution_clock::now();
            auto res = hnsw.searchKnnWithStats(qv, (size_t)Kmax, (size_t)ef, visited_nodes);
            auto t_q1 = std::chrono::high_resolution_clock::now();
            per_query_time_s[qi] = std::chrono::duration<double>(t_q1 - t_q0).count();
            per_query_visited[qi] = visited_nodes;
            pred[qi].reserve(Kmax);
            while(!res.empty()){
                idx_t id = (idx_t)res.top().second;
                // If using internal IDs for recall, convert labels back to internal id.
                if(args.use_internal_ids){
                    // HierarchicalNSW doesn't provide reverse map label->internal directly; assume labels==internal when index was built accordingly.
                    // If labels differ, caller should not use --use_internal_ids.
                }
                pred[qi].push_back(id);
                res.pop();
            }
            std::reverse(pred[qi].begin(), pred[qi].end());
            while((int)pred[qi].size() < Kmax) pred[qi].push_back(pred[qi].empty()?0:pred[qi].back());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1-t0).count();
        double qps = sec>0.0 ? (double)eval_nq/sec : 0.0;
        double ms_per_q = sec*1000.0 / (double)eval_nq;
        double visited_avg = eval_nq ? std::accumulate(per_query_visited.begin(), per_query_visited.end(), 0.0) / (double)eval_nq : 0.0;

        // For each K compute recall and print / csv
        for(int k : args.Ks){
            float recall = 0.0f;
            if(eval_nq){
                double sum=0.0;
                for(idx_t qi=0; qi<eval_nq; ++qi){
                    idx_t orig_i = use_subset ? subset_ids[qi] : qi;
                    int gt_row = (gt_row_map.empty() ? (int)orig_i : gt_row_map[orig_i]);
                    const auto& R = pred[qi];
                    int hit=0; int kk = std::min(k, (int)R.size());
                    const int* g = gt.data() + (size_t)gt_row*dgt;
                    for(int t=0;t<kk;++t){
                        int id=(int)R[t];
                        for(int j=0;j<std::min(k, dgt); ++j){ if(g[j]==id){ ++hit; break; } }
                    }
                    sum += (double)hit / std::min((double)k, (double)dgt);
                }
                recall = (float)(sum / (double)eval_nq);
            }
            std::cout << "EF="<<ef<<" K="<<k
                      <<" | QPS="<<qps
                      <<" | ms/q="<<ms_per_q
                      <<" | L2CallsAvg="<<visited_avg
                      <<" | Recall="<<recall << "\n";
            std::ofstream ofs(args.csv_path, std::ios::app);
            if(ofs){
                std::stringstream clusters_ss; if(!args.cluster_ids.empty()){ for(size_t ci=0; ci<args.cluster_ids.size(); ++ci){ if(ci) clusters_ss<<';'; clusters_ss<<args.cluster_ids[ci]; } }
                ofs << args.dir << ','
                    << args.index_path << ','
                    << base_path << ','
                    << args.load_synth_prefix << ','
                    << ef << ',' << k << ','
                    << nb << ',' << nq << ',' << eval_nq << ',' << dim << ','
                    << qps << ',' << ms_per_q << ',' << visited_avg << ',' << recall << ',' << clusters_ss.str() << '\n';
            }
        }

        // Per-query CSV output (write after timing; not included in sec/qps)
        if(!args.per_query_csv_path.empty()){
            bool new_pcsv = !std::filesystem::exists(args.per_query_csv_path) || std::filesystem::file_size(args.per_query_csv_path)==0;
            std::ofstream pofs(args.per_query_csv_path, std::ios::app);
            if(!pofs){
                std::cerr << "WARNING: cannot open per-query csv: " << args.per_query_csv_path << "\n";
            } else {
                if(new_pcsv){
                    pofs << "data_dir,index_path,base_path,prefix,ef,qid,orig_qid,cluster,search_time_s,l2_calls";
                    for(size_t idx=0; idx<args.Ks.size(); ++idx){ pofs << ",recall@" << args.Ks[idx]; }
                    pofs << "\n";
                }
                pofs << std::fixed << std::setprecision(6);
                for(idx_t qi=0; qi<eval_nq; ++qi){
                    idx_t orig_i = use_subset ? subset_ids[qi] : qi;
                    int cluster_val = query_clusters.empty() ? -1 : query_clusters[orig_i];
                    int gt_row = (gt_row_map.empty() ? (int)orig_i : gt_row_map[orig_i]);
                    pofs << args.dir << ','
                         << args.index_path << ','
                         << base_path << ','
                         << args.load_synth_prefix << ','
                         << ef << ',' << qi << ',' << orig_i << ',' << cluster_val << ','
                         << per_query_time_s[qi] << ',' << per_query_visited[qi];
                    for(int k : args.Ks){
                        int denom = std::min(k, dgt);
                        int hit=0; const auto& R = pred[qi];
                        for(int j=0;j<denom;++j){
                            int g = gt[(size_t)gt_row*dgt + j];
                            for(int t=0; t<std::min(k, (int)R.size()); ++t){ if((int)R[t]==g){ ++hit; break; } }
                        }
                        float r = denom>0 ? (float)hit/(float)denom : 0.0f;
                        pofs << ',' << r;
                    }
                    pofs << '\n';
                }
            }
        }
    }

    std::cout << "Done. CSV -> "<<args.csv_path<<"\n";
    return 0;
}

