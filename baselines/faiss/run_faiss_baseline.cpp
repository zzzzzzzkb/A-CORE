// run_faiss_baseline.cpp
// Faiss-based vector search baseline for cluster-level evaluation.
// Supports multiple index types: flat (brute-force), hnsw, ivfflat.
//
// Usage examples:
//   # Exact search (IndexFlatL2):
//   ./run_faiss_baseline ../data/clip-webvid-2.5M/ \
//     --load_synth_prefix ./outputs_webvid_real_clusters/s1 \
//     --index_type=flat \
//     --K=10 \
//     --csv=webvid_faiss_flat_baseline.csv
//
//   # HNSW search:
//   ./run_faiss_baseline ../data/clip-webvid-2.5M/ \
//     --load_synth_prefix ./outputs_webvid_real_clusters/s1 \
//     --index_type=hnsw --hnsw_M=32 --hnsw_efSearch=200 \
//     --K=10 \
//     --csv=webvid_faiss_hnsw_baseline.csv
//
//   # IVFFlat search:
//   ./run_faiss_baseline ../data/clip-webvid-2.5M/ \
//     --load_synth_prefix ./outputs_webvid_real_clusters/s1 \
//     --index_type=ivfflat --nlist=1000 --nprobe=10 \
//     --K=10 \
//     --csv=webvid_faiss_ivfflat_baseline.csv

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/MetricType.h>

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
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using idx_t = faiss::idx_t;

// ----- I/O helpers (same format as run_hnsw_baseline) -----
static idx_t read_fbin(const char* f, std::vector<float>& out, int& dim) {
    FILE* fp = fopen(f, "rb");
    if (!fp) { perror(f); std::exit(1); }
    int n = 0, d = 0;
    fread(&n, 4, 1, fp);
    fread(&d, 4, 1, fp);
    out.resize((idx_t)n * (idx_t)d);
    size_t tot = (size_t)n * (size_t)d;
    if (fread(out.data(), sizeof(float), tot, fp) != tot) {
        std::cerr << "read " << f << " fail\n";
        std::exit(1);
    }
    fclose(fp);
    dim = d;
    return (idx_t)n;
}

static idx_t read_ibin(const char* f, std::vector<int>& out, int& dim) {
    FILE* fp = fopen(f, "rb");
    if (!fp) { perror(f); std::exit(1); }
    int n = 0, d = 0;
    fread(&n, 4, 1, fp);
    fread(&d, 4, 1, fp);
    out.resize((idx_t)n * (idx_t)d);
    size_t tot = (size_t)n * (size_t)d;
    if (fread(out.data(), sizeof(int), tot, fp) != tot) {
        std::cerr << "read " << f << " fail\n";
        std::exit(1);
    }
    fclose(fp);
    dim = d;
    return (idx_t)n;
}

static std::string first_existing(const std::string& dir,
                                   const std::vector<std::string>& names) {
    for (auto& n : names) {
        auto p = dir + "/" + n;
        if (std::filesystem::exists(p)) return p;
    }
    return dir + "/" + names.front();
}

// ---- list parsing helpers ----
static void append_grid_token(const std::string& tok, std::vector<int>& out) {
    if (tok.empty()) return;
    size_t c1 = tok.find(':');
    if (c1 == std::string::npos) { out.push_back(std::stoi(tok)); return; }
    size_t c2 = tok.find(':', c1 + 1);
    auto to_i = [](const std::string& s) { return std::stoi(s); };
    if (c2 == std::string::npos) {
        int a = to_i(tok.substr(0, c1));
        int b = to_i(tok.substr(c1 + 1));
        if (a <= b) { for (int x = a; x <= b; ++x) out.push_back(x); }
        else        { for (int x = a; x >= b; --x) out.push_back(x); }
        return;
    }
    int a = to_i(tok.substr(0, c1));
    int s = to_i(tok.substr(c1 + 1, c2 - (c1 + 1)));
    int b = to_i(tok.substr(c2 + 1));
    if (s == 0) return;
    if ((long long)(b - a) * (long long)s < 0) return;
    if (s > 0) { for (int x = a; x <= b; x += s) out.push_back(x); }
    else       { for (int x = a; x >= b; x += s) out.push_back(x); }
}

static std::vector<int> parse_int_list_or_grid(const std::string& s) {
    std::vector<int> v;
    std::string buf;
    for (char c : s) {
        if (c == ',' || c == ' ') {
            if (!buf.empty()) { append_grid_token(buf, v); buf.clear(); }
        } else buf.push_back(c);
    }
    if (!buf.empty()) append_grid_token(buf, v);
    return v;
}

static std::vector<int> parse_int_list_simple(const std::string& s) {
    std::vector<int> v;
    std::string buf;
    for (char c : s) {
        if (c == ',' || c == ' ') {
            if (!buf.empty()) { v.push_back(std::stoi(buf)); buf.clear(); }
        } else buf.push_back(c);
    }
    if (!buf.empty()) v.push_back(std::stoi(buf));
    return v;
}

// ---- Args ----
enum IndexType { FLAT, HNSW, IVFFLAT };

struct Args {
    std::string dir;
    IndexType index_type = FLAT;
    std::vector<int> Ks;              // recall@K list
    std::string load_synth_prefix;    // prefix for xq/gt/labels
    std::string csv_path = "faiss_baseline_results.csv";
    std::string per_query_csv_path;
    std::vector<int> cluster_ids;
    int batch_size = 0;
    std::string gt_row_map_path;
    // HNSW params
    int hnsw_M = 32;
    std::vector<int> hnsw_efSearch;   // ef_search list
    // IVFFlat params
    int nlist = 1000;
    std::vector<int> nprobe;          // nprobe list
    // Save/load index
    std::string save_index_path;
    std::string load_index_path;
};

static void usage() {
    std::cout <<
        "Usage:\n"
        "  run_faiss_baseline <data_dir>\n"
        "    --load_synth_prefix <path>       # reads <path>.xq.fbin / <path>.gt.ibin\n"
        "                                      and <path>.labels.ibin for per-query cluster IDs\n"
        "    --index_type=flat|hnsw|ivfflat   # index type (default: flat)\n"
        "    --K=1,10,100                     # recall@K list (default 10)\n"
        "    --csv <path>                     # output overall csv file\n"
        "    --per_query_csv <path>           # output per-query csv\n"
        "    --clusters=1,2,3                 # only evaluate queries whose cluster id in list\n"
        "    --batch_size <int>               # max queries per cluster (0=all)\n"
        "    --gt_row_map <path>              # ibin mapping: query idx -> GT row idx\n"
        "  HNSW options:\n"
        "    --hnsw_M=32                      # HNSW M parameter\n"
        "    --hnsw_efSearch=100,200,500      # ef_search list or grid (e.g. 100:100:500)\n"
        "  IVFFlat options:\n"
        "    --nlist=1000                     # number of clusters for IVF\n"
        "    --nprobe=1,10,100                # nprobe list or grid\n"
        "  Index I/O:\n"
        "    --save_index <path>              # save built index to file\n"
        "    --load_index <path>              # load pre-built index from file\n";
}

static Args parse_args(int argc, char** argv) {
    if (argc < 2) { usage(); std::exit(1); }
    Args a;
    a.dir = argv[1];
    int i = 2;
    for (; i < argc; ++i) {
        std::string s = argv[i];
        auto need_next = [&]() {
            if (i + 1 >= argc) { std::cerr << "missing arg after " << s << "\n"; std::exit(1); }
            return std::string(argv[++i]);
        };
        if (s.rfind("--", 0) == 0) {
            size_t eq = s.find('=');
            std::string key = s.substr(2, (eq == std::string::npos ? s.size() - 2 : eq - 2));
            std::string val = (eq == std::string::npos ? "" : s.substr(eq + 1));
            auto ensure_val = [&]() { if (val.empty()) { val = need_next(); } };

            if (key == "load_synth_prefix") { ensure_val(); a.load_synth_prefix = val; }
            else if (key == "index_type") {
                ensure_val();
                if (val == "flat") a.index_type = FLAT;
                else if (val == "hnsw") a.index_type = HNSW;
                else if (val == "ivfflat") a.index_type = IVFFLAT;
                else { std::cerr << "Unknown index_type: " << val << "\n"; std::exit(1); }
            }
            else if (key == "K") { ensure_val(); a.Ks = parse_int_list_simple(val); }
            else if (key == "csv") { ensure_val(); a.csv_path = val; }
            else if (key == "per_query_csv") { ensure_val(); a.per_query_csv_path = val; }
            else if (key == "clusters") { ensure_val(); a.cluster_ids = parse_int_list_simple(val); }
            else if (key == "batch_size") { ensure_val(); a.batch_size = std::stoi(val); }
            else if (key == "gt_row_map") { ensure_val(); a.gt_row_map_path = val; }
            else if (key == "hnsw_M") { ensure_val(); a.hnsw_M = std::stoi(val); }
            else if (key == "hnsw_efSearch") { ensure_val(); a.hnsw_efSearch = parse_int_list_or_grid(val); }
            else if (key == "nlist") { ensure_val(); a.nlist = std::stoi(val); }
            else if (key == "nprobe") { ensure_val(); a.nprobe = parse_int_list_or_grid(val); }
            else if (key == "save_index") { ensure_val(); a.save_index_path = val; }
            else if (key == "load_index") { ensure_val(); a.load_index_path = val; }
            else { std::cerr << "Unknown option: " << s << "\n"; usage(); std::exit(1); }
        } else {
            std::cerr << "Unexpected positional arg: " << s << "\n"; usage(); std::exit(1);
        }
    }
    if (a.load_synth_prefix.empty()) {
        std::cerr << "ERROR: --load_synth_prefix is required\n"; std::exit(1);
    }
    if (a.Ks.empty()) a.Ks = {10};
    if (a.hnsw_efSearch.empty()) a.hnsw_efSearch = {200};
    if (a.nprobe.empty()) a.nprobe = {10};
    // de-duplicate & sort
    auto dedup = [](std::vector<int>& v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    dedup(a.Ks);
    dedup(a.hnsw_efSearch);
    dedup(a.nprobe);
    return a;
}

// ---- Batch search helper ----
// Performs faiss batch search and returns results in pred[qi][j] format.
static void faiss_batch_search(faiss::Index* index, const float* queries,
                               idx_t nq, int K,
                               std::vector<idx_t>& labels_out,
                               std::vector<float>& distances_out) {
    labels_out.resize(nq * K);
    distances_out.resize(nq * K);
    index->search(nq, queries, K, distances_out.data(), labels_out.data());
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // ---- Load base vectors ----
    std::string base_path = first_existing(args.dir,
        {"base.2.5M.fbin", "base.10M.fbin", "base.1M.fbin"});
    std::vector<float> xb;
    int dim = 0;
    idx_t nb = read_fbin(base_path.c_str(), xb, dim);
    std::cout << "Base vectors: nb=" << nb << " dim=" << dim << " from " << base_path << "\n";

    // ---- Load queries & GT ----
    std::string xqf = args.load_synth_prefix + ".xq.fbin";
    std::string gtf = args.load_synth_prefix + ".gt.ibin";
    if (!std::filesystem::exists(xqf) || !std::filesystem::exists(gtf)) {
        std::cerr << "ERROR: missing prefix files: " << args.load_synth_prefix << "\n";
        return 1;
    }
    std::vector<float> xq;
    int dq = 0;
    idx_t nq = read_fbin(xqf.c_str(), xq, dq);
    if (dq != dim) { std::cerr << "Dim mismatch xq=" << dq << " base=" << dim << "\n"; return 1; }
    std::vector<int> gt;
    int dgt = 0;
    idx_t ngt_rows = read_ibin(gtf.c_str(), gt, dgt);
    if (ngt_rows < nq) { std::cerr << "gt rows < nq\n"; return 1; }
    std::cout << "Queries: nq=" << nq << " dim=" << dq << ", GT rows=" << ngt_rows << " dgt=" << dgt << "\n";

    // ---- Cluster labels & subset filtering ----
    std::vector<int> query_clusters;
    std::vector<idx_t> subset_ids;
    {
        std::string labelsf = args.load_synth_prefix + ".labels.ibin";
        if (std::filesystem::exists(labelsf)) {
            int cdim = 0;
            std::vector<int> raw;
            idx_t rows = read_ibin(labelsf.c_str(), raw, cdim);
            if (rows < nq) { std::cerr << "labels rows < nq\n"; return 1; }
            query_clusters.resize(nq);
            for (idx_t i = 0; i < nq; ++i) query_clusters[i] = raw[(size_t)i * cdim];
        } else {
            if (!args.cluster_ids.empty() || args.batch_size > 0) {
                std::cerr << "ERROR: --clusters/--batch_size provided but labels file not found: "
                          << labelsf << "\n";
                return 1;
            }
        }
        if (!query_clusters.empty() && (!args.cluster_ids.empty() || args.batch_size > 0)) {
            std::set<int> keep(args.cluster_ids.begin(), args.cluster_ids.end());
            std::unordered_map<int, int> kept_per_cluster;
            for (idx_t i = 0; i < nq; ++i) {
                int cid = query_clusters[i];
                if (!keep.empty() && !keep.count(cid)) continue;
                if (args.batch_size > 0 && kept_per_cluster[cid] >= args.batch_size) continue;
                subset_ids.push_back(i);
                ++kept_per_cluster[cid];
            }
        }
    }
    bool use_subset = !subset_ids.empty();
    idx_t eval_nq = use_subset ? (idx_t)subset_ids.size() : nq;

    // Optional GT row remapping
    std::vector<int> gt_row_map;
    if (!args.gt_row_map_path.empty()) {
        int dmap = 0;
        std::vector<int> rawmap;
        idx_t rmap_rows = read_ibin(args.gt_row_map_path.c_str(), rawmap, dmap);
        if (rmap_rows < nq) { std::cerr << "gt_row_map rows < nq\n"; return 1; }
        gt_row_map.resize(nq);
        for (idx_t i = 0; i < nq; ++i) gt_row_map[i] = rawmap[(size_t)i * dmap];
    }

    // ---- Prepare subset query vectors for batch search ----
    std::vector<float> xq_eval(eval_nq * dim);
    for (idx_t qi = 0; qi < eval_nq; ++qi) {
        idx_t orig_i = use_subset ? subset_ids[qi] : qi;
        std::memcpy(xq_eval.data() + qi * dim, xq.data() + orig_i * dim, dim * sizeof(float));
    }

    // ---- Build or load index ----
    int Kmax = *std::max_element(args.Ks.begin(), args.Ks.end());
    if (Kmax <= 0) Kmax = 1;

    std::unique_ptr<faiss::Index> index;

    // Try loading pre-built index
    if (!args.load_index_path.empty() && std::filesystem::exists(args.load_index_path)) {
        std::cout << "Loading pre-built index from: " << args.load_index_path << "\n";
        index.reset(faiss::read_index(args.load_index_path.c_str()));
        if ((idx_t)index->ntotal != nb) {
            std::cerr << "WARNING: loaded index ntotal=" << index->ntotal
                      << " != base nb=" << nb << "\n";
        }
    } else {
        std::cout << "Building faiss index...\n";
        auto t_build0 = std::chrono::high_resolution_clock::now();

        switch (args.index_type) {
        case FLAT: {
            std::cout << "  Index type: IndexFlatL2 (exact brute-force)\n";
            index = std::make_unique<faiss::IndexFlatL2>(dim);
            index->add(nb, xb.data());
            break;
        }
        case HNSW: {
            std::cout << "  Index type: IndexHNSWFlat, M=" << args.hnsw_M << "\n";
            auto hnsw_idx = new faiss::IndexHNSWFlat(dim, args.hnsw_M, faiss::METRIC_L2);
            hnsw_idx->hnsw.efConstruction = 200;  // match HNSW baseline efc
            index.reset(hnsw_idx);
            index->add(nb, xb.data());
            break;
        }
        case IVFFLAT: {
            std::cout << "  Index type: IndexIVFFlat, nlist=" << args.nlist << "\n";
            // Use a quantizer for IVFFlat
            auto quantizer = new faiss::IndexFlatL2(dim);
            auto ivf_idx = new faiss::IndexIVFFlat(quantizer, dim, args.nlist, faiss::METRIC_L2);
            ivf_idx->train(nb, xb.data());
            ivf_idx->add(nb, xb.data());
            index.reset(ivf_idx);
            break;
        }
        }

        auto t_build1 = std::chrono::high_resolution_clock::now();
        double build_sec = std::chrono::duration<double>(t_build1 - t_build0).count();
        std::cout << "  Build time: " << build_sec << " s\n";

        // Save index if requested
        if (!args.save_index_path.empty()) {
            std::cout << "  Saving index to: " << args.save_index_path << "\n";
            faiss::write_index(index.get(), args.save_index_path.c_str());
        }
    }

    // ---- Prepare CSV ----
    bool new_csv = !std::filesystem::exists(args.csv_path) ||
                   std::filesystem::file_size(args.csv_path) == 0;
    {
        std::ofstream ofs(args.csv_path, std::ios::app);
        if (!ofs) { std::cerr << "WARNING: cannot open csv: " << args.csv_path << "\n"; }
        else if (new_csv) {
            ofs << "data_dir,base_path,prefix,index_type,search_param,K,nb,nq,subset_nq,dim,"
                << "QPS,ms_per_query,recall,clusters\n";
        }
    }

    // ---- Run batch search ----
    std::string index_type_str;
    switch (args.index_type) {
        case FLAT: index_type_str = "flat"; break;
        case HNSW: index_type_str = "hnsw"; break;
        case IVFFLAT: index_type_str = "ivfflat"; break;
    }

    // Determine the search parameter list (efSearch for HNSW, nprobe for IVFFlat, none for flat)
    std::vector<int> search_params;
    if (args.index_type == FLAT) {
        search_params = {0}; // placeholder
    } else if (args.index_type == HNSW) {
        search_params = args.hnsw_efSearch;
    } else {
        search_params = args.nprobe;
    }

    std::cout << "\n===== Faiss Baseline =====\n";
    std::cout << "Index type: " << index_type_str << "\n";
    std::cout << "Base nb=" << nb << " dim=" << dim << " Queries nq=" << nq;
    if (use_subset) std::cout << " (subset=" << eval_nq << ")";
    std::cout << "\n";
    if (use_subset) {
        std::cout << "Selected clusters:";
        for (int cid : args.cluster_ids) std::cout << ' ' << cid;
        std::cout << "\n";
    }
    if (args.batch_size > 0) std::cout << "Batch size per cluster: " << args.batch_size << "\n";
    std::cout << "K list:"; for (int k : args.Ks) std::cout << ' ' << k;
    std::cout << " (Kmax=" << Kmax << ")\n";
    if (args.index_type != FLAT) {
        std::cout << "Search params:";
        for (int p : search_params) std::cout << ' ' << p;
        std::cout << "\n";
    }

    for (int sp : search_params) {
        // Set search parameter
        if (args.index_type == HNSW) {
            auto hnsw_idx = dynamic_cast<faiss::IndexHNSWFlat*>(index.get());
            if (hnsw_idx) hnsw_idx->hnsw.efSearch = sp;
        } else if (args.index_type == IVFFLAT) {
            auto ivf_idx = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
            if (ivf_idx) ivf_idx->nprobe = sp;
        }

        // Batch search
        std::vector<idx_t> labels;
        std::vector<float> distances;

        auto t0 = std::chrono::high_resolution_clock::now();
        faiss_batch_search(index.get(), xq_eval.data(), eval_nq, Kmax, labels, distances);
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        double qps = sec > 0.0 ? (double)eval_nq / sec : 0.0;
        double ms_per_q = sec * 1000.0 / (double)eval_nq;

        // Build pred array from batch results
        std::vector<std::vector<idx_t>> pred(eval_nq);
        for (idx_t qi = 0; qi < eval_nq; ++qi) {
            pred[qi].resize(Kmax);
            for (int j = 0; j < Kmax; ++j) {
                pred[qi][j] = labels[qi * Kmax + j];
            }
        }

        // For each K compute recall
        for (int k : args.Ks) {
            float recall = 0.0f;
            if (eval_nq) {
                double sum = 0.0;
                for (idx_t qi = 0; qi < eval_nq; ++qi) {
                    idx_t orig_i = use_subset ? subset_ids[qi] : qi;
                    int gt_row = (gt_row_map.empty() ? (int)orig_i : gt_row_map[orig_i]);
                    const auto& R = pred[qi];
                    int hit = 0;
                    int kk = std::min(k, (int)R.size());
                    const int* g = gt.data() + (size_t)gt_row * dgt;
                    for (int t = 0; t < kk; ++t) {
                        int id = (int)R[t];
                        if (id < 0) continue; // faiss returns -1 for missing
                        for (int j = 0; j < std::min(k, dgt); ++j) {
                            if (g[j] == id) { ++hit; break; }
                        }
                    }
                    sum += (double)hit / std::min((double)k, (double)dgt);
                }
                recall = (float)(sum / (double)eval_nq);
            }

            std::string param_str = (args.index_type == FLAT) ? "N/A" : std::to_string(sp);
            std::cout << "param=" << param_str << " K=" << k
                      << " | QPS=" << qps
                      << " | ms/q=" << ms_per_q
                      << " | Recall=" << recall << "\n";

            std::ofstream ofs(args.csv_path, std::ios::app);
            if (ofs) {
                std::stringstream clusters_ss;
                if (!args.cluster_ids.empty()) {
                    for (size_t ci = 0; ci < args.cluster_ids.size(); ++ci) {
                        if (ci) clusters_ss << ';';
                        clusters_ss << args.cluster_ids[ci];
                    }
                }
                ofs << args.dir << ','
                    << base_path << ','
                    << args.load_synth_prefix << ','
                    << index_type_str << ','
                    << param_str << ','
                    << k << ',' << nb << ',' << nq << ',' << eval_nq << ',' << dim << ','
                    << qps << ',' << ms_per_q << ',' << recall << ','
                    << clusters_ss.str() << '\n';
            }
        }

        // ---- Per-query CSV output ----
        if (!args.per_query_csv_path.empty()) {
            bool new_pcsv = !std::filesystem::exists(args.per_query_csv_path) ||
                            std::filesystem::file_size(args.per_query_csv_path) == 0;
            std::ofstream pofs(args.per_query_csv_path, std::ios::app);
            if (!pofs) {
                std::cerr << "WARNING: cannot open per-query csv: "
                          << args.per_query_csv_path << "\n";
            } else {
                if (new_pcsv) {
                    pofs << "data_dir,base_path,prefix,index_type,search_param,"
                         << "qid,orig_qid,cluster,search_time_s";
                    for (size_t idx = 0; idx < args.Ks.size(); ++idx)
                        pofs << ",recall@" << args.Ks[idx];
                    pofs << "\n";
                }
                // For per-query timing, we report total_time / eval_nq as approximation
                // (faiss batch search doesn't give per-query timing)
                double avg_time_s = sec / (double)eval_nq;
                pofs << std::fixed << std::setprecision(6);
                for (idx_t qi = 0; qi < eval_nq; ++qi) {
                    idx_t orig_i = use_subset ? subset_ids[qi] : qi;
                    int cluster_val = query_clusters.empty() ? -1 : query_clusters[orig_i];
                    int gt_row = (gt_row_map.empty() ? (int)orig_i : gt_row_map[orig_i]);

                    std::string param_str = (args.index_type == FLAT) ? "N/A" : std::to_string(sp);
                    pofs << args.dir << ','
                         << base_path << ','
                         << args.load_synth_prefix << ','
                         << index_type_str << ','
                         << param_str << ','
                         << qi << ',' << orig_i << ',' << cluster_val << ','
                         << avg_time_s;

                    for (int k : args.Ks) {
                        int denom = std::min(k, dgt);
                        int hit = 0;
                        const auto& R = pred[qi];
                        for (int j = 0; j < denom; ++j) {
                            int g = gt[(size_t)gt_row * dgt + j];
                            for (int t = 0; t < std::min(k, (int)R.size()); ++t) {
                                if ((int)R[t] == g) { ++hit; break; }
                            }
                        }
                        float r = denom > 0 ? (float)hit / (float)denom : 0.0f;
                        pofs << ',' << r;
                    }
                    pofs << '\n';
                }
            }
        }
    }

    std::cout << "Done. CSV -> " << args.csv_path << "\n";
    return 0;
}
