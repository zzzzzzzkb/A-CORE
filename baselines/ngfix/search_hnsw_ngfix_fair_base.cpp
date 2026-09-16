#include "ngfixlib/graph/hnsw_ngfix.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using idx_t = size_t;

enum class MetricKind {
    L2,
    IP,
    COSINE,
};

static inline float unified_l2_sq(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

static inline float unified_dot(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline float unified_raw_score(const float* a, const float* b, int dim, MetricKind metric) {
    if (metric == MetricKind::L2) {
        return unified_l2_sq(a, b, dim);
    }
    // For IP/COSINE we keep smaller-is-better score as negative dot.
    return -unified_dot(a, b, dim);
}

struct SetupTiming {
    double base_load_s = 0.0;
    double index_load_s = 0.0;
    double queries_load_s = 0.0;
    double gt_load_s = 0.0;
    double labels_load_s = 0.0;
    double cluster_build_s = 0.0;
};

struct RunTiming {
    int ef = 0;
    double search_wall_s = 0.0;
    double eval_outside_timer_s = 0.0;
    double qps = 0.0;
    float recall = 0.0f;
};

static idx_t read_fbin(const char* f, std::vector<float>& out, int& dim) {
    FILE* fp = fopen(f, "rb");
    if (!fp) {
        perror(f);
        std::exit(1);
    }
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
    if (!fp) {
        perror(f);
        std::exit(1);
    }
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

static std::string first_existing(const std::string& dir, const std::vector<std::string>& names) {
    for (auto& n : names) {
        auto p = dir + "/" + n;
        if (std::filesystem::exists(p)) {
            return p;
        }
    }
    return dir + "/" + names.front();
}

static void ngfix_search_l0_heaps(
    ngfixlib::HNSW_NGFix<float>& index,
    const float* q,
    size_t ef,
    std::vector<idx_t>& out_top_ids,
    int& step_count,
    std::vector<ngfixlib::vl_type>& visited,
    ngfixlib::vl_type& visited_tag,
    MetricKind metric) {

    out_top_ids.clear();
    step_count = 0;
    if (index.n == 0) {
        return;
    }

    struct MinCmp {
        bool operator()(const std::pair<float, idx_t>& a, const std::pair<float, idx_t>& b) const {
            return a.first > b.first;
        }
    };

    using DistId = std::pair<float, idx_t>;

    const size_t n = (size_t)index.n;
    if (visited.size() != n) {
        visited.assign(n, 0);
        visited_tag = 1;
    } else {
        ++visited_tag;
        if (visited_tag == 0) {
            std::fill(visited.begin(), visited.end(), 0);
            visited_tag = 1;
        }
    }

    std::priority_queue<DistId, std::vector<DistId>, MinCmp> cand;
    std::priority_queue<DistId> top;

    auto mark = [&](idx_t id) {
        if (id >= n || visited[id] == visited_tag) {
            return false;
        }
        visited[id] = visited_tag;
        ++step_count;
        return true;
    };

    auto push_both = [&](idx_t id, float d) {
        cand.emplace(d, id);
        top.emplace(d, id);
        if (top.size() > ef) {
            top.pop();
        }
    };

    idx_t ep = (idx_t)index.entry_point;
    if (ep >= n) {
        ep = 0;
    }
    if (mark(ep)) {
        float d0 = unified_raw_score(index.getData(ep), q, static_cast<int>(index.dim), metric);
        push_both(ep, d0);
    }

    float lb = top.empty() ? std::numeric_limits<float>::infinity() : top.top().first;

    while (!cand.empty()) {
        auto cu = cand.top();
        if (cu.first > lb) {
            break;
        }
        cand.pop();

        auto [outs, sz, st] = index.getNeighbors(cu.second);
        for (int i = st; i < st + sz; ++i) {
            idx_t v = (idx_t)outs[i];
            if (!mark(v)) {
                continue;
            }
            float d = unified_raw_score(index.getData(v), q, static_cast<int>(index.dim), metric);
            if (top.size() < ef || d < lb) {
                cand.emplace(d, v);
                top.emplace(d, v);
                if (top.size() > ef) {
                    top.pop();
                }
                lb = top.top().first;
            }
        }
    }

    std::vector<DistId> buf;
    while (!top.empty()) {
        buf.push_back(top.top());
        top.pop();
    }
    std::sort(buf.begin(), buf.end(), [](auto& a, auto& b) { return a.first < b.first; });
    out_top_ids.reserve(buf.size());
    for (auto& p : buf) {
        out_top_ids.push_back(p.second);
    }
}

static float compute_recall_subset(const std::vector<std::vector<idx_t>>& results,
                                   const std::vector<int>& gt,
                                   int k,
                                   const std::vector<idx_t>& qids) {
    int nq = (int)results.size();
    if (nq == 0) {
        return 0.f;
    }
    int per_row = (int)gt.size() / nq;
    int expect = std::min(k, per_row);

    long long tot_hit = 0, tot_need = 0;
    for (auto qi : qids) {
        int hit = 0;
        const auto& R = results[qi];
        for (int j = 0; j < expect; ++j) {
            int g = gt[(size_t)qi * per_row + j];
            if (g < 0) {
                break;
            }
            for (int t = 0; t < std::min((int)R.size(), k); ++t) {
                if ((int)R[t] == g) {
                    ++hit;
                    break;
                }
            }
        }
        tot_hit += hit;
        tot_need += expect;
    }
    return tot_need > 0 ? (float)tot_hit / (float)tot_need : 0.f;
}

static std::vector<int> parse_ef_list(const std::string& raw) {
    std::vector<int> out;
    std::string s = raw;
    if (s.find(':') != std::string::npos) {
        std::stringstream ss(s);
        std::string a_str, st_str, b_str;
        if (std::getline(ss, a_str, ':') && std::getline(ss, st_str, ':') && std::getline(ss, b_str, ':')) {
            int a = std::stoi(a_str), st = std::stoi(st_str), b = std::stoi(b_str);
            if (st == 0) {
                out.push_back(std::max(1, a));
                return out;
            }
            if ((st > 0 && a > b) || (st < 0 && a < b)) {
                std::swap(a, b);
            }
            if (st > 0) {
                for (int x = a; x <= b; x += st) {
                    out.push_back(std::max(1, x));
                }
            } else {
                for (int x = a; x >= b; x += st) {
                    out.push_back(std::max(1, x));
                }
            }
            return out;
        }
    }

    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            out.push_back(std::max(1, std::stoi(tok)));
        }
    }
    if (out.empty()) {
        out.push_back(100);
    }
    return out;
}

struct Args {
    std::string dir;
    std::string index_path;
    std::string load_synth_prefix;
    std::string dist = "l2";

    int k = 10;
    int k_collect = 100;
    int ef = 200;
    std::vector<int> ef_list;

    std::string csv_out = "baseline_hnsw_ngfix_fair.csv";
    std::string per_query_csv_out;
};

static void usage() {
    std::cout
        << "Usage:\n  search_hnsw_ngfix_fair_base <data_dir> [index_path]\n"
        << "    --k <int> --k_collect <int> --ef <int> --ef_list <int|a:s:b|v1,v2,...>\n"
        << "    --load_synth_prefix <path>\n"
    << "    --dist <ip|l2|cosine>\n"
        << "    --csv_out <path> --per_query_csv <path>\n";
}

static Args parse_args(int argc, char** argv) {
    if (argc < 2) {
        usage();
        std::exit(1);
    }

    Args a;
    a.dir = argv[1];
    a.index_path = (argc > 2 && argv[2][0] != '-') ? argv[2] : a.dir + "/base_ngfix.index";

    int i = (argc > 2 && argv[2][0] != '-') ? 3 : 2;
    for (; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) {
                std::cerr << "missing after " << s << "\n";
                std::exit(1);
            }
            return std::string(argv[++i]);
        };

        if (s == "--k") {
            a.k = std::stoi(next());
        } else if (s == "--k_collect") {
            a.k_collect = std::stoi(next());
        } else if (s == "--ef") {
            a.ef = std::stoi(next());
        } else if (s == "--ef_list") {
            a.ef_list = parse_ef_list(next());
        } else if (s == "--load_synth_prefix") {
            a.load_synth_prefix = next();
        } else if (s == "--dist") {
            a.dist = next();
        } else if (s == "--csv_out") {
            a.csv_out = next();
        } else if (s == "--per_query_csv") {
            a.per_query_csv_out = next();
        } else {
            std::cerr << "Unknown arg " << s << "\n";
            usage();
            std::exit(1);
        }
    }

    if (a.load_synth_prefix.empty()) {
        std::cerr << "ERROR: --load_synth_prefix is required\n";
        std::exit(1);
    }
    if (a.ef_list.empty()) {
        a.ef_list.push_back(std::max(1, a.ef));
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    SetupTiming setup_timing;

    MetricKind metric_kind = MetricKind::IP;
    ngfixlib::Metric ngfix_metric = ngfixlib::IP_float;
    if (args.dist == "l2") {
        metric_kind = MetricKind::L2;
        ngfix_metric = ngfixlib::L2_float;
    } else if (args.dist == "ip") {
        metric_kind = MetricKind::IP;
        ngfix_metric = ngfixlib::IP_float;
    } else if (args.dist == "cosine") {
        metric_kind = MetricKind::COSINE;
        ngfix_metric = ngfixlib::IP_float;
    } else {
        std::cerr << "Unknown --dist: " << args.dist << "\n";
        return 1;
    }

    std::string base_path = first_existing(args.dir, {"base.fbin", "base.2.5M.fbin", "base.10M.fbin"});
    std::vector<float> xb;
    int dim = 0;
    idx_t nb = 0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        nb = read_fbin(base_path.c_str(), xb, dim);
        auto t1 = std::chrono::high_resolution_clock::now();
        setup_timing.base_load_s = std::chrono::duration<double>(t1 - t0).count();
    }

    auto t_index0 = std::chrono::high_resolution_clock::now();
    ngfixlib::HNSW_NGFix<float> index(ngfix_metric, args.index_path);
    auto t_index1 = std::chrono::high_resolution_clock::now();
    setup_timing.index_load_s = std::chrono::duration<double>(t_index1 - t_index0).count();
    if ((size_t)index.n != nb) {
        std::cerr << "Warning: base rows " << nb << " vs index n " << (size_t)index.n << "\n";
    }

    std::string xqf = args.load_synth_prefix + ".xq.fbin";
    std::string gtf = args.load_synth_prefix + ".gt.ibin";
    std::string labf = args.load_synth_prefix + ".labels.ibin";

    std::vector<float> xq;
    int dq = 0;
    idx_t nq = 0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        nq = read_fbin(xqf.c_str(), xq, dq);
        auto t1 = std::chrono::high_resolution_clock::now();
        setup_timing.queries_load_s = std::chrono::duration<double>(t1 - t0).count();
    }
    if (dq != dim) {
        std::cerr << "Dim mismatch in xq: " << dq << " vs base " << dim << "\n";
        return 1;
    }

    std::vector<int> gt_full;
    int d_gtk = 0;
    idx_t ngt_rows = 0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        ngt_rows = read_ibin(gtf.c_str(), gt_full, d_gtk);
        auto t1 = std::chrono::high_resolution_clock::now();
        setup_timing.gt_load_s = std::chrono::duration<double>(t1 - t0).count();
    }
    if (ngt_rows < nq) {
        std::cerr << "Loaded gt rows < nq\n";
        return 1;
    }

    std::vector<int> gt((size_t)nq * args.k, -1);
    for (idx_t i = 0; i < nq; ++i) {
        std::copy(gt_full.data() + (size_t)i * d_gtk,
                  gt_full.data() + (size_t)i * d_gtk + std::min(args.k, d_gtk),
                  gt.begin() + (size_t)i * args.k);
    }

    std::vector<int> labels;
    int d_lab = 0;
    idx_t n_lab = 0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        n_lab = read_ibin(labf.c_str(), labels, d_lab);
        auto t1 = std::chrono::high_resolution_clock::now();
        setup_timing.labels_load_s = std::chrono::duration<double>(t1 - t0).count();
    }
    if (n_lab != nq || d_lab != 1) {
        std::cerr << "Loaded labels size mismatch\n";
        return 1;
    }

    std::vector<std::vector<idx_t>> clusters;
    std::vector<idx_t> active_qids;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        int maxlab = -1;
        for (int v : labels) {
            if (v > maxlab) {
                maxlab = v;
            }
        }
        clusters.assign((size_t)std::max(1, maxlab + 1), {});
        for (idx_t i = 0; i < nq; ++i) {
            int c = labels[(size_t)i];
            if (c >= 0) {
                clusters[(size_t)c].push_back(i);
            }
        }
        for (const auto& C : clusters) {
            active_qids.insert(active_qids.end(), C.begin(), C.end());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        setup_timing.cluster_build_s = std::chrono::duration<double>(t1 - t0).count();
    }

    std::cout << "[Baseline] clusters=" << clusters.size() << ", active_queries=" << active_qids.size() << "\n";

    struct Row {
        int ef;
        double qps;
        float recall;
        double avg_visited;
    };
    std::vector<Row> rows;
    std::vector<RunTiming> run_timings;

    for (size_t eidx = 0; eidx < args.ef_list.size(); ++eidx) {
        int ef_cur = std::max(1, args.ef_list[eidx]);
        std::cout << "\n=== RUN baseline ef=" << ef_cur << " ===\n";

        std::vector<std::vector<idx_t>> out_topk(nq);
        std::vector<double> per_query_time(nq, 0.0);
        std::vector<float> per_query_recall(nq, 0.0f);
        std::vector<int> per_query_visited(nq, 0);

        auto t0 = std::chrono::high_resolution_clock::now();

        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            const auto& C = clusters[ci];
            if (C.empty()) {
                continue;
            }

#pragma omp parallel for schedule(dynamic, 8) if(C.size() > 1)
            for (long long i = 0; i < (long long)C.size(); ++i) {
                idx_t qi = C[(size_t)i];
                auto qs = std::chrono::high_resolution_clock::now();

                thread_local std::vector<ngfixlib::vl_type> visited;
                thread_local ngfixlib::vl_type visited_tag = 1;
                std::vector<idx_t> res;
                int steps = 0;
                ngfix_search_l0_heaps(index,
                                      xq.data() + (size_t)qi * dim,
                                      (size_t)ef_cur,
                                      res,
                                      steps,
                                      visited,
                                      visited_tag,
                                      metric_kind);
                per_query_visited[qi] = steps;
                if ((int)res.size() > args.k_collect) {
                    res.resize((size_t)args.k_collect);
                }

                auto qe = std::chrono::high_resolution_clock::now();
                per_query_time[qi] = std::chrono::duration<double>(qe - qs).count();
                out_topk[qi] = std::move(res);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double total_time_seconds = std::chrono::duration<double>(t1 - t0).count();
        double qps = (double)active_qids.size() / std::max(1e-12, total_time_seconds);

        auto te0 = std::chrono::high_resolution_clock::now();
        float recall = compute_recall_subset(out_topk, gt, args.k, active_qids);

        int per_row = (int)gt.size() / (int)nq;
        int expect = std::min(args.k, per_row);
        for (idx_t qi : active_qids) {
            int hit = 0;
            const auto& R = out_topk[qi];
            for (int j = 0; j < expect; ++j) {
                int g = gt[(size_t)qi * per_row + j];
                if (g < 0) {
                    break;
                }
                for (int t = 0; t < std::min((int)R.size(), args.k); ++t) {
                    if ((int)R[t] == g) {
                        ++hit;
                        break;
                    }
                }
            }
            per_query_recall[qi] = (expect > 0) ? ((float)hit / (float)expect) : 0.0f;
        }
        auto te1 = std::chrono::high_resolution_clock::now();
        double eval_outside_timer_s = std::chrono::duration<double>(te1 - te0).count();

        double avg_visited = 0.0;
        for (idx_t qi : active_qids) {
            avg_visited += (double)per_query_visited[qi];
        }
        avg_visited /= std::max<size_t>(1, active_qids.size());

        std::cout << "Total time: " << total_time_seconds << " s  |  QPS: " << qps << "\n";
        std::cout << "Recall@" << args.k << ": " << std::fixed << std::setprecision(4) << recall << "\n";

        rows.push_back(Row{ef_cur, qps, recall, avg_visited});
        run_timings.push_back(RunTiming{ef_cur, total_time_seconds, eval_outside_timer_s, qps, recall});

        if (!args.per_query_csv_out.empty()) {
            std::ios_base::openmode mode = std::ios::out;
            if (eidx > 0) {
                mode |= std::ios::app;
            }
            std::ofstream qout(args.per_query_csv_out, mode);
            if (qout) {
                if (eidx == 0) {
                    qout << "ef,qid,search_time_s,visited,recall\n";
                }
                qout << std::fixed << std::setprecision(6);
                for (idx_t qi : active_qids) {
                    qout << ef_cur << "," << qi << "," << per_query_time[qi] << "," << per_query_visited[qi] << ","
                         << per_query_recall[qi] << "\n";
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
            fout << "ef,qps,recall,avg_visited\n";
            fout << std::fixed << std::setprecision(6);
            for (const auto& r : rows) {
                fout << r.ef << "," << r.qps << "," << r.recall << "," << r.avg_visited << "\n";
            }
            std::cout << "\n[CSV] wrote " << rows.size() << " rows to " << args.csv_out << "\n";
        }
    }

    std::cout << "\n=== Timing Breakdown (printed after search, not in QPS timer) ===\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[Setup] base_load_s=" << setup_timing.base_load_s << "\n";
    std::cout << "[Setup] index_load_s=" << setup_timing.index_load_s << "\n";
    std::cout << "[Setup] queries_load_s=" << setup_timing.queries_load_s << "\n";
    std::cout << "[Setup] gt_load_s=" << setup_timing.gt_load_s << "\n";
    std::cout << "[Setup] labels_load_s=" << setup_timing.labels_load_s << "\n";
    std::cout << "[Setup] cluster_build_s=" << setup_timing.cluster_build_s << "\n";

    for (const auto& rt : run_timings) {
        std::cout << "[Run ef=" << rt.ef << "] search_wall_s=" << rt.search_wall_s
                  << " qps=" << rt.qps
                  << " recall=" << rt.recall
                  << " eval_outside_timer_s=" << rt.eval_outside_timer_s
                  << "\n";
    }

    return 0;
}
