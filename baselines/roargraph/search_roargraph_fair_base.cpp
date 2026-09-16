#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "../ZHENGXIN/roargraph/include/efanna2e/parameters.h"
#include "../ZHENGXIN/roargraph/include/efanna2e/util.h"
#include "../ZHENGXIN/roargraph/include/index_bipartite.h"

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
    return -unified_dot(a, b, dim);
}

static void roargraph_search_l0_heaps(efanna2e::IndexBipartite& index,
                                      const float* q,
                                      size_t ef,
                                      MetricKind metric,
                                      std::vector<idx_t>& out_top_ids,
                                      size_t& out_visited) {
    out_top_ids.clear();
    out_visited = 0;
    if (ef == 0 || index.GetSizeOfDataset() == 0) {
        return;
    }

    using DistId = std::pair<float, idx_t>;
    struct MinCmp {
        bool operator()(const DistId& a, const DistId& b) const { return a.first > b.first; }
    };

    const size_t n = index.GetSizeOfDataset();
    const int dim = (int)index.GetDimension();
    const float* xb = index.GetBasePointSet();
    auto& graph = index.GetProjectionGraph();
    const idx_t ep = (idx_t)index.GetProjectionEntryPoint();

    std::vector<uint8_t> visited(n, 0);
    std::priority_queue<DistId, std::vector<DistId>, MinCmp> cand;
    std::priority_queue<DistId> top;

    if (ep < n) {
        const float d0 = unified_raw_score(xb + ep * (size_t)dim, q, dim, metric);
        visited[ep] = 1;
        ++out_visited;
        cand.emplace(d0, ep);
        top.emplace(d0, ep);
    }

    float lb = top.empty() ? std::numeric_limits<float>::infinity() : top.top().first;
    while (!cand.empty()) {
        const auto cu = cand.top();
        if (cu.first > lb) {
            break;
        }
        cand.pop();

        const auto& nbrs = graph[cu.second];
        for (size_t i = 0; i < nbrs.size(); ++i) {
            const idx_t v = (idx_t)nbrs[i];
            if (v >= n || visited[v]) {
                continue;
            }
            visited[v] = 1;
            ++out_visited;
            const float d = unified_raw_score(xb + v * (size_t)dim, q, dim, metric);
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
    buf.reserve(top.size());
    while (!top.empty()) {
        buf.push_back(top.top());
        top.pop();
    }
    std::sort(buf.begin(), buf.end(), [](const DistId& a, const DistId& b) { return a.first < b.first; });
    out_top_ids.reserve(buf.size());
    for (const auto& p : buf) {
        out_top_ids.push_back(p.second);
    }
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

    int k = 10;
    int k_collect = 100;
    int ef = 200;
    std::vector<int> ef_list;

    std::string dist = "l2";
    int num_threads = 1;

    std::string csv_out = "baseline_roargraph_fair.csv";
    std::string per_query_csv_out;
};

static void usage() {
    std::cout
        << "Usage:\n  search_roargraph_fair_base <data_dir> [projection_index_path]\n"
        << "    --k <int> --k_collect <int> --ef <int> --ef_list <int|a:s:b|v1,v2,...>\n"
        << "    --load_synth_prefix <path>\n"
        << "    --dist <ip|l2|cosine> --num_threads <int>\n"
        << "    --csv_out <path> --per_query_csv <path>\n";
}

static Args parse_args(int argc, char** argv) {
    if (argc < 2) {
        usage();
        std::exit(1);
    }

    Args a;
    a.dir = argv[1];
    a.index_path = (argc > 2 && argv[2][0] != '-') ? argv[2] : a.dir + "/projection_graph.index";

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
        } else if (s == "--num_threads") {
            a.num_threads = std::max(1, std::stoi(next()));
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
    if (a.k_collect < a.k) {
        a.k_collect = a.k;
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    SetupTiming setup_timing;

    std::string base_path = first_existing(args.dir, {"base.2.5M.fbin", "base.10M.fbin", "base.fbin"});

    std::vector<float> xb;
    int dim = 0;
    idx_t nb = 0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        nb = read_fbin(base_path.c_str(), xb, dim);
        auto t1 = std::chrono::high_resolution_clock::now();
        setup_timing.base_load_s = std::chrono::duration<double>(t1 - t0).count();
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

    MetricKind metric_kind = MetricKind::IP;
    efanna2e::Metric dist_metric = efanna2e::INNER_PRODUCT;
    if (args.dist == "l2") {
        metric_kind = MetricKind::L2;
        dist_metric = efanna2e::L2;
    } else if (args.dist == "ip") {
        metric_kind = MetricKind::IP;
        dist_metric = efanna2e::INNER_PRODUCT;
    } else if (args.dist == "cosine") {
        metric_kind = MetricKind::COSINE;
        dist_metric = efanna2e::COSINE;
    } else {
        std::cerr << "Unsupported --dist: " << args.dist << "\n";
        return 1;
    }

    if (!std::filesystem::exists(args.index_path)) {
        std::cerr << "projection index does not exist: " << args.index_path << "\n";
        return 1;
    }

    efanna2e::IndexBipartite index((unsigned)dq, nb, dist_metric, nullptr);

    auto t_index0 = std::chrono::high_resolution_clock::now();
    index.LoadSearchNeededData(base_path.c_str(), "");
    index.LoadProjectionGraph(args.index_path.c_str());
    auto t_index1 = std::chrono::high_resolution_clock::now();
    setup_timing.index_load_s = std::chrono::duration<double>(t_index1 - t_index0).count();

    omp_set_num_threads(args.num_threads);
    index.InitVisitedListPool(args.num_threads);

    float* query_ptr = xq.data();

    if (index.need_normalize) {
        for (idx_t i = 0; i < nq; ++i) {
            efanna2e::normalize<float>(query_ptr + i * (idx_t)dq, (uint32_t)dq);
        }
    }

    std::cout << "[RoarGraph Fair] clusters=" << clusters.size() << ", active_queries=" << active_qids.size() << "\n";

    struct Row {
        int ef;
        double qps;
        float recall;
        double avg_visited;
    };
    std::vector<Row> rows;
    std::vector<RunTiming> run_timings;

    efanna2e::Parameters parameters;
    parameters.Set<uint32_t>("num_threads", (uint32_t)args.num_threads);

    for (size_t eidx = 0; eidx < args.ef_list.size(); ++eidx) {
        int ef_cur = std::max(1, args.ef_list[eidx]);
        uint32_t L_pq = (uint32_t)ef_cur;
        if ((int)L_pq < args.k) {
            std::cerr << "Skip ef=" << ef_cur << " because L_pq(ef) < k" << "\n";
            continue;
        }
        parameters.Set<uint32_t>("L_pq", L_pq);
        const size_t run_collect = std::min<size_t>((size_t)args.k_collect, (size_t)L_pq);

        std::cout << "\n=== RUN roargraph L_pq=" << L_pq << " (ef=" << ef_cur
                  << ", collect=" << run_collect << ") ===\n";

        std::vector<std::vector<idx_t>> out_topk(nq);
        std::vector<double> per_query_time(nq, 0.0);
        std::vector<float> per_query_recall(nq, 0.0f);
        std::vector<size_t> per_query_visited(nq, 0);

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

                std::vector<idx_t> res;
                size_t visited = 0;
                roargraph_search_l0_heaps(index,
                                          query_ptr + (size_t)qi * (size_t)dq,
                                          (size_t)L_pq,
                                          metric_kind,
                                          res,
                                          visited);
                per_query_visited[qi] = visited;
                if (res.size() > run_collect) {
                    res.resize(run_collect);
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
                std::cout << "[Per-Query CSV] wrote nq=" << active_qids.size() << " rows to " << args.per_query_csv_out
                          << "\n";
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
