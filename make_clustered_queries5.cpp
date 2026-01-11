// A-CORE synthetic clustered query generator
//
// This tool samples clustered queries on the unit sphere with
// configurable cluster count, within-cluster angular spread,
// separation between cluster centers, and several optional
// noise models. It can also recompute ground-truth nearest
// neighbours (GT) against a provided base dataset.
//
// For typical command-line usage and dataset paths, please
// refer to the README of this code package and the paper's
// experimental section.

#include <bits/stdc++.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std; using idx_t = size_t;

// simple progress printer (not thread-safe; call from single thread)
static void print_progress(const string &tag, size_t cur, size_t total, chrono::high_resolution_clock::time_point t0){
    double progress = total==0 ? 1.0 : (double)cur / (double)total;
    auto now = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(now - t0).count();
    double eta = progress <= 0.0 ? 0.0 : elapsed * (1.0 - progress) / max(progress, 1e-12);
    cerr.setf(std::ios::fixed); cerr<<setprecision(1);
    cerr << "["<<tag<<"] " << (100.0*progress) << "% ("<<cur<<"/"<<total<<")  elapsed="<<elapsed<<"s  ETA~"<<eta<<"s\r";
    cerr.flush();
    if (cur==total) cerr << "\n";
}

/************ fbin/ibin I/O ************/
static idx_t read_fbin_header(const char* f, int& n, int& d) {
    FILE* fp = fopen(f, "rb"); if (!fp) { perror(f); exit(1); }
    if (fread(&n,4,1,fp)!=1 || fread(&d,4,1,fp)!=1){ cerr<<"Bad fbin header: "<<f<<"\n"; fclose(fp); exit(1); }
    fclose(fp); return (idx_t)n;
}
static idx_t read_fbin_all(const char* f, vector<float>& out, int& dim) {
    FILE* fp = fopen(f, "rb"); if (!fp) { perror(f); exit(1); }
    int n=0,d=0; if (fread(&n,4,1,fp)!=1 || fread(&d,4,1,fp)!=1){ cerr<<"Bad fbin header: "<<f<<"\n"; fclose(fp); exit(1); }
    out.resize((idx_t)n*(idx_t)d);
    if (fread(out.data(), sizeof(float), (size_t)n*(size_t)d, fp) != (size_t)n*(size_t)d){ cerr<<"Bad fbin read: "<<f<<"\n"; fclose(fp); exit(1); }
    fclose(fp); dim=d; return (idx_t)n;
}
static void read_fbin_block(FILE* fp, int d, idx_t start, int count, float* buf) {
    long long header = 8; // int n, int d
    long long offset = header + (long long)start * d * sizeof(float);
    if (fseeko(fp, offset, SEEK_SET) != 0) { perror("fseeko"); exit(1); }
    size_t need = (size_t)count * (size_t)d;
    if (fread(buf, sizeof(float), need, fp) != need) { cerr<<"Short read at block\n"; exit(1); }
}
static void write_fbin(const char* f, const float* data, int n, int d){
    FILE* fp=fopen(f,"wb"); if(!fp){perror(f); exit(1);}
    fwrite(&n,4,1,fp); fwrite(&d,4,1,fp);
    fwrite(data,sizeof(float),(size_t)n*(size_t)d,fp); fclose(fp);
}
static void write_ibin(const char* f, const int* data, int n, int d){
    FILE* fp=fopen(f,"wb"); if(!fp){perror(f); exit(1);}
    fwrite(&n,4,1,fp); fwrite(&d,4,1,fp);
    fwrite(data,sizeof(int),(size_t)n*(size_t)d,fp); fclose(fp);
}

/************ math utils ************/
static inline double dotv(const float* a, const float* b, int d){
    double s=0; for(int i=0;i<d;++i) s += (double)a[i]*(double)b[i]; return s;
}
static inline double nrm2(const float* a, int d){
    double s=0; for(int i=0;i<d;++i) s += (double)a[i]*(double)a[i]; return sqrt(max(1e-30, s));
}
static inline void normalize_vec(float* a, int d){
    double n = nrm2(a,d); if (n>0) for(int i=0;i<d;++i) a[i] = (float)(a[i]/n);
}
static inline void normalize_block(float* a, int rows, int d){
#pragma omp parallel for schedule(static) if(rows*d>100000)
    for (int i=0;i<rows;++i) normalize_vec(a + (size_t)i*d, d);
}

/************ 参数 ************/
struct Args {
    // 生成自制查询
    string real_q;
    string out_prefix = "synth";
    int clusters = 10;
    int gt_only = 0;  // 只生成GT（在已存在的 out_prefix.xq.fbin 上）
    int qpc = 100;
    double sigma_deg = 2.0;      // 簇内角度标准差（度），|N(0,σ)| 截断
    double theta_min_deg = 0.0;  // 簇内角度下限（度），0 表示不启用
    double min_sep_deg = 15.0;   // 中心间最小夹角（度）
    double outlier_frac = 0.0;
    double outlier_deg_min = 25.0, outlier_deg_max = 60.0;
    int shuffle = 0;             // 0=按簇输出，1=打乱
    unsigned seed = 42;
    int print_n = 5;

    // 生成 GT
    string base_path;            // 提供则生成 GT
    int k_gt = 10;
    idx_t nb_use = 0;            // 0=用全部 base
    int block_size = 100000;     // base 分块大小
    int normalize_base = 1;      // 对 base 归一化
    int nthreads = 0;            // OpenMP 线程数（0=不设置）

    // A) 每簇配置文件
    string clusters_cfg;         // 每行: qpc,sigma_deg,theta_min_deg

    // B) 纯命令行分布采样
    int qpc_min = 0, qpc_max = 0;      // 0 表示不用范围，仍用 --qpc
    string qpc_dist = "uniform";       // "uniform" 或 "lognormal"
    double qpc_logmean = 5.3;          // lognormal 参数
    double qpc_logstd  = 0.6;
    double sigma_deg_min = 0.0, sigma_deg_max = 0.0;         // 0 表示用 --sigma_deg
    double theta_min_deg_min = 0.0, theta_min_deg_max = 0.0; // 0 表示用 --theta_min_deg
    // 在 struct Args 末尾追加
    string noise_model = "halfnormal"; // "halfnormal"(现状) 或 "tangent_gauss"
    double per_query_sigma_logstd = 0.0; // 每个 query 的 σ 按 lognormal 抖动，0 表示不用
    double heavy_tail_p = 0.0;          // 重尾概率（例如 0.05）
    double heavy_tail_mult = 3.0;       // 命中重尾时的尺度放大倍数
    double anisotropy_lambda = 1.0;     // 切平面 rank-1 各向异性系数，1=各向同性
    double center_jitter_deg = 0.0;     // 簇中心抖动角度（度），0 表示不用
    // ===== [ADD in struct Args] 新噪声模型参数 =====
    double t_df = 5.0;                 // tangent_t: Student-t 自由度 (>2)
    int    mix_m = 0;                  // tangent_mix: 分量数 (0/1 => 退化为单高斯)
    string mix_weights = "";           // 逗号分隔的权重，如 "0.7,0.3"
    string mix_sigma_mult = "";        // 各分量尺度乘子，如 "1.0,2.0"
    string mix_lambda = "";            // 各分量 rank-1 各向异性，如 "1.0,4.0"

    int    subspace_rank = 0;          // tangent_subspace: 子空间维度 k (0=>不启用)
    int    reuse_subspace = 1;         // 每簇固定子空间(1)或每样本重采(0)

    double ring_theta_deg = 0.0;       // ring: 环形中心角度 θ0 (度)
    double ring_width_deg = 0.0;       // ring: 环形角度标准差 (度)

    double corr_frac = 0.0;            // tangent_corr: 簇内共享漂移方差比例 ∈[0,1]

};
static void usage(){
    cerr <<
      "make_clustered_queries2 --real_q <query.fbin> [--out_prefix s1]\n"
      "  --clusters 10 --qpc 100 --sigma_deg 2 --theta_min_deg 0 --min_sep_deg 30\n"
      "  --outlier_frac 0.1 --outlier_deg_min 25 --outlier_deg_max 60\n"
      "  [--shuffle 0|1] [--seed 42] [--print_n 5]\n"
      "  [--base base.fbin] [--k_gt 10] [--nb_use 0] [--block_size 100000]\n"
      "  [--normalize_base 1] [--nthreads 16]\n"
      "  # A) 每簇配置文件:\n"
      "  [--clusters_cfg clusters_cfg.txt]\n"
      "  # B) 分布采样:\n"
      "  [--qpc_min 60 --qpc_max 300 --qpc_dist uniform|lognormal --qpc_logmean 5.3 --qpc_logstd 0.6]\n"
      "  [--sigma_deg_min 5 --sigma_deg_max 35] [--theta_min_deg_min 2 --theta_min_deg_max 12]\n"
      "  # 仅生成 GT:\n"
      "  [--gt_only]\n";
}
static Args parse_args(int argc, char** argv){
    Args a;
    for(int i=1;i<argc;++i){
        string s = argv[i];
        auto next = [&](){ if(i+1>=argc){ cerr<<"Missing value after "<<s<<"\n"; exit(1);} return string(argv[++i]); };
        if (s=="--real_q") a.real_q = next();
        else if (s=="--out_prefix") a.out_prefix = next();
        else if (s=="--gt_only") a.gt_only = 1;
        else if (s=="--clusters") a.clusters = stoi(next());
        else if (s=="--qpc") a.qpc = stoi(next());
        else if (s=="--sigma_deg") a.sigma_deg = stod(next());
        else if (s=="--theta_min_deg") a.theta_min_deg = stod(next());
        else if (s=="--min_sep_deg") a.min_sep_deg = stod(next());
        else if (s=="--outlier_frac") a.outlier_frac = stod(next());
        else if (s=="--outlier_deg_min") a.outlier_deg_min = stod(next());
        else if (s=="--outlier_deg_max") a.outlier_deg_max = stod(next());
        else if (s=="--shuffle") a.shuffle = stoi(next());
        else if (s=="--seed") a.seed = (unsigned)stoul(next());
        else if (s=="--print_n") a.print_n = stoi(next());
        else if (s=="--base") a.base_path = next();
        else if (s=="--k_gt") a.k_gt = stoi(next());
        else if (s=="--nb_use") a.nb_use = (idx_t)stoull(next());
        else if (s=="--block_size") a.block_size = stoi(next());
        else if (s=="--normalize_base") a.normalize_base = stoi(next());
        else if (s=="--nthreads") a.nthreads = stoi(next());
        else if (s=="--clusters_cfg") a.clusters_cfg = next();
        else if (s=="--qpc_min") a.qpc_min = stoi(next());
        else if (s=="--qpc_max") a.qpc_max = stoi(next());
        else if (s=="--qpc_dist") a.qpc_dist = next();
        else if (s=="--qpc_logmean") a.qpc_logmean = stod(next());
        else if (s=="--qpc_logstd")  a.qpc_logstd  = stod(next());
        else if (s=="--sigma_deg_min") a.sigma_deg_min = stod(next());
        else if (s=="--sigma_deg_max") a.sigma_deg_max = stod(next());
        else if (s=="--theta_min_deg_min") a.theta_min_deg_min = stod(next());
        else if (s=="--theta_min_deg_max") a.theta_min_deg_max = stod(next());
        else if (s=="--noise_model") a.noise_model = next();
        else if (s=="--per_query_sigma_logstd") a.per_query_sigma_logstd = stod(next());
        else if (s=="--heavy_tail_p") a.heavy_tail_p = stod(next());
        else if (s=="--heavy_tail_mult") a.heavy_tail_mult = stod(next());
        else if (s=="--anisotropy_lambda") a.anisotropy_lambda = stod(next());
        else if (s=="--center_jitter_deg") a.center_jitter_deg = stod(next());
        else if (s=="--t_df") a.t_df = stod(next());
        else if (s=="--mix_m") a.mix_m = stoi(next());
        else if (s=="--mix_weights") a.mix_weights = next();
        else if (s=="--mix_sigma_mult") a.mix_sigma_mult = next();
        else if (s=="--mix_lambda") a.mix_lambda = next();
        else if (s=="--subspace_rank") a.subspace_rank = stoi(next());
        else if (s=="--reuse_subspace") a.reuse_subspace = stoi(next());
        else if (s=="--ring_theta_deg") a.ring_theta_deg = stod(next());
        else if (s=="--ring_width_deg") a.ring_width_deg = stod(next());
        else if (s=="--corr_frac") a.corr_frac = stod(next());
        else { cerr<<"Unknown arg: "<<s<<"\n"; usage(); exit(1); }
    }
    if (a.clusters <= 0 || a.qpc <= 0) { cerr<<"clusters/qpc must be >0\n"; exit(1); }
    if (a.k_gt <= 0) { cerr<<"k_gt must be >0\n"; exit(1); }
    if (a.block_size <= 0) a.block_size = 100000;
    if (a.theta_min_deg < 0.0) a.theta_min_deg = 0.0;
    return a;
}

/************ 生成球面样本相关 ************/
static inline double ang_between(const float* a, const float* b, int d){
    double c = max(-1.0, min(1.0, dotv(a,b,d)));
    return acos(c);
}
/* 生成与 c 正交的随机单位向量 u（切向方向） */
static void random_unit_tangent(const float* c, int d, std::mt19937& rng, vector<float>& u){
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    u.resize(d);
    for(int i=0;i<d;++i) u[i] = gauss(rng);
    // 去掉径向分量
    double proj = dotv(u.data(), c, d);
    for(int i=0;i<d;++i) u[i] -= (float)(proj * c[i]);
    double nu = nrm2(u.data(), d);
    if (nu < 1e-12) {
        int k = 0; for (int i=1;i<d;++i) if (fabs(c[i]) < fabs(c[k])) k=i;
        fill(u.begin(), u.end(), 0.0f); u[k] = 1.0f;
        proj = dotv(u.data(), c, d);
        for(int i=0;i<d;++i) u[i] -= (float)(proj * c[i]);
        nu = nrm2(u.data(), d);
    }
    for(int i=0;i<d;++i) u[i] = (float)(u[i] / nu);
}

// ===== [ADD] clamp & split =====
template<typename T> static inline T clampv(T x, T lo, T hi){ return x<lo?lo:(x>hi?hi:x); }

static inline vector<double> parse_csv_doubles(const string& s){
    vector<double> v;
    if (s.empty()) return v;
    string t=s; for(char& c:t){ if(c=='|') c=','; } // 容忍 |
    stringstream ss(t); string tok;
    while(getline(ss,tok,',')){ if(!tok.empty()) v.push_back(stod(tok)); }
    return v;
}

// ===== [ADD] 强制最小角度：若 ang(q,c)<theta_min，则沿 q 的切向方向把点推到 theta_min =====
static inline void enforce_theta_min_radians(const float* c, int d, double theta_min, float* q, std::mt19937& rng){
    if (theta_min<=0) return;
    double cosqc = clampv(dotv(q,c,d), -1.0, 1.0);
    double ang = acos(cosqc);
    if (ang >= theta_min) return;

    vector<float> u(d);
    for (int i=0;i<d;++i) u[i] = q[i] - (float)(cosqc) * c[i]; // 切向分量
    double nu = nrm2(u.data(), d);
    if (nu < 1e-20){
        random_unit_tangent(c, d, rng, u);
    } else {
        for (int i=0;i<d;++i) u[i] = (float)(u[i] / nu);
    }
    double ct = cos(theta_min), st = sin(theta_min);
    for (int i=0;i<d;++i) q[i] = (float)(ct * c[i] + st * u[i]);
    normalize_vec(q, d);
}

/* 在中心 c 附近，按给定角度 theta（弧度）生成点：q = cosθ c + sinθ u */
static void sample_on_sphere(const float* c, int d, double theta, std::mt19937& rng, float* out){
    vector<float> u; random_unit_tangent(c,d,rng,u);
    double ct = cos(theta), st = sin(theta);
    for(int i=0;i<d;++i) out[i] = (float)(ct * c[i] + st * u[i]);
    normalize_vec(out,d); // 稳一手
}
/* 选中心（fartherst-first + 最小夹角约束） */
static vector<idx_t> pick_centers(const vector<float>& X, idx_t n, int d,
                                  int k, double min_sep_deg, std::mt19937& rng){
    vector<float> tmp(X);
    for (idx_t i=0;i<n;++i) normalize_vec(tmp.data() + (size_t)i*d, d);

    auto ang = [&](idx_t a, idx_t b){ return ang_between(tmp.data()+a*d, tmp.data()+b*d, d); };
    double min_sep = min_sep_deg * M_PI / 180.0;

    std::uniform_int_distribution<idx_t> uni(0, n-1);
    vector<idx_t> centers; centers.reserve(k);

    centers.push_back(uni(rng));
    vector<double> mindist(n, 1e100);
    for (idx_t i=0;i<n;++i) mindist[i] = ang(i, centers[0]);

    while ((int)centers.size() < k) {
        idx_t best = 0; double bd = -1;
        for (idx_t i=0;i<n;++i){ double v = mindist[i]; if (v > bd) { bd = v; best = i; } }
        bool ok = true;
        for (auto c : centers) { if (ang(best, c) < min_sep) { ok = false; break; } }
        if (ok) {
            centers.push_back(best);
            for (idx_t i=0;i<n;++i){
                double v = ang(i, best);
                if (v < mindist[i]) mindist[i] = v;
            }
        } else {
            mindist[best] = -1e100;
            bool all_neg = true; for (idx_t i=0;i<n;++i) if (mindist[i] > -1e50){ all_neg=false; break; }
            if (all_neg) {
                min_sep *= 0.8;
                for (idx_t i=0;i<n;++i) mindist[i] = ang(i, centers.back());
            }
        }
    }
    return centers;
}
/* 将 z 投影到 c 的切平面 */
static inline void project_to_tangent(const float* c, float* z, int d){
    double proj = dotv(z, c, d);
    for (int i=0;i<d;++i) z[i] = (float)(z[i] - proj * c[i]);
}

/* 切平面各向同性/各向异性高斯噪声：q = normalize(c + g) */
static void sample_tangent_gauss(
    const float* c, int d,
    double sigma_t,                 // 切平面高斯的“每维”标准差
    double anisotropy_lambda,       // >=1，沿随机切向方向的方差放大倍数
    std::mt19937& rng,
    float* out
){
    static thread_local std::normal_distribution<float> gauss01(0.0f, 1.0f);
    // 1) 生成各向同性高斯
    vector<float> z(d);
    for (int i=0;i<d;++i) z[i] = gauss01(rng);
    project_to_tangent(c, z.data(), d);

    // 2) 可选：rank-1 各向异性（把 z 在 w 方向的分量放大）
    if (anisotropy_lambda > 1.0){
        vector<float> w; random_unit_tangent(c, d, rng, w);
        double zw = dotv(z.data(), w.data(), d);
        double alpha = sqrt(max(1.0, anisotropy_lambda)) - 1.0;
        for (int i=0;i<d;++i) z[i] = (float)(z[i] + alpha * zw * w[i]);
    }

    // 3) 叠加到中心并归一化
    for (int i=0;i<d;++i) out[i] = (float)(c[i] + sigma_t * z[i]);
    normalize_vec(out, d);
}
// ===== [ADD] Student-t 切平面噪声
static void sample_tangent_t(
    const float* c, int d, double df, double sigma_scale, // sigma_scale: 见调用处标定
    double anisotropy_lambda, std::mt19937& rng, float* out)
{
    int p = max(1, d-1);
    static thread_local std::normal_distribution<double> g01(0.0,1.0);
    std::gamma_distribution<double> gammad(df/2.0, 2.0); // => chi2(df)

    // 1) z ~ N(0, I_p) in tangent
    vector<double> z(d, 0.0);
    for (int i=0;i<d;++i) z[i] = g01(rng);
    // project
    double proj = dotv((float*)z.data(), c, d);
    for (int i=0;i<d;++i) z[i] -= proj * c[i];

    // 2) t = z / sqrt(u/df)
    double u = max(1e-12, gammad(rng));
    double s = 1.0 / sqrt(u/df);
    for (int i=0;i<d;++i) z[i] *= s;

    // 3) rank-1 anisotropy
    if (anisotropy_lambda > 1.0){
        vector<float> w; random_unit_tangent(c, d, rng, w);
        double zw = 0.0; for(int i=0;i<d;++i) zw += z[i]*w[i];
        double alpha = sqrt(max(1.0, anisotropy_lambda)) - 1.0;
        for (int i=0;i<d;++i) z[i] += alpha * zw * w[i];
    }

    // 4) add & normalize
    for (int i=0;i<d;++i) out[i] = (float)(c[i] + sigma_scale * z[i]);
    normalize_vec(out, d);
}
// ===== [ADD] Laplace 切平面噪声（各维独立 Laplace 再投影，近似标定）
static inline double sample_laplace01(std::mt19937& rng){
    // 标准 Laplace(0,1): 逆变换
    static thread_local std::uniform_real_distribution<double> u(0.0,1.0);
    double v = u(rng); return (v<0.5) ? log(2.0*v) : -log(2.0*(1.0-v));
}
static void sample_tangent_laplace(
    const float* c, int d, double sigma_scale, // 见调用处标定
    double anisotropy_lambda, std::mt19937& rng, float* out)
{
    int p = max(1, d-1);
    vector<double> z(d, 0.0);
    for (int i=0;i<d;++i) z[i] = sample_laplace01(rng); // Var≈2

    // 投影切平面
    double proj = 0.0; for (int i=0;i<d;++i) proj += z[i]*c[i];
    for (int i=0;i<d;++i) z[i] -= proj * c[i];

    // rank-1 各向异性
    if (anisotropy_lambda > 1.0){
        vector<float> w; random_unit_tangent(c, d, rng, w);
        double zw = 0.0; for(int i=0;i<d;++i) zw += z[i]*w[i];
        double alpha = sqrt(max(1.0, anisotropy_lambda)) - 1.0;
        for (int i=0;i<d;++i) z[i] += alpha * zw * w[i];
    }

    for (int i=0;i<d;++i) out[i] = (float)(c[i] + sigma_scale * z[i]);
    normalize_vec(out, d);
}
// ===== [ADD] 解析 & 采样混合分量
static inline int sample_categorical(const vector<double>& w, std::mt19937& rng){
    static thread_local std::uniform_real_distribution<double> u01(0.0,1.0);
    double r = u01(rng), acc=0.0;
    for (int i=0;i<(int)w.size();++i){ acc += w[i]; if (r <= acc) return i; }
    return (int)w.size()-1;
}
// ===== [ADD] 构造簇的 k 维切平面正交基 (Gram-Schmidt)
static void build_tangent_subspace_basis(const float* c, int d, int k, std::mt19937& rng, vector<float>& Uk){
    k = max(1, min(k, d-1));
    Uk.assign((size_t)d*(size_t)k, 0.0f);
    for (int j=0;j<k;++j){
        vector<float> v; random_unit_tangent(c, d, rng, v);
        // 正交到之前的列
        for (int t=0;t<j;++t){
            double proj=0.0; for(int i=0;i<d;++i) proj += v[i]*Uk[(size_t)i*k + t];
            for (int i=0;i<d;++i) v[i] -= (float)(proj * Uk[(size_t)i*k + t]);
        }
        normalize_vec(v.data(), d);
        for (int i=0;i<d;++i) Uk[(size_t)i*k + j] = v[i];
    }
}

// z_k ~ N(0,I_k); z = U_k z_k; q=normalize(c + s*z)
static void sample_in_subspace(
    const float* c, int d, const vector<float>& Uk, int k, double sigma_scale,
    double anisotropy_lambda, std::mt19937& rng, float* out)
{
    static thread_local std::normal_distribution<double> g01(0.0,1.0);
    vector<double> z_k(k); for (int j=0;j<k;++j) z_k[j] = g01(rng);

    // 可选 rank-1: 放大第一个基方向
    if (anisotropy_lambda > 1.0 && k>0){
        double alpha = sqrt(max(1.0, anisotropy_lambda));
        z_k[0] *= alpha;
    }

    vector<double> z(d,0.0);
    for (int i=0;i<d;++i){
        double s = 0.0;
        for (int j=0;j<k;++j) s += (double)Uk[(size_t)i*k + j] * z_k[j];
        z[i] = s;
    }
    for (int i=0;i<d;++i) out[i] = (float)(c[i] + sigma_scale * z[i]);
    normalize_vec(out, d);
}
// ===== [ADD] 环形角度分布（围绕 θ0 的截断高斯）
static void sample_ring_on_sphere(
    const float* c, int d, double theta0, double width, double theta_min,
    std::mt19937& rng, float* out)
{
    static thread_local std::normal_distribution<double> g01(0.0,1.0);
    double t;
    do {
        t = theta0 + width * g01(rng);
        t = clampv(t, 0.0, M_PI);
    } while (t < max(0.0, theta_min) || t > M_PI);
    sample_on_sphere(c, d, t, rng, out);
    normalize_vec(out, d);
}
// ===== [ADD] 生成切平面各向同性高斯向量（封装一下）
static void sample_tangent_gauss_vec(const float* c, int d, double sigma_t, std::mt19937& rng, vector<float>& z){
    static thread_local std::normal_distribution<float> g01(0.0f,1.0f);
    z.resize(d);
    for (int i=0;i<d;++i) z[i] = g01(rng);
    project_to_tangent(c, z.data(), d);
    double nz = nrm2(z.data(), d); if (nz < 1e-12) { random_unit_tangent(c,d,rng,z); }
    for (int i=0;i<d;++i) z[i] = (float)(sigma_t * z[i]);
}

/* 让中心本身也轻微抖动（角度近似 ~ center_jitter_deg） */
static void jitter_center_on_sphere(
    const float* c, int d, double jitter_deg, std::mt19937& rng, float* out)
{
    if (jitter_deg <= 0.0){
        for (int i=0;i<d;++i) out[i] = c[i];
        return;
    }
    double jr = jitter_deg * M_PI / 180.0;
    // 用切平面高斯完成中心抖动，σ_t 选择 jr/sqrt(d-1)
    double sigma_t = jr / sqrt(max(1, d-1));
    sample_tangent_gauss(c, d, sigma_t, 1.0, rng, out); // 不加各向异性
}

/************ GT（块式暴力精确 NN） ************/
struct TopKBuffer {
    int k;
    vector<float> dist;  // size k
    vector<int>   id;    // size k
    float worst;
    int worst_pos;
    TopKBuffer(int kk=1):k(kk),dist(kk, std::numeric_limits<float>::infinity()),id(kk,-1),worst(INFINITY),worst_pos(0){}
    inline void reset(){
        std::fill(dist.begin(), dist.end(), std::numeric_limits<float>::infinity());
        std::fill(id.begin(),   id.end(),   -1);
        worst = std::numeric_limits<float>::infinity(); worst_pos = 0;
    }
    inline void finalize_worst(){
        worst = -1.0f; worst_pos = 0;
        for (int i=0;i<k;++i){ if (dist[i] > worst){ worst = dist[i]; worst_pos = i; } }
    }
    inline void consider(float d, int idx){
        if (d >= worst) return;
        dist[worst_pos] = d; id[worst_pos] = idx;
        worst = dist[0]; worst_pos = 0;
        for (int i=1;i<k;++i) if (dist[i] > worst){ worst = dist[i]; worst_pos = i; }
    }
};
static inline float l2sq(const float* a, const float* b, int d){
    float s=0.f;
    for (int i=0;i<d;++i){ float df = a[i]-b[i]; s += df*df; }
    return s;
}
static void compute_gt_bruteforce_fbin(
    const string& base_path, int d, const float* Q, int nq, int k,
    idx_t nb_use, int block_size, int normalize_base, int nthreads,
    vector<int>& gt_out // nq * k
){
    int nb_hdr=0, d_hdr=0; read_fbin_header(base_path.c_str(), nb_hdr, d_hdr);
    if (d_hdr != d){ cerr<<"[GT] Dim mismatch: base d="<<d_hdr<<" vs query d="<<d<<"\n"; exit(1); }
    idx_t nb_total = (idx_t)nb_hdr;
    if (nb_use==0 || nb_use>nb_total) nb_use = nb_total;
    std::cout<<"[GT] Computing exact NN (brute-force) on "<<nb_use<<" base vectors, nq="<<nq<<", k="<<k<<"\n";

    gt_out.assign((size_t)nq * (size_t)k, -1);

    vector<TopKBuffer> bufs((size_t)nq, TopKBuffer(k));
    for (int i=0; i<nq; ++i) bufs[i].finalize_worst();

    FILE* fp = fopen(base_path.c_str(), "rb"); if (!fp){ perror(base_path.c_str()); exit(1); }
    vector<float> block; block.resize((size_t)block_size * (size_t)d);

    if (nthreads > 0) {
#ifdef _OPENMP
        omp_set_num_threads(nthreads);
#endif
    }

    auto t0 = chrono::high_resolution_clock::now();
    idx_t processed = 0;
    for (idx_t start=0; start<nb_use; start += (idx_t)block_size){
        int cur = (int)min<idx_t>((idx_t)block_size, nb_use - start);
        read_fbin_block(fp, d, start, cur, block.data());
        if (normalize_base) normalize_block(block.data(), cur, d);

#pragma omp parallel for schedule(static)
        for (int qi = 0; qi < nq; ++qi){
            TopKBuffer& B = bufs[(size_t)qi];
            const float* q = Q + (size_t)qi * d;
            for (int bi=0; bi<cur; ++bi){
                const float* xb = block.data() + (size_t)bi * d;
                float dist = l2sq(q, xb, d);
                B.consider(dist, (int)(start + bi));
            }
        }
            processed += cur;
        auto now = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(now - t0).count();
        double progress = (double)processed / (double)nb_use;
        double eta = elapsed * (1.0 - progress) / max(progress, 1e-9);

        cerr.setf(std::ios::fixed); cerr<<setprecision(2);
        cerr << "[GT] progress " << (100.0*progress) << "%  "
            << "blk=" << cur << "  elapsed=" << elapsed << "s  "
            << "ETA~" << eta << "s\n";
    }
    fclose(fp);

#pragma omp parallel for schedule(static)
    for (int qi=0; qi<nq; ++qi){
        vector<int> ord(k); iota(ord.begin(), ord.end(), 0);
        auto &B = bufs[(size_t)qi];
        stable_sort(ord.begin(), ord.end(), [&](int a, int b){ return B.dist[a] < B.dist[b]; });
        for (int j=0;j<k;++j){
            gt_out[(size_t)qi * k + j] = B.id[ord[j]];
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double sec = chrono::duration<double>(t1-t0).count();
    cerr << "[GT] Done. base_used="<< nb_use << "  nq="<<nq<<"  k="<<k
         << "  block_size="<<block_size << "  time="<< sec <<" s\n";
}

/************ 工具：Jaccard 计算（两个长度为 k 的列表） ************/
static double jaccard_topk(const int* a, const int* b, int k){
    // 将 -1 过滤掉（极少见，防御）
    vector<int> va, vb; va.reserve(k); vb.reserve(k);
    for (int i=0;i<k;++i){ if (a[i]>=0) va.push_back(a[i]); }
    for (int i=0;i<k;++i){ if (b[i]>=0) vb.push_back(b[i]); }
    sort(va.begin(), va.end());
    va.erase(unique(va.begin(), va.end()), va.end());
    sort(vb.begin(), vb.end());
    vb.erase(unique(vb.begin(), vb.end()), vb.end());
    int i=0,j=0, inter=0;
    while(i<(int)va.size() && j<(int)vb.size()){
        if (va[i]==vb[j]){ ++inter; ++i; ++j; }
        else if (va[i]<vb[j]) ++i; else ++j;
    }
    int uni = (int)va.size() + (int)vb.size() - inter;
    if (uni==0) return 1.0; // 两空集
    return (double)inter / (double)uni;
}

/************ 主流程 ************/
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    Args args = parse_args(argc, argv);

    if (args.nthreads > 0) {
    #ifdef _OPENMP
        omp_set_num_threads(args.nthreads);
    #endif
    }

    // ===== 仅生成（或重生成）GT =====
    if (args.gt_only) {
        if (args.base_path.empty() || args.out_prefix.empty()) {
            cerr << "ERROR: --gt_only 需要 --out_prefix <prefix> 和 --base <base.fbin>\n";
            return 1;
        }
        // 读取 out_prefix.xq.fbin 作为查询
        vector<float> Q; int d=0;
        string f_q = args.out_prefix + ".xq.fbin";
        idx_t nq = read_fbin_all(f_q.c_str(), Q, d);
        for (idx_t i=0; i<nq; ++i) normalize_vec(Q.data() + (size_t)i*d, d);
        cerr << "[GT] Loaded queries: n=" << nq << " d=" << d << "\n";

        // 计算 GT
        vector<int> gt;
        compute_gt_bruteforce_fbin(
            args.base_path, d, Q.data(), (int)nq, args.k_gt,
            args.nb_use, args.block_size, args.normalize_base, args.nthreads,
            gt
        );

        // 保存
        string f_gt = args.out_prefix + ".gt.ibin";
        write_ibin(f_gt.c_str(), gt.data(), (int)nq, args.k_gt);
        cerr << "[GT] Wrote: " << f_gt << " (" << nq << " x " << args.k_gt << ")\n";
        return 0;
    }

    // ===== 常规：读取真实查询、选中心、按簇采样 =====
    if (args.real_q.empty()) { cerr<<"ERROR: --real_q is required\n"; usage(); return 1; }

    // 读取真实查询，并归一化一次（稳妥）
    vector<float> Xq; int d=0; idx_t nq = read_fbin_all(args.real_q.c_str(), Xq, d);
    for (idx_t i=0;i<nq;++i) normalize_vec(Xq.data() + (size_t)i*d, d);
    cerr << "Loaded real queries: n="<<nq<<", d="<<d<<"\n";

    mt19937 rng(args.seed);

    // 选中心
    auto centers = pick_centers(Xq, nq, d, args.clusters, args.min_sep_deg, rng);
    cerr << "Picked centers: " << centers.size() << " (min_sep_deg="<<args.min_sep_deg<<")\n";

    // === 组装每簇参数：优先 clusters_cfg，其次命令行分布，最后用全局默认 ===
    vector<int>    qpc_v(args.clusters, args.qpc);
    vector<double> sigma_v(args.clusters, args.sigma_deg);
    vector<double> thetamin_v(args.clusters, args.theta_min_deg);

    // A) 如提供配置文件，用其覆盖
    if (!args.clusters_cfg.empty()){
        ifstream fin(args.clusters_cfg);
        if (!fin) { cerr<<"Cannot open --clusters_cfg "<<args.clusters_cfg<<"\n"; return 1; }
        string line; int c=0;
        while (c<args.clusters && std::getline(fin, line)){
            if (line.empty()) continue;
            // 允许逗号/空格分隔: qpc,sigma_deg,theta_min_deg
            std::replace(line.begin(), line.end(), ',', ' ');
            stringstream ss(line);
            int qpc; double sdeg, tmin;
            if (!(ss >> qpc >> sdeg >> tmin)) { cerr<<"Bad line in clusters_cfg: "<<line<<"\n"; return 1; }
            qpc_v[c]     = max(1, qpc);
            sigma_v[c]   = max(1e-6, sdeg);
            thetamin_v[c]= max(0.0, tmin);
            ++c;
        }
        if (c < args.clusters){
            cerr<<"[WARN] clusters_cfg provides "<<c<<" rows < --clusters "<<args.clusters<<"; "
                   "the remaining clusters will use defaults and/or sampling.\n";
        }
    }

    // B) 纯命令行：对仍是默认值的簇进行分布采样（或对全部簇采样——按需可改）
    bool use_qpc_range = (args.qpc_min>0 && args.qpc_max>0 && args.qpc_max>=args.qpc_min);
    bool use_sigma_rng = (args.sigma_deg_min>0 && args.sigma_deg_max>0 && args.sigma_deg_max>=args.sigma_deg_min);
    bool use_tmin_rng  = (args.theta_min_deg_min>0 && args.theta_min_deg_max>0 &&
                          args.theta_min_deg_max>=args.theta_min_deg_min);

    std::uniform_int_distribution<int> qpc_uni(max(1, args.qpc_min), max(1, args.qpc_max));
    std::lognormal_distribution<double> qpc_logn(args.qpc_logmean, args.qpc_logstd);

    std::uniform_real_distribution<double> sigma_uni(
        (use_sigma_rng?args.sigma_deg_min:args.sigma_deg),
        (use_sigma_rng?args.sigma_deg_max:args.sigma_deg)
    );
    std::uniform_real_distribution<double> tmin_uni(
        (use_tmin_rng?args.theta_min_deg_min:args.theta_min_deg),
        (use_tmin_rng?args.theta_min_deg_max:args.theta_min_deg)
    );

    for (int c=0;c<args.clusters;++c){
        if (use_qpc_range){
            if (args.qpc_dist=="lognormal"){
                qpc_v[c] = max(1, (int)llround(qpc_logn(rng)));
            }else{
                qpc_v[c] = qpc_uni(rng);
            }
        }
        if (use_sigma_rng){
            sigma_v[c] = max(1e-6, sigma_uni(rng));
        }
        if (use_tmin_rng){
            thetamin_v[c] = max(0.0, tmin_uni(rng));
        }
    }

    // 打印简要参数（前若干簇）
    cerr << "Per-cluster params (first 8):\n";
    for (int c=0; c<min(8, args.clusters); ++c){
        cerr << "  c="<<c<<" qpc="<<qpc_v[c]
             <<" sigma_deg="<<sigma_v[c]
             <<" theta_min_deg="<<thetamin_v[c]<<"\n";
    }

    // 生成簇内样本
    vector<float> Y; Y.reserve((size_t)args.clusters * (size_t)args.qpc * d);
    vector<int> labels; labels.reserve((size_t)args.clusters * (size_t)args.qpc);

    auto t0_clusters = chrono::high_resolution_clock::now();
    for (int c = 0; c < args.clusters; ++c){
        int per_c = qpc_v[c];
        const float* center = Xq.data() + (size_t)centers[(size_t)c] * d;

        double sig_rad_c      = sigma_v[c]    * M_PI / 180.0;
        double theta_min_rad_c= thetamin_v[c] * M_PI / 180.0;
        if (sig_rad_c <= 0) sig_rad_c = 1e-6;
        std::normal_distribution<double> n01(0.0, sig_rad_c);

        auto sample_theta_c = [&](){
            // 半正态 + 3σ 截断
            double t;
            do { t = fabs(n01(rng)); } while (t > 3.0 * sig_rad_c + 1e-12);
            if (t < theta_min_rad_c) t = theta_min_rad_c;
            // 限制到 [0, pi]（极端 σ 时的稳健）
            if (t > M_PI) t = M_PI;
            return t;
        };
        vector<float> center_buf(d);
        const float* center_raw = Xq.data() + (size_t)centers[(size_t)c] * d;
        const float* center_use = center_raw;
        if (args.center_jitter_deg > 0.0){
        jitter_center_on_sphere(center_raw, d, args.center_jitter_deg, rng, center_buf.data());
        center_use = center_buf.data();
        }
        for (int i=0;i<per_c;++i){
        vector<float> q(d);

        // 统一准备：常用量
    const int    p = max(1, d-1);
    const double sigma_rad     = sig_rad_c;        // 本簇角度 std（弧度）
    const double theta_min_rad = theta_min_rad_c;  // 本簇角度下限（弧度）

    if (args.noise_model == "tangent_gauss"){
        double sigma_t = sigma_rad / sqrt((double)p);

        // 每 query 异质性
        if (args.per_query_sigma_logstd > 0.0){
            std::lognormal_distribution<double> lnd(0.0, args.per_query_sigma_logstd);
            sigma_t *= lnd(rng);
        }
        // 可选重尾
        if (args.heavy_tail_p > 0.0){
            std::uniform_real_distribution<double> u01(0.0,1.0);
            if (u01(rng) < args.heavy_tail_p) sigma_t *= max(1.0, args.heavy_tail_mult);
        }

        sample_tangent_gauss(center_use, d, sigma_t, args.anisotropy_lambda, rng, q.data());
        enforce_theta_min_radians(center_use, d, theta_min_rad, q.data(), rng);
    }
    else if (args.noise_model == "tangent_t"){
        // Student-t 重尾：标定让角度 std ≈ sigma_deg
        double df = max(2.1, args.t_df);
        double sigma_scale = sigma_rad / sqrt( (double)p * df/(df-2.0) );
        sample_tangent_t(center_use, d, df, sigma_scale, args.anisotropy_lambda, rng, q.data());
        enforce_theta_min_radians(center_use, d, theta_min_rad, q.data(), rng);
    }
    else if (args.noise_model == "tangent_laplace"){
        // Laplace（各维独立，Var=2），标定让角度 std ≈ sigma_deg
        double sigma_scale = sigma_rad / sqrt(2.0 * (double)p);
        sample_tangent_laplace(center_use, d, sigma_scale, args.anisotropy_lambda, rng, q.data());
        enforce_theta_min_radians(center_use, d, theta_min_rad, q.data(), rng);
    }
    else if (args.noise_model == "tangent_mix"){
        // 解析混合参数（为简单起见，这里用 thread_local 缓存一次）
        static thread_local bool mix_inited = false;
        static thread_local vector<double> W, S, L; // 权重, 尺度乘子, 各向异性
        if (!mix_inited){
            int M = max(0, args.mix_m);
            W = parse_csv_doubles(args.mix_weights);
            S = parse_csv_doubles(args.mix_sigma_mult);
            L = parse_csv_doubles(args.mix_lambda);
            if (M<=1 || (int)W.size()!=M || (int)S.size()!=M || (int)L.size()!=M){
                M = 1; W = {1.0}; S = {1.0}; L = {args.anisotropy_lambda};
            }
            double sw=0; for(double x:W) sw+=x; if (sw<=0) sw=1.0;
            for (double& x:W) x = x/sw;
            mix_inited = true;
        }
        int m = sample_categorical(W, rng);
        double sigma_t = (sigma_rad / sqrt((double)p)) * S[m];
        double lam     = max(1.0, L[m]);
        sample_tangent_gauss(center_use, d, sigma_t, lam, rng, q.data());
        enforce_theta_min_radians(center_use, d, theta_min_rad, q.data(), rng);
    }
    else if (args.noise_model == "tangent_subspace"){
        int k = max(1, min(args.subspace_rank, p));
        // 每簇构造一次子空间（用 thread_local 简易缓存）
        static thread_local vector<float> Uk; // d x k
        static thread_local int Uk_valid_cluster = -1;
        if (!args.reuse_subspace || Uk_valid_cluster != c){
            build_tangent_subspace_basis(center_use, d, k, rng, Uk);
            Uk_valid_cluster = c;
        }
        double sigma_scale = sigma_rad / sqrt((double)k);
        sample_in_subspace(center_use, d, Uk, k, sigma_scale, args.anisotropy_lambda, rng, q.data());
        enforce_theta_min_radians(center_use, d, theta_min_rad, q.data(), rng);
    }
    else if (args.noise_model == "ring"){
        double theta0 = args.ring_theta_deg * M_PI / 180.0;
        double width  = max(1e-9, args.ring_width_deg * M_PI / 180.0);
        sample_ring_on_sphere(center_use, d, theta0, width, theta_min_rad, rng, q.data());
        // ring 已在角度域控制，无需额外 enforce
    }
    else if (args.noise_model == "tangent_corr"){
        // 相关漂移：δ_c + ε_i，方差拆分
        double sigma_total = sigma_rad / sqrt((double)p);
        double frac = clampv(args.corr_frac, 0.0, 1.0);
        double sigma_shared = sqrt(frac) * sigma_total;
        double sigma_ind    = sqrt(1.0 - frac) * sigma_total;

        // 每簇共享漂移 δ_c：构造一次
        static thread_local vector<float> delta_c;
        static thread_local int delta_valid_cluster = -1;
        if (delta_valid_cluster != c){
            sample_tangent_gauss_vec(center_use, d, sigma_shared, rng, delta_c);
            delta_valid_cluster = c;
        }

        // 独立扰动 ε_i
        vector<float> eps;
        sample_tangent_gauss_vec(center_use, d, sigma_ind, rng, eps);

        // 可选各向异性（作用在 ε 上）
        if (args.anisotropy_lambda > 1.0){
            vector<float> w; random_unit_tangent(center_use, d, rng, w);
            double ew = 0.0; for(int t=0;t<d;++t) ew += eps[t]*w[t];
            double alpha = sqrt(max(1.0, args.anisotropy_lambda)) - 1.0;
            for (int t=0;t<d;++t) eps[t] = (float)(eps[t] + alpha * ew * w[t]);
        }

        // 叠加：q = normalize(c + δ_c + ε)
        for (int t=0;t<d;++t) q[t] = (float)(center_use[t] + delta_c[t] + eps[t]);
        normalize_vec(q.data(), d);
        enforce_theta_min_radians(center_use, d, theta_min_rad, q.data(), rng);
    }
    else {
        // 旧逻辑：halfnormal（保留原行文）
        double theta = sample_theta_c();
        sample_on_sphere(center_use, d, theta, rng, q.data());
        normalize_vec(q.data(), d);
    }

    Y.insert(Y.end(), q.begin(), q.end());
    labels.push_back(c);
}

        // print per-cluster progress
        print_progress("CLUSTERS", (size_t)(c+1), (size_t)args.clusters, t0_clusters);
    }
    // 离群点（在 [outlier_deg_min, outlier_deg_max] 均匀采样角度）
    const idx_t n_inlier = (idx_t)Y.size()/d;
    const idx_t n_out = (idx_t)llround(args.outlier_frac * (double)n_inlier);
    if (n_out > 0){
        uniform_int_distribution<idx_t> pick_center(0, (idx_t)centers.size()-1);
        uniform_real_distribution<double> udeg(args.outlier_deg_min, args.outlier_deg_max);
        for (idx_t t=0;t<n_out;++t){
            idx_t ci = pick_center(rng);
            const float* center = Xq.data() + (size_t)centers[ci] * d;
            double theta = udeg(rng) * M_PI / 180.0;
            vector<float> q(d);
            sample_on_sphere(center, d, theta, rng, q.data());
            normalize_vec(q.data(), d);
            Y.insert(Y.end(), q.begin(), q.end());
            labels.push_back(-1);
        }
    }

    // 输出顺序
    vector<idx_t> order(Y.size() / d);
    iota(order.begin(), order.end(), 0);
    if (args.shuffle) shuffle(order.begin(), order.end(), rng);

    vector<float> Y_shuf; Y_shuf.reserve(Y.size());
    vector<int> labels_shuf; labels_shuf.reserve(labels.size());
    for (idx_t i=0;i<order.size();++i){
        idx_t id = order[i];
        Y_shuf.insert(Y_shuf.end(), Y.begin() + (size_t)id*d, Y.begin() + (size_t)(id+1)*d);
        labels_shuf.push_back(labels[(size_t)id]);
    }

    // 保存查询/标签/中心
    string f_xq = args.out_prefix + ".xq.fbin";
    string f_lab= args.out_prefix + ".labels.ibin";
    string f_cen= args.out_prefix + ".centers.ibin";
    write_fbin(f_xq.c_str(), Y_shuf.data(), (int)(Y_shuf.size()/d), d);
    write_ibin(f_lab.c_str(), labels_shuf.data(), (int)labels_shuf.size(), 1);
    // centers is vector<idx_t> (size_t). Convert to int vector for ibin writer.
    vector<int> centers_int(centers.begin(), centers.end());
    write_ibin(f_cen.c_str(), centers_int.data(), (int)centers_int.size(), 1);
    cerr << "Wrote:\n  " << f_xq << "\n  " << f_lab << "\n  " << f_cen << "\n";

    // 写每簇元信息（便于训练 join）
    {
        string f_meta = args.out_prefix + ".clusters.csv";
        FILE* fp = fopen(f_meta.c_str(), "wb");
        if (fp){
            fprintf(fp, "cluster,center_idx,qpc,sigma_deg,theta_min_deg\n");
            for (int c=0;c<args.clusters;++c){
                fprintf(fp, "%d,%d,%d,%.6f,%.6f\n",
                    c, (int)centers[(size_t)c], qpc_v[c], sigma_v[c], thetamin_v[c]);
            }
            fclose(fp);
            cerr << "Wrote cluster meta: " << f_meta << "\n";
        } else {
            perror("fopen clusters.csv");
        }
    }

    // 打印前几条范数
    int print_n = min(args.print_n, (int)(Y_shuf.size()/d));
    cout.setf(std::ios::fixed); cout<<setprecision(6);
    cout << "\nFirst " << print_n << " L2 norms (should ~1.0):\n";
    for (int i=0;i<print_n;++i){
        const float* q = Y_shuf.data() + (size_t)i*d;
        double n = nrm2(q,d);
        cout << "qid="<<i<<"\t||q||="<< n << "\n";
    }
    cout.unsetf(std::ios::floatfield);

    // ===== 生成 GT（可选）=====
    vector<int> gt; // nq_gen * k_gt
    int nq_gen = (int)(Y_shuf.size()/d);
    if (!args.base_path.empty()){
        int base_n=0, base_d=0;
        read_fbin_header(args.base_path.c_str(), base_n, base_d);
        if (base_d != d){ cerr<<"[GT] ERROR: base dim="<<base_d<<" != query dim="<<d<<"\n"; return 1; }

        compute_gt_bruteforce_fbin(
            args.base_path, d, Y_shuf.data(), nq_gen, args.k_gt,
            args.nb_use, args.block_size, args.normalize_base, args.nthreads,
            gt
        );
        string f_gt = args.out_prefix + ".gt.ibin";
        write_ibin(f_gt.c_str(), gt.data(), nq_gen, args.k_gt);
        cerr << "Wrote GT: " << f_gt << " ("<< nq_gen << " x " << args.k_gt << ")\n";
    } else {
        cerr << "[GT] base not provided; skip GT generation.\n";
    }

    // ===== 计算并输出“每个 query 与其最近 query 的 top-k Jaccard” =====
    if (!gt.empty()){
        vector<int> nearest(nq_gen, -1);
        vector<double> best_cos(nq_gen, -2.0);
        // 找到每个 query 在查询集合中的最近邻（角度最小，即余弦最大）
        atomic<size_t> nq_done(0);
        std::mutex print_mtx;
        auto t0_near = chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static) if(nq_gen > 256)
        for (int i=0;i<nq_gen;++i){
            const float* qi = Y_shuf.data() + (size_t)i*d;
            double bcos = -2.0; int bj = -1;
            for (int j=0; j<nq_gen; ++j){
                if (j==i) continue;
                const float* qj = Y_shuf.data() + (size_t)j*d;
                double c = dotv(qi, qj, d);
                if (c > bcos){ bcos = c; bj = j; }
            }
            nearest[i] = bj; best_cos[i] = bcos;

            size_t done = ++nq_done;
            if (done % 100 == 0 || done == (size_t)nq_gen){
                std::lock_guard<std::mutex> lk(print_mtx);
                print_progress("NEAREST", done, (size_t)nq_gen, t0_near);
            }
        }

        vector<double> jvals; jvals.reserve(nq_gen);
        for (int i=0;i<nq_gen;++i){
            int j = nearest[i];
            if (j<0){ jvals.push_back(0.0); continue; }
            const int* gi = gt.data() + (size_t)i * args.k_gt;
            const int* gj = gt.data() + (size_t)j * args.k_gt;
            double jac = jaccard_topk(gi, gj, args.k_gt);
            jvals.push_back(jac);
        }

        // 统计
        vector<double> tmp = jvals;
        sort(tmp.begin(), tmp.end());
        auto pct = [&](double p){
            if (tmp.empty()) return 0.0;
            double idx = p * (tmp.size()-1);
            size_t i = (size_t)floor(idx), j = (size_t)ceil(idx);
            if (i==j) return tmp[i];
            double w = idx - i;
            return (1.0-w)*tmp[i] + w*tmp[j];
        };
        double mean = 0.0; for (double v: jvals) mean += v; mean /= max(1,(int)jvals.size());

        cout.setf(std::ios::fixed); cout<<setprecision(4);
        cout << "\nNearest-query top-" << args.k_gt << " Jaccard summary (vs nearest query by angle):\n";
        cout << "  count="<< jvals.size()
             << "  mean="<< mean
             << "  median="<< pct(0.5)
             << "  p10="<< pct(0.10)
             << "  p90="<< pct(0.90)
             << "  min="<< (tmp.empty()?0.0:tmp.front())
             << "  max="<< (tmp.empty()?0.0:tmp.back()) << "\n";

        // 打印前 print_n 个样例
        int show = min(args.print_n, nq_gen);
        cout << "Examples (first "<< show << "):\n";
        for (int i=0;i<show;++i){
            int j = nearest[i];
            double cosij = best_cos[i];
            double ang_deg = acos(max(-1.0, min(1.0, cosij))) * 180.0 / M_PI;
            double jac = jaccard_topk(gt.data() + (size_t)i*args.k_gt,
                                      gt.data() + (size_t)j*args.k_gt,
                                      args.k_gt);
            cout << "  q="<< i
                 << "  nearest_q="<< j
                 << "  cos="<< setprecision(6) << cosij
                 << "  ang_deg="<< setprecision(2) << ang_deg
                 << "  jaccard="<< setprecision(4) << jac << "\n";
        }
        cout.unsetf(std::ios::floatfield);
    } else {
        cerr << "[INFO] GT not available; skip nearest-query Jaccard computation.\n";
    }

    return 0;
}
