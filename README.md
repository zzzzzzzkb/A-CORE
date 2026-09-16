# A-CORE: Anchor-based Context Reuse for Efficient Batched Vector Search

Official implementation of **A-CORE**, an anchor-based context reuse framework that accelerates
batched approximate nearest neighbor (ANN) search by exploiting in-batch query similarity,
*without* modifying the underlying index structure.

> Kaibo Zhang, Zhengxin Li, Wentong Zhang, Lanting Fang, Kaiyu Feng, Miao Xie.
> *A-CORE: Anchor-based Context Reuse for Efficient Batched Vector Search.* (VLDB)

A-CORE nominates a representative **anchor** query per batch, runs a single standard search to
materialize a reusable **Centroid Context**, then warm-starts the remaining queries via
selective re-ranking (**Selective-L**). A lightweight adaptive-configuration module
(LightGBM predictors) chooses near-optimal search parameters $(ef_c, ef_w, L)$ from inexpensive
batch-level features under a target-recall constraint.

---

## 目录结构

```
A-CORE/
├── CMakeLists.txt                  # 构建系统
├── README.md
├── requirements.txt                # Python 依赖（训练脚本）
├── include/                        # header-only 依赖（已内置）
│   ├── hnswlib/                    # HNSW 库（header-only）
│   └── json.hpp                    # JSON 单头文件
├── src/                            # A-CORE 核心
│   ├── run_acore.cpp               # 主框架：Centroid Context + Selective-L + 自适应配置（标签聚类版）
│   ├── run_acore_online_grouping.cpp  # 最新流程（先聚类再查询）：Projected-Greedy 在线分组 + Context Reuse
│   ├── projected_greedy_cluster.cpp   # 在线分组（Online grouping, Sec 5.3）独立实现
│   ├── run_train_get_all_k_onlytop.cpp  # 训练标签生成（(ef_c,ef_w,L) 网格搜索）
│   ├── export_feature_csvs_perk.py      # 特征导出/过滤（按 k 生成 filtered_AB_rows）
│   ├── augment_features.cpp             # 特征增强（probe search → *.with_new_feats.csv）
│   ├── make_clustered_queries5.cpp      # 合成 OOD batch 生成（farthest-first + 角度扰动）
│   └── train_full_conditional_and_recall_newfeat.py  # LightGBM 三预测器训练
├── dataset_process/                # TREC(SessionTrack)/WikiAnswers 预处理
├── baselines/                      # 基线
│   ├── hnsw/                       # HNSW
│   ├── nsg/                        # NSG（建图 + 搜索）
│   ├── taumng/                     # τ-MNG（建图 + 搜索）
│   ├── roargraph/                  # RoarGraph（外部依赖）
│   ├── ngfix/                      # NGFix（外部依赖）
│   └── faiss/                      # Faiss（外部依赖）
├── third_party/                    # 小型头文件库（efanna2e / taumng / rvamana）
└── scripts/                        # 复现脚本
```

---

## 论文组件 → 代码文件对照

| 论文组件 | 章节 | 对应代码 | 提交 |
|---|---|---|---|
| A-CORE 核心 Context Reuse | 5.1 | `src/run_acore.cpp`、`src/run_acore_online_grouping.cpp` | 必须 |
| Centroid Context / warm-start | 5.1.2 | 同上（`WarmHierarchicalNSW` / `continueFromSnapshotL0`） | 必须 |
| Selective-L | 5.1.2 | 同上（`--L_mode rankm/recall`） | 必须 |
| Adaptive Configuration | 5.2 | 同上（三阶段预测器 `Πc/Πw/ΠL`） | 必须 |
| LightGBM predictor train/inference | 5.2 | `src/train_full_conditional_and_recall_newfeat.py`（训练）；框架内 `load_from_txt`（推理） | 必须 |
| 特征导出/增强（训练前处理） | 5.2.3 | `src/export_feature_csvs_perk.py` + `src/augment_features.cpp` | 必须 |
| Online grouping | 5.3 / 6.3.3 | `src/projected_greedy_cluster.cpp`、`src/run_acore_online_grouping.cpp` | 必须 |
| HNSW+A-CORE | 主实验 | `include/hnswlib/` + 框架（默认后端 HNSW） | 必须 |
| synthetic workload generator | 6.1 | `src/make_clustered_queries5.cpp` | 必须 |
| TREC/WikiAnswers 预处理 | real workload | `dataset_process/*.py` | 必须有脚本/说明 |
| training-label grid search | Adaptive Config | `src/run_train_get_all_k_onlytop.cpp` | 必须 |
| evaluation scripts | Recall@10 / QPS | `scripts/` + 各基线 wrapper | 必须 |
| 原始大数据集本体 | — | 不上传，见「数据准备」下载/获取说明 | 说明即可 |

---

## 依赖

### C++（编译核心与自包含基线）

| 依赖 | 版本/说明 |
|---|---|
| CMake | >= 3.16 |
| GCC | >= 9（需支持 C++17） |
| OpenMP | 编译时自动探测 |

`hnswlib` 与 `json.hpp` 已随仓库内置（`include/`），无需额外安装。

### Python（仅训练自适应配置模块需要）

```bash
pip install -r requirements.txt
```

依赖：`numpy`、`pandas`、`lightgbm`、`joblib`、`scikit-learn`。

### 外部基线（可选）

| 基线 | 来源 | 配置 |
|---|---|---|
| RoarGraph | https://github.com/matchyc/RoarGraph | `-DROARGRAPH_DIR=<path>` |
| NGFix | https://github.com/BlindingStars/NGFix | `-DNGFIX_DIR=<path>` |
| Faiss | https://github.com/facebookresearch/faiss | `-DFAISS_INSTALL_DIR=<install>` |
| RVAMANA | DiskANN 的 RobustVamana | 见 `scripts/0_setup_external.sh` |

未提供外部依赖时，CMake 会自动跳过对应目标，不影响核心与自包含基线构建。

---

## 编译

```bash
cd A-CORE
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

可选：启用外部基线

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DROARGRAPH_DIR=/path/to/RoarGraph \
  -DNGFIX_DIR=/path/to/NGFix \
  -DFAISS_INSTALL_DIR=/path/to/faiss/install
make -j
```

生成的可执行文件位于 `build/`：

- 核心：`run_acore`（标签聚类版）、`run_acore_online_grouping`（先聚类再查询）、
  `projected_greedy_cluster`（在线分组）、`run_train_get_all_k_onlytop`、`make_clustered_queries5`
- 基线：`run_hnsw_baseline`、`build_nsg`、`search_nsg`、`search_nsg_cluster`、
  `tming_build`、`search_mrng`、（可选）`search_roargraph_fair_base`、
  `search_hnsw_ngfix_fair_base`、`run_faiss_baseline`

---

## 数据准备

A-CORE 在三个大规模跨模态数据集上评测：

| 数据集 | 规模 | 维度 | 索引向量 |
|---|---|---|---|
| Text-to-Image (t2i) | 10M | 200 | 图像/多模态嵌入 |
| LAION | 10M | 512 | 图像嵌入 |
| WebVid | 2.5M | 512 | 视频嵌入 |

查询由 CLIP ViT-B/32 文本编码器编码并 L2 归一化。

### 数据文件格式

- 向量：`.fbin`（`int32 n` + `int32 d` + `n*d` 个 `float32`）
- 查询：`.xq.fbin`（同上）
- ground truth：`.gt.ibin`（`int32 n` + `int32 k` + `n*k` 个 `int32`）
- 簇标签：`.labels.ibin`（`int32 n` + `n` 个 `int32`）

目录约定（可参考 `scripts/1_prepare_data.sh`）：

```
data/
├── clip-webvid-2.5M/{base.2.5M.fbin, query.10k.fbin}
├── laion-10M/{base.10M.fbin, query.10k.fbin}
└── t2i-10M/{base.10M.fbin, query.public.100K.fbin}
```

> **原始数据集本体无需上传**，只需提供获取/处理说明。三个数据集均为公开数据：
>
> - **LAION**：https://laion.ai （LAION-5B 子集）
> - **WebVid**：https://github.com/m-bain/webvid
> - **Text-to-Image (t2i)**：公开 text-to-image 检索数据集
>
> 查询统一用 CLIP ViT-B/32 文本编码器编码。

### 真实数据集（TREC / WikiAnswers）与格式转换

论文 6.2.2 使用**真实查询 workload**（非合成扰动）。真实数据不是向量，而是**文本查询**，需要先做
「下载原始文本 → 转格式 → CLIP 编码 → 聚类 → 算 GT」这一整条转换，最终得到与合成数据**完全相同**的
`s1.*` 二进制格式，才能喂给 A-CORE。脚本在 `dataset_process/`：

| 脚本 | 输入 → 输出 | 作用 |
|---|---|---|
| `convert_sessiontrack_to_jsonl.py` | `sessiontrack2013.xml` → `topic_queries.jsonl` | TREC SessionTrack XML 转 JSONL（按 topic 去重） |
| `encode_topic_queries_clip.py` | `topic_queries.jsonl` → `clip_topic_vectors/s1.*` | CLIP 编码 + KMeans 聚类 |
| `build_topic_cluster_version.py` | `clip_topic_vectors/s1.*` → `clip_topic_vectors_topic/s1.*` | 严格按 topic 分簇（1 topic = 1 簇） |
| `encode_wikianswers_clip.py` | HuggingFace WikiAnswers → `clip_wikianswers_vectors/s1.*` | 流式下载 + CLIP 编码（每行 = 1 簇） |

**完整转换流程（以 TREC 为例）：**

```bash
cd dataset_process

# ① 原始 XML → JSONL（TREC SessionTrack 2013）
python convert_sessiontrack_to_jsonl.py          # 读 sessiontrack2013.xml，写 topic_queries.jsonl

# ② CLIP 编码 + KMeans 聚类 → s1.xq.fbin / s1.labels.ibin / s1.centers.ibin
python encode_topic_queries_clip.py \
  --input topic_queries.jsonl \
  --outdir clip_topic_vectors --prefix s1 --clusters 50

# ③（可选）严格 topic 分簇版本：1 个 topic = 1 个簇
python build_topic_cluster_version.py \
  --topic-jsonl topic_queries.jsonl \
  --xq clip_topic_vectors/s1.xq.fbin \
  --queries-csv clip_topic_vectors/s1.queries.csv \
  --outdir clip_topic_vectors_topic --prefix s1

# ④ 计算 ground truth（对真实查询在 base 上暴力搜，--gt_only）
../build/make_clustered_queries5 --gt_only \
  --out_prefix clip_topic_vectors_topic/s1 \
  --base ../data/webvid_base.fbin \
  --k_gt 100 --normalize_base 0 --nthreads 24
```

WikiAnswers 同理，直接跑 `encode_wikianswers_clip.py`（内部流式下载，无需手动下载 8GB 原始文件）。

**转换产物 `s1.*` 格式**（与合成数据完全一致，可直接 `--load_synth_prefix` 使用）：

| 文件 | 格式 | 说明 |
|---|---|---|
| `s1.xq.fbin` | `int32 n` + `int32 d` + `n*d float32` | 查询向量（CLIP 编码 + L2 归一化） |
| `s1.labels.ibin` | `int32 n` + `n int32` | 每个查询的簇标签 |
| `s1.centers.ibin` | `int32 k` + `k int32` | 每个簇的中心（质心最近点） |
| `s1.clusters.csv` | CSV | 每簇统计（cluster, center_idx, qpc, sigma_deg, theta_min_deg） |
| `s1.gt.ibin` | `int32 n` + `n*k int32` | ground truth（由 `--gt_only` 计算） |
| `s1.queries.csv` | CSV | 向量行号 ↔ topicid ↔ 原文映射（可追溯） |

> **合成数据 vs 真实数据**：合成数据用 `make_clustered_queries5`（farthest-first 选中心 + 角度扰动）生成；
> 真实数据用上述 `dataset_process/` 脚本把 TREC/WikiAnswers 文本编码后按 topic 分簇。两者最终都是 `s1.*`
> 格式，A-CORE 与各基线的 `--load_synth_prefix` 对两者一视同仁。

---

## 完整流程（先聚类，再查询）

### Step 1 — 构建索引

```bash
# HNSW（A-CORE 默认后端）
./build/run_hnsw_baseline ...        # 或使用任意 HNSW 建图工具生成 *.hnsw
# NSG
./build/build_nsg <base.fbin> <out.nsg> <knn.graph> 64 200 200 --threads 20
# τ-MNG
./build/tming_build <base.fbin> <out.mrng> <knn.graph> 64 200 1 --tau=0.01
```

### Step 2 — 聚类生成合成 OOD batch

```bash
./build/make_clustered_queries5 \
  --real_q data/t2i-10M/query.public.100K.fbin \
  --out_prefix outputs_t2i_new_train_clusters/s1 \
  --clusters 30 \
  --qpc_min 60 --qpc_max 300 --qpc_dist uniform --qpc_logmean 5.3 --qpc_logstd 0.6 \
  --sigma_deg_min 35 --sigma_deg_max 75 \
  --theta_min_deg_min 2 --theta_min_deg_max 12 \
  --min_sep_deg 60 \
  --base data/t2i-10M/base.10M.fbin --k_gt 100 --normalize_base 1 \
  --seed 42 --nthreads 24
```

### Step 3 — 生成训练标签（离线网格搜索）

```bash
./build/run_train_get_all_k_onlytop data/laion-10M/ data/laion_base.hnsw \
  --load_synth_prefix outputs_laion_new_train_clusters/s1 \
  --k 10 \
  --efc_list 1000,1500,2000,...,8000 \
  --efw_list 100,200,300,...,1000 \
  --L_list 0.2,0.4,0.6,...,3.0 \
  --Rstar 0.9 --delta 0.003 \
  --csv_out train_runs_laion.csv
```

### Step 4 — 训练自适应配置模块（完整 4 步链路）

网格搜索产物 `train_runs_*.csv` 不能直接训练，还需经过「特征导出 → 特征增强」两步：

```bash
# 4a) 特征导出/过滤：按 k 生成 filtered_AB_rows_k{1,10,100}.csv
python src/export_feature_csvs_perk.py \
  --csv train_runs_laion.csv \
  --outdir filtered_csvs_laion \
  --rstars 0.75,0.76,...,0.99 \
  --delta 0.015 --gain_eps 0.002 \
  --ks 1,10,100

# 4b) 特征增强：probe search，输出 filtered_AB_rows_k10.with_new_feats.csv
./build/augment_features data/laion-10M \
  filtered_csvs_laion/filtered_AB_rows_k10.csv \
  --load_synth_prefix outputs_laion_new_train2/s1 \
  --index data/laion_base.hnsw \
  --k 10

# 4c) 训练 LightGBM 三预测器（用增强后的 CSV）
python src/train_full_conditional_and_recall_newfeat.py \
  --csv filtered_csvs_laion/filtered_AB_rows_k10.with_new_feats.csv \
  --outdir model_out_laion \
  --alpha 0.9 --seed 42
# 产出：model_out_laion/{model_A_efc_mono.txt, model_B_efw_mono.txt, model_rank_m_mono.txt, meta.json}
```

> 一键执行：`bash scripts/4_train_models.sh` 已封装上述 4 步（网格搜索 → `export_feature_csvs_perk.py` → `augment_features` → 训练）。

### Step 5 — 运行 A-CORE 查询

```bash
./build/run_acore data/clip-webvid-2.5M/ data/webvid_base.hnsw \
  --clusters 100 --k 10 --ef 2000 \
  --load_synth_prefix outputs_webvid_new_train2_test2/s1 \
  --model_dir model_out_webvid \
  --model_A model_A_efc_mono.txt \
  --model_B model_B_efw_mono.txt \
  --model_rank model_rank_m_mono.txt \
  --R_target 0.86:0.02:1 \
  --csv_out webvid_acore.csv \
  --per_query_csv webvid_acore_perq.csv
```

> 完整可一键执行的流程见 `scripts/`（见下文）。

---

## 复现论文实验

| 内容 | 脚本 |
|---|---|
| 数据准备（合成） | `scripts/1_prepare_data.sh` |
| 数据准备（真实 TREC/WikiAnswers） | `scripts/1b_prepare_real_data.sh` |
| 索引构建 | `scripts/2_build_index.sh` |
| 聚类生成 batch | `scripts/3_cluster_queries.sh` |
| 训练自适应配置 | `scripts/4_train_models.sh` |
| 运行 A-CORE | `scripts/5_run_acore.sh` |
| 运行基线 | `scripts/6_run_baselines.sh` |
| 一键复现 | `scripts/reproduce_all.sh` |

论文中的主要结果对应关系：

- **RQ1（Recall@10–QPS 曲线）**：`scripts/5_run_acore.sh` + `scripts/6_run_baselines.sh`
- **RQ2/RQ3（消融）**：通过 `train_full_conditional_and_recall_newfeat_ablation*.py` 的 `--drop` 参数实现
- **RQ4（自适应配置 vs Ada-ef）**：`run_acore` 的 `--R_target` 网格

---

## 引用

```bibtex
@article{zhang2026acore,
  title     = {A-CORE: Anchor-based Context Reuse for Efficient Batched Vector Search},
  author    = {Zhang, Kaibo and Li, Zhengxin and Zhang, Wentong and Fang, Lanting and Feng, Kaiyu and Xie, Miao},
  journal   = {Proceedings of the VLDB Endowment},
  year      = {2026}
}
```
