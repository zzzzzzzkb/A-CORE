# A-CORE: Anchor-based Context Reuse for Efficient Batched Vector Search

**English** | [简体中文](README.md)

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

## Directory Structure

```
A-CORE/
├── CMakeLists.txt                  # Build system
├── README.md                       # Chinese README
├── README_EN.md                    # English README (this file)
├── requirements.txt                # Python dependencies (training scripts)
├── include/                        # header-only dependencies (bundled)
│   ├── hnswlib/                    # HNSW library (header-only)
│   └── json.hpp                    # Single-header JSON
├── src/                            # A-CORE core
│   ├── run_acore.cpp               # Main framework: Centroid Context + Selective-L + adaptive config (label-clustering version)
│   ├── run_acore_online_grouping.cpp  # Latest pipeline (cluster first, then query): Projected-Greedy online grouping + Context Reuse
│   ├── projected_greedy_cluster.cpp   # Standalone implementation of online grouping (Sec 5.3)
│   ├── run_train_get_all_k_onlytop.cpp  # Training label generation ((ef_c,ef_w,L) grid search)
│   ├── export_feature_csvs_perk.py      # Feature export/filtering (filtered_AB_rows per k)
│   ├── augment_features.cpp             # Feature augmentation (probe search → *.with_new_feats.csv)
│   ├── make_clustered_queries5.cpp      # Synthetic OOD batch generation (farthest-first + angular perturbation)
│   └── train_full_conditional_and_recall_newfeat.py  # LightGBM three-predictor training (with --drop/--keep ablation)
├── dataset_process/                # TREC(SessionTrack)/WikiAnswers preprocessing
├── baselines/                      # Baselines
│   ├── hnsw/                       # HNSW
│   ├── nsg/                        # NSG (graph construction + search)
│   ├── taumng/                     # τ-MNG (graph construction + search)
│   ├── roargraph/                  # RoarGraph (external dependency)
│   └── ngfix/                      # NGFix (external dependency)
├── third_party/                    # Small header-only libraries (efanna2e / taumng / rvamana)
└── scripts/                        # Reproduction scripts
```

---


## Dependencies

### C++ (for compiling the core and self-contained baselines)

| Dependency | Version/Notes |
|---|---|
| CMake | >= 3.16 |
| GCC | >= 9 (must support C++17) |
| OpenMP | Auto-detected at build time |

`hnswlib` and `json.hpp` are bundled in the repository (`include/`); no extra installation needed.

### Python (only needed for training the adaptive-configuration module)

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `lightgbm`, `joblib`, `scikit-learn`.

### External Baselines (optional)

| Baseline | Source | Configuration |
|---|---|---|
| RoarGraph | https://github.com/matchyc/RoarGraph | `-DROARGRAPH_DIR=<path>` |
| NGFix | https://github.com/BlindingStars/NGFix | `-DNGFIX_DIR=<path>` |
| RVAMANA | RobustVamana from DiskANN | see `scripts/0_setup_external.sh` |

If the external dependencies are not provided, CMake automatically skips the corresponding targets,
without affecting the core or the self-contained baselines.

---

## Build

```bash
cd A-CORE
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

Optional: enable external baselines

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DROARGRAPH_DIR=/path/to/RoarGraph \
  -DNGFIX_DIR=/path/to/NGFix
make -j
```

The resulting executables are placed in `build/`:

- Core: `run_acore` (label-clustering version), `run_acore_online_grouping` (cluster first, then query),
  `projected_greedy_cluster` (online grouping), `run_train_get_all_k_onlytop`, `make_clustered_queries5`
- Baselines: `run_hnsw_baseline`, `build_nsg`, `search_nsg`, `search_nsg_cluster`,
  `tming_build`, `search_mrng`, (optional) `search_roargraph_fair_base`,
  `search_hnsw_ngfix_fair_base`

---

## Data Preparation

A-CORE is evaluated on three large-scale cross-modal datasets:

| Dataset | Scale | Dimension | Index vectors |
|---|---|---|---|
| Text-to-Image (t2i) | 10M | 200 | Image/multimodal embeddings |
| LAION | 10M | 512 | Image embeddings |
| WebVid | 2.5M | 512 | Video embeddings |

Queries are encoded by the CLIP ViT-B/32 text encoder and L2-normalized.

### Data File Format

- Vectors: `.fbin` (`int32 n` + `int32 d` + `n*d` `float32`)
- Queries: `.xq.fbin` (same as above)
- Ground truth: `.gt.ibin` (`int32 n` + `int32 k` + `n*k` `int32`)
- Cluster labels: `.labels.ibin` (`int32 n` + `n` `int32`)

Directory convention (see `scripts/1_prepare_data.sh`):

```
data/
├── clip-webvid-2.5M/{base.2.5M.fbin, query.10k.fbin}
├── laion-10M/{base.10M.fbin, query.10k.fbin}
└── t2i-10M/{base.10M.fbin, query.public.100K.fbin}
```

> **The raw datasets themselves do not need to be uploaded**; only acquisition instructions are
> provided. All three datasets (t2i-10M / laion-10M / webvid-2.5M) are downloaded from Zenodo,
> sourced from the RoarGraph project:
>
> **Download URL**: https://zenodo.org/records/11090378 (DOI: 10.5281/zenodo.11090378, CC BY 4.0)
>
> This record contains (base vectors / queries / ground truth / RoarGraph index):
>
> | File | Size | Description |
> |---|---|---|
> | `clip.webvid.base.2.5M.fbin` | 5.1 GB | WebVid base vectors (2.5M×512) |
> | `webvid.query.train.2.5M.fbin` | 5.1 GB | WebVid training queries |
> | `laion.query.10k.fbin` | 20.5 MB | LAION queries (10k) |
> | `webvid.query.10k.fbin` | 20.5 MB | WebVid queries (10k) |
> | `laion.gt.10k.ibin` / `t2i.gt.10k.ibin` / `webvid.gt.10k.ibin` | 40 MB each | ground truth |
> | `laion_10M_roar.index` / `t2i_10M_roar.index` / `webvid_2.5M_roar.index` | — | Pre-built RoarGraph index (optional) |
>
> Queries are uniformly encoded by the CLIP ViT-B/32 text encoder and L2-normalized.

### Obtaining the Text-to-Image (t2i) Dataset

t2i-10M is **Yandex's text-to-image dataset** (`yandex-research` ann-datasets). RoarGraph's
`prepare_data.sh t2i-10M` downloads it automatically (see https://github.com/matchyc/RoarGraph):

```bash
git clone --recursive https://github.com/matchyc/RoarGraph.git
cd RoarGraph
bash prepare_data.sh t2i-10M        # downloads to ./data/t2i-10M
```

Original files (`storage.yandexcloud.net`):

| File | URL |
|---|---|
| base vectors (10M×200) | `https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.10M.fbin` |
| training queries | `https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin` (take first 10M) |
| public queries | `https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin` (take first 10k) |
| GT | `https://zenodo.org/records/11090378/files/t2i.gt.10k.ibin` |

### Real-World Datasets (TREC / WikiAnswers) and Format Conversion

Section 6.2.1 of the paper uses a **real query workload** (not synthetic perturbation). Real data
are not vectors but **text queries**, so the full conversion pipeline
"download raw text → convert format → CLIP encoding → clustering → compute GT" must be run first,
producing a `s1.*` binary format **identical** to the synthetic data, which can then be fed to
A-CORE. The scripts are in `dataset_process/`:

| Script | Input → Output | Purpose |
|---|---|---|
| `convert_sessiontrack_to_jsonl.py` | `sessiontrack2014.xml` → `topic_queries.jsonl` | Convert TREC Session 2014 XML to JSONL (deduplicated by topic) |
| `encode_topic_queries_clip.py` | `topic_queries.jsonl` → `clip_topic_vectors/s1.*` | CLIP encoding + KMeans clustering |
| `build_topic_cluster_version.py` | `clip_topic_vectors/s1.*` → `clip_topic_vectors_topic/s1.*` | Strict topic-based clustering (1 topic = 1 cluster) |
| `encode_wikianswers_clip.py` | HuggingFace WikiAnswers → `clip_wikianswers_vectors/s1.*` | Streaming download + CLIP encoding (each line = 1 cluster) |

**Full conversion pipeline (TREC as example):**

```bash
cd dataset_process

# ① Raw XML → JSONL (TREC Session 2014, data: https://trec.nist.gov/data/session2014.html)
python convert_sessiontrack_to_jsonl.py sessiontrack2014.xml   # reads XML, writes topic_queries.jsonl

# ② CLIP encoding + KMeans clustering → s1.xq.fbin / s1.labels.ibin / s1.centers.ibin
python encode_topic_queries_clip.py \
  --input topic_queries.jsonl \
  --outdir clip_topic_vectors --prefix s1 --clusters 50

# ③ (Optional) strict topic clustering version: 1 topic = 1 cluster
python build_topic_cluster_version.py \
  --topic-jsonl topic_queries.jsonl \
  --xq clip_topic_vectors/s1.xq.fbin \
  --queries-csv clip_topic_vectors/s1.queries.csv \
  --outdir clip_topic_vectors_topic --prefix s1

# ④ Compute ground truth (brute force on base for real queries, --gt_only)
../build/make_clustered_queries5 --gt_only \
  --out_prefix clip_topic_vectors_topic/s1 \
  --base ../data/webvid_base.fbin \
  --k_gt 100 --normalize_base 0 --nthreads 24
```

WikiAnswers works the same way: just run `encode_wikianswers_clip.py` (it streams the download
internally; no need to manually download the 8 GB raw file).

**Converted `s1.*` format** (identical to synthetic data, ready to use via `--load_synth_prefix`):

| File | Format | Description |
|---|---|---|
| `s1.xq.fbin` | `int32 n` + `int32 d` + `n*d float32` | Query vectors (CLIP-encoded + L2-normalized) |
| `s1.labels.ibin` | `int32 n` + `n int32` | Cluster label of each query |
| `s1.centers.ibin` | `int32 k` + `k int32` | Center of each cluster (nearest point to centroid) |
| `s1.clusters.csv` | CSV | Per-cluster statistics (cluster, center_idx, qpc, sigma_deg, theta_min_deg) |
| `s1.gt.ibin` | `int32 n` + `n*k int32` | ground truth (computed by `--gt_only`) |
| `s1.queries.csv` | CSV | Vector row index ↔ topicid ↔ raw text mapping (traceable) |

> **Synthetic vs. real data**: Synthetic data is generated by `make_clustered_queries5`
> (farthest-first center selection + angular perturbation); real data is produced by the
> `dataset_process/` scripts above, which encode TREC/WikiAnswers text and cluster by topic.
> Both end up in the `s1.*` format, and A-CORE plus all baselines treat them identically via
> `--load_synth_prefix`.

---

## Full Pipeline (cluster first, then query)

### Step 1 — Build Index

```bash
# HNSW (A-CORE default backend)
./build/run_hnsw_baseline ...        # or any HNSW graph builder that produces *.hnsw
# NSG
./build/build_nsg <base.fbin> <out.nsg> <knn.graph> 64 200 200 --threads 20
# τ-MNG
./build/tming_build <base.fbin> <out.mrng> <knn.graph> 64 200 1 --tau=0.01
```

### Step 2 — Cluster to Generate Synthetic OOD Batches (Training + Test Sets)

Synthetic data is split by **200 clusters for training / 50 clusters for testing** (corresponding
to `C_train = 200` in the paper). The training set is used for Steps 3–4 (grid search, feature
augmentation, training), and the test set (or real data) is used for Step 5 inference.

```bash
# One-click generation (scripts/3_cluster_queries.sh produces both *_train(200) and *_test(50))
bash scripts/3_cluster_queries.sh

# Or manually (WebVid training set with 200 clusters as example)
./build/make_clustered_queries5 \
  --real_q data/clip-webvid-2.5M/query.10k.fbin \
  --out_prefix outputs/webvid_train/s1 \
  --clusters 200 \
  --qpc_min 60 --qpc_max 300 --qpc_dist uniform --qpc_logmean 5.3 --qpc_logstd 0.6 \
  --sigma_deg_min 5 --sigma_deg_max 35 \
  --theta_min_deg_min 2 --theta_min_deg_max 12 \
  --min_sep_deg 60 \
  --base data/clip-webvid-2.5M/base.2.5M.fbin --k_gt 100 --normalize_base 0 \
  --seed 42 --nthreads 24
```

### Step 3 — Generate Training Labels (Offline Grid Search)

The grid matches Section 6.1 of the paper: `ef_c ∈ [1000,10000] step 500`, `ef_w ∈ [100,1000] step 50`,
`L ∈ [0.2,3.0] step 0.2`.

```bash
./build/run_train_get_all_k_onlytop data/laion-10M/ data/laion_base.hnsw \
  --load_synth_prefix outputs/laion_train/s1 \
  --k 10 \
  --efc_list 1000,1500,2000,...,10000 \
  --efw_list 100,150,200,...,1000 \
  --L_list 0.2,0.4,0.6,...,3.0 \
  --Rstar 0.9 --delta 0.003 \
  --csv_out train_runs_laion.csv
```

### Step 4 — Train the Adaptive-Configuration Module (full 4-step pipeline)

The grid-search output `train_runs_*.csv` cannot be trained on directly; it must go through two
more steps: "feature export → feature augmentation".

```bash
# 4a) Feature export/filtering: generate filtered_AB_rows_k{1,10,100}.csv per k
python src/export_feature_csvs_perk.py \
  --csv train_runs_laion.csv \
  --outdir filtered_csvs_laion \
  --rstars 0.75,0.76,...,0.99 \
  --delta 0.015 --gain_eps 0.002 \
  --ks 1,10,100

# 4b) Feature augmentation: probe search, outputs filtered_AB_rows_k10.with_new_feats.csv
./build/augment_features data/laion-10M \
  filtered_csvs_laion/filtered_AB_rows_k10.csv \
  --load_synth_prefix outputs/laion_train/s1 \
  --index data/laion_base.hnsw \
  --k 10

# 4c) Train the three LightGBM predictors (using the augmented CSV)
python src/train_full_conditional_and_recall_newfeat.py \
  --csv filtered_csvs_laion/filtered_AB_rows_k10.with_new_feats.csv \
  --outdir model_out_laion \
  --alpha 0.9 --seed 42
# Outputs: model_out_laion/{model_A_efc_mono.txt, model_B_efw_mono.txt, model_rank_m_mono.txt, meta.json}
```

> One-click execution: `bash scripts/4_train_models.sh` wraps the entire pipeline
> "grid search → feature export → feature augmentation → training → inference",
> where Steps 1–4 use the training set and Step 5 uses the test set (or real data).

### Step 5 — Run A-CORE Queries (inference on test set / real data)

> Note: when `--load_synth_prefix` provides `labels.ibin`, the actual number of clusters is
> determined by the labels, and the `--clusters` argument is ignored (passing 50 here is only for
> semantic consistency).

```bash
# Synthetic test set (50 clusters) inference
./build/run_acore data/clip-webvid-2.5M/ data/webvid_base.hnsw \
  --clusters 50 --k 10 --ef 2000 \
  --load_synth_prefix outputs/webvid_test/s1 \
  --model_dir model_out \
  --model_A model_A_efc_mono.txt \
  --model_B model_B_efw_mono.txt \
  --model_rank model_rank_m_mono.txt \
  --R_target 0.86:0.02:1 \
  --csv_out webvid_acore.csv \
  --per_query_csv webvid_acore_perq.csv

# Real data (topic) inference: just replace --load_synth_prefix with the prefix produced by dataset_process
#   --load_synth_prefix dataset_process/clip_topic_vectors_topic/s1
```

> For end-to-end reproduction use `scripts/reproduce_all.sh`. It performs a preflight check at the
> start, and if datasets/indexes are missing it reports a clear error and points you to the README
> sections "Data Preparation" and "Step 1".

---

## Reproducing the Paper Experiments

| Content | Script |
|---|---|
| Data preparation (synthetic) | `scripts/1_prepare_data.sh` |
| Data preparation (real TREC/WikiAnswers) | `scripts/1b_prepare_real_data.sh` |
| Index construction | `scripts/2_build_index.sh` |
| Clustering to generate batches | `scripts/3_cluster_queries.sh` |
| Train adaptive configuration | `scripts/4_train_models.sh` |
| Run A-CORE | `scripts/5_run_acore.sh` |
| Run baselines | `scripts/6_run_baselines.sh` |
| Ablation study | `scripts/7_ablation.sh` |
| End-to-end reproduction (datasets & indexes must be prepared first) | `scripts/reproduce_all.sh` |

---

## Citation

```bibtex
@article{zhang2026acore,
  title     = {A-CORE: Anchor-based Context Reuse for Efficient Batched Vector Search},
  author    = {Zhang, Kaibo and Li, Zhengxin and Zhang, Wentong and Fang, Lanting and Feng, Kaiyu and Xie, Miao},
  journal   = {Proceedings of the VLDB Endowment},
  year      = {2026}
}
```
