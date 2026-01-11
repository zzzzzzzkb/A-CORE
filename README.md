# A-CORE:Anchor-based Context Reuse in Adaptive Batched Out-of-Distribution Vector Search

This repository contains the core implementation used in the experiments. It provides a complete pipeline to

1. generate synthetic clustered queries,
2. collect per-cluster training statistics from an HNSW index,
3. train conditional LightGBM models, and
4. run the final adaptive search.

## File overview

- `make_clustered_queries5.cpp`  
  Synthetic clustered query generator. Produces clustered query sets and (optionally) ground-truth nearest neighbours against a base dataset.

- `run_train_get_all_k_onlytop.cpp`  
  HNSW-based feature collector. Runs searches on the clustered queries and writes per-cluster / per-query statistics to a CSV file for model training.

- `train_full_conditional_and_recall_newfeat.py`  
  LightGBM training script. Trains the conditional models used by A-CORE from the CSV features.

- `run_ACORE.cpp`  
  Final A-CORE search executable. Uses the trained models to adapt search parameters at query / cluster level.

## Dependencies

### C++

- A C++17 compiler (e.g., `g++` or `clang++`).
- [hnswlib](https://github.com/nmslib/hnswlib) available in the include path.
- CMake or a simple Makefile can be used; the examples below use direct compilation commands.

### Python

- Python 3.8+
- `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `joblib`.

Example installation:

```bash
pip install numpy pandas scikit-learn lightgbm joblib
```

## Step 1: Build the C++ tools

Example (adjust include paths and flags as needed):

```bash
# make_clustered_queries5
 g++ -O3 -std=c++17 make_clustered_queries5.cpp -o make_clustered_queries5

# run_train_get_all_k_onlytop
 g++ -O3 -std=c++17 run_train_get_all_k_onlytop.cpp -o run_train_get_all_k_onlytop -I/path/to/hnswlib

# run_ACORE
 g++ -O3 -std=c++17 run_ACORE.cpp -o run_ACORE -I/path/to/hnswlib
```

You may need to add `-fopenmp` if you wish to use OpenMP parallelism.

## Step 2: Generate clustered queries

Use `make_clustered_queries5` to generate synthetic clustered queries from a base dataset (and optionally recompute ground-truth neighbours).

Conceptually, you provide:

- `--real_q`: an existing query or base feature file in FBIN format,
- `--out_prefix`: where the generated clustered queries will be written,
- cluster configuration: `--clusters`, `--qpc_min/max`, angular spreads, etc.,
- optional GT arguments: `--base`, `--k_gt`, `--normalize_base`, `--nthreads`.

Example (paths and exact hyperparameters are for illustration only):

```bash
./make_clustered_queries5 \
  --real_q PATH/TO/base.fbin \
  --out_prefix outputs/s1 \
  --clusters 40 \
  --qpc_min 80 --qpc_max 200 --qpc_dist uniform \
  --sigma_deg_min 5 --sigma_deg_max 35 \
  --theta_min_deg_min 2 --theta_min_deg_max 12 \
  --min_sep_deg 60 \
  --base PATH/TO/base.fbin --k_gt 100 --normalize_base 1 \
  --seed 42 --nthreads 24
```

This step produces (among others) an `out_prefix.xq.fbin` and, if GT is enabled, the corresponding GT files used later for evaluation.

## Step 3: Collect training data

Once the clustered queries and GT are available and an HNSW index is built for the base data, run `run_train_get_all_k_onlytop` to collect training features.

You provide:

- the data / index directory,
- the path to the HNSW index file,
- `--load_synth_prefix`: the prefix produced by `make_clustered_queries5`,
- search configuration grids over `efc`, `ef_warm`, and `L`.

Example usage pattern (simplified):

```bash
./run_train_get_all_k_onlytop \
  DATA_DIR INDEX_PATH \
  --load_synth_prefix outputs/s1 \
  --k_collect 100 \
  --csv_out training_features.csv \
  --efc_list 64,128,256 \
  --efw_list 64,128,256 \
  --L_list 0.5,1.0,1.5
```

The program writes a CSV file (`training_features.csv`) with per-cluster features and recall statistics, which will be used by the Python training script.

## Step 4: Train conditional models

Run the Python script on the CSV generated in Step 3:

```bash
python train_full_conditional_and_recall_newfeat.py \
  --csv training_features.csv \
  --outdir model_out \
  --alpha 0.85 \
  --seed 42
```

This produces LightGBM TXT model files such as:

- `model_A_efc.txt` and `model_A_efc_mono.txt`
- `model_B_efw.txt` and `model_B_efw_mono.txt`
- `model_rank_m.txt` and `model_rank_m_mono.txt`

along with corresponding `.pkl` and feature-importance CSVs.

## Step 5: Run A-CORE search

Finally, use `run_ACORE` to run the adaptive search using the trained models.

Typical arguments include:

- dataset / index paths,
- `--clusters`, `--k`, `--ef`, `--ef_warm`,
- `--load_synth_prefix`: the same clustered-query prefix as before,
- `--model_dir`: directory containing the TXT models produced in Step 4,
- model file names: `--model_A`, `--model_B`, `--model_rank`,
- target recall range `--R_target`,
- optional CSV output for logging.

Example (schematic):

```bash
./run_ACORE \
  DATA_DIR INDEX_PATH \
  --k 10 \
  --load_synth_prefix outputs/s1 \
  --model_dir model_out \
  --model_A model_A_efc_mono.txt \
  --model_B model_B_efw_mono.txt \
  --model_rank model_rank_m_mono.txt \
  --R_target 0.85:0.05:0.95 \
  --csv_out acore_results.csv
```

Please adapt paths and hyperparameters to match the exact setup described in the paper.

## Reproducibility notes

- All randomised components expose seeds (e.g., `--seed` in the generator and training pipeline).
- The exact hyperparameter values used in the paper can be provided in a separate configuration file or script if required by the reviewing process.
- The code is written to be self-contained apart from standard libraries, hnswlib, and the listed Python dependencies.
