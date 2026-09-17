#!/usr/bin/env bash
# 运行基线对比（HNSW / NSG / τ-MNG / RoarGraph / RVAMANA / NGFix）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
SYN_PREFIX="${SYN_PREFIX:-$ROOT/outputs/laion_test/s1}"

mkdir -p "$ROOT/results"

# ---- NSG ----
"$BIN/search_nsg" \
  "$DATA_DIR/laion-10M/base.10M.fbin" \
  "$DATA_DIR/laion.nsg" \
  "$SYN_PREFIX.xq.fbin" \
  "$SYN_PREFIX.gt.ibin" \
  --K=1,10,100 --L=300:500:5000 --runs=1 --warmup=0 \
  --csv="$ROOT/results/laion_nsg.csv"

# ---- τ-MNG ----
"$BIN/search_mrng" \
  "$DATA_DIR/laion-10M/base.10M.fbin" \
  "$DATA_DIR/laion-10M.mrng" \
  "$SYN_PREFIX.xq.fbin" \
  "$SYN_PREFIX.gt.ibin" \
  --K=1,10,100 --L=300:5000:500 --S=8 --runs=1 --warmup=0 --seed=42 \
  --csv="$ROOT/results/laion_mrng.csv"

# ---- HNSW ----
"$BIN/run_hnsw_baseline" "$DATA_DIR/laion-10M" "$DATA_DIR/laion_base.hnsw" \
  --load_synth_prefix "$SYN_PREFIX" \
  --K=1,10,100 --EF=500:500:5000 \
  --csv="$ROOT/results/laion_hnsw_baseline.csv"

# ---- RoarGraph（需 -DROARGRAPH_DIR 编译）----
if [ -x "$BIN/search_roargraph_fair_base" ]; then
  OMP_NUM_THREADS=1 "$BIN/search_roargraph_fair_base" \
    "$DATA_DIR/laion-10M" "$DATA_DIR/laion_roargraph.index" \
    --k 10 --k_collect 100 \
    --ef_list 100,200,300,400,500,600,700,1000 \
    --load_synth_prefix "$SYN_PREFIX" --dist l2 --num_threads 1 \
    --csv_out "$ROOT/results/laion_roargraph.csv" --per_query_csv "$ROOT/results/laion_roargraph_perq.csv"
fi

# ---- NGFix（需 -DNGFIX_DIR 编译）----
if [ -x "$BIN/search_hnsw_ngfix_fair_base" ]; then
  OMP_NUM_THREADS=1 "$BIN/search_hnsw_ngfix_fair_base" "$DATA_DIR/laion-10M" \
    "$DATA_DIR/laion-10M_HNSW_NGFix.index" \
    --k 10 --k_collect 100 --ef_list 100,200,400,800,1000 \
    --load_synth_prefix "$SYN_PREFIX" \
    --csv_out "$ROOT/results/laion_ngfix.csv" --per_query_csv "$ROOT/results/laion_ngfix_perq.csv"
fi

# ---- RVAMANA（DiskANN，二进制需另行编译）----
# ./rvamana22 -mode search -data <base.fbin> -graph <rvamana.ivf> \
#   -qfile <xq.fbin> -gt <gt.ibin> -K 1,10,100 -L 32:32:512 -threads 1 -csv <out.csv>

echo "[OK] 基线结果已写入 $ROOT/results/"
