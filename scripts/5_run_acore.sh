#!/usr/bin/env bash
# Step 5：运行 A-CORE 查询（先聚类，再查询）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/run_acore"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
INDEX="${INDEX:-$DATA_DIR/webvid_base.hnsw}"
SYN_PREFIX="${SYN_PREFIX:-$ROOT/outputs/webvid/s1}"
MODEL_DIR="${MODEL_DIR:-$ROOT/model_out}"

CLUSTERS="${CLUSTERS:-100}"
K="${K:-10}"
EF="${EF:-2000}"
R_TARGET="${R_TARGET:-0.86:0.02:1}"

mkdir -p "$ROOT/results"

"$BIN" "$DATA_DIR/clip-webvid-2.5M/" "$INDEX" \
  --clusters "$CLUSTERS" \
  --k "$K" \
  --ef "$EF" \
  --load_synth_prefix "$SYN_PREFIX" \
  --model_dir "$MODEL_DIR" \
  --model_A model_A_efc_mono.txt \
  --model_B model_B_efw_mono.txt \
  --model_rank model_rank_m_mono.txt \
  --R_target "$R_TARGET" \
  --csv_out "$ROOT/results/webvid_acore.csv" \
  --per_query_csv "$ROOT/results/webvid_acore_perq.csv"

echo "[OK] 结果写入 $ROOT/results/webvid_acore.csv"
