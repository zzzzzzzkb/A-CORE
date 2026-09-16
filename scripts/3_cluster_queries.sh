#!/usr/bin/env bash
# Step 2：聚类生成合成 OOD batch（farthest-first 选中心 + 角度扰动）
# 输出前缀含 .xq.fbin / .labels.ibin / .gt.ibin
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/make_clustered_queries5"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"

mkdir -p "$OUT_DIR"

# WebVid
"$BIN" \
  --real_q "$DATA_DIR/clip-webvid-2.5M/query.10k.fbin" \
  --out_prefix "$OUT_DIR/webvid/s1" \
  --clusters 100 \
  --qpc_min 60 --qpc_max 300 --qpc_dist uniform --qpc_logmean 5.3 --qpc_logstd 0.6 \
  --sigma_deg_min 5 --sigma_deg_max 35 \
  --theta_min_deg_min 2 --theta_min_deg_max 12 \
  --min_sep_deg 60 \
  --base "$DATA_DIR/clip-webvid-2.5M/base.2.5M.fbin" --k_gt 100 --normalize_base 0 \
  --seed 42 --nthreads 24

# LAION
"$BIN" \
  --real_q "$DATA_DIR/laion-10M/query.10k.fbin" \
  --out_prefix "$OUT_DIR/laion/s1" \
  --clusters 200 \
  --qpc_min 60 --qpc_max 300 --qpc_dist uniform --qpc_logmean 5.3 --qpc_logstd 0.6 \
  --sigma_deg_min 5 --sigma_deg_max 35 \
  --theta_min_deg_min 2 --theta_min_deg_max 12 \
  --min_sep_deg 60 \
  --base "$DATA_DIR/laion-10M/base.10M.fbin" --k_gt 100 --normalize_base 0 \
  --seed 66 --nthreads 24

# Text-to-Image
"$BIN" \
  --real_q "$DATA_DIR/t2i-10M/query.public.100K.fbin" \
  --out_prefix "$OUT_DIR/t2i/s1" \
  --clusters 30 \
  --qpc_min 60 --qpc_max 300 --qpc_dist uniform --qpc_logmean 5.3 --qpc_logstd 0.6 \
  --sigma_deg_min 35 --sigma_deg_max 75 \
  --theta_min_deg_min 2 --theta_min_deg_max 12 \
  --min_sep_deg 60 \
  --base "$DATA_DIR/t2i-10M/base.10M.fbin" --k_gt 100 --normalize_base 1 \
  --seed 42 --nthreads 24

echo "[OK] 合成 batch 已生成到 $OUT_DIR"
