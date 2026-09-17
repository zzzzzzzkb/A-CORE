#!/usr/bin/env bash
# Step 2：聚类生成合成 OOD batch（farthest-first 选中心 + 角度扰动）
# 每个数据集生成两份：训练集（200 簇，用于网格搜索/增强/训练） + 测试集（50 簇，用于推理）
# 输出前缀含 .xq.fbin / .labels.ibin / .gt.ibin
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/make_clustered_queries5"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs}"

TRAIN_CLUSTERS="${TRAIN_CLUSTERS:-200}"
TEST_CLUSTERS="${TEST_CLUSTERS:-50}"
NTHREADS="${NTHREADS:-24}"

mkdir -p "$OUT_DIR"

gen () {
  local real_q="$1" out_prefix="$2" clusters="$3" base="$4" normalize_base="$5" seed="$6"
  "$BIN" \
    --real_q "$real_q" \
    --out_prefix "$out_prefix" \
    --clusters "$clusters" \
    --qpc_min 60 --qpc_max 300 --qpc_dist uniform --qpc_logmean 5.3 --qpc_logstd 0.6 \
    --sigma_deg_min 5 --sigma_deg_max 35 \
    --theta_min_deg_min 2 --theta_min_deg_max 12 \
    --min_sep_deg 60 \
    --base "$base" --k_gt 100 --normalize_base "$normalize_base" \
    --seed "$seed" --nthreads "$NTHREADS"
}

# WebVid（训练 + 测试）
gen "$DATA_DIR/clip-webvid-2.5M/query.10k.fbin" "$OUT_DIR/webvid_train/s1" "$TRAIN_CLUSTERS" "$DATA_DIR/clip-webvid-2.5M/base.2.5M.fbin" 0 42
gen "$DATA_DIR/clip-webvid-2.5M/query.10k.fbin" "$OUT_DIR/webvid_test/s1"  "$TEST_CLUSTERS"  "$DATA_DIR/clip-webvid-2.5M/base.2.5M.fbin" 0 43

# LAION（训练 + 测试）
gen "$DATA_DIR/laion-10M/query.10k.fbin" "$OUT_DIR/laion_train/s1" "$TRAIN_CLUSTERS" "$DATA_DIR/laion-10M/base.10M.fbin" 0 66
gen "$DATA_DIR/laion-10M/query.10k.fbin" "$OUT_DIR/laion_test/s1"  "$TEST_CLUSTERS"  "$DATA_DIR/laion-10M/base.10M.fbin" 0 67

# Text-to-Image（训练 + 测试）
gen "$DATA_DIR/t2i-10M/query.public.100K.fbin" "$OUT_DIR/t2i_train/s1" "$TRAIN_CLUSTERS" "$DATA_DIR/t2i-10M/base.10M.fbin" 1 42
gen "$DATA_DIR/t2i-10M/query.public.100K.fbin" "$OUT_DIR/t2i_test/s1"  "$TEST_CLUSTERS"  "$DATA_DIR/t2i-10M/base.10M.fbin" 1 43

echo "[OK] 合成 batch 已生成到 $OUT_DIR"
echo "     训练集：*_train/s1（${TRAIN_CLUSTERS} 簇）"
echo "     测试集：*_test/s1（${TEST_CLUSTERS} 簇）"
