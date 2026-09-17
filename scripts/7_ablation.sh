#!/usr/bin/env bash
# 消融实验（RQ3）：训练时 drop 指定特征 -> 用 run_acore 推理 -> 汇总 summary.csv
# 用法：bash scripts/7_ablation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/run_acore"
DATA_DIR="${DATA_DIR:-$ROOT/data/clip-webvid-2.5M}"
INDEX="${INDEX:-$ROOT/data/webvid_base.hnsw}"
SYN_PREFIX="${SYN_PREFIX:-$ROOT/outputs/webvid/s1}"
CSV="${CSV:-$ROOT/model_out/filtered/filtered_AB_rows_k10.with_new_feats.csv}"

OUTROOT="${OUTROOT:-$ROOT/exp_ablation}"
K="${K:-10}"
EF="${EF:-2000}"
CLUSTERS="${CLUSTERS:-100}"
RGRID="${RGRID:-0.8:0.05:0.95}"
ALPHA="${ALPHA:-0.9}"
SEED="${SEED:-42}"

mkdir -p "$OUTROOT"
echo "[INFO] 消融结果目录: $OUTROOT"

run_one () {
  local EXP="$1"           # 实验名
  local DROP_LIST="$2"     # 要 drop 的特征（空串=不删，即完整模型）

  local OUTDIR="$OUTROOT/$EXP"
  mkdir -p "$OUTDIR"

  echo "[TRAIN] $EXP (drop: ${DROP_LIST:-<none>})"
  if [[ -z "${DROP_LIST}" ]]; then
    python "$ROOT/src/train_full_conditional_and_recall_newfeat.py" \
      --csv "$CSV" --outdir "$OUTDIR" --alpha "$ALPHA" --seed "$SEED"
  else
    python "$ROOT/src/train_full_conditional_and_recall_newfeat.py" \
      --csv "$CSV" --outdir "$OUTDIR" --alpha "$ALPHA" --seed "$SEED" \
      --drop "$DROP_LIST"
  fi

  echo "[INFER] $EXP"
  "$BIN" "$DATA_DIR" "$INDEX" \
    --clusters "$CLUSTERS" --k "$K" --ef "$EF" \
    --load_synth_prefix "$SYN_PREFIX" \
    --model_dir "$OUTDIR" \
    --model_A model_A_efc_mono.txt \
    --model_B model_B_efw_mono.txt \
    --model_rank model_rank_m_mono.txt \
    --R_target "$RGRID" \
    --csv_out "$OUTROOT/grid_${EXP}.csv" \
    --per_query_csv "$OUTROOT/perq_${EXP}.csv"
}

# ---- 完整模型（不 drop）----
run_one "full" ""

# ---- 单特征消融（逐个 drop 一个特征，可按需增减）----
FEATS=(
  cluster_size cluster_size_log log1p_cluster_size cluster_density
  cluster_radius_p50 cluster_radius_p90 radius_skew
  dist_centroid_to_entryL0 entry_dist_norm dist_centroid_top1_smallEF
  overlap128_vs_256 jaccard128_vs_256
  lid_probe256_k10 rc_probe256_k10 expansion2k_over_k_probe256_k10
  lid_k_efc rc_k_efc expand2k_over_k_efc
)
for f in "${FEATS[@]}"; do
  run_one "drop_${f}" "$f"
done

# ---- 汇总 grid_* 到 summary.csv ----
export OUTROOT
python - "$OUTROOT" <<'PY'
import os, glob, csv, sys
root = sys.argv[1]
out = os.path.join(root, "summary.csv")
rows = []
for fp in sorted(glob.glob(os.path.join(root, "grid_*.csv"))):
    exp = os.path.basename(fp)[5:-4]  # strip 'grid_' and '.csv'
    with open(fp, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"exp": exp, **r})
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["exp", "target_recall", "qps", "recall"])
    w.writeheader()
    w.writerows(rows)
print(f"[OK] 消融汇总写入 {out}")
PY
