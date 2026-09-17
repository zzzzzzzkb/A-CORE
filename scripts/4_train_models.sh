#!/usr/bin/env bash
# 训练自适应配置模块（对齐 /data/RoarGraph-main/src/ablation2.sh 的真实流程）
# 完整链路：
#   1) 网格搜索生成训练标签        run_train_get_all_k_onlytop
#   2) 特征导出/过滤（按 k）        export_feature_csvs_perk.py
#   3) 特征增强（probe search）     augment_features
#   4) 训练 LightGBM 三预测器       train_full_conditional_and_recall_newfeat.py（ablation2 版，对齐论文特征）
#   5) 推理（评测 Recall@k / QPS）   run_acore
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data/clip-webvid-2.5M}"
INDEX="${INDEX:-$ROOT/data/webvid_base.hnsw}"
# 训练/推理用的查询前缀。合成数据用 outputs/webvid/s1；真实 topic 数据用 dataset_process 产出的 clip_topic_vectors_topic/s1
SYN_PREFIX="${SYN_PREFIX:-$ROOT/outputs/webvid/s1}"
MODEL_OUT="${MODEL_OUT:-$ROOT/model_out}"

# ---- 与 ablation2.sh 对齐的参数 ----
K="${K:-10}"
EF="${EF:-2000}"
CLUSTERS="${CLUSTERS:-100}"
RGRID="${RGRID:-0.8:0.05:0.95}"
ALPHA="${ALPHA:-0.9}"
SEED="${SEED:-42}"

# ---- 网格搜索 / 过滤参数 ----
RSTAR="${RSTAR:-0.9}"
DELTA="${DELTA:-0.003}"
GAIN_EPS="${GAIN_EPS:-0.002}"
KS="${KS:-1,10,100}"
RSTARS="${RSTARS:-0.80,0.85,0.88,0.90,0.92,0.94,0.96,0.98}"

mkdir -p "$MODEL_OUT/filtered" "$ROOT/results"

# ---- 1) 生成训练标签（(ef_c,ef_w,L) 网格搜索）----
TRAIN_CSV="$MODEL_OUT/train_runs.csv"
"$ROOT/build/run_train_get_all_k_onlytop" "$DATA_DIR" "$INDEX" \
  --load_synth_prefix "$SYN_PREFIX" \
  --k "$K" \
  --efc_list 1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000 \
  --efw_list 100,200,300,400,500,600,700,800,900,1000 \
  --L_list 0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3 \
  --Rstar "$RSTAR" --delta "$DELTA" \
  --csv_out "$TRAIN_CSV"

# ---- 2) 特征导出/过滤：按 k 生成 filtered_AB_rows_k{1,10,100}.csv ----
python "$ROOT/src/export_feature_csvs_perk.py" \
  --csv "$TRAIN_CSV" \
  --outdir "$MODEL_OUT/filtered" \
  --rstars "$RSTARS" \
  --delta "$DELTA" \
  --gain_eps "$GAIN_EPS" \
  --ks "$KS"

# ---- 3) 特征增强：probe search，输出 *.with_new_feats.csv ----
FEAT_CSV="$MODEL_OUT/filtered/filtered_AB_rows_k${K}.csv"
"$ROOT/build/augment_features" "$DATA_DIR" "$FEAT_CSV" \
  --load_synth_prefix "$SYN_PREFIX" \
  --index "$INDEX" \
  --k "$K"

# ---- 4) 训练 LightGBM 三预测器（对齐论文特征）----
python "$ROOT/src/train_full_conditional_and_recall_newfeat.py" \
  --csv "${FEAT_CSV%.csv}.with_new_feats.csv" \
  --outdir "$MODEL_OUT" \
  --alpha "$ALPHA" --seed "$SEED"

# ---- 5) 推理（评测 Recall@k / QPS，与 ablation2.sh 一致）----
"$ROOT/build/run_acore" "$DATA_DIR" "$INDEX" \
  --clusters "$CLUSTERS" --k "$K" --ef "$EF" \
  --load_synth_prefix "$SYN_PREFIX" \
  --model_dir "$MODEL_OUT" \
  --model_A model_A_efc_mono.txt \
  --model_B model_B_efw_mono.txt \
  --model_rank model_rank_m_mono.txt \
  --R_target "$RGRID" \
  --csv_out "$ROOT/results/acore_grid.csv" \
  --per_query_csv "$ROOT/results/acore_perq.csv"

echo "[OK] 模型已保存到 $MODEL_OUT"
echo "     产物：model_A_efc_mono.txt / model_B_efw_mono.txt / model_rank_m_mono.txt / meta.json"
echo "[OK] 推理结果：$ROOT/results/acore_grid.csv"
