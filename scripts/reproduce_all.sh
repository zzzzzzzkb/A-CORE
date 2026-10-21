#!/usr/bin/env bash
# 端到端复现：数据 -> 索引 -> 聚类 -> 训练 -> A-CORE 查询 -> 基线
#
# 用法：
#   bash scripts/reproduce_all.sh                    # 全量：三个合成数据集（默认）
#   bash scripts/reproduce_all.sh --dataset webvid   # 单数据集核心实验（webvid / laion / t2i）
#   bash scripts/reproduce_all.sh --dataset real     # 真实数据集（TREC / WikiAnswers）
#
# 前置条件：已按 README「数据准备」「Step 1」准备好数据集与索引（.fbin / .hnsw / KNN graph）
#   - 单数据集模式只检查该数据集所需的文件
#   - real 模式会自动下载/编码真实查询，并基于已训练好的模型推理
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:?缺少数据集名（webvid / laion / t2i / real）}"
      shift 2
      ;;
    *)
      echo "未知参数: $1" >&2
      exit 1
      ;;
  esac
done

# ============ 数据集配置表（单数据集模式用） ============
D_SUBDIR=""; D_BASE=""; D_QUERY=""; D_INDEX=""; D_TRAIN=""; D_TEST=""
case "$DATASET" in
  ""|all) ;;
  webvid)
    D_SUBDIR="clip-webvid-2.5M"; D_BASE="base.2.5M.fbin"; D_QUERY="query.10k.fbin"
    D_INDEX="$ROOT/data/webvid_base.hnsw"
    D_TRAIN="$ROOT/outputs/webvid_train/s1"; D_TEST="$ROOT/outputs/webvid_test/s1"
    ;;
  laion)
    D_SUBDIR="laion-10M"; D_BASE="base.10M.fbin"; D_QUERY="query.10k.fbin"
    D_INDEX="$ROOT/data/laion_base.hnsw"
    D_TRAIN="$ROOT/outputs/laion_train/s1"; D_TEST="$ROOT/outputs/laion_test/s1"
    ;;
  t2i)
    D_SUBDIR="t2i-10M"; D_BASE="base.10M.fbin"; D_QUERY="query.public.100K.fbin"
    D_INDEX="$ROOT/data/t2i_base.hnsw"
    D_TRAIN="$ROOT/outputs/t2i_train/s1"; D_TEST="$ROOT/outputs/t2i_test/s1"
    ;;
  real) ;;
  *)
    echo "未知数据集: $DATASET（可选 webvid / laion / t2i / real）" >&2
    exit 1
    ;;
esac

# ============ Preflight 检查 ============
preflight() {
  local missing=()
  local need=()
  case "$DATASET" in
    ""|all)
      need=(
        # 数据集 base + 查询（README「数据准备」）
        "$ROOT/data/clip-webvid-2.5M/base.2.5M.fbin"
        "$ROOT/data/clip-webvid-2.5M/query.10k.fbin"
        "$ROOT/data/laion-10M/base.10M.fbin"
        "$ROOT/data/laion-10M/query.10k.fbin"
        "$ROOT/data/t2i-10M/base.10M.fbin"
        "$ROOT/data/t2i-10M/query.public.100K.fbin"
        # NSG / τ-MNG 建图所需的 KNN graph（README「Step 1」）
        "$ROOT/data/knng/clip-webvid-2.5M/base.100NN.graph"
        "$ROOT/data/knng/laion-10M/base.100NN.graph"
      )
      ;;
    real)
      need=(
        # 真实查询需在 base 上算 GT（HNSW 索引会在后续自动构建）
        "$ROOT/data/clip-webvid-2.5M/base.2.5M.fbin"
      )
      ;;
    *)
      need=(
        "$ROOT/data/$D_SUBDIR/$D_BASE"
        "$ROOT/data/$D_SUBDIR/$D_QUERY"
      )
      ;;
  esac
  for f in "${need[@]}"; do
    [ -e "$f" ] || missing+=("$f")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "[PREFLIGHT] 缺少前置数据集/索引文件，请按 README「数据准备」与「Step 1」准备后再运行：" >&2
    for m in "${missing[@]}"; do
      echo "    Missing $m" >&2
    done
    echo "    Please prepare them according to README (数据准备 / Step 1 — 构建索引)." >&2
    exit 1
  fi
  echo "[PREFLIGHT] 前置数据集/索引检查通过"
}
preflight

echo "==================== 编译 ===================="
mkdir -p "$ROOT/build" && cd "$ROOT/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd "$ROOT"

# ============ 索引构建（HNSW，缺失时自动构图） ============
build_hnsw() {
  local data_dir="$1" index="$2"
  if [ -f "$index" ]; then
    echo "[HNSW] 索引已存在: $index"
  else
    echo "[HNSW] 构建索引: $data_dir -> $index"
    "$ROOT/build/run_hnsw_baseline" "$data_dir" "$index" --build_only
  fi
}
case "$DATASET" in
  ""|all)
    build_hnsw "$ROOT/data/clip-webvid-2.5M" "$ROOT/data/webvid_base.hnsw"
    build_hnsw "$ROOT/data/laion-10M" "$ROOT/data/laion_base.hnsw"
    build_hnsw "$ROOT/data/t2i-10M" "$ROOT/data/t2i_base.hnsw"
    ;;
  real)
    build_hnsw "$ROOT/data/clip-webvid-2.5M" "$ROOT/data/webvid_base.hnsw"
    ;;
  *)
    build_hnsw "$ROOT/data/$D_SUBDIR" "$D_INDEX"
    ;;
esac

# ============ 单数据集核心实验 ============
if [[ -n "$DATASET" && "$DATASET" != "all" && "$DATASET" != "real" ]]; then
  echo "==================== 数据准备（下载 + 合成 batch，$DATASET） ===================="
  bash "$ROOT/scripts/1_prepare_data.sh" "$DATASET"

  echo "==================== 训练自适应配置 ===================="
  DATA_DIR="$ROOT/data/$D_SUBDIR" INDEX="$D_INDEX" \
    TRAIN_PREFIX="$D_TRAIN" TEST_PREFIX="$D_TEST" \
    MODEL_OUT="$ROOT/model_out" \
    bash "$ROOT/scripts/4_train_models.sh"

  echo "==================== 运行 A-CORE ===================="
  DATA_DIR="$ROOT/data/$D_SUBDIR" INDEX="$D_INDEX" \
    TEST_PREFIX="$D_TEST" MODEL_DIR="$ROOT/model_out" TAG="$DATASET" \
    bash "$ROOT/scripts/5_run_acore.sh"

  echo ""
  echo "[ALL DONE] 结果见 $ROOT/results/${DATASET}_acore.csv"
  exit 0
fi

# ============ 真实数据集（TREC / WikiAnswers） ============
if [[ "$DATASET" == "real" ]]; then
  echo "==================== 数据准备（真实数据集：下载/编码/聚类/GT） ===================="
  bash "$ROOT/scripts/1_prepare_data.sh" real

  echo "==================== 运行 A-CORE（真实查询） ===================="
  if [ ! -d "$ROOT/model_out" ] || [ ! -f "$ROOT/model_out/model_A_efc_mono.txt" ]; then
    echo "[ERROR] 未找到已训练好的模型 $ROOT/model_out/，" >&2
    echo "        请先执行: bash scripts/reproduce_all.sh --dataset webvid" >&2
    exit 1
  fi
  DATA_DIR="$ROOT/data/clip-webvid-2.5M" INDEX="$ROOT/data/webvid_base.hnsw" \
    TEST_PREFIX="$ROOT/dataset_process/clip_topic_vectors_topic/s1" \
    MODEL_DIR="$ROOT/model_out" TAG="real_topic" \
    bash "$ROOT/scripts/5_run_acore.sh"

  echo ""
  echo "[ALL DONE] 结果见 $ROOT/results/real_topic_acore.csv"
  exit 0
fi

# ============ 全量（默认） ============
echo "==================== 数据准备（下载 + 合成 batch + 真实数据） ===================="
bash "$ROOT/scripts/1_prepare_data.sh"

echo "==================== 索引构建（NSG / τ-MNG） ===================="
bash "$ROOT/scripts/2_build_index.sh"

echo "==================== 训练自适应配置 ===================="
bash "$ROOT/scripts/4_train_models.sh"

echo "==================== 运行 A-CORE ===================="
bash "$ROOT/scripts/5_run_acore.sh"

echo "==================== 运行基线 ===================="
bash "$ROOT/scripts/6_run_baselines.sh"

echo ""
echo "[ALL DONE] 结果见 $ROOT/results/"
