#!/usr/bin/env bash
# 端到端复现：数据 -> 索引 -> 聚类 -> 训练 -> A-CORE 查询 -> 基线
# 前置条件：已按 README「数据准备」「Step 1」准备好数据集与索引（.fbin / .hnsw / KNN graph）
# 用法：bash scripts/reproduce_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============ Preflight 检查：前置数据集 / 索引是否就绪 ============
preflight() {
  local missing=()
  local need=(
    # 数据集 base + 查询（README「数据准备」）
    "$ROOT/data/clip-webvid-2.5M/base.2.5M.fbin"
    "$ROOT/data/clip-webvid-2.5M/query.10k.fbin"
    "$ROOT/data/laion-10M/base.10M.fbin"
    "$ROOT/data/laion-10M/query.10k.fbin"
    "$ROOT/data/t2i-10M/base.10M.fbin"
    "$ROOT/data/t2i-10M/query.public.100K.fbin"
    # HNSW 索引（README「Step 1」）
    "$ROOT/data/webvid_base.hnsw"
    "$ROOT/data/laion_base.hnsw"
    "$ROOT/data/t2i_base.hnsw"
    # NSG / τ-MNG 建图所需的 KNN graph（README「Step 1」）
    "$ROOT/data/knng/clip-webvid-2.5M/base.100NN.graph"
    "$ROOT/data/knng/laion-10M/base.100NN.graph"
  )
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

echo "==================== 数据准备 ===================="
bash "$ROOT/scripts/1_prepare_data.sh"

echo "==================== 索引构建 ===================="
bash "$ROOT/scripts/2_build_index.sh"

echo "==================== 聚类生成 batch ===================="
bash "$ROOT/scripts/3_cluster_queries.sh"

echo "==================== 训练自适应配置 ===================="
bash "$ROOT/scripts/4_train_models.sh"

echo "==================== 运行 A-CORE ===================="
bash "$ROOT/scripts/5_run_acore.sh"

echo "==================== 运行基线 ===================="
bash "$ROOT/scripts/6_run_baselines.sh"

echo ""
echo "[ALL DONE] 结果见 $ROOT/results/"
