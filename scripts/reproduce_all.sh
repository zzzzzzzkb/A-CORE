#!/usr/bin/env bash
# 一键复现：数据 -> 索引 -> 聚类 -> 训练 -> A-CORE 查询 -> 基线
# 用法：bash scripts/reproduce_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
