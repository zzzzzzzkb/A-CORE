#!/usr/bin/env bash
# 拉取并编译外部基线依赖（RoarGraph / NGFix / DiskANN(RVAMANA)）
# 用法：bash scripts/0_setup_external.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/third_party"
mkdir -p "$EXT"

echo "==== 以下外部依赖默认不构建，仅在你需要复现对应基线时安装 ===="

# ---------------- RoarGraph ----------------
if [ ! -d "$EXT/RoarGraph" ]; then
  echo "[RoarGraph] cloning ..."
  git clone --recursive https://github.com/matchyc/RoarGraph.git "$EXT/RoarGraph"
fi
echo "[RoarGraph] 编译时配置: -DROARGRAPH_DIR=$EXT/RoarGraph"

# ---------------- NGFix ----------------
if [ ! -d "$EXT/NGFix" ]; then
  echo "[NGFix] cloning ..."
  git clone --recursive https://github.com/BlindingStars/NGFix.git "$EXT/NGFix"
fi
echo "[NGFix] 编译时配置: -DNGFIX_DIR=$EXT/NGFix"

# ---------------- DiskANN (RVAMANA) ----------------
if [ ! -d "$EXT/DiskANN" ]; then
  echo "[DiskANN] cloning ..."
  git clone --recursive https://github.com/microsoft/DiskANN.git "$EXT/DiskANN"
fi
echo "[RVAMANA] 使用 DiskANN 的 RobustVamana 构建索引，见其 README"

echo ""
echo "安装完成后，重新配置并编译："
echo "  cd $ROOT/build"
echo "  cmake .. -DROARGRAPH_DIR=$EXT/RoarGraph -DNGFIX_DIR=$EXT/NGFix"
echo "  make -j"
