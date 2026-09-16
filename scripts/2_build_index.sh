#!/usr/bin/env bash
# 索引构建：HNSW / NSG / τ-MNG
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
THREADS="${THREADS:-20}"

echo "==== NSG 建图 ===="
# 依赖预计算的 KNN 图 (base.100NN.graph)
"$BIN/build_nsg" "$DATA_DIR/clip-webvid-2.5M/base.2.5M.fbin" "$DATA_DIR/clip-webvid.nsg" \
  "$DATA_DIR/knng/clip-webvid-2.5M/base.100NN.graph" 64 200 200 --threads "$THREADS"
"$BIN/build_nsg" "$DATA_DIR/laion-10M/base.10M.fbin" "$DATA_DIR/laion.nsg" \
  "$DATA_DIR/knng/laion-10M/base.100NN.graph" 64 200 200 --threads "$THREADS"

echo "==== τ-MNG 建图 ===="
"$BIN/tming_build" "$DATA_DIR/clip-webvid-2.5M/base.2.5M.fbin" "$DATA_DIR/clip-webvid-2.5M.mrng" \
  "$DATA_DIR/knng/clip-webvid-2.5M/base.100NN.graph" 64 200 1 --tau=0.01
"$BIN/tming_build" "$DATA_DIR/laion-10M/base.10M.fbin" "$DATA_DIR/laion-10M.mrng" \
  "$DATA_DIR/knng/laion-10M/base.100NN.graph" 64 200 1 --tau=0.01

echo "[OK] HNSW 索引请用 hnswlib 建图（生成 *.hnsw），或直接使用现成索引文件。"
