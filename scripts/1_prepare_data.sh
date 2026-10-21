#!/usr/bin/env bash
# 数据准备：三大数据集（webvid / laion / t2i）下载 + 合成 OOD batch 生成 + 真实数据集（TREC/WikiAnswers）获取与格式转换
#
# 用法：
#   bash scripts/1_prepare_data.sh            # 全部（默认）
#   bash scripts/1_prepare_data.sh webvid     # 只下载 webvid + 生成合成 batch
#   bash scripts/1_prepare_data.sh laion      # 只下载 laion  + 生成合成 batch
#   bash scripts/1_prepare_data.sh t2i        # 只下载 t2i    + 生成合成 batch
#   bash scripts/1_prepare_data.sh real       # 只处理真实数据集（TREC/WikiAnswers）
#
# 说明：
#   - 下载幂等（文件已存在则跳过），wget -c 支持断点续传
#   - 三大数据集按仓库约定布局放置于 data/<dataset>/
#   - 合成 batch 用 make_clustered_queries5 生成（训练 200 簇 / 测试 50 簇）
#   - laion base 无现成单文件，需从 LAION-400M 图像嵌入 .npy 分片拼装（约 10 个分片，体积较大）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
DATASET="${1:-${DATASET:-all}}"

ZENODO="https://zenodo.org/records/11090378/files"
YANDEX="https://storage.yandexcloud.net/yandex-research/ann-datasets"
LAION_NPY="https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings/images"

mkdir -p "$DATA_DIR"

download() {
  local url="$1" out="$2"
  if [ -s "$out" ]; then
    echo "[SKIP] 已存在: $out"
  else
    mkdir -p "$(dirname "$out")"
    echo "[DOWNLOAD] $url"
    echo "        -> $out"
    wget -c "$url" -O "$out"
  fi
}

# ---------------- WebVid ----------------
prep_webvid() {
  local dir="$DATA_DIR/clip-webvid-2.5M"
  download "$ZENODO/clip.webvid.base.2.5M.fbin?download=1" "$dir/base.2.5M.fbin"
  download "$ZENODO/webvid.query.10k.fbin?download=1"   "$dir/query.10k.fbin"
}

# ---------------- t2i ----------------
prep_t2i() {
  local dir="$DATA_DIR/t2i-10M"
  download "$YANDEX/T2I/base.10M.fbin"          "$dir/base.10M.fbin"
  download "$YANDEX/T2I/query.public.100K.fbin" "$dir/query.public.100K.fbin"
}

# ---------------- LAION ----------------
prep_laion() {
  local dir="$DATA_DIR/laion-10M"
  download "$ZENODO/laion.query.10k.fbin?download=1" "$dir/query.10k.fbin"

  # base 向量：LAION-400M 图像嵌入按 .npy 分片发布（跳过 index 8），需要拼装成 base.10M.fbin
  if [ -s "$dir/base.10M.fbin" ]; then
    echo "[SKIP] 已存在: $dir/base.10M.fbin"
    return 0
  fi

  echo "[LAION] base 向量需从 .npy 分片拼装（体积较大，请耐心等待）"
  local i
  for i in 0 1 2 3 4 5 6 7 9 10; do
    download "$LAION_NPY/img_emb_${i}.npy" "$dir/img_emb_${i}.npy"
  done

  python3 - "$dir" <<'PY'
import os, sys
import numpy as np

out_dir = sys.argv[1]
idxs = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
files = [os.path.join(out_dir, f"img_emb_{i}.npy") for i in idxs]

# 维度取自第一个分片
dim = int(np.load(files[0], mmap_mode="r").shape[1])
out_path = os.path.join(out_dir, "base.10M.fbin")
total = 0
with open(out_path, "wb") as f:
    f.write(np.array([0, dim], dtype=np.int32).tobytes())  # 占位 n，最后回填
    for fn in files:
        x = np.load(fn, mmap_mode="r").astype(np.float32).reshape(-1, dim)
        total += x.shape[0]
        f.write(x.tobytes())
    f.seek(0)
    f.write(np.array([total, dim], dtype=np.int32).tobytes())
print(f"[OK] laion base 写入 {out_path} (n={total}, dim={dim})")
PY
}

# ---------------- 合成 OOD batch 生成 ----------------
gen_synthetic() {
  local bin="$ROOT/build/make_clustered_queries5"
  if [ ! -x "$bin" ]; then
    echo "[ERROR] 未找到 $bin，请先编译：" >&2
    echo "        mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j" >&2
    exit 1
  fi
  echo "==== 生成合成 OOD batch（make_clustered_queries5）===="
  local d="$DATASET"
  [[ "$d" == "all" ]] && d=""
  DATASET="$d" bash "$ROOT/scripts/3_cluster_queries.sh"
}

# ---------------- 真实数据集 ----------------
prep_real() {
  echo "==== 真实数据集（TREC/WikiAnswers）获取 + 格式转换 ===="
  bash "$ROOT/scripts/1b_prepare_real_data.sh"
}

# ---------------- 主流程 ----------------
case "$DATASET" in
  all)
    prep_webvid; prep_laion; prep_t2i
    gen_synthetic
    prep_real
    ;;
  webvid|laion|t2i)
    "prep_${DATASET}"
    gen_synthetic
    ;;
  real)
    prep_real
    ;;
  *)
    echo "未知数据集: $DATASET（可选 all/webvid/laion/t2i/real）" >&2
    exit 1
    ;;
esac

echo ""
echo "[OK] 数据准备完成。数据目录: $DATA_DIR"
