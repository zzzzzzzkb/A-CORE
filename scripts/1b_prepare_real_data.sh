#!/usr/bin/env bash
# 真实数据集（TREC SessionTrack / WikiAnswers）预处理：文本 → 格式转换 → CLIP 编码 → 聚类 → GT
# 用法：bash scripts/1b_prepare_real_data.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DP="$ROOT/dataset_process"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
BASE_FBIN="${BASE_FBIN:-$DATA_DIR/clip-webvid-2.5M/base.2.5M.fbin}"

echo "==== 真实数据预处理（依赖 torch / open_clip / requests，见 requirements.txt）===="
cd "$DP"

# ---------- TREC Session 2014 ----------
echo "[TREC] ① XML → JSONL"
# 需要先下载 TREC Session 2014 数据（https://trec.nist.gov/data/session2014.html）解压出 XML 放到 dataset_process/ 下
if [ -f sessiontrack2014.xml ]; then
  python convert_sessiontrack_to_jsonl.py sessiontrack2014.xml
else
  echo "[SKIP] 未找到 sessiontrack2014.xml，请先下载 TREC Session 2014 原始 XML 到 $DP/"
fi

echo "[TREC] ② CLIP 编码 + KMeans 聚类"
if [ -f topic_queries.jsonl ]; then
  python encode_topic_queries_clip.py \
    --input topic_queries.jsonl \
    --outdir clip_topic_vectors --prefix s1 --clusters 50
fi

echo "[TREC] ③ 严格 topic 分簇版本"
if [ -f clip_topic_vectors/s1.xq.fbin ]; then
  python build_topic_cluster_version.py \
    --topic-jsonl topic_queries.jsonl \
    --xq clip_topic_vectors/s1.xq.fbin \
    --queries-csv clip_topic_vectors/s1.queries.csv \
    --outdir clip_topic_vectors_topic --prefix s1
fi

# ---------- WikiAnswers ----------
echo "[WikiAnswers] 流式下载 + CLIP 编码（每行 = 1 簇）"
python encode_wikianswers_clip.py

# ---------- 计算 GT ----------
echo "[GT] 对真实查询在 base 上暴力计算 ground truth"
for PREFIX in clip_topic_vectors_topic/s1 clip_wikianswers_vectors/s1; do
  if [ -f "$DP/$PREFIX.xq.fbin" ]; then
    "$ROOT/build/make_clustered_queries5" --gt_only \
      --out_prefix "$DP/$PREFIX" \
      --base "$BASE_FBIN" \
      --k_gt 100 --normalize_base 0 --nthreads 24
  fi
done

echo ""
echo "[OK] 真实数据转换完成。产物目录："
echo "  $DP/clip_topic_vectors/           （KMeans 分簇）"
echo "  $DP/clip_topic_vectors_topic/     （严格 topic 分簇）"
echo "  $DP/clip_wikianswers_vectors/     （WikiAnswers）"
echo ""
echo "用 --load_synth_prefix 指向对应 s1 前缀即可用于 A-CORE / 基线评测，例如："
echo "  --load_synth_prefix $DP/clip_topic_vectors_topic/s1"
