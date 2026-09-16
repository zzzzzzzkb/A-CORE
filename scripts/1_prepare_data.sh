#!/usr/bin/env bash
# 数据准备：将原始向量/查询/GT 转成 A-CORE 需要的 .fbin/.ibin 布局
# 数据集较大（LAION/T2I 各 10M、WebVid 2.5M），请按实际情况修改路径。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
mkdir -p "$DATA_DIR"

echo "[INFO] 数据目录: $DATA_DIR"
echo ""
echo "请将数据集按如下布局放置（向量为 .fbin，GT 为 .gt.ibin）："
cat <<EOF

$DATA_DIR/
├── clip-webvid-2.5M/
│   ├── base.2.5M.fbin          # 2.5M x 512 float32
│   └── query.10k.fbin          # 真实查询，用于生成合成 batch
├── laion-10M/
│   ├── base.10M.fbin           # 10M x 512
│   └── query.10k.fbin
└── t2i-10M/
    ├── base.10M.fbin           # 10M x 200
    └── query.public.100K.fbin
EOF

echo ""
echo "提示："
echo "  - 向量由 CLIP ViT-B/32 编码并 L2 归一化（t2i 归一化 base=1，WebVid/LAION 归一化 base=0）"
echo "  - 若使用 numpy 导出 .fbin，可参考 src/ 中现有导出逻辑："
echo "      4 字节 n + 4 字节 d + n*d 个 float32"
