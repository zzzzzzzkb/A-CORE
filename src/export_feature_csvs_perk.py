#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 k = 1,10,100 分别提取特征并构造 (recall, 访问节点/配置) 前沿样本，输出到不同 CSV。

与 `export_feature_csvs.py` 类似，但这里针对每个 k：
1. 将对应的 recall 列映射为统一的 `recall_at_k` 供已有逻辑使用。
2. 将对应的 k 专属 overlap/jaccard 列映射为统一的 `overlap128_vs_256` / `jaccard128_vs_256`。
3. 调用现有的 `build_tables_for_rstars`（不训练，只做样本筛选），再把原始行（A/B/Gate/Rank 所引用的）导出。

输出文件示例（默认 k=1,10,100）：
  filtered_AB_rows_k1.csv
  filtered_TG_rows_k1.csv
  filtered_TR_rows_k1.csv
  ...(k10, k100 同理)

如果某个 k 的专用 overlap/jaccard 列缺失，则退化为使用通用列 `overlap128_vs_256` / `jaccard128_vs_256`。
"""

import argparse
import os
import pandas as pd
import numpy as np
from typing import List, Set

# 复用已有函数（不强制使用 load_runs，因为原 CSV 没有统一的 recall_at_k 列，需要我们按 k 动态映射）
from train_full_conditional_and_recall_4 import build_tables_for_rstars


# 本工具可能在较老的 pandas 环境运行（如 0.x），兼容 on_bad_lines / error_bad_lines 差异
def read_csv_robust(path: str, known_cols: Set[str]) -> pd.DataFrame:
    """更健壮地读取 CSV：
    - 先只读表头拿到可用列
    - 将 usecols 限定为我们已知/需要的列，避免异常行的额外分隔符影响
    - 若 C 引擎失败，则退回 Python 引擎并跳过异常行
    - 自动丢弃 Unnamed:* 列
    """
    # 先读取表头
    head = pd.read_csv(path, nrows=0)
    file_cols = [c for c in head.columns if not str(c).startswith("Unnamed:")]
    usecols = [c for c in file_cols if c in known_cols]
    if not usecols:
        # 如果交集为空，就使用文件中所有列，再在读完后做裁剪
        usecols = file_cols

    # 尝试 C 引擎
    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception:
        # 退回 Python 引擎，并兼容老版本 pandas 的参数
        try:
            df = pd.read_csv(path, usecols=usecols, engine="python", on_bad_lines="warn")
        except TypeError:
            # 老版本 pandas（<1.3）
            df = pd.read_csv(path, usecols=usecols, engine="python", error_bad_lines=False, warn_bad_lines=True)  # type: ignore[arg-type]

    # 丢掉 Unnamed 列（以防）并只保留需要列
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed:")]]
    df = df[[c for c in df.columns if c in known_cols or True]]  # 已经限定 usecols，这里只是确保无异常
    return df


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="输入包含 recall_at_1/10/100 等列的运行记录 CSV")
    ap.add_argument("--outdir", required=True, help="输出目录")
    ap.add_argument("--rstars", type=str, default="0.80,0.85,0.88,0.90,0.92,0.94,0.96,0.98",
                    help="逗号分隔的 R* 列表，用于前沿筛选")
    ap.add_argument("--delta", type=float, default=0.003,
                    help="前沿可行条件：recall>=R*+delta 时 (efc,efw) 可行")
    ap.add_argument("--gain_eps", type=float, default=0.002,
                    help="Gate/Rank 的敏感性与饱和带阈值")
    ap.add_argument("--ks", type=str, default="1,10,100",
                    help="需要处理的 k 列表，对应 recall_at_{k} 以及 overlap/jaccard 的 k 专属列")
    return ap.parse_args()


def ensure_columns(df: pd.DataFrame, required: List[str]):
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺少必要列: {miss}")


def make_view_for_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """为指定 k 构造一个视图：
    - recall_at_k <- recall_at_{k}
    - overlap128_vs_256 / jaccard128_vs_256 <- 对应 k 专属列（若存在）
    - 删除其它 k 的 recall/overlap/jaccard 专属列，仅保留当前 k 与通用列。
    """
    df_k = df.copy()
    recall_col = f"recall_at_{k}"
    if recall_col not in df_k.columns:
        raise ValueError(f"找不到列: {recall_col}")
    df_k["recall_at_k"] = df_k[recall_col].astype(float)

    # overlap/jaccard：若存在 k 专属列，则先删除通用列再以重命名方式替换；否则保留通用列
    overlap_k = f"overlap128_vs_256_k{k}"
    jaccard_k = f"jaccard128_vs_256_k{k}"
    if overlap_k in df_k.columns:
        # 先删通用列，再用 k 专属列替换为通用名
        df_k = df_k.drop(columns=["overlap128_vs_256"], errors="ignore")
        df_k.rename(columns={overlap_k: "overlap128_vs_256"}, inplace=True)
        df_k["overlap128_vs_256"] = pd.to_numeric(df_k["overlap128_vs_256"], errors="coerce")
    if jaccard_k in df_k.columns:
        df_k = df_k.drop(columns=["jaccard128_vs_256"], errors="ignore")
        df_k.rename(columns={jaccard_k: "jaccard128_vs_256"}, inplace=True)
        df_k["jaccard128_vs_256"] = pd.to_numeric(df_k["jaccard128_vs_256"], errors="coerce")

    # 删除非当前 k 的 per-k 列（recall/overlap/jaccard）
    # 注意保留映射后的 recall_at_k，不要误删
    drop_cols = []
    for c in df_k.columns:
        if c.startswith("recall_at_") and c != "recall_at_k":
            drop_cols.append(c)
        if c.startswith("overlap128_vs_256_k") and c != overlap_k:
            drop_cols.append(c)
        if c.startswith("jaccard128_vs_256_k") and c != jaccard_k:
            drop_cols.append(c)
    if drop_cols:
        df_k = df_k.drop(columns=drop_cols, errors="ignore")

    return df_k


def export_filtered_rows(df: pd.DataFrame, TA: pd.DataFrame, TB: pd.DataFrame, TG: pd.DataFrame, TR: pd.DataFrame,
                         outdir: str, k: int):
    """复用现有脚本里的行提取逻辑，输出带 k 后缀的文件。"""
    orig_cols = df.columns.tolist()

    # A/B: 使用 TA/TB 的 orig_idx
    if "orig_idx" in TA.columns or "orig_idx" in TB.columns:
        idxs_ab = []
        if "orig_idx" in TA.columns:
            idxs_ab.extend(TA["orig_idx"].dropna().astype(int).tolist())
        if "orig_idx" in TB.columns:
            idxs_ab.extend(TB["orig_idx"].dropna().astype(int).tolist())
        seen = set(); ordered = []
        for x in idxs_ab:
            if x not in seen:
                seen.add(x); ordered.append(x)
        df_ab = df.loc[ordered] if ordered else pd.DataFrame(columns=orig_cols)
    else:
        df_ab = pd.DataFrame(columns=orig_cols)
    ab_path = os.path.join(outdir, f"filtered_AB_rows_k{k}.csv")
    df_ab.to_csv(ab_path, index=False)

    # Gate 样本
    if "orig_idx" in TG.columns:
        idxs_tg = TG["orig_idx"].dropna().astype(int).tolist()
        seen = set(); ordered = []
        for x in idxs_tg:
            if x not in seen:
                seen.add(x); ordered.append(x)
        df_tg = df.loc[ordered] if ordered else pd.DataFrame(columns=orig_cols)
    else:
        df_tg = pd.DataFrame(columns=orig_cols)
    tg_path = os.path.join(outdir, f"filtered_TG_rows_k{k}.csv")
    df_tg.to_csv(tg_path, index=False)

    # Rank 样本
    if "orig_idx" in TR.columns:
        idxs_tr = TR["orig_idx"].dropna().astype(int).tolist()
        seen = set(); ordered = []
        for x in idxs_tr:
            if x not in seen:
                seen.add(x); ordered.append(x)
        df_tr = df.loc[ordered] if ordered else pd.DataFrame(columns=orig_cols)
    else:
        df_tr = pd.DataFrame(columns=orig_cols)
    tr_path = os.path.join(outdir, f"filtered_TR_rows_k{k}.csv")
    df_tr.to_csv(tr_path, index=False)

    print(f"[k={k}] written: {ab_path} (rows={len(df_ab)}), {tg_path} (rows={len(df_tg)}), {tr_path} (rows={len(df_tr)})")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    # 我们仅关心这些列（其余列即使存在也不参与本次导出）
    known_cols: Set[str] = set([
        "cluster_id","cluster_size","cluster_density",
        "cluster_radius_p50","cluster_radius_p90","radius_skew",
        "dist_centroid_to_entryL0","entry_dist_norm","dist_centroid_top1_smallEF",
        # overlap/jaccard（通用 + per-k）
        "overlap128_vs_256","jaccard128_vs_256",
        "overlap128_vs_256_k1","overlap128_vs_256_k10","overlap128_vs_256_k100",
        "jaccard128_vs_256_k1","jaccard128_vs_256_k10","jaccard128_vs_256_k100",
        # 其他可选指标（不一定使用，但保留以便下游分析）
        "lid_probe256_k10","rc_probe256_k10","expansion2k_over_k_probe256_k10",
        "lid_k_efc","rc_k_efc","expand2k_over_k_efc",
        # 搜索参数/性能
        "efc","ef_warm","L_aligned","qps_run",
        # recall per-k
        "recall_at_1","recall_at_10","recall_at_100",
        # tie-breaker 择优字段
        "centroid_steps","visited_mean",
    ])

    df = read_csv_robust(args.csv, known_cols)

    # 构造与训练脚本 load_runs 同步的派生列
    if "cluster_size" in df.columns:
        df["cluster_size"] = pd.to_numeric(df["cluster_size"], errors="coerce")
        if "log1p_cluster_size" not in df.columns:
            df["log1p_cluster_size"] = np.log1p(df["cluster_size"].astype(float))
        if "cluster_size_log" not in df.columns:
            df["cluster_size_log"] = np.log1p(df["cluster_size"].astype(float))

    # 基础必需列（不含 recall_at_k，因为我们会动态映射）
    base_required = [
        "cluster_id","cluster_size","cluster_density",
        "cluster_radius_p50","cluster_radius_p90","radius_skew",
        "dist_centroid_to_entryL0","entry_dist_norm","dist_centroid_top1_smallEF",
        "efc","ef_warm","L_aligned","qps_run"
    ]
    ensure_columns(df, base_required)

    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    R_grid = [float(x) for x in args.rstars.split(',') if x.strip()]

    print(f"Processing ks={ks} with R* grid={R_grid}, delta={args.delta}, gain_eps={args.gain_eps}")

    for k in ks:
        print(f"\n=== [k={k}] BUILD FRONTIER TABLES ===")
        df_k = make_view_for_k(df, k)
        # 调用现有逻辑构造样本
        TA, TB, TG, TR, feat_base = build_tables_for_rstars(
            df_k, R_grid, delta=args.delta, gain_eps=args.gain_eps, dedup_same_cfg=True, show_progress=False
        )
        print(f"[k={k}] Rows: A={len(TA)} B={len(TB)} Gate={len(TG)} Rank={len(TR)}")
        export_filtered_rows(df_k, TA, TB, TG, TR, args.outdir, k)

    print("\n=== DONE (per-k export) ===")


if __name__ == "__main__":
    main()
