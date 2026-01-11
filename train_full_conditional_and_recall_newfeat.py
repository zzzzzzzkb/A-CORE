#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A-CORE model training script.

This script trains the LightGBM models used in A-CORE:
    - Head A predicts the construction ef (efc),
    - Head B predicts the warm-search ef (ef_warm),
    - Head Rank-M predicts m* = L_aligned * ef_warm.

Input:
    A CSV file produced by run_train_get_all_k_onlytop.cpp, containing
    per-cluster features and recall statistics.

Output:
    A set of LightGBM TXT models and auxiliary metadata, which are
    later consumed by run_ACORE.cpp.

Feature ablations are controlled through the ABLATION_KEEP list and
the command-line flags --drop / --keep / --keep_file.

For end-to-end usage, see the README accompanying this code package.
"""

import argparse, os, json
from typing import List
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error

# ============ 消融：保留列表（初始化包含所有可用特征） ============
# 默认全保留；可通过命令行 --drop / --keep / --keep_file 覆盖
DEFAULT_ABLATION_KEEP: List[str] = [
    # 基础簇与几何
    "cluster_size", "cluster_size_log", "log1p_cluster_size",
    "cluster_density", 
    "cluster_radius_p50", "cluster_radius_p90", "radius_skew",
    "dist_centroid_to_entryL0", "entry_dist_norm", "dist_centroid_top1_smallEF",
    "overlap128_vs_256", "jaccard128_vs_256",
    # probe 相关
    "lid_probe256_k10", "rc_probe256_k10", "expansion2k_over_k_probe256_k10",
    # efc 相关输入
    "lid_k_efc", "rc_k_efc", "expand2k_over_k_efc",
    # 由目标派生的输入
    "log_efc", "log_efw", "efw_over_efc",
    # 目标本身在部分 head 被当作输入
    "efc", "ef_warm",
    # 任务目标
    "recall_at_k",
]
ABLATION_KEEP: List[str] = DEFAULT_ABLATION_KEEP.copy()
# -------- Columns（与原脚本一致） --------
REQ_BASE = [
    "cluster_id",
    "cluster_size","cluster_density",
    "cluster_radius_p50","cluster_radius_p90","radius_skew",
    "dist_centroid_to_entryL0","entry_dist_norm",
    "dist_centroid_top1_smallEF",
    "overlap128_vs_256","jaccard128_vs_256",
    "efc","ef_warm","L_aligned",
    "recall_at_k","qps_run"
]
NEW_PROBE = ["lid_probe256_k10","rc_probe256_k10","expansion2k_over_k_probe256_k10"]
NEW_EFC   = ["lid_k_efc","rc_k_efc","expand2k_over_k_efc","expand2k_over_k_efc"]  # 兼容两种命名
DERIVED_FILL_ORDER = [
    "cluster_size","cluster_density",
    "cluster_radius_p50","cluster_radius_p90","radius_skew",
    "dist_centroid_to_entryL0","entry_dist_norm",
    "dist_centroid_top1_smallEF",
    "overlap128_vs_256","jaccard128_vs_256",
    "efc","ef_warm","L_aligned",
] + NEW_PROBE + NEW_EFC

# -------- Utils（与原脚本一致） --------
def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fillna_and_clip(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].replace([np.inf, -np.inf], np.nan)
        if s.isna().all():
            df[c] = 0.0
            continue
        med = np.nanmedian(s.values.astype(float))
        if not np.isfinite(med):
            med = 0.0
        df[c] = s.fillna(med).astype(float)
    return df

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    if "cluster_size_log" not in df.columns:
        df["cluster_size_log"] = np.log1p(df["cluster_size"].astype(float))
    if "log1p_cluster_size" not in df.columns:
        df["log1p_cluster_size"] = np.log1p(df["cluster_size"].astype(float))

    df["log_efc"] = np.log1p(df["efc"].clip(lower=0).astype(float))
    df["log_efw"] = np.log1p(df["ef_warm"].clip(lower=0).astype(float))
    df["efw_over_efc"] = df["ef_warm"].astype(float) / np.maximum(1.0, df["efc"].astype(float))

    for c in NEW_PROBE + ["lid_k_efc","rc_k_efc","expand2k_over_k_efc"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            lo, hi = np.nanpercentile(s, 1), np.nanpercentile(s, 99)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                df[c] = s.clip(lower=lo, upper=hi)
            else:
                df[c] = s
    return df

# --- 新增：回归（带单调） ---
'''
    params = dict(
        objective=("huber" if use_huber else "regression"),
        n_estimators=1600, learning_rate=0.03,
        num_leaves=63, subsample=0.9, colsample_bytree=0.9,
        min_data_in_leaf=48, min_gain_to_split=1e-2,
        reg_lambda=3.5, max_bin=255, random_state=seed,
'''
def lgb_reg_fit(X, y, seed=42, mono=None, use_huber=True):
    params = dict(
        objective=("huber" if use_huber else "regression"),
        n_estimators=1600, learning_rate=0.03,
        num_leaves=63, subsample=0.9, colsample_bytree=0.9,
        min_data_in_leaf=48, min_gain_to_split=1e-2,
        reg_lambda=3.5, max_bin=255, random_state=seed,
    )
    if mono is not None:
        params["monotone_constraints"] = mono
        params["monotone_constraints_method"] = "advanced"
        params["monotone_penalty"] = 1.0

    model = lgb.LGBMRegressor(**params)
    y_tr = np.log1p(y.astype(float))
    model.fit(X, y_tr)
    pred = np.expm1(model.predict(X))
    mae = float(mean_absolute_error(y, pred))
    return model, mae, params

def lgb_quantile_fit(X: pd.DataFrame, y: np.ndarray, alpha=0.85, seed=42):
    params = dict(
        objective="quantile", alpha=alpha,
        n_estimators=1600, learning_rate=0.03,
        num_leaves=63, subsample=0.9, colsample_bytree=0.9,
        min_data_in_leaf=24, min_gain_to_split=1e-3,
        reg_lambda=2.5, max_bin=255, random_state=seed
    )
    model = lgb.LGBMRegressor(**params)
    y_tr = np.log1p(y.astype(float))
    model.fit(X, y_tr)
    pred = np.expm1(model.predict(X))
    mae = float(mean_absolute_error(y, pred))
    return model, mae, params


def save_feature_importance(model, cols: List[str], path_csv: str):
    try:
        imp = model.booster_.feature_importance(importance_type="gain")
        df_imp = pd.DataFrame({"feature": cols, "gain": imp})
        df_imp.sort_values("gain", ascending=False).to_csv(path_csv, index=False)
    except Exception:
        pass

# === 新增：构造单调性约束向量的工具 ===
def build_monotone_vector(cols, rules):
    """
    cols: 本模型用到的特征列顺序
    rules: dict[str -> -1|0|+1]
    未出现在 rules 的列默认 0（不约束）
    """
    return [int(rules.get(c, 0)) for c in cols]

# === 新增：根据 ABLATION_KEEP 过滤特征 ===
def apply_ablation(feat_list: List[str]) -> List[str]:
    return [f for f in feat_list if f in ABLATION_KEEP]

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop", type=str, default="",
                help="逗号/空格分隔的特征名列表；从默认集合中删除这些特征")
    ap.add_argument("--keep", type=str, default="",
                    help="逗号/空格分隔的特征名列表；仅保留这些特征（若与 --drop 同时给，先 keep 后 drop）")
    ap.add_argument("--keep_file", type=str, default="",
                    help="包含要保留特征名单的文本文件（每行一个）；与 --keep 取并集")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
 # ---- 解析 --drop/--keep/--keep_file 并覆盖全局 ABLATION_KEEP ----
    # ---- 解析 --drop/--keep/--keep_file 并覆盖全局 ABLATION_KEEP ----
    def _split_list(s: str) -> List[str]:
        # 逗号/空格分隔，去重但保序
        toks = [x.strip() for tok in s.replace(",", " ").split() for x in [tok] if x.strip()]
        seen, out = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    # 起点：默认全保留
    keep_set: set = set(DEFAULT_ABLATION_KEEP)

    # 收集 keep 源（命令行 + 文件），所有分支都有默认值，避免未定义
    keep_from_cli: set = set(_split_list(args.keep)) if args.keep else set()
    keep_from_file: set = set()
    if args.keep_file:
        with open(args.keep_file, "r", encoding="utf-8") as f:
            keep_from_file = {ln.strip() for ln in f if ln.strip()}

    if keep_from_cli or keep_from_file:
        # 只保留 DEFAULT 中存在的特征，避免拼写错误引入垃圾名
        keep_set = (keep_from_cli | keep_from_file) & set(DEFAULT_ABLATION_KEEP)
        if not keep_set:
            raise ValueError("keep 集合为空：请检查 --keep/--keep_file 内容是否为有效特征名")

    # 应用 drop（同样只在 DEFAULT 中有效）
    if args.drop:
        for f in _split_list(args.drop):
            if f in keep_set:
                keep_set.remove(f)

    if not keep_set:
        raise ValueError("消融后 ABLATION_KEEP 为空，请至少保留一个特征")

    # 以 DEFAULT 的原始顺序稳定输出（避免特征顺序随 set 打乱）
    global ABLATION_KEEP
    ABLATION_KEEP = [f for f in DEFAULT_ABLATION_KEEP if f in keep_set]

    # 这里保持与原脚本一致的 need（数据文件通常已有所有列；消融仅影响训练特征的使用）
    need = set(REQ_BASE + NEW_PROBE + ["lid_k_efc","rc_k_efc","expand2k_over_k_efc"])
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 缺少字段: {miss}")

    df = ensure_numeric(df, list(need) + ["centroid_steps","visited_mean"])
    df = fillna_and_clip(df, DERIVED_FILL_ORDER)
    df = add_derived(df)

    feat_base = [
        "cluster_size","cluster_size_log","log1p_cluster_size",
        "cluster_density","cluster_radius_p50","cluster_radius_p90","radius_skew",
        "dist_centroid_to_entryL0","entry_dist_norm",
        "dist_centroid_top1_smallEF",
        "overlap128_vs_256","jaccard128_vs_256",
    ]

    # --- 基础方向规则（通用于三头） ---
    base_rules = {
        # 难→贵：+1
        "cluster_radius_p50": +1,
        "cluster_radius_p90": +1,
        "radius_skew": +1,
        "dist_centroid_to_entryL0": 0,
        "entry_dist_norm": 0,
        "dist_centroid_top1_smallEF": 0,
        # 易→便宜：-1
        "cluster_density": -1,
        "overlap128_vs_256": +1,
        "jaccard128_vs_256": +1,
        # 目标更高→更贵：+1
        "recall_at_k": +1,
        # cluster_size / log 特征：难以统一判断，保持 0
        "cluster_size": 0,
        "cluster_size_log": 0,
        "log1p_cluster_size": 0,
    }

    # --- A: efc ---
    feats_A_raw = feat_base + [
        "lid_probe256_k10","rc_probe256_k10","expansion2k_over_k_probe256_k10",
        "recall_at_k",
    ]
    feats_A = apply_ablation(feats_A_raw)
    if len(feats_A) == 0:
        raise ValueError("Head A(efc) 的特征在消融后为空，请在 ABLATION_KEEP 中至少保留一个与 A 相关的特征。")
    XA = df[feats_A]; yA = df["efc"].values.astype(float)

    rules_A = dict(base_rules)
    rules_A.update({
        "lid_probe256_k10": +1,
        "rc_probe256_k10": +1,
        "expansion2k_over_k_probe256_k10": +1,
    })
    mono_A = build_monotone_vector(feats_A, rules_A)

    mA, maeA, paramsA = lgb_quantile_fit(XA, yA, alpha=args.alpha, seed=args.seed)
    mA_mono, maeA_mono, paramsA_mono = lgb_reg_fit(XA, yA, seed=args.seed, mono=mono_A, use_huber=True)

    # --- B: ef_warm ---
    feats_B_raw = feat_base + [
        "lid_k_efc","rc_k_efc","expand2k_over_k_efc","efc","log_efc",
        "recall_at_k",
    ]
    feats_B = apply_ablation(feats_B_raw)
    if len(feats_B) == 0:
        raise ValueError("Head B(ef_warm) 的特征在消融后为空，请在 ABLATION_KEEP 中至少保留一个与 B 相关的特征。")
    XB = df[feats_B]; yB = df["ef_warm"].values.astype(float)

    rules_B = dict(base_rules)
    rules_B.update({
        "lid_k_efc": +1, "rc_k_efc": +1, "expand2k_over_k_efc": +1,
        # 候选越多，warm 需求不应更大 → 非增
        "efc": -1, "log_efc": -1,
    })
    mono_B = build_monotone_vector(feats_B, rules_B)
    mB_mono, maeB_mono, paramsB_mono = lgb_reg_fit(XB, yB, seed=args.seed, mono=mono_B, use_huber=True)
    mB, maeB, paramsB = lgb_quantile_fit(XB, yB, alpha=args.alpha, seed=args.seed)

    # --- Rank-M: m* = L_aligned * ef_warm ---
    feats_M_raw = feat_base + [
        "lid_k_efc","rc_k_efc","expand2k_over_k_efc",
        "efc","ef_warm","log_efc","log_efw","efw_over_efc",
        "recall_at_k",
    ]
    feats_M = apply_ablation(feats_M_raw)
    if len(feats_M) == 0:
        raise ValueError("Head Rank-M 的特征在消融后为空，请在 ABLATION_KEEP 中至少保留一个与 Rank-M 相关的特征。")
    XM = df[feats_M]
    yM = (df["L_aligned"].values.astype(float) * df["ef_warm"].values.astype(float))

    rules_M = dict(base_rules)
    rules_M.update({
        "lid_k_efc": +1, "rc_k_efc": +1, "expand2k_over_k_efc": +1,
        # m* 的“最小质量”对 efc 合理设定非增；对 ef_warm/比值不强约束（0），避免自相矛盾
        "efc": -1, "log_efc": -1,
        "ef_warm": 0, "log_efw": 0, "efw_over_efc": 0,
    })
    mono_M = build_monotone_vector(feats_M, rules_M)
    mM_mono, maeM_mono, paramsM_mono = lgb_reg_fit(XM, yM, seed=args.seed, mono=mono_M, use_huber=True)
    mM, maeM, paramsM = lgb_quantile_fit(XM, yM, alpha=args.alpha, seed=args.seed)

    # --- Save models ---
    pA = os.path.join(args.outdir, "model_A_efc.txt")
    mA_mono.booster_.save_model(os.path.join(args.outdir, "model_A_efc_mono.txt"))

    pB = os.path.join(args.outdir, "model_B_efw.txt")
    mB_mono.booster_.save_model(os.path.join(args.outdir, "model_B_efw_mono.txt"))

    pM = os.path.join(args.outdir, "model_rank_m.txt")  # align with C++ consumer
    mM_mono.booster_.save_model(os.path.join(args.outdir, "model_rank_m_mono.txt"))

    mA.booster_.save_model(pA)
    mB.booster_.save_model(pB)
    mM.booster_.save_model(pM)

    joblib.dump(mA, os.path.join(args.outdir, "model_A_efc.pkl"))
    joblib.dump(mA_mono, os.path.join(args.outdir, "model_A_efc_mono.pkl"))

    joblib.dump(mB, os.path.join(args.outdir, "model_B_efw.pkl"))
    joblib.dump(mB_mono, os.path.join(args.outdir, "model_B_efw_mono.pkl"))

    joblib.dump(mM, os.path.join(args.outdir, "model_rank_m.pkl"))
    joblib.dump(mM_mono, os.path.join(args.outdir, "model_rank_m_mono.pkl"))

    # feature importance (基于消融后实际使用的特征)
    save_feature_importance(mA, feats_A, os.path.join(args.outdir, "fi_A_efc.csv"))
    save_feature_importance(mB, feats_B, os.path.join(args.outdir, "fi_B_efw.csv"))
    save_feature_importance(mM, feats_M, os.path.join(args.outdir, "fi_rank_m.csv"))

    meta = {
        "ablation_keep": ABLATION_KEEP,
        "features_base_all": [
            "cluster_size","cluster_size_log","log1p_cluster_size",
            "cluster_density","cluster_radius_p50","cluster_radius_p90","radius_skew",
            "dist_centroid_to_entryL0","entry_dist_norm",
            "dist_centroid_top1_smallEF",
            "overlap128_vs_256","jaccard128_vs_256",
        ],
        "features_A_used": feats_A,
        "features_B_used": feats_B,
        "features_rank_m_used": feats_M,
        "alpha": args.alpha,
        "seed": args.seed,
        "mae_A": maeA,
        "mae_B": maeB,
        "mae_rank_m": maeM,
        "notes": "Rank-M predicts m* = L_aligned * ef_warm; features filtered by ABLATION_KEEP.",
    }
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("==== TRAIN SUMMARY (Ablation) ====")
    print(f"Rows={df.shape[0]}")
    print(f"MAE(A efc)     = {maeA:.3f}")
    print(f"MAE(B ef_warm) = {maeB:.3f}")
    print(f"MAE(Rank m*)   = {maeM:.3f}")
    print(f"Models saved to: {args.outdir}")
    print(f"[A] used feats: {feats_A}")
    print(f"[B] used feats: {feats_B}")
    print(f"[M] used feats: {feats_M}")
    for tag, model, feats in [("A", mA, feats_A), ("B", mB, feats_B), ("RankM", mM, feats_M)]:
        imp = model.booster_.feature_importance(importance_type="gain")
        order = np.argsort(-imp)[:10]
        print(f"[{tag}] top-10: " + ", ".join(f"{feats[i]}:{imp[i]:.1f}" for i in order))

    print("=======================")
    print(f"MAE_MONO(A efc)     = {maeA_mono:.3f}")
    print(f"MAE_MONO(B ef_warm) = {maeB_mono:.3f}")
    print(f"MAE_MONO(Rank m*)   = {maeM_mono:.3f}")

if __name__ == "__main__":
    main()
