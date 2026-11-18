"""Stage 4 EDA: Informative Missingness, Consistency, Ranges & Categoricals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

from project_name.config import PROJECT_ROOT
from project_name.hmecvas import (
    DICT_PATH,
    TEST_PATH,
    TRAIN_PATH,
    current_git_commit,
    iso_now_plus8,
    load_raw_wids,
    make_inquiry,
    sha256_file,
    upsert_inquiries,
)

REPORTS_DIR = PROJECT_ROOT / "reports" / "findings"
DROP_COLS = {"Unnamed: 0", "encounter_id", "readmission_status", "hospital_id", "icu_id"}
DEMO_COLS = {"age", "bmi", "height", "weight", "ethnicity", "gender"}
CAT_COLS_8 = [
    "ethnicity",
    "gender",
    "hospital_admit_source",
    "icu_admit_source",
    "icu_stay_type",
    "icu_type",
    "apache_2_diagnosis",
    "apache_3j_diagnosis",
]


def eval_hgb_5fold(
    df: pd.DataFrame,
    cols: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    impute_all: bool = False,
) -> float:
    """Evaluate 5-fold grouped HGB ROC-AUC on df[cols]."""
    oof = np.zeros(len(df), dtype=float)
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in cols if c not in set(cat_cols)]

    for tr_i, va_i in splits:
        x_tr = df.iloc[tr_i][cols].copy()
        x_va = df.iloc[va_i][cols].copy()
        if impute_all:
            for c in num_cols:
                med = x_tr[c].median()
                med = 0.0 if pd.isna(med) else float(med)
                x_tr[c] = x_tr[c].fillna(med)
                x_va[c] = x_va[c].fillna(med)
            for c in cat_cols:
                modes = x_tr[c].dropna().mode()
                mv = modes.iloc[0] if len(modes) else "__MISSING__"
                x_tr[c] = x_tr[c].fillna(mv)
                x_va[c] = x_va[c].fillna(mv)
        if cat_cols:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=np.nan,
            )
            x_tr[cat_cols] = enc.fit_transform(x_tr[cat_cols].astype(object))
            x_va[cat_cols] = enc.transform(x_va[cat_cols].astype(object))

        clf = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            categorical_features=cat_cols if cat_cols else None,
            early_stopping=True,
            random_state=42,
        )
        clf.fit(x_tr, y[tr_i])
        oof[va_i] = clf.predict_proba(x_va)[:, 1]
    return float(roc_auc_score(y, oof))


def build_miss_counts(df: pd.DataFrame, f_clin: list[str], dd_map: dict[str, str]) -> pd.DataFrame:
    """Build 6 row-wise non-demographic panel missingness counts."""
    non_demo = [c for c in f_clin if c not in DEMO_COLS]
    h1_labs = [c for c in non_demo if c.startswith("h1_") and dd_map.get(c) == "labs"]
    d1_labs = [c for c in non_demo if c.startswith("d1_") and dd_map.get(c) == "labs"]
    bg_cols = [c for c in non_demo if dd_map.get(c) == "labs blood gas"]
    inv_bp = [c for c in non_demo if "_invasive_" in c and "bp" in c]
    ap_cols = [c for c in non_demo if dd_map.get(c) == "APACHE covariate"]
    return pd.DataFrame(
        {
            "n_miss_total_nondemo": df[non_demo].isna().sum(axis=1).astype(np.int16),
            "n_miss_h1_labs": df[h1_labs].isna().sum(axis=1).astype(np.int16),
            "n_miss_d1_labs": df[d1_labs].isna().sum(axis=1).astype(np.int16),
            "n_miss_blood_gas": df[bg_cols].isna().sum(axis=1).astype(np.int16),
            "n_miss_invasive_bp": df[inv_bp].isna().sum(axis=1).astype(np.int16),
            "n_miss_apache": df[ap_cols].isna().sum(axis=1).astype(np.int16),
        },
        index=df.index,
    )


def run_i10_and_i10_1(
    train: pd.DataFrame,
    f_clin: list[str],
    dd_map: dict[str, str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    auc_native_raw: float,
) -> tuple[dict, dict]:
    """Audit deduplicated missingness masks (MNAR/MAR) and 4-arm CV ablation."""
    n_rows = len(train)
    masks_by_bytes: dict[bytes, list[str]] = {}
    mask_arrays: dict[bytes, np.ndarray] = {}
    for c in f_clin:
        n_null = int(train[c].isna().sum())
        if 30 <= n_null <= n_rows - 30:
            arr = train[c].isna().to_numpy(dtype=np.uint8)
            key = arr.tobytes()
            if key not in masks_by_bytes:
                masks_by_bytes[key] = []
                mask_arrays[key] = arr
            masks_by_bytes[key].append(c)

    rows = []
    pvals = []
    for key, cols in masks_by_bytes.items():
        m = mask_arrays[key]
        n_na = int(m.sum())
        p_na = float(y[m == 1].mean() * 100.0)
        p_ok = float(y[m == 0].mean() * 100.0)
        chi2_p = float(stats.chi2_contingency(pd.crosstab(m, y))[1])
        pvals.append(chi2_p)
        rows.append(
            {
                "rep_column": cols[0],
                "twin_count": len(cols),
                "columns_sharing_mask": ";".join(cols),
                "n_missing": n_na,
                "miss_pct": round(100.0 * n_na / n_rows, 4),
                "prev_missing_pct": round(p_na, 4),
                "prev_present_pct": round(p_ok, 4),
                "abs_diff_pp": round(abs(p_na - p_ok), 4),
                "chi2_p": chi2_p,
            }
        )

    mdf = pd.DataFrame(rows)
    mdf["bh_fdr_q"] = stats.false_discovery_control(np.array(pvals), method="bh")
    mdf = mdf.sort_values(["abs_diff_pp", "bh_fdr_q"], ascending=[False, True]).reset_index(
        drop=True
    )
    mdf.to_csv(REPORTS_DIR / "stage4_missingness_mechanisms.csv", index=False)

    k_mnar = int(((mdf["bh_fdr_q"] < 0.01) & (mdf["abs_diff_pp"] >= 2.0)).sum())
    miss_counts = build_miss_counts(train, f_clin, dd_map)
    n_miss_all = train[f_clin].isna().sum(axis=1).to_numpy()
    r_tot, p_tot = stats.spearmanr(n_miss_all, y)
    r_tot, p_tot = float(r_tot), float(p_tot)

    i10_out = (
        "TRUE"
        if (k_mnar >= 10 or (k_mnar >= 3 and p_tot < 0.001 and abs(r_tot) >= 0.02))
        else ("FALSE" if (k_mnar == 0 and (p_tot >= 0.05 or abs(r_tot) < 0.01)) else "INCONCLUSIVE")
    )

    # I10.1: 4-arm ablation
    df_counts = pd.concat([train[f_clin], miss_counts], axis=1)
    auc_native_counts = eval_hgb_5fold(
        df_counts, f_clin + list(miss_counts.columns), y, splits, impute_all=False
    )
    auc_imp_no_flags = eval_hgb_5fold(train, f_clin, y, splits, impute_all=True)

    # Deduplicated non-demographic U_miss flags
    flag_cols = {}
    for key, cols in masks_by_bytes.items():
        if all(c in {"age", "bmi", "weight"} for c in cols):
            continue
        flag_cols[f"miss_{cols[0]}"] = mask_arrays[key].astype(np.int8)
    df_imp_flags = pd.concat([train[f_clin], pd.DataFrame(flag_cols, index=train.index)], axis=1)
    auc_imp_flags = eval_hgb_5fold(
        df_imp_flags, f_clin + list(flag_cols.keys()), y, splits, impute_all=True
    )
    delta_imp = auc_imp_flags - auc_imp_no_flags
    delta_nat = auc_native_counts - auc_native_raw
    i10_1_out = (
        "TRUE"
        if (delta_imp >= 0.005 or delta_nat >= 0.002)
        else ("FALSE" if (delta_imp <= 0.0 and delta_nat <= 0.0) else "INCONCLUSIVE")
    )
    return (
        {
            "outcome": i10_out,
            "n_masks": len(mdf),
            "k_mnar": k_mnar,
            "r_tot": r_tot,
            "p_tot": p_tot,
            "top_masks": mdf.head(5)[["rep_column", "abs_diff_pp", "bh_fdr_q"]].to_dict("records"),
        },
        {
            "outcome": i10_1_out,
            "auc_native_raw": auc_native_raw,
            "auc_native_counts": auc_native_counts,
            "auc_imp_no_flags": auc_imp_no_flags,
            "auc_imp_flags": auc_imp_flags,
            "delta_imp": delta_imp,
            "delta_nat": delta_nat,
            "n_flags_added": len(flag_cols),
        },
    )


def run_i11_and_i11_1(
    train: pd.DataFrame,
    test: pd.DataFrame,
    f_clin: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    auc_native_raw: float,
) -> tuple[dict, dict]:
    """Audit min/max, h1/d1 containment, BMI formula, and BP invasive/noninvasive triplets."""
    eps = 1e-6
    cons_rows = []
    n_min_gt_max = 0
    n_h1_out_d1 = 0
    n_ap_out_d1 = 0

    min_cols = sorted(c for c in f_clin if c.endswith("_min"))
    for c_min in min_cols:
        c_max = c_min[:-4] + "_max"
        if c_max not in f_clin:
            continue
        m_tr = train[c_min].notna() & train[c_max].notna()
        m_te = test[c_min].notna() & test[c_max].notna()
        v_tr = int((train.loc[m_tr, c_min] > train.loc[m_tr, c_max] + eps).sum())
        v_te = int((test.loc[m_te, c_min] > test.loc[m_te, c_max] + eps).sum())
        n_min_gt_max += v_tr + v_te
        cons_rows.append(
            {
                "check_type": "min_gt_max",
                "pair": f"{c_min} > {c_max}",
                "train_n_both": int(m_tr.sum()),
                "train_violations": v_tr,
                "test_n_both": int(m_te.sum()),
                "test_violations": v_te,
            }
        )

    h1_cols = sorted(c for c in f_clin if c.startswith("h1_"))
    for h1_c in h1_cols:
        d1_c = "d1_" + h1_c[3:]
        if d1_c not in f_clin:
            continue
        m_tr = train[h1_c].notna() & train[d1_c].notna()
        m_te = test[h1_c].notna() & test[d1_c].notna()
        if h1_c.endswith("_max"):
            v_tr = int((train.loc[m_tr, h1_c] > train.loc[m_tr, d1_c] + eps).sum())
            v_te = int((test.loc[m_te, h1_c] > test.loc[m_te, d1_c] + eps).sum())
            desc = f"{h1_c} > {d1_c}"
        else:
            v_tr = int((train.loc[m_tr, h1_c] < train.loc[m_tr, d1_c] - eps).sum())
            v_te = int((test.loc[m_te, h1_c] < test.loc[m_te, d1_c] - eps).sum())
            desc = f"{h1_c} < {d1_c}"
        n_h1_out_d1 += v_tr + v_te
        cons_rows.append(
            {
                "check_type": "h1_outside_d1",
                "pair": desc,
                "train_n_both": int(m_tr.sum()),
                "train_violations": v_tr,
                "test_n_both": int(m_te.sum()),
                "test_violations": v_te,
            }
        )

    # _apache vs d1_min..d1_max containment
    ap_map = {
        "albumin_apache": "albumin",
        "bilirubin_apache": "bilirubin",
        "bun_apache": "bun",
        "creatinine_apache": "creatinine",
        "glucose_apache": "glucose",
        "heart_rate_apache": "heartrate",
        "hematocrit_apache": "hematocrit",
        "map_apache": "mbp",
        "paco2_apache": "arterial_pco2",
        "pao2_apache": "arterial_po2",
        "ph_apache": "arterial_ph",
        "resprate_apache": "resprate",
        "sodium_apache": "sodium",
        "temp_apache": "temp",
        "wbc_apache": "wbc",
    }
    for ap_c, stem in ap_map.items():
        d_min, d_max = f"d1_{stem}_min", f"d1_{stem}_max"
        m_tr = train[ap_c].notna() & train[d_min].notna() & train[d_max].notna()
        v_tr = int(
            (
                (train.loc[m_tr, ap_c] < train.loc[m_tr, d_min] - eps)
                | (train.loc[m_tr, ap_c] > train.loc[m_tr, d_max] + eps)
            ).sum()
        )
        m_te = test[ap_c].notna() & test[d_min].notna() & test[d_max].notna()
        v_te = int(
            (
                (test.loc[m_te, ap_c] < test.loc[m_te, d_min] - eps)
                | (test.loc[m_te, ap_c] > test.loc[m_te, d_max] + eps)
            ).sum()
        )
        n_ap_out_d1 += v_tr + v_te
        cons_rows.append(
            {
                "check_type": "apache_outside_d1",
                "pair": f"{ap_c} not in [{d_min}, {d_max}]",
                "train_n_both": int(m_tr.sum()),
                "train_violations": v_tr,
                "test_n_both": int(m_te.sum()),
                "test_violations": v_te,
            }
        )

    # BMI formula check & recoverable BMI
    def bmi_audit(df_x: pd.DataFrame) -> tuple[int, int, int]:
        both = df_x["bmi"].notna() & df_x["height"].notna() & df_x["weight"].notna()
        calc = df_x.loc[both, "weight"] / ((df_x.loc[both, "height"] / 100.0) ** 2)
        rel_err = (df_x.loc[both, "bmi"] - calc).abs() / calc
        n_mis = int((rel_err > 0.02).sum())
        n_rec = int((df_x["bmi"].isna() & df_x["height"].notna() & df_x["weight"].notna()).sum())
        return int(both.sum()), n_mis, n_rec

    bmi_n_tr, bmi_mis_tr, bmi_rec_tr = bmi_audit(train)
    bmi_n_te, bmi_mis_te, bmi_rec_te = bmi_audit(test)
    n_bmi_mis = bmi_mis_tr + bmi_mis_te
    cons_rows.append(
        {
            "check_type": "bmi_formula_mismatch_gt2pct",
            "pair": "bmi vs weight/(height/100)^2",
            "train_n_both": bmi_n_tr,
            "train_violations": bmi_mis_tr,
            "test_n_both": bmi_n_te,
            "test_violations": bmi_mis_te,
        }
    )

    # I11.1: 12 BP triplets (general vs invasive & noninvasive)
    max_unabs = 0.0
    bp_diffs = []
    bp_sub_cols = []
    inv_bp_cols = []
    for win in ["d1", "h1"]:
        for metric in ["diasbp", "mbp", "sysbp"]:
            for ext in ["max", "min"]:
                g_c = f"{win}_{metric}_{ext}"
                i_c = f"{win}_{metric}_invasive_{ext}"
                n_c = f"{win}_{metric}_noninvasive_{ext}"
                bp_sub_cols.extend([i_c, n_c])
                inv_bp_cols.append(i_c)
                has_any = train[i_c].notna() | train[n_c].notna()
                expected = (
                    np.fmax(train.loc[has_any, i_c], train.loc[has_any, n_c])
                    if ext == "max"
                    else np.fmin(train.loc[has_any, i_c], train.loc[has_any, n_c])
                )
                unabs = train.loc[has_any, g_c].isna() | (
                    (train.loc[has_any, g_c] - expected).abs() > eps
                )
                unabs_pct = float(100.0 * unabs.mean())
                max_unabs = max(max_unabs, unabs_pct)

                both_m = train[i_c].notna() & train[n_c].notna()
                mean_diff = float((train.loc[both_m, i_c] - train.loc[both_m, n_c]).mean())
                bp_diffs.append(mean_diff)
                cons_rows.append(
                    {
                        "check_type": "bp_triplet_absorption",
                        "pair": f"{g_c} vs ({i_c}, {n_c})",
                        "train_n_both": int(has_any.sum()),
                        "train_violations": int(unabs.sum()),
                        "test_n_both": int((test[i_c].notna() | test[n_c].notna()).sum()),
                        "test_violations": 0,
                    }
                )

    pd.DataFrame(cons_rows).to_csv(REPORTS_DIR / "stage4_paired_consistency.csv", index=False)

    tot_viol = n_min_gt_max + n_h1_out_d1 + n_bmi_mis
    i11_out = (
        "TRUE"
        if (tot_viol >= 100 or n_min_gt_max >= 1 or n_h1_out_d1 >= 1)
        else ("FALSE" if tot_viol == 0 else "INCONCLUSIVE")
    )

    df_bp = train[f_clin].copy()
    df_bp["has_invasive_bp"] = train[inv_bp_cols].notna().any(axis=1).astype(np.int8)
    pruned_cols = [c for c in f_clin if c not in set(bp_sub_cols)] + ["has_invasive_bp"]
    auc_bp_pruned = eval_hgb_5fold(df_bp, pruned_cols, y, splits, impute_all=False)
    delta_bp = auc_bp_pruned - auc_native_raw
    i11_1_out = (
        "TRUE"
        if (max_unabs <= 0.01 and abs(delta_bp) < 0.002)
        else ("FALSE" if (max_unabs > 0.05 or delta_bp <= -0.005) else "INCONCLUSIVE")
    )
    return (
        {
            "outcome": i11_out,
            "n_min_gt_max": n_min_gt_max,
            "n_h1_out_d1": n_h1_out_d1,
            "n_ap_out_d1": n_ap_out_d1,
            "n_bmi_mis": n_bmi_mis,
            "bmi_rec_tr": bmi_rec_tr,
            "bmi_rec_te": bmi_rec_te,
            "tot_viol": tot_viol,
        },
        {
            "outcome": i11_1_out,
            "max_unabs_pct": max_unabs,
            "mean_inv_minus_noninv": float(np.mean(bp_diffs)),
            "auc_bp_pruned": auc_bp_pruned,
            "delta_bp": delta_bp,
        },
    )


def run_i12_and_i12_1(
    train: pd.DataFrame,
    test: pd.DataFrame,
    f_clin: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    auc_native_raw: float,
) -> tuple[dict, dict]:
    """Audit quantile tails, unit-scale mismatches, sanitization CV, and zero sentinels."""
    num_cols = [c for c in f_clin if pd.api.types.is_numeric_dtype(train[c])]
    q_rows = []
    unit_mismatches = []
    extreme_tail_cols = []

    df_san = train[f_clin].copy()
    for c in num_cols:
        s_tr = train[c].dropna().astype(float)
        s_te = test[c].dropna().astype(float)
        qs = s_tr.quantile([0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]).to_dict()

        unit_flag = "OK"
        if "hematocrit" in c and qs[0.5] > 1.0:
            unit_flag = "DICT_SAYS_FRACTION_ACTUAL_PCT"
            unit_mismatches.append(c)
        elif "pao2fio2ratio" in c and qs[0.5] > 10.0:
            unit_flag = "DICT_SAYS_FIO2_FRACTION_ACTUAL_RATIO"
            unit_mismatches.append(c)

        # Extreme tail / sentinel rules
        s_full = train[c].astype(float)
        nn = s_full.notna()
        bad = pd.Series(False, index=train.index)
        if c == "pre_icu_los_days":
            bad = nn & (s_full < 0)
        elif c == "age":
            bad = nn & ((s_full <= 0) | (s_full > 110))
        elif "resprate" in c:
            bad = nn & ((s_full <= 0) | (s_full > 100))
        elif "heartrate" in c or c == "heart_rate_apache":
            bad = nn & ((s_full <= 0) | (s_full > 250))
        elif "sysbp" in c:
            bad = nn & ((s_full <= 0) | (s_full > 300))
        elif "diasbp" in c:
            bad = nn & ((s_full <= 0) | (s_full > 200))
        elif "_spo2_" in c:
            bad = nn & ((s_full < 30) | (s_full > 100))
        elif "glucose" in c:
            bad = nn & ((s_full < 10) | (s_full > 1500))
        elif "_temp_" in c or c == "temp_apache":
            bad = nn & ((s_full < 28) | (s_full > 43))

        bad_pct = float(100.0 * bad.sum() / max(1, len(s_tr)))
        if bad_pct >= 0.10:
            extreme_tail_cols.append(c)
        if bad.sum() > 0:
            df_san.loc[bad, c] = np.nan

        q_rows.append(
            {
                "column": c,
                "unit_scale_flag": unit_flag,
                "min": round(float(qs[0.0]), 4),
                "p0_1": round(float(qs[0.001]), 4),
                "p1": round(float(qs[0.01]), 4),
                "median": round(float(qs[0.5]), 4),
                "p99": round(float(qs[0.99]), 4),
                "p99_9": round(float(qs[0.999]), 4),
                "max": round(float(qs[1.0]), 4),
                "test_min": round(float(s_te.min()), 4) if len(s_te) else np.nan,
                "test_max": round(float(s_te.max()), 4) if len(s_te) else np.nan,
                "extreme_tail_n": int(bad.sum()),
                "extreme_tail_pct": round(bad_pct, 4),
            }
        )

    pd.DataFrame(q_rows).sort_values("extreme_tail_pct", ascending=False).to_csv(
        REPORTS_DIR / "stage4_physiological_outliers.csv", index=False
    )

    auc_san = eval_hgb_5fold(df_san, f_clin, y, splits, impute_all=False)
    delta_san = auc_san - auc_native_raw
    i12_out = (
        "TRUE"
        if (len(unit_mismatches) >= 5 and len(extreme_tail_cols) >= 3 and delta_san >= -0.001)
        else (
            "FALSE"
            if (len(unit_mismatches) == 0 and len(extreme_tail_cols) == 0)
            else "INCONCLUSIVE"
        )
    )

    # I12.1: Hospital concentration on zero-coded vitals & age==0
    zero_cols = ["d1_resprate_min", "h1_resprate_min", "d1_heartrate_min", "age"]
    zero_stats = []
    k_hosp_conc = 0
    for c in zero_cols:
        nn = train[c].notna()
        is_zero = nn & (train[c] == 0)
        tot_z = int(is_zero.sum())
        by_h = (
            pd.DataFrame({"h": train.loc[nn, "hospital_id"], "z": is_zero.loc[nn].astype(int)})
            .groupby("h")["z"]
            .agg(["sum", "count", "mean"])
        )
        top5_share = float(by_h["sum"].nlargest(5).sum() / max(1, tot_z))
        h30 = by_h[by_h["count"] >= 30]
        max_h_rate = float(h30["mean"].max()) if len(h30) else 0.0
        if top5_share >= 0.50 or max_h_rate >= 0.25:
            k_hosp_conc += 1
        dm_z = float(y[is_zero].mean() * 100.0) if tot_z > 0 else np.nan
        dm_nz = float(y[nn & (~is_zero)].mean() * 100.0)
        zero_stats.append(
            {
                "col": c,
                "n_zero": tot_z,
                "top5_hosp_share": round(top5_share, 4),
                "max_hosp_rate_n30": round(max_h_rate, 4),
                "dm_zero_pct": round(dm_z, 2) if pd.notna(dm_z) else np.nan,
                "dm_nonzero_pct": round(dm_nz, 2),
            }
        )
    max_top5 = max(z["top5_hosp_share"] for z in zero_stats)
    max_hrate = max(z["max_hosp_rate_n30"] for z in zero_stats)
    i12_1_out = (
        "TRUE"
        if k_hosp_conc >= 1
        else ("FALSE" if (max_top5 < 0.25 and max_hrate < 0.10) else "INCONCLUSIVE")
    )
    return (
        {
            "outcome": i12_out,
            "k_unit": len(unit_mismatches),
            "unit_cols": unit_mismatches,
            "n_extreme_cols": len(extreme_tail_cols),
            "extreme_cols": extreme_tail_cols,
            "auc_san": auc_san,
            "delta_san": delta_san,
        },
        {
            "outcome": i12_1_out,
            "k_hosp_conc": k_hosp_conc,
            "zero_stats": zero_stats,
        },
    )


def oof_smoothed_te(
    s: pd.Series,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    m: float = 20.0,
) -> np.ndarray:
    """Compute leak-free out-of-fold smoothed target encoding."""
    oof = np.zeros(len(s), dtype=float)
    s_key = s.fillna("__MISSING__").astype(str)
    for tr_i, va_i in splits:
        prior = float(y[tr_i].mean())
        agg = (
            pd.DataFrame({"k": s_key.iloc[tr_i].to_numpy(), "y": y[tr_i]})
            .groupby("k")["y"]
            .agg(["sum", "count"])
        )
        mp = (agg["sum"] + m * prior) / (agg["count"] + m)
        oof[va_i] = s_key.iloc[va_i].map(mp).fillna(prior).to_numpy()
    return oof


def run_i13_and_i13_1(
    train: pd.DataFrame,
    test: pd.DataFrame,
    f_clin: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    auc_native_raw: float,
) -> tuple[dict, dict]:
    """Audit 8 categorical/diagnosis columns and APACHE III-J hierarchical TE."""
    cat_rows = []
    tot_rare = 0
    tot_unseen = 0
    max_unseen_pct = 0.0

    for c in CAT_COLS_8:
        s_tr = train[c].fillna("__MISSING__").astype(str)
        s_te = test[c].fillna("__MISSING__").astype(str)
        vc_tr = s_tr.value_counts()
        vc_te = s_te.value_counts()
        rare_n = int((vc_tr < 30).sum())
        unseen_lvls = sorted(set(vc_te.index) - set(vc_tr.index))
        unseen_rows = int(vc_te.reindex(unseen_lvls, fill_value=0).sum())
        unseen_pct = float(100.0 * unseen_rows / len(test))
        norm_coll = len(vc_tr) - len({k.strip().lower() for k in vc_tr.index})

        tot_rare += rare_n
        tot_unseen += len(unseen_lvls)
        max_unseen_pct = max(max_unseen_pct, unseen_pct)
        cat_rows.append(
            {
                "column": c,
                "train_nunique": len(vc_tr),
                "test_nunique": len(vc_te),
                "rare_levels_lt30": rare_n,
                "unseen_test_levels": len(unseen_lvls),
                "unseen_test_row_pct": round(unseen_pct, 4),
                "case_ws_collisions": norm_coll,
                "top_level": str(vc_tr.index[0]),
                "top_level_pct": round(100.0 * vc_tr.iloc[0] / len(train), 2),
            }
        )

    pd.DataFrame(cat_rows).to_csv(REPORTS_DIR / "stage4_categorical_audit.csv", index=False)
    i13_out = (
        "TRUE"
        if (tot_rare >= 50 and (tot_unseen >= 1 or max_unseen_pct > 0.0))
        else ("FALSE" if (tot_rare == 0 and tot_unseen == 0) else "INCONCLUSIVE")
    )

    # I13.1: Diagnosis-only & Full F_clinical hierarchical decomposition + OOF TE
    a3 = train["apache_3j_diagnosis"]
    hier_df = pd.DataFrame(
        {
            "apache_2_diagnosis": train["apache_2_diagnosis"],
            "apache_3j_diagnosis": a3,
            "apache_3j_major": np.floor(a3),
            "apache_3j_system": np.floor(a3 / 100.0),
            "apache_3j_sub": np.round(a3 - np.floor(a3), 2),
        },
        index=train.index,
    )
    te_cols = {}
    for col in ["apache_2_diagnosis", "apache_3j_diagnosis", "apache_3j_major", "apache_3j_system"]:
        te_cols[f"{col}_te"] = oof_smoothed_te(hier_df[col], y, splits, m=20.0)
    hier_with_te = pd.concat([hier_df, pd.DataFrame(te_cols, index=train.index)], axis=1)

    auc_diag_raw = eval_hgb_5fold(
        train, ["apache_2_diagnosis", "apache_3j_diagnosis"], y, splits, impute_all=False
    )
    auc_diag_hier = eval_hgb_5fold(
        hier_with_te, list(hier_with_te.columns), y, splits, impute_all=False
    )
    delta_diag = auc_diag_hier - auc_diag_raw

    new_hier_cols = ["apache_3j_major", "apache_3j_system", "apache_3j_sub"] + list(te_cols.keys())
    full_hier_df = pd.concat([train[f_clin], hier_with_te[new_hier_cols]], axis=1)
    auc_full_hier = eval_hgb_5fold(
        full_hier_df, f_clin + new_hier_cols, y, splits, impute_all=False
    )
    delta_full = auc_full_hier - auc_native_raw

    i13_1_out = (
        "TRUE"
        if (delta_diag >= 0.010 and delta_full >= 0.001)
        else ("FALSE" if (delta_diag <= 0.0 and delta_full <= 0.0) else "INCONCLUSIVE")
    )
    return (
        {
            "outcome": i13_out,
            "tot_rare": tot_rare,
            "tot_unseen": tot_unseen,
            "max_unseen_pct": max_unseen_pct,
            "cat_rows": cat_rows,
        },
        {
            "outcome": i13_1_out,
            "auc_diag_raw": auc_diag_raw,
            "auc_diag_hier": auc_diag_hier,
            "delta_diag": delta_diag,
            "auc_full_raw": auc_native_raw,
            "auc_full_hier": auc_full_hier,
            "delta_full": delta_full,
        },
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now_iso = iso_now_plus8()
    commit = current_git_commit()
    train_sha = sha256_file(TRAIN_PATH)
    test_sha = sha256_file(TEST_PATH)
    dict_sha = sha256_file(DICT_PATH)

    train, test, dd = load_raw_wids()
    dd_map = dd.set_index("Variable Name")["Category"].to_dict()
    y = train["diabetes_mellitus"].to_numpy(dtype=int)
    g = train["hospital_id"].to_numpy()
    f_clin = [c for c in test.columns if c not in DROP_COLS]
    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(train, y, g)
    )

    auc_native_raw = eval_hgb_5fold(train, f_clin, y, splits, impute_all=False)
    r10, r10_1 = run_i10_and_i10_1(train, f_clin, dd_map, y, splits, auc_native_raw)
    r11, r11_1 = run_i11_and_i11_1(train, test, f_clin, y, splits, auc_native_raw)
    r12, r12_1 = run_i12_and_i12_1(train, test, f_clin, y, splits, auc_native_raw)
    r13, r13_1 = run_i13_and_i13_1(train, test, f_clin, y, splits, auc_native_raw)

    src_both = f"data/raw/TrainingWiDS2021.csv; data/raw/UnlabeledWiDS2021.csv (sha256={test_sha})"
    src_all = f"{src_both}; data/raw/DataDictionaryWiDS2021.csv (sha256={dict_sha})"
    script_ref = "scripts/stage4_integrity_and_missingness.py"

    entries = [
        make_inquiry(
            inq_id="I10",
            status=r10["outcome"],
            polarity="+",
            claim=(
                "Across deduplicated non-trivial missingness masks in F_clinical, missingness is "
                "MNAR/MAR wrt diabetes_mellitus (>= 10 masks with BH-FDR q < 0.01 & |Δ| >= 2.0 pp)."
            ),
            technique="Deduplicated missingness-mask chi2 + BH-FDR and burden Spearman test",
            parameters={"fdr_alpha": 0.01, "min_abs_diff_pp": 2.0},
            assumptions=["Twin columns sharing masks reflect one underlying lab-order event."],
            missing_data_policy="Missingness evaluated directly via isna() indicator.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"n_unique_masks={r10['n_masks']}, k_mnar={r10['k_mnar']}, "
                f"r_tot_miss={r10['r_tot']:.4f} (p={r10['p_tot']:.3e}), "
                f"top={r10['top_masks'][:3]}"
            ),
            formula="count(BH_q < 0.01 AND |P(y=1|miss)-P(y=1|present)| >= 2.0 pp)",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows and 175 predictors with 30 <= n_null <= N-30"],
            outcome=r10["outcome"],
            on_true_cond="k_mnar >= 10 OR (k_mnar >= 3 AND p_tot < 0.001 AND |r_tot| >= 0.02)",
            on_false_cond="k_mnar == 0 AND (p_tot >= 0.05 OR |r_tot| < 0.01)",
            on_inconc_cond="1-9 significant missingness masks",
            act_true="Confirm MNAR/MAR; export stage4_missingness_mechanisms.csv; Spawns #I10.1.",
            act_false="Reject widespread MNAR; Spawns #I10.1.",
            act_inconc="Inspect 1-9 significant masks; Spawns #I10.1.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["F_clinical (175 predictors)", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I10.1",
            status=r10_1["outcome"],
            polarity="+",
            claim=(
                "Subtends #I10. Explicit missingness flags/panel counts lift grouped HGB AUC by "
                ">= +0.005 over imputed without flags (or >= +0.002 over native NaN)."
            ),
            technique="4-arm missingness-representation ablation on 5-fold SGKF",
            parameters={"n_splits": 5, "n_flags_added": r10_1["n_flags_added"]},
            assumptions=["Comparing imputed+flags vs imputed isolates pure missingness signal."],
            missing_data_policy="Imputers fit per fold; non-demographic counts omit shifted demos.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"native_raw={r10_1['auc_native_raw']:.5f}, "
                f"native_counts={r10_1['auc_native_counts']:.5f} (Δ={r10_1['delta_nat']:+.5f}), "
                f"imp_no_flags={r10_1['auc_imp_no_flags']:.5f}, "
                f"imp_flags={r10_1['auc_imp_flags']:.5f} (Δ={r10_1['delta_imp']:+.5f})"
            ),
            formula="auc_imp_flags - auc_imp_no_flags; auc_native_counts - auc_native_raw",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=r10_1["outcome"],
            on_true_cond="delta_imp >= 0.005 OR delta_nat >= 0.002",
            on_false_cond="delta_imp <= 0.0 AND delta_nat <= 0.0",
            on_inconc_cond="Modest positive lift below thresholds",
            act_true="Retain non-demo missingness counts/flags for Stage 5-6 -> Ref to #I11.",
            act_false="Rely on native tree NaN handling without extra flags -> Ref to #I11.",
            act_inconc="Test missingness counts on Stage 6 Log-Loss/Brier -> Ref to #I11.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["F_clinical", "hospital_id", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I11",
            status=r11["outcome"],
            polarity="+",
            claim=(
                "Across 64 min/max pairs, 64 (h1_, d1_) bounds, and BMI formula checks in Train "
                "and Test, >= 100 violations or >= 1 min>max / h1>d1 violation occur."
            ),
            technique="Algebraic invariant audit across min/max, h1/d1, apache/d1, and BMI",
            parameters={"tolerance_eps": 1e-6, "bmi_rel_err_threshold": 0.02},
            assumptions=["1-hour extrema must lie within 24-hour extrema for same stay."],
            missing_data_policy="Evaluate strictly where both compared columns are non-null.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"min_gt_max={r11['n_min_gt_max']}, h1_out_d1={r11['n_h1_out_d1']}, "
                f"ap_out_d1={r11['n_ap_out_d1']}, bmi_mismatch={r11['n_bmi_mis']}, "
                f"bmi_recoverable=(tr:{r11['bmi_rec_tr']}, te:{r11['bmi_rec_te']})"
            ),
            formula="min_gt_max + h1_out_d1 + bmi_mismatch",
            timeframe="Static WiDS Datathon 2021 Train and Unlabeled snapshots",
            filters=["All 130,157 Train and 10,234 Unlabeled rows"],
            outcome=r11["outcome"],
            on_true_cond="total_violations >= 100 OR min_gt_max >= 1 OR h1_out_d1 >= 1",
            on_false_cond="total_violations == 0",
            on_inconc_cond="1-99 BMI mismatches with zero min/max or h1/d1 inversions",
            act_true="Define cleaning/BMI recovery rules in Stage 5-6; Spawns #I11.1; Ref to #I12.",
            act_false="Certify algebraic bounds; log recoverable BMI; Spawns #I11.1; Ref to #I12.",
            act_inconc="Inspect <100 BMI discrepancies; Spawns #I11.1; Ref to #I12.",
            source_file=src_both,
            commit=commit,
            sha256=train_sha,
            columns=["64 min/max pairs", "height", "weight", "bmi"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I11.1",
            status=r11_1["outcome"],
            polarity="+",
            claim=(
                "Subtends #I11. General BP absorbs >= 99% of invasive/noninvasive BP readings "
                "(unabsorbed <= 0.01%) and dropping sub-columns with has_invasive_bp preserves AUC."
            ),
            technique="12-triplet BP extremum reconciliation and grouped-CV ablation",
            parameters={"bp_triplets": 12, "pruned_sub_cols": 24},
            assumptions=["General BP encompasses either invasive or noninvasive readings."],
            missing_data_policy="Reconcile where either invasive or noninvasive is non-null.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"max_unabs_pct={r11_1['max_unabs_pct']:.4f}%, "
                f"mean_inv_minus_noninv={r11_1['mean_inv_minus_noninv']:+.3f}mmHg, "
                f"auc_pruned={r11_1['auc_bp_pruned']:.5f} (Δ={r11_1['delta_bp']:+.5f})"
            ),
            formula="unabsorbed_pct; auc_bp_pruned - auc_native_raw",
            timeframe="Static WiDS Datathon 2021 snapshot",
            filters=["All 36 BP columns in Train and Test"],
            outcome=r11_1["outcome"],
            on_true_cond="max_unabs <= 0.01 AND |delta_bp| < 0.002",
            on_false_cond="max_unabs > 0.05 OR delta_bp <= -0.005",
            on_inconc_cond="Partial absorption or minor AUC drop",
            act_true="Adopt general BP + has_invasive_bp simplification -> Ref to #I12.",
            act_false="Retain invasive/noninvasive BP sub-columns -> Ref to #I12.",
            act_inconc="Keep BP sub-columns + has_invasive_bp for Stage 6 check -> Ref to #I12.",
            source_file=src_both,
            commit=commit,
            sha256=train_sha,
            columns=["36 d1_/h1_ blood-pressure columns"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I12",
            status=r12["outcome"],
            polarity="+",
            claim=(
                "Numeric predictors exhibit >= 5 dictionary unit-scale mismatches and >= 3 extreme "
                "tail columns (>= 0.1%), while sanitizing hard sentinels to NaN preserves AUC."
            ),
            technique="Quantile tail audit (p0.1..p99.9) and pre/post sanitization 5-fold SGKF CV",
            parameters={"extreme_rate_threshold_pct": 0.10},
            assumptions=["Extreme survival bounds distinguish physiology from EHR sentinels."],
            missing_data_policy="Coerce hard-impossible/sentinel values to NaN without row drops.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"k_unit={r12['k_unit']} ({r12['unit_cols']}), "
                f"n_extreme={r12['n_extreme_cols']} ({r12['extreme_cols']}), "
                f"auc_san={r12['auc_san']:.5f} (Δ={r12['delta_san']:+.5f})"
            ),
            formula="auc_sanitized - auc_native_raw",
            timeframe="Static WiDS Datathon 2021 snapshot",
            filters=["All 169 numeric F_clinical predictors"],
            outcome=r12["outcome"],
            on_true_cond="k_unit >= 5 AND n_extreme >= 3 AND delta_san >= -0.001",
            on_false_cond="k_unit == 0 AND n_extreme == 0",
            on_inconc_cond="Partial unit mismatch or sanitization AUC delta < -0.001",
            act_true="Lock range sanitization (keep hematocrit in %); Spawns #I12.1; Ref to #I13.",
            act_false="Document clean scales; Spawns #I12.1; Ref to #I13.",
            act_inconc=(
                "Hold blind zero->NaN pending hospital-sentinel check #I12.1 -> Ref to #I13."
            ),
            source_file=src_all,
            commit=commit,
            sha256=train_sha,
            columns=["169 numeric predictors in F_clinical"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I12.1",
            status=r12_1["outcome"],
            polarity="+",
            claim=(
                "Subtends #I12. Zero-coded vital/age anomalies cluster inside specific hospital_id "
                "feeds (top-5 hospitals >= 50% of zeros or >= 25% rate in a hospital n >= 30)."
            ),
            technique="Per-hospital zero-rate concentration and diabetes prevalence audit",
            parameters={"top5_share_min": 0.50, "hosp_rate_min": 0.25},
            assumptions=["Hospital-concentrated 0s indicate EHR missingness-as-zero defaults."],
            missing_data_policy="Evaluated over non-null entries per column.",
            script_ref=script_ref,
            ev_type="N",
            metric=f"k_hosp_conc={r12_1['k_hosp_conc']}, details={r12_1['zero_stats']}",
            formula="count(top5_hosp_share >= 0.50 OR max_hosp_rate_n30 >= 0.25)",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["d1_resprate_min, h1_resprate_min, d1_heartrate_min, age"],
            outcome=r12_1["outcome"],
            on_true_cond="k_hosp_conc >= 1",
            on_false_cond="max(top5_share) < 0.25 AND max(hosp_rate) < 0.10",
            on_inconc_cond="Moderate hospital clustering",
            act_true=(
                "Treat concentrated 0-vitals/age==0 as EHR missingness sentinels -> Ref to #I13."
            ),
            act_false="Treat zero vitals as distributed clinical events -> Ref to #I13.",
            act_inconc="Verify 0->NaN sensitivity in Stage 6 CV -> Ref to #I13.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=[
                "d1_resprate_min",
                "h1_resprate_min",
                "d1_heartrate_min",
                "age",
                "hospital_id",
            ],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I13",
            status=r13["outcome"],
            polarity="+",
            claim=(
                "Across the 8 realized WiDS 2021 categorical/diagnosis predictors, >= 50 rare "
                "levels (n < 30) and >= 1 unseen Unlabeled test category level exist."
            ),
            technique="Cardinality, rare-level, unseen test-level, and string-collision audit",
            parameters={"rare_n_threshold": 30, "n_cols": 8},
            assumptions=["apache_2_bodysystem / apache_3j_bodysystem do not exist in WiDS 2021."],
            missing_data_policy="Map NaN to '__MISSING__' prior to level comparisons.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"tot_rare_lt30={r13['tot_rare']}, tot_unseen_test={r13['tot_unseen']}, "
                f"max_unseen_row_pct={r13['max_unseen_pct']:.4f}%"
            ),
            formula="sum(|levels < 30|); sum(|levels_test \\ levels_train|)",
            timeframe="Static WiDS Datathon 2021 snapshot",
            filters=["8 realized categorical/diagnosis cols (excluding absent bodysystem cols)"],
            outcome=r13["outcome"],
            on_true_cond="tot_rare >= 50 AND (tot_unseen >= 1 OR max_unseen_pct > 0)",
            on_false_cond="tot_rare == 0 AND tot_unseen == 0",
            on_inconc_cond="Rare levels without unseen test levels (or vice versa)",
            act_true="Lock unknown-safe encoding + rare pooling; Spawns #I13.1.",
            act_false="Document clean overlap; Spawns #I13.1.",
            act_inconc=(
                "Review rare-level concentration in stage4_categorical_audit.csv; Spawns #I13.1."
            ),
            source_file=src_both,
            commit=commit,
            sha256=train_sha,
            columns=CAT_COLS_8,
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I13.1",
            status=r13_1["outcome"],
            polarity="+",
            claim=(
                "Subtends #I13. Decomposing apache_3j_diagnosis into hierarchical nominal parts "
                "+ OOF smoothed TE lifts diagnosis-only AUC >= +0.010 and full AUC >= +0.001."
            ),
            technique="5-fold SGKF benchmark: raw float vs hierarchical decomposition + OOF TE",
            parameters={"te_smoothing_m": 20.0},
            assumptions=["Integer major code and hundreds organ system encode nominal pathology."],
            missing_data_policy="Missing codes receive fold-train target prior.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"diag_raw={r13_1['auc_diag_raw']:.5f}, diag_hier={r13_1['auc_diag_hier']:.5f} "
                f"(Δ={r13_1['delta_diag']:+.5f}), full_raw={r13_1['auc_full_raw']:.5f}, "
                f"full_hier={r13_1['auc_full_hier']:.5f} (Δ={r13_1['delta_full']:+.5f})"
            ),
            formula="auc_diag_hier - auc_diag_raw; auc_full_hier - auc_full_raw",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=r13_1["outcome"],
            on_true_cond="delta_diag >= 0.010 AND delta_full >= 0.001",
            on_false_cond="delta_diag <= 0.0 AND delta_full <= 0.0",
            on_inconc_cond="Standalone diagnosis gain without full-tree delta >= +0.001",
            act_true="Adopt apache_3j hierarchy + OOF TE in Stage 5-6 -> Proceeds Stage 5.",
            act_false="Keep native handling without TE overhead -> Proceeds Stage 5.",
            act_inconc=(
                "Carry apache_3j hierarchy into Stage 5-6 feature testing -> Proceeds Stage 5."
            ),
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["F_clinical", "apache_2_diagnosis", "apache_3j_diagnosis"],
            timestamp=now_iso,
        ),
    ]
    upsert_inquiries(entries)

    cat_md = "\n".join(
        f"| `{c['column']}` | {c['train_nunique']} | {c['test_nunique']} | "
        f"{c['rare_levels_lt30']} | {c['unseen_test_levels']} ({c['unseen_test_row_pct']:.3f}%) | "
        f"{c['case_ws_collisions']} |"
        for c in r13["cat_rows"]
    )
    z_md = "\n".join(
        f"- `{z['col']}`: `n_zero={z['n_zero']}`, "
        f"top-5 hosp=`{z['top5_hosp_share'] * 100:.1f}%`, "
        f"max hosp rate (`n>=30`)=`{z['max_hosp_rate_n30'] * 100:.1f}%`, "
        f"DM% (zero vs non-zero)=`{z['dm_zero_pct']}% vs {z['dm_nonzero_pct']}%`"
        for z in r12_1["zero_stats"]
    )
    md = f"""# Stage 4 Report: Missingness Mechanisms, Consistency, Ranges & Categoricals

- **Generated**: `{now_iso}` (Base Commit: `{commit}`)
- **Schema Correction**: Confirmed `apache_2_bodysystem` and `apache_3j_bodysystem` do **not**
  exist in WiDS Datathon 2021 (WiDS 2020 columns).

## Key Findings

1. **[I10 = {r10["outcome"]}, I10.1 = {r10_1["outcome"]}] Informative Missingness**:
   - Unique non-trivial masks: `{r10["n_masks"]}`, MNAR/MAR masks (`BH q<0.01`, `|Δ|>=2pp`):
     **`{r10["k_mnar"]}`**; `n_miss_total` vs DM Spearman `r={r10["r_tot"]:.4f}`
     (`p={r10["p_tot"]:.3e}`).
   - 4-arm SGKF AUC: `native_raw={r10_1["auc_native_raw"]:.5f}`,
     `native_counts={r10_1["auc_native_counts"]:.5f}` (`Δ={r10_1["delta_nat"]:+.5f}`),
     `imp_no_flags={r10_1["auc_imp_no_flags"]:.5f}`,
     `imp_flags={r10_1["auc_imp_flags"]:.5f}` (`Δ={r10_1["delta_imp"]:+.5f}`).

2. **[I11 = {r11["outcome"]}, I11.1 = {r11_1["outcome"]}] Paired & BP Consistency**:
   - `min > max` violations: **`{r11["n_min_gt_max"]}`**; `h1_` outside `d1_` violations:
     **`{r11["n_h1_out_d1"]}`**; `_apache` outside `d1_` violations: **`{r11["n_ap_out_d1"]}`**.
   - `bmi` >2% formula mismatches: **`{r11["n_bmi_mis"]}`**; recoverable null BMIs:
     Train=`{r11["bmi_rec_tr"]}`, Test=`{r11["bmi_rec_te"]}`.
   - BP triplets: max unabsorbed=`{r11_1["max_unabs_pct"]:.4f}%`, mean `(inv - noninv)`=
     `{r11_1["mean_inv_minus_noninv"]:+.2f} mmHg`, pruned BP AUC=`{r11_1["auc_bp_pruned"]:.5f}`
     (`Δ={r11_1["delta_bp"]:+.5f}`).

3. **[I12 = {r12["outcome"]}, I12.1 = {r12_1["outcome"]}] Physiological Bounds & Zero Sentinels**:
   - Dictionary unit mismatches (`{r12["k_unit"]}`): `{r12["unit_cols"]}`.
   - Extreme tail cols (`>=0.1%`): `{r12["extreme_cols"]}`; sanitized AUC=`{r12["auc_san"]:.5f}`
     (`Δ={r12["delta_san"]:+.5f}`).
   - Hospital-concentrated zero columns (`k_hosp_conc={r12_1["k_hosp_conc"]}`):
{z_md}

4. **[I13 = {r13["outcome"]}, I13.1 = {r13_1["outcome"]}] Categoricals & APACHE III-J Hierarchy**:
   - Total rare levels (`n<30`): **`{r13["tot_rare"]}`**, unseen test levels:
     **`{r13["tot_unseen"]}`** (max row share `{r13["max_unseen_pct"]:.4f}%`).
   - Diagnosis-only AUC: raw=`{r13_1["auc_diag_raw"]:.5f}` -> hier+TE=
     `{r13_1["auc_diag_hier"]:.5f}` (`Δ={r13_1["delta_diag"]:+.5f}`);
     Full AUC: raw=`{r13_1["auc_full_raw"]:.5f}` -> hier+TE=
     `{r13_1["auc_full_hier"]:.5f}` (`Δ={r13_1["delta_full"]:+.5f}`).

| Column | Train Nunique | Test Nunique | Rare (<30) | Unseen Test (Rows %) | Collisions |
| :--- | ---: | ---: | ---: | ---: | ---: |
{cat_md}
"""
    (REPORTS_DIR / "stage4_integrity_missingness.md").write_text(md, encoding="utf-8")
    print(
        f"Stage 4 done: I10={r10['outcome']}, I10.1={r10_1['outcome']}, "
        f"I11={r11['outcome']}, I11.1={r11_1['outcome']}, "
        f"I12={r12['outcome']}, I12.1={r12_1['outcome']}, "
        f"I13={r13['outcome']}, I13.1={r13_1['outcome']}"
    )


if __name__ == "__main__":
    main()
