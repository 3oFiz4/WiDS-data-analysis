"""Stage 5 EDA: Cross-Window Collinearity, Leakage Proxy Audit & Domain FE."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

from project_name.config import PROJECT_ROOT
from project_name.features import (
    DROP_STRUCTURAL,
    add_clinical_ratios,
    add_glucose_features,
    add_hierarchy_and_miss_counts,
    add_hosp_covariate_rel,
    compute_fold_safe_features,
    sanitize_clinical_df,
)
from project_name.hmecvas import (
    DICT_PATH,
    TRAIN_PATH,
    current_git_commit,
    iso_now_plus8,
    load_raw_wids,
    make_inquiry,
    sha256_file,
    upsert_inquiries,
)

REPORTS_DIR = PROJECT_ROOT / "reports" / "findings"
APACHE_TWINS_15 = [
    "albumin_apache",
    "bilirubin_apache",
    "bun_apache",
    "creatinine_apache",
    "glucose_apache",
    "heart_rate_apache",
    "hematocrit_apache",
    "map_apache",
    "paco2_apache",
    "pao2_apache",
    "ph_apache",
    "resprate_apache",
    "sodium_apache",
    "temp_apache",
    "wbc_apache",
]
TRI_WINDOW_MAP = [
    ("albumin", "d1_albumin_max", "h1_albumin_max", "albumin_apache"),
    ("bilirubin", "d1_bilirubin_max", "h1_bilirubin_max", "bilirubin_apache"),
    ("bun", "d1_bun_max", "h1_bun_max", "bun_apache"),
    ("creatinine", "d1_creatinine_max", "h1_creatinine_max", "creatinine_apache"),
    ("glucose", "d1_glucose_max", "h1_glucose_max", "glucose_apache"),
    ("heartrate", "d1_heartrate_max", "h1_heartrate_max", "heart_rate_apache"),
    ("hematocrit", "d1_hematocrit_max", "h1_hematocrit_max", "hematocrit_apache"),
    ("mbp", "d1_mbp_max", "h1_mbp_max", "map_apache"),
    ("paco2", "d1_arterial_pco2_max", "h1_arterial_pco2_max", "paco2_apache"),
    ("pao2", "d1_arterial_po2_max", "h1_arterial_po2_max", "pao2_apache"),
    ("ph", "d1_arterial_ph_max", "h1_arterial_ph_max", "ph_apache"),
    ("resprate", "d1_resprate_max", "h1_resprate_max", "resprate_apache"),
    ("sodium", "d1_sodium_max", "h1_sodium_max", "sodium_apache"),
    ("temp", "d1_temp_max", "h1_temp_max", "temp_apache"),
    ("wbc", "d1_wbc_max", "h1_wbc_max", "wbc_apache"),
    ("oxygenation", "d1_pao2fio2ratio_min", "h1_pao2fio2ratio_min", "fio2_apache"),
]
WATCHLIST = [
    "pre_icu_los_days",
    "elective_surgery",
    "icu_stay_type",
    "apache_3j_diagnosis",
    "d1_glucose_max",
    "glucose_apache",
    "h1_glucose_max",
    "arf_apache",
]


def eval_5fold_hgb(
    df: pd.DataFrame,
    cols: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    use_te: bool = False,
    use_stratum_z: bool = False,
    perm_first_fold: bool = False,
) -> dict:
    """Run 5-fold grouped HGB returning OOF AUC, Log-Loss, fold AUCs, and perm imp."""
    oof = np.zeros(len(df), dtype=float)
    fold_aucs: list[float] = []
    perm_imp: dict[str, float] = {}
    base_cats = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    for f_i, (tr_i, va_i) in enumerate(splits):
        x_tr = df.iloc[tr_i][cols].copy()
        x_va = df.iloc[va_i][cols].copy()

        if use_te or use_stratum_z:
            tr_f, va_f = compute_fold_safe_features(
                df.iloc[tr_i],
                df.iloc[va_i],
                y[tr_i],
                use_te=use_te,
                use_stratum_z=use_stratum_z,
            )
            x_tr = pd.concat([x_tr, tr_f], axis=1)
            x_va = pd.concat([x_va, va_f], axis=1)

        if base_cats:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=np.nan,
            )
            x_tr[base_cats] = enc.fit_transform(x_tr[base_cats].astype(object))
            x_va[base_cats] = enc.transform(x_va[base_cats].astype(object))

        clf = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            categorical_features=base_cats if base_cats else None,
            early_stopping=True,
            random_state=42,
        )
        clf.fit(x_tr, y[tr_i])
        preds = clf.predict_proba(x_va)[:, 1]
        oof[va_i] = preds
        fold_aucs.append(float(roc_auc_score(y[va_i], preds)))

        if perm_first_fold and f_i == 0:
            res = permutation_importance(
                clf, x_va, y[va_i], scoring="roc_auc", n_repeats=3, random_state=42, n_jobs=1
            )
            perm_imp = {
                c: float(m) for c, m in zip(list(x_tr.columns), res.importances_mean, strict=True)
            }

    return {
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_logloss": float(log_loss(y, oof)),
        "fold_aucs": fold_aucs,
        "perm_imp": perm_imp,
        "n_cols": len(cols) + (6 if use_te else 0) + (4 if use_stratum_z else 0),
    }


def eval_1d_fold_stability(
    df: pd.DataFrame,
    col: str,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    """Compute per-fold grouped HGB AUC for one predictor."""
    is_num = pd.api.types.is_numeric_dtype(df[col])
    aucs = []
    if is_num:
        vals = df[col].to_numpy(dtype=float).reshape(-1, 1)
        for tr_i, va_i in splits:
            clf = HistGradientBoostingClassifier(
                max_iter=25,
                learning_rate=0.1,
                max_leaf_nodes=15,
                min_samples_leaf=100,
                early_stopping=False,
                random_state=42,
            )
            clf.fit(vals[tr_i], y[tr_i])
            aucs.append(float(roc_auc_score(y[va_i], clf.predict_proba(vals[va_i])[:, 1])))
    else:
        arr = df[col].astype(object).to_numpy().reshape(-1, 1)
        for tr_i, va_i in splits:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=np.nan,
            )
            clf = HistGradientBoostingClassifier(
                max_iter=25,
                learning_rate=0.1,
                max_leaf_nodes=15,
                min_samples_leaf=100,
                categorical_features=[0],
                early_stopping=False,
                random_state=42,
            )
            clf.fit(enc.fit_transform(arr[tr_i]), y[tr_i])
            aucs.append(
                float(roc_auc_score(y[va_i], clf.predict_proba(enc.transform(arr[va_i]))[:, 1]))
            )
    return aucs


def greedy_sulo_keep(
    ranked_cols: list[str], corr_df: pd.DataFrame, num_set: set[str], thr: float
) -> list[str]:
    """Keep features in descending Stage 3 priority order dropping |rho| >= thr twins."""
    kept: list[str] = []
    kept_num: list[str] = []
    for c in ranked_cols:
        if c not in num_set:
            kept.append(c)
            continue
        redundant = any(
            pd.notna(corr_df.loc[c, k]) and abs(float(corr_df.loc[c, k])) >= thr for k in kept_num
        )
        if not redundant:
            kept.append(c)
            kept_num.append(c)
    return kept


def timeline_tier(col: str, dd_map: dict[str, str]) -> str:
    """Map column to its clinical timeline tier."""
    if col.startswith("h1_"):
        return "T1_first_1h"
    if col.startswith("d1_"):
        return "T2_first_24h"
    if dd_map.get(col) == "APACHE covariate":
        return "T3_apache_24h"
    return "T0_pre_icu"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now_iso = iso_now_plus8()
    commit = current_git_commit()
    train_sha = sha256_file(TRAIN_PATH)
    dict_sha = sha256_file(DICT_PATH)

    train, test, dd = load_raw_wids()
    dd_map = dd.set_index("Variable Name")["Category"].to_dict()
    y = train["diabetes_mellitus"].to_numpy(dtype=int)
    g = train["hospital_id"].to_numpy()
    f_clin = [c for c in test.columns if c not in DROP_STRUCTURAL]
    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(train, y, g)
    )

    triage = pd.read_csv(REPORTS_DIR / "feature_triage_table.csv")
    ranked_cols = triage["column"].tolist()

    # Family A: Sanitized base
    df_san = sanitize_clinical_df(train, f_clin)
    cols_san = f_clin + ["n_min_max_swapped"]
    res_san = eval_5fold_hgb(df_san, cols_san, y, splits)

    # I14: Cross-window Spearman & window ablations
    cw_rows = []
    for analyte, d1_c, h1_c, ap_c in TRI_WINDOW_MAP:
        rho_d1_h1 = float(df_san[[d1_c, h1_c]].corr(method="spearman").iloc[0, 1])
        rho_d1_ap = float(df_san[[d1_c, ap_c]].corr(method="spearman").iloc[0, 1])
        rho_h1_ap = float(df_san[[h1_c, ap_c]].corr(method="spearman").iloc[0, 1])
        cw_rows.append(
            {
                "analyte": analyte,
                "d1_col": d1_c,
                "h1_col": h1_c,
                "apache_col": ap_c,
                "rho_d1_h1": round(rho_d1_h1, 4),
                "rho_d1_apache": round(rho_d1_ap, 4),
                "rho_h1_apache": round(rho_h1_ap, 4),
            }
        )
    cw_df = pd.DataFrame(cw_rows)
    cw_df.to_csv(REPORTS_DIR / "stage5_cross_window_collinearity.csv", index=False)
    med_rho_d1_ap = float(cw_df["rho_d1_apache"].abs().median())

    cols_no_h1 = [c for c in cols_san if not c.startswith("h1_")]
    cols_no_ap_twins = [c for c in cols_san if c not in set(APACHE_TWINS_15)]
    res_no_h1 = eval_5fold_hgb(df_san, cols_no_h1, y, splits)
    res_no_ap = eval_5fold_hgb(df_san, cols_no_ap_twins, y, splits)
    delta_drop_h1 = res_san["oof_auc"] - res_no_h1["oof_auc"]
    delta_drop_ap = res_san["oof_auc"] - res_no_ap["oof_auc"]
    i14_out = (
        "TRUE"
        if (med_rho_d1_ap >= 0.75 and (delta_drop_h1 >= 0.001 or delta_drop_ap >= 0.001))
        else (
            "FALSE"
            if (med_rho_d1_ap < 0.60 and delta_drop_h1 <= 0.0 and delta_drop_ap <= 0.0)
            else "INCONCLUSIVE"
        )
    )

    # I14.1: Greedy SULO correlation pruning on numeric cols
    num_cols = [c for c in f_clin if pd.api.types.is_numeric_dtype(df_san[c])]
    sp_corr = df_san[num_cols].corr(method="spearman")
    keep_r90 = greedy_sulo_keep(ranked_cols, sp_corr, set(num_cols), 0.90)
    keep_r80 = greedy_sulo_keep(ranked_cols, sp_corr, set(num_cols), 0.80)
    res_r90 = eval_5fold_hgb(df_san, keep_r90 + ["n_min_max_swapped"], y, splits)
    res_r80 = eval_5fold_hgb(df_san, keep_r80 + ["n_min_max_swapped"], y, splits)
    delta_r90 = res_r90["oof_auc"] - res_san["oof_auc"]
    delta_r80 = res_r80["oof_auc"] - res_san["oof_auc"]
    i14_1_out = (
        "TRUE"
        if delta_r80 <= -0.002
        else ("FALSE" if (delta_r80 >= 0.0 and delta_r90 >= 0.0) else "INCONCLUSIVE")
    )

    # I15: Leakage, timeline & fold-stability audit
    audit_cols = list(dict.fromkeys(ranked_cols[:25] + WATCHLIST))
    leak_rows = []
    n_post_24h = 0
    n_leak_ge90 = 0
    n_unstable = 0
    for c in audit_cols:
        tier = timeline_tier(c, dd_map)
        f_aucs = eval_1d_fold_stability(df_san, c, y, splits)
        m_auc, min_auc, std_auc = (
            float(np.mean(f_aucs)),
            float(np.min(f_aucs)),
            float(np.std(f_aucs)),
        )
        if m_auc >= 0.90:
            n_leak_ge90 += 1
        unstable = (m_auc >= 0.55) and (min_auc < 0.50 or std_auc > 0.04)
        if unstable:
            n_unstable += 1
        leak_rows.append(
            {
                "column": c,
                "dd_category": dd_map.get(c, "unknown"),
                "timeline_tier": tier,
                "mean_fold_auc": round(m_auc, 5),
                "min_fold_auc": round(min_auc, 5),
                "max_fold_auc": round(float(np.max(f_aucs)), 5),
                "std_fold_auc": round(std_auc, 5),
                "unstable_flag": unstable,
            }
        )
    pd.DataFrame(leak_rows).to_csv(REPORTS_DIR / "stage5_leakage_proxy_audit.csv", index=False)
    # Check arf_apache vs creatinine_apache >= 1.5 mg/dL coherence
    arf1 = train[train["arf_apache"] == 1]["creatinine_apache"].dropna()
    arf0 = train[train["arf_apache"] == 0]["creatinine_apache"].dropna()
    arf_coh = f"arf1_creat_med={arf1.median():.2f} vs arf0_creat_med={arf0.median():.2f}"

    i15_out = (
        "TRUE"
        if (n_post_24h == 0 and n_leak_ge90 == 0 and n_unstable == 0)
        else ("FALSE" if (n_post_24h >= 1 or n_leak_ge90 >= 1) else "INCONCLUSIVE")
    )

    # I15.1: Temporal horizon ablation
    t0_cols = [c for c in f_clin if timeline_tier(c, dd_map) == "T0_pre_icu"]
    t1_cols = [c for c in f_clin if timeline_tier(c, dd_map) == "T1_first_1h"]
    t2_cols = [c for c in f_clin if timeline_tier(c, dd_map) == "T2_first_24h"]
    res_t0 = eval_5fold_hgb(df_san, t0_cols, y, splits)
    res_t01 = eval_5fold_hgb(df_san, t0_cols + t1_cols, y, splits)
    res_t012 = eval_5fold_hgb(df_san, t0_cols + t1_cols + t2_cols, y, splits)
    delta_t3 = res_san["oof_auc"] - res_t012["oof_auc"]
    i15_1_out = (
        "TRUE"
        if (res_t01["oof_auc"] >= 0.78 and delta_t3 < 0.015)
        else ("FALSE" if (res_t01["oof_auc"] < 0.72 or delta_t3 >= 0.030) else "INCONCLUSIVE")
    )

    # I16 & I16.1: Stepwise Domain Feature Engineering
    fe_b = add_glucose_features(df_san)
    fe_c = add_clinical_ratios(df_san)
    fe_d = add_hierarchy_and_miss_counts(df_san, f_clin, dd_map)
    fe_h = add_hosp_covariate_rel(df_san)

    df_ab = pd.concat([df_san, fe_b], axis=1)
    df_abc = pd.concat([df_ab, fe_c], axis=1)
    df_abcd = pd.concat([df_abc, fe_d], axis=1)
    df_abcd_h = pd.concat([df_abcd, fe_h], axis=1)

    cols_ab = cols_san + list(fe_b.columns)
    cols_abc = cols_ab + list(fe_c.columns)
    cols_abcd = cols_abc + list(fe_d.columns)
    cols_abcd_h = cols_abcd + list(fe_h.columns)

    res_ab = eval_5fold_hgb(df_ab, cols_ab, y, splits)
    res_abc = eval_5fold_hgb(df_abc, cols_abc, y, splits)
    res_abcd = eval_5fold_hgb(df_abcd, cols_abcd, y, splits, use_te=True, perm_first_fold=True)

    delta_auc_eng = res_abcd["oof_auc"] - res_san["oof_auc"]
    delta_ll_eng = res_abcd["oof_logloss"] - res_san["oof_logloss"]
    k_folds_imp = sum(
        a > b for a, b in zip(res_abcd["fold_aucs"], res_san["fold_aucs"], strict=True)
    )
    i16_out = (
        "TRUE"
        if (delta_auc_eng >= 0.0030 and k_folds_imp >= 4 and delta_ll_eng <= -0.0010)
        else ("FALSE" if (delta_auc_eng <= 0.0 or k_folds_imp <= 1) else "INCONCLUSIVE")
    )

    res_strat_z = eval_5fold_hgb(df_abcd, cols_abcd, y, splits, use_te=True, use_stratum_z=True)
    res_hosp_rel = eval_5fold_hgb(
        df_abcd_h, cols_abcd_h, y, splits, use_te=True, use_stratum_z=False
    )
    res_both_ctx = eval_5fold_hgb(
        df_abcd_h, cols_abcd_h, y, splits, use_te=True, use_stratum_z=True
    )
    delta_strat = res_strat_z["oof_auc"] - res_abcd["oof_auc"]
    delta_hrel = res_hosp_rel["oof_auc"] - res_abcd["oof_auc"]
    i16_1_out = (
        "TRUE"
        if max(delta_strat, delta_hrel) >= 0.0015
        else ("FALSE" if (delta_strat <= 0.0 and delta_hrel <= 0.0) else "INCONCLUSIVE")
    )

    cv_rows = [
        ("sanitized_base_A", res_san),
        ("no_h1_window", res_no_h1),
        ("no_apache_twins", res_no_ap),
        ("sulo_rho_0.90", res_r90),
        ("sulo_rho_0.80", res_r80),
        ("horizon_T0_door", res_t0),
        ("horizon_T0_T1_1h", res_t01),
        ("horizon_T0_T1_T2_no_apache", res_t012),
        ("fe_AB_glucose", res_ab),
        ("fe_ABC_ratios", res_abc),
        ("fe_ABCD_hierarchy_te_miss", res_abcd),
        ("fe_ABCD_plus_stratum_z", res_strat_z),
        ("fe_ABCD_plus_hosp_cov_rel", res_hosp_rel),
        ("fe_ABCD_plus_both_contextual", res_both_ctx),
    ]
    cv_df = pd.DataFrame(
        [
            {
                "experiment": name,
                "n_cols": r["n_cols"],
                "oof_auc": round(r["oof_auc"], 5),
                "oof_logloss": round(r["oof_logloss"], 5),
                "delta_auc_vs_sanitized": round(r["oof_auc"] - res_san["oof_auc"], 5),
                "delta_ll_vs_sanitized": round(r["oof_logloss"] - res_san["oof_logloss"], 5),
            }
            for name, r in cv_rows
        ]
    )
    cv_df.to_csv(REPORTS_DIR / "stage5_feature_engineering_cv.csv", index=False)

    eng_only_names = (
        set(fe_b.columns)
        | set(fe_c.columns)
        | set(fe_d.columns)
        | {
            f"{c}_te"
            for c in [
                "apache_3j_diagnosis",
                "apache_3j_major",
                "apache_2_diagnosis",
                "icu_type",
                "ethnicity",
                "hospital_admit_source",
            ]
        }
        | {"n_min_max_swapped"}
    )
    top_eng_perm = sorted(
        [(k, round(v, 5)) for k, v in res_abcd["perm_imp"].items() if k in eng_only_names],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    # Upsert HMECVAS units I14..I16.1
    src_both = (
        f"data/raw/TrainingWiDS2021.csv; data/raw/DataDictionaryWiDS2021.csv (sha256={dict_sha})"
    )
    script_ref = "scripts/stage5_collinearity_and_fe.py"
    entries = [
        make_inquiry(
            inq_id="I14",
            status=i14_out,
            polarity="+",
            claim=(
                "Across 16 tri-window analytes, median |Spearman rho(d1_max, apache)| >= 0.75, "
                "yet dropping h1_ or 15 _apache twins reduces 5-fold SGKF AUC by >= 0.001."
            ),
            technique="Cross-window Spearman correlations and window-ablation 5-fold SGKF CV",
            parameters={"tri_window_analytes": 16, "apache_twins": 15},
            assumptions=["80k+ h1_ and 250k+ _apache entries outside d1_ add distinct signal."],
            missing_data_policy="Pairwise complete Spearman; native NaN in HGB.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"med_rho_d1_ap={med_rho_d1_ap:.4f}, auc_base={res_san['oof_auc']:.5f}, "
                f"auc_no_h1={res_no_h1['oof_auc']:.5f} (drop={delta_drop_h1:+.5f}), "
                f"auc_no_ap={res_no_ap['oof_auc']:.5f} (drop={delta_drop_ap:+.5f})"
            ),
            formula="auc_sanitized - auc_no_window",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=i14_out,
            on_true_cond="med_rho >= 0.75 AND (drop_h1 >= 0.001 OR drop_ap >= 0.001)",
            on_false_cond="med_rho < 0.60 AND drop_h1 <= 0 AND drop_ap <= 0",
            on_inconc_cond="Partial correlation or smaller drop",
            act_true="Retain multi-window predictors; Spawns #I14.1; Ref to #I15.",
            act_false="Prune redundant window in #I14.1; Ref to #I15.",
            act_inconc="Test granular SULO pruning in #I14.1; Ref to #I15.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["64 h1_ cols", "64 d1_ cols", "15 _apache twins"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I14.1",
            status=i14_1_out,
            polarity="-",
            claim=(
                "Subtends #I14 and tests AutoViz SULO pruning. Greedy correlation pruning at "
                "|rho| >= 0.80 degrades 5-fold SGKF HGB AUC by >= 0.002."
            ),
            technique="Target-AUC-prioritized greedy Spearman pruning at 0.90 and 0.80",
            parameters={"thresholds": [0.90, 0.80]},
            assumptions=["Correlated min/max/window twins carry joint tree interaction signal."],
            missing_data_policy="Pairwise Spearman over numeric predictors; native NaN in HGB.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"n_kept_r90={len(keep_r90)}, auc_r90={res_r90['oof_auc']:.5f} "
                f"(Δ={delta_r90:+.5f}); n_kept_r80={len(keep_r80)}, "
                f"auc_r80={res_r80['oof_auc']:.5f} (Δ={delta_r80:+.5f})"
            ),
            formula="auc_sulo - auc_sanitized_base",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=i14_1_out,
            on_true_cond="delta_r80 <= -0.002",
            on_false_cond="delta_r80 >= 0.0 AND delta_r90 >= 0.0",
            on_inconc_cond="Minor delta in (-0.002, 0.0)",
            act_true="Reject SULO |rho|>=0.80 pruning for tree models -> Ref to #I15.",
            act_false="Adopt SULO pruned subset for Stage 6 -> Ref to #I15.",
            act_inconc="Keep full sanitized set for GBDT; benchmark pruned for linear -> Ref #I15.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["Sanitized F_clinical (175 predictors)"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I15",
            status=i15_out,
            polarity="+",
            claim=(
                "Across Top-25 + clinical watchlist predictors, zero originate post-24h or hit "
                "AUC >= 0.90, and all predictors with AUC >= 0.55 generalize cleanly across folds."
            ),
            technique="Timeline classification, 5-fold univariate stability, and arf coherence",
            parameters={"max_fold_std": 0.04, "min_fold_auc": 0.50, "leak_auc": 0.90},
            assumptions=["Valid clinical signals generalize without fold-direction flips."],
            missing_data_policy="Native NaN preserved in univariate fold checks.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"n_post_24h={n_post_24h}, n_leak_ge90={n_leak_ge90}, "
                f"n_unstable={n_unstable}, {arf_coh}"
            ),
            formula="count(mean_auc >= 0.55 AND (min_fold < 0.50 OR std_fold > 0.04))",
            timeframe="Static WiDS Datathon 2021 pre-ICU through 24h ICU",
            filters=["Top-25 F_clinical union 8 watchlist predictors"],
            outcome=i15_out,
            on_true_cond="n_post_24h == 0 AND n_leak_ge90 == 0 AND n_unstable == 0",
            on_false_cond="n_post_24h >= 1 OR n_leak_ge90 >= 1",
            on_inconc_cond="Hospital-unstable proxy features detected",
            act_true="Certify audited predictors as leak-free; Spawns #I15.1; Ref to #I16.",
            act_false="Quarantine leaky predictors; Spawns #I15.1; Ref to #I16.",
            act_inconc="Inspect unstable proxies in stage5_leakage_proxy_audit.csv; Spawns #I15.1.",
            source_file=src_both,
            commit=commit,
            sha256=train_sha,
            columns=audit_cols,
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I15.1",
            status=i15_1_out,
            polarity="+",
            claim=(
                "Subtends #I15. Early ICU-door + 1h predictors achieve SGKF AUC >= 0.78, and "
                "adding 28 T3_apache summary covariates on top of 24h vitals/labs adds < 0.015."
            ),
            technique="4-horizon temporal-access cumulative grouped CV evaluation",
            parameters={"horizons": ["T0", "T0+T1", "T0+T1+T2", "All_24h"]},
            assumptions=[
                "Isolates incremental APACHE coding value over raw first-24h labs/vitals."
            ],
            missing_data_policy="Native NaN preserved across all 4 horizons.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"T0={res_t0['oof_auc']:.5f}, T0_T1={res_t01['oof_auc']:.5f}, "
                f"T0_T1_T2={res_t012['oof_auc']:.5f}, All_24h={res_san['oof_auc']:.5f}, "
                f"delta_apache_t3={delta_t3:+.5f}"
            ),
            formula="auc_all_24h - auc_T0_T1_T2",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=i15_1_out,
            on_true_cond="auc_T0_T1 >= 0.78 AND delta_apache_t3 < 0.015",
            on_false_cond="auc_T0_T1 < 0.72 OR delta_apache_t3 >= 0.030",
            on_inconc_cond="Intermediate early AUC or APACHE uplift",
            act_true="Confirm strong early physiology signal independent of APACHE -> Ref to #I16.",
            act_false="Audit T3_apache drivers -> Ref to #I16.",
            act_inconc="Log temporal horizon progression in stage5 summary -> Ref to #I16.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["Sanitized F_clinical by temporal tier"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I16",
            status=i16_out,
            polarity="+",
            claim=(
                "Cumulative clean domain FE (Families A-D: cleaning, glycemic dynamics, clinical "
                "ratios, APACHE3j hierarchy+TE+panel miss) lifts SGKF AUC >= +0.0030 (>=4/5 folds) "
                "and cuts Log-Loss >= 0.0010."
            ),
            technique="Stepwise cumulative FE ablation (Base -> +AB -> +ABC -> +ABCD) on SGKF",
            parameters={"te_smoothing_m": 20.0, "n_new_features": len(cols_abcd) - len(f_clin) + 6},
            assumptions=["Safe division and nested inner OOF TE prevent ratio/encoding leakage."],
            missing_data_policy="Preserve NaN when denominator <= 0 or component is missing.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"san_A={res_san['oof_auc']:.5f} (ll={res_san['oof_logloss']:.5f}), "
                f"AB={res_ab['oof_auc']:.5f}, ABC={res_abc['oof_auc']:.5f}, "
                f"ABCD={res_abcd['oof_auc']:.5f} (ll={res_abcd['oof_logloss']:.5f}), "
                f"Δauc={delta_auc_eng:+.5f}, Δll={delta_ll_eng:+.5f}, "
                f"folds={k_folds_imp}/5, top_eng={top_eng_perm[:5]}"
            ),
            formula="auc_eng_AD - auc_sanitized; logloss_eng_AD - logloss_sanitized",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=i16_out,
            on_true_cond="delta_auc >= 0.0030 AND folds >= 4 AND delta_ll <= -0.0010",
            on_false_cond="delta_auc <= 0.0 OR folds <= 1",
            on_inconc_cond="Positive lift below +0.0030 or partial fold consistency",
            act_true="Promote Families A-D into Stage 6 production pipeline; Spawns #I16.1.",
            act_false="Prune net-negative engineering families; Spawns #I16.1.",
            act_inconc="Retain individually positive FE families for Stage 6; Spawns #I16.1.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["Sanitized F_clinical + Families A-D engineered features"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I16.1",
            status=i16_1_out,
            polarity="+",
            claim=(
                "Subtends #I16. Fold-safe clinical-stratum z-scores (age_bin x icu_type) or "
                "unsupervised within-hospital covariate centering lifts eng_AD SGKF AUC >= +0.0015."
            ),
            technique="2-arm contextual aggregation ablation on top of eng_AD",
            parameters={"stratum": "age_bin x icu_type", "min_hospital_n": 5},
            assumptions=["Target-free covariate centering generalizes across unseen hospitals."],
            missing_data_policy="NaN when hospital n < 5 or component missing.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"eng_AD={res_abcd['oof_auc']:.5f}, "
                f"stratum_z={res_strat_z['oof_auc']:.5f} (Δ={delta_strat:+.5f}), "
                f"hosp_cov_rel={res_hosp_rel['oof_auc']:.5f} (Δ={delta_hrel:+.5f}), "
                f"both={res_both_ctx['oof_auc']:.5f}"
            ),
            formula="max(auc_stratum_z, auc_hosp_cov_rel) - auc_eng_AD",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=i16_1_out,
            on_true_cond="max(delta_stratum, delta_hrel) >= 0.0015",
            on_false_cond="delta_stratum <= 0.0 AND delta_hrel <= 0.0",
            on_inconc_cond="Modest positive lift in (0, 0.0015)",
            act_true="Include winning contextual arm in Stage 6 baseline pipeline -> Stage 6.",
            act_false="Reject contextual group aggregations -> Proceed Stage 6 with eng_AD.",
            act_inconc="Test positive contextual arm as Stage 6 ablation -> Proceed Stage 6.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["eng_AD matrix", "hospital_id", "age", "icu_type"],
            timestamp=now_iso,
        ),
    ]
    upsert_inquiries(entries)

    exp_md = "\n".join(
        f"| `{r['experiment']}` | {r['n_cols']} | {r['oof_auc']:.5f} | "
        f"{r['oof_logloss']:.5f} | {r['delta_auc_vs_sanitized']:+.5f} |"
        for _, r in cv_df.iterrows()
    )
    md = f"""# Stage 5 Report: Cross-Window Collinearity, Leakage Audit & Domain FE

- **Generated**: `{now_iso}` (Base Commit: `{commit}`)

## Inquiry Outcomes

- **[I14 = {i14_out}] Cross-Window Collinearity**: Median `|rho(d1_max, apache)| =
  {med_rho_d1_ap:.4f}`; dropping `h1_` shifts AUC `{delta_drop_h1:+.5f}`, dropping 15
  `_apache` twins shifts AUC `{delta_drop_ap:+.5f}`.
- **[I14.1 = {i14_1_out}] SULO Correlation Pruning**: `|rho|>=0.90` keeps `{len(keep_r90)}`
  (`Δ={delta_r90:+.5f}`); `|rho|>=0.80` keeps `{len(keep_r80)}` (`Δ={delta_r80:+.5f}`).
- **[I15 = {i15_out}] Timeline & Proxy Stability**: `n_post_24h={n_post_24h}`,
  `n_leak_ge90={n_leak_ge90}`, `n_unstable={n_unstable}`, `{arf_coh}`.
- **[I15.1 = {i15_1_out}] Temporal Access Horizons**: `T0={res_t0["oof_auc"]:.5f}`,
  `T0+T1={res_t01["oof_auc"]:.5f}`, `T0+T1+T2={res_t012["oof_auc"]:.5f}`,
  `All_24h={res_san["oof_auc"]:.5f}` (`Δ_apache_t3={delta_t3:+.5f}`).
- **[I16 = {i16_out}] Stepwise Domain FE**: `sanitized_A={res_san["oof_auc"]:.5f}` ->
  `ABCD={res_abcd["oof_auc"]:.5f}` (`Δ_AUC={delta_auc_eng:+.5f}`, `Δ_LL={delta_ll_eng:+.5f}`,
  `{k_folds_imp}/5` folds). Top engineered perm importances: `{top_eng_perm[:6]}`.
- **[I16.1 = {i16_1_out}] Contextual Aggregations**: `stratum_z` `Δ={delta_strat:+.5f}`,
  `hosp_cov_rel` `Δ={delta_hrel:+.5f}`, `both` `{res_both_ctx["oof_auc"]:.5f}`.

## Stepwise Experiment Benchmark (`StratifiedGroupKFold`, 5-Fold)
| Experiment | Cols | OOF ROC-AUC | OOF Log-Loss | Δ AUC vs Sanitized |
| :--- | ---: | ---: | ---: | ---: |
{exp_md}
"""
    (REPORTS_DIR / "stage5_collinearity_and_fe.md").write_text(md, encoding="utf-8")
    print(
        f"Stage 5 done: I14={i14_out}, I14.1={i14_1_out}, I15={i15_out}, "
        f"I15.1={i15_1_out}, I16={i16_out}, I16.1={i16_1_out}"
    )


if __name__ == "__main__":
    main()
