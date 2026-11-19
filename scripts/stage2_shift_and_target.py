"""Stage 2 EDA: Facility Collisions, Adversarial Shift & Target Prevalence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from project_name.config import PROJECT_ROOT
from project_name.hmecvas import (
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
DROP_STAGE1 = {"Unnamed: 0", "encounter_id", "readmission_status"}
DUP_ENCOUNTERS_167 = [
    159417,
    244601,
    150518,
    264918,
    225543,
    147789,
    234525,
    189191,
    197188,
]


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Return 95% Wilson score interval (lower, upper) in [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, float(center - half)), min(1.0, float(center + half)))


def compute_psi(tr_vals: np.ndarray, te_vals: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index using Train deciles."""
    tr = tr_vals[~np.isnan(tr_vals)]
    te = te_vals[~np.isnan(te_vals)]
    if len(tr) < 10 or len(te) < 10:
        return 0.0
    qs = np.unique(np.quantile(tr, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    tr_cnt = np.histogram(tr, bins=qs)[0]
    te_cnt = np.histogram(te, bins=qs)[0]
    tr_p = np.clip(tr_cnt / len(tr), 1e-6, None)
    te_p = np.clip(te_cnt / len(te), 1e-6, None)
    return float(np.sum((te_p - tr_p) * np.log(te_p / tr_p)))


def fit_eval_hgb_cv(
    df: pd.DataFrame,
    y: np.ndarray,
    cols: list[str],
    cat_cols: list[str],
    splitter,
    groups: np.ndarray | None = None,
    impute_all: bool = False,
    impute_subset: set[str] | None = None,
    compute_perm_first_fold: bool = False,
) -> tuple[float, dict[str, float]]:
    """Run 5-fold CV HistGradientBoostingClassifier and return OOF ROC-AUC."""
    oof = np.zeros(len(df), dtype=float)
    perm_imp: dict[str, float] = {}
    splits = splitter.split(df, y, groups) if groups is not None else splitter.split(df, y)
    num_cols = [c for c in cols if c not in set(cat_cols)]

    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        x_tr = df.iloc[tr_idx][cols].copy()
        x_va = df.iloc[va_idx][cols].copy()

        if impute_all:
            for c in num_cols:
                med = x_tr[c].median()
                med = 0.0 if pd.isna(med) else float(med)
                x_tr[c] = x_tr[c].fillna(med)
                x_va[c] = x_va[c].fillna(med)
            for c in cat_cols:
                if c in cols:
                    modes = x_tr[c].dropna().mode()
                    mv = modes.iloc[0] if len(modes) else "__MISSING__"
                    x_tr[c] = x_tr[c].fillna(mv)
                    x_va[c] = x_va[c].fillna(mv)
        elif impute_subset:
            for c in impute_subset:
                if c in x_tr.columns:
                    med = x_tr[c].median()
                    med = 0.0 if pd.isna(med) else float(med)
                    x_tr[c] = x_tr[c].fillna(med)
                    x_va[c] = x_va[c].fillna(med)

        active_cats = [c for c in cat_cols if c in cols]
        if active_cats:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=np.nan,
            )
            x_tr[active_cats] = enc.fit_transform(x_tr[active_cats].astype(object))
            x_va[active_cats] = enc.transform(x_va[active_cats].astype(object))

        clf = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            categorical_features=active_cats if active_cats else None,
            early_stopping=True,
            random_state=42,
        )
        clf.fit(x_tr, y[tr_idx])
        oof[va_idx] = clf.predict_proba(x_va)[:, 1]

        if compute_perm_first_fold and fold_idx == 0:
            res = permutation_importance(
                clf, x_va, y[va_idx], scoring="roc_auc", n_repeats=3, random_state=42, n_jobs=1
            )
            perm_imp = {c: float(m) for c, m in zip(cols, res.importances_mean, strict=True)}

    return float(roc_auc_score(y, oof)), perm_imp


def run_i4_and_i4_1(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict]:
    """Audit icu_id-to-hospital_id collisions and StratifiedGroupKFold balance."""
    tr_sub = train[["hospital_id", "icu_id"]].assign(split="train")
    te_sub = test[["hospital_id", "icu_id"]].assign(split="test")
    both = pd.concat([tr_sub, te_sub], ignore_index=True)

    grp_all = (
        both.groupby("icu_id", dropna=False)
        .agg(n_hosp_all=("hospital_id", "nunique"), n_rows_all=("hospital_id", "size"))
        .reset_index()
    )
    grp_tr = (
        tr_sub.groupby("icu_id", dropna=False)
        .agg(n_hosp_tr=("hospital_id", "nunique"), n_rows_tr=("hospital_id", "size"))
        .reset_index()
    )
    grp_te = (
        te_sub.groupby("icu_id", dropna=False)
        .agg(n_hosp_te=("hospital_id", "nunique"), n_rows_te=("hospital_id", "size"))
        .reset_index()
    )
    icu_map = grp_all.merge(grp_tr, on="icu_id", how="left").merge(grp_te, on="icu_id", how="left")
    for c in ["n_hosp_tr", "n_rows_tr", "n_hosp_te", "n_rows_te"]:
        icu_map[c] = icu_map[c].fillna(0).astype(int)
    icu_map = icu_map.sort_values(["n_hosp_all", "n_rows_all"], ascending=False)
    icu_map.to_csv(REPORTS_DIR / "stage2_icu_hospital_mapping.csv", index=False)

    comb_multi = float((icu_map["n_hosp_all"] >= 2).mean())
    tr_only_map = icu_map[icu_map["n_rows_tr"] > 0]
    tr_multi = float((tr_only_map["n_hosp_tr"] >= 2).mean())
    max_h_all = int(icu_map["n_hosp_all"].max())
    max_h_tr = int(tr_only_map["n_hosp_tr"].max())
    nmi_tr = float(
        normalized_mutual_info_score(train["hospital_id"].to_numpy(), train["icu_id"].to_numpy())
    )
    hosp_sizes = train.groupby("hospital_id").size()
    hosp_q = hosp_sizes.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()

    i4_out = (
        "TRUE"
        if (comb_multi >= 0.50 and tr_multi > 0.0)
        else ("FALSE" if (comb_multi == 0.0 and tr_multi == 0.0) else "INCONCLUSIVE")
    )

    # I4.1: StratifiedGroupKFold verification & icu_id smoothed target-encoding leakage
    y_tr = train["diabetes_mellitus"].to_numpy(dtype=int)
    g_tr = train["hospital_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_ids = np.zeros(len(train), dtype=int)
    fold_shares, fold_rates = [], []
    hosp_fold_sets = []
    for f_i, (_, va_idx) in enumerate(sgkf.split(train, y_tr, g_tr)):
        fold_ids[va_idx] = f_i
        fold_shares.append(100.0 * len(va_idx) / len(train))
        fold_rates.append(100.0 * y_tr[va_idx].mean())
        hosp_fold_sets.append(set(g_tr[va_idx]))

    fold_share_rng = float(max(fold_shares) - min(fold_shares))
    fold_rate_rng = float(max(fold_rates) - min(fold_rates))
    max_h_overlap = max(
        len(hosp_fold_sets[a] & hosp_fold_sets[b]) for a in range(5) for b in range(a + 1, 5)
    )
    dup_mask = train["encounter_id"].isin(DUP_ENCOUNTERS_167)
    dup_folds = int(len(set(fold_ids[dup_mask])))

    def oof_icu_te_auc(splitter, use_groups: bool) -> float:
        oof = np.zeros(len(train), dtype=float)
        icu_s = train["icu_id"].fillna("__MISSING__").astype(str)
        it = splitter.split(train, y_tr, g_tr) if use_groups else splitter.split(train, y_tr)
        for tr_i, va_i in it:
            prior = float(y_tr[tr_i].mean())
            tmp = pd.DataFrame({"icu": icu_s.iloc[tr_i].to_numpy(), "y": y_tr[tr_i]})
            agg = tmp.groupby("icu")["y"].agg(["sum", "count"])
            enc_map = (agg["sum"] + 10.0 * prior) / (agg["count"] + 10.0)
            oof[va_i] = icu_s.iloc[va_i].map(enc_map).fillna(prior).to_numpy()
        return float(roc_auc_score(y_tr, oof))

    auc_rand = oof_icu_te_auc(skf, False)
    auc_grp = oof_icu_te_auc(sgkf, True)

    bal_ok = (
        fold_share_rng <= 5.0 and fold_rate_rng <= 1.0 and max_h_overlap == 0 and dup_folds == 1
    )
    leak_ok = auc_rand >= 0.55 and auc_grp < 0.52
    i4_1_out = "TRUE" if (bal_ok and leak_ok) else ("FALSE" if not bal_ok else "INCONCLUSIVE")

    return (
        {
            "outcome": i4_out,
            "comb_multi": comb_multi,
            "tr_multi": tr_multi,
            "max_h_all": max_h_all,
            "max_h_tr": max_h_tr,
            "nmi_tr": nmi_tr,
            "hosp_q": {str(k): int(v) for k, v in hosp_q.items()},
            "n_icu_all": len(icu_map),
            "n_icu_tr": len(tr_only_map),
        },
        {
            "outcome": i4_1_out,
            "fold_share_rng": fold_share_rng,
            "fold_rate_rng": fold_rate_rng,
            "max_h_overlap": max_h_overlap,
            "dup_folds": dup_folds,
            "auc_rand": auc_rand,
            "auc_grp": auc_grp,
        },
    )


def run_i5_and_i5_1(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict]:
    """Run out-of-hospital Adversarial Validation and missingness-vs-value ablation."""
    f_clin = [c for c in test.columns if c not in (DROP_STAGE1 | {"hospital_id", "icu_id"})]
    tr_df = train[f_clin + ["hospital_id", "icu_id"]].copy()
    te_df = test[f_clin + ["hospital_id", "icu_id"]].copy()
    tr_df["is_test"] = 0
    te_df["is_test"] = 1
    comb = pd.concat([tr_df, te_df], ignore_index=True)

    y_adv = comb["is_test"].to_numpy(dtype=int)
    g_adv = comb["hospital_id"].to_numpy()
    cat_cols = [c for c in f_clin if not pd.api.types.is_numeric_dtype(comb[c])]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_raw_grp, perm_raw = fit_eval_hgb_cv(
        comb, y_adv, f_clin, cat_cols, sgkf, g_adv, compute_perm_first_fold=True
    )
    auc_raw_rand, _ = fit_eval_hgb_cv(comb, y_adv, f_clin, cat_cols, skf)
    auc_imp_grp, _ = fit_eval_hgb_cv(comb, y_adv, f_clin, cat_cols, sgkf, g_adv, impute_all=True)
    auc_no_anthro, _ = fit_eval_hgb_cv(
        comb,
        y_adv,
        f_clin,
        cat_cols,
        sgkf,
        g_adv,
        impute_subset={"age", "bmi", "weight"},
    )
    auc_with_icu, _ = fit_eval_hgb_cv(comb, y_adv, f_clin + ["icu_id"], cat_cols, sgkf, g_adv)

    miss_df = comb[f_clin].isna().astype(np.int8)
    miss_df.columns = [f"{c}_isna" for c in f_clin]
    auc_miss_only, _ = fit_eval_hgb_cv(miss_df, y_adv, list(miss_df.columns), [], sgkf, g_adv)

    # Univariate KS / PSI / Chi2 drift across F_clinical + icu_id
    drift_rows = []
    for c in f_clin + ["icu_id"]:
        is_num = pd.api.types.is_numeric_dtype(comb[c])
        miss_tr = float(train[c].isna().mean() * 100.0)
        miss_te = float(test[c].isna().mean() * 100.0)
        if is_num:
            v_tr = train[c].dropna().to_numpy(dtype=float)
            v_te = test[c].dropna().to_numpy(dtype=float)
            ks_res = stats.ks_2samp(v_tr, v_te) if len(v_tr) > 0 and len(v_te) > 0 else None
            ks_stat = float(ks_res.statistic) if ks_res else 0.0
            ks_p = float(ks_res.pvalue) if ks_res else 1.0
            psi = compute_psi(v_tr, v_te)
            tvd, chi2_p = np.nan, np.nan
        else:
            ks_stat, ks_p, psi = np.nan, np.nan, np.nan
            s_tr = train[c].fillna("__MISSING__").astype(str)
            s_te = test[c].fillna("__MISSING__").astype(str)
            cats_all = sorted(set(s_tr.unique()) | set(s_te.unique()))
            c_tr = s_tr.value_counts().reindex(cats_all, fill_value=0).to_numpy()
            c_te = s_te.value_counts().reindex(cats_all, fill_value=0).to_numpy()
            p_tr, p_te = c_tr / c_tr.sum(), c_te / c_te.sum()
            tvd = float(0.5 * np.abs(p_tr - p_te).sum())
            _, chi2_p, _, _ = stats.chi2_contingency(np.vstack([c_tr, c_te]))
            chi2_p = float(chi2_p)
        drift_rows.append(
            {
                "column": c,
                "is_numeric": is_num,
                "miss_pct_train": round(miss_tr, 4),
                "miss_pct_test": round(miss_te, 4),
                "abs_miss_shift_pp": round(abs(miss_tr - miss_te), 4),
                "ks_stat": round(ks_stat, 6) if pd.notna(ks_stat) else np.nan,
                "ks_p": ks_p,
                "psi_10bin": round(psi, 6) if pd.notna(psi) else np.nan,
                "cat_tvd": round(tvd, 6) if pd.notna(tvd) else np.nan,
                "cat_chi2_p": chi2_p,
                "adv_perm_importance": round(perm_raw.get(c, 0.0), 6),
            }
        )
    drift_df = pd.DataFrame(drift_rows).sort_values(
        ["adv_perm_importance", "psi_10bin"], ascending=[False, False]
    )
    drift_df.to_csv(REPORTS_DIR / "stage2_adversarial_validation.csv", index=False)

    i5_out = "TRUE" if auc_raw_grp >= 0.65 else ("FALSE" if auc_raw_grp < 0.58 else "INCONCLUSIVE")
    delta_miss = auc_raw_grp - auc_imp_grp
    delta_icu = auc_with_icu - auc_raw_grp
    i5_1_out = (
        "TRUE"
        if ((auc_miss_only >= 0.60 and delta_miss >= 0.03) or delta_icu >= 0.03)
        else (
            "FALSE"
            if (auc_miss_only < 0.55 and delta_miss < 0.01 and delta_icu < 0.01)
            else "INCONCLUSIVE"
        )
    )
    top_drivers = drift_df.head(10)[["column", "adv_perm_importance", "psi_10bin", "ks_stat"]]
    return (
        {
            "outcome": i5_out,
            "auc_raw_grp": auc_raw_grp,
            "auc_raw_rand": auc_raw_rand,
            "top_drivers": top_drivers.to_dict(orient="records"),
        },
        {
            "outcome": i5_1_out,
            "auc_imp_grp": auc_imp_grp,
            "auc_miss_only": auc_miss_only,
            "auc_with_icu": auc_with_icu,
            "auc_no_anthro": auc_no_anthro,
            "delta_miss": delta_miss,
            "delta_icu": delta_icu,
            "n_psi_ge_01": int((drift_df["psi_10bin"] >= 0.10).sum()),
        },
    )


def add_strata_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Create age_bin and bmi_bin columns with explicit __MISSING__ category."""
    out = pd.DataFrame(index=df.index)
    age_bins = pd.cut(
        df["age"],
        bins=[-np.inf, 45, 65, 80, np.inf],
        labels=["<45", "45-64", "65-79", "80+"],
        right=False,
    )
    bmi_bins = pd.cut(
        df["bmi"],
        bins=[-np.inf, 18.5, 25, 30, 40, np.inf],
        labels=["<18.5", "18.5-24.9", "25-29.9", "30-39.9", "40+"],
        right=False,
    )
    out["age_bin"] = age_bins.astype(object).fillna("__MISSING__").astype(str)
    out["bmi_bin"] = bmi_bins.astype(object).fillna("__MISSING__").astype(str)
    for col in [
        "ethnicity",
        "icu_type",
        "elective_surgery",
        "gender",
        "icu_admit_source",
        "arf_apache",
    ]:
        out[col] = df[col].fillna("__MISSING__").astype(str)
    return out


def run_i6_and_i6_1(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict]:
    """Audit inter-hospital target spread, subgroup prevalence, and missing-stratum bias."""
    y = train["diabetes_mellitus"].to_numpy(dtype=int)
    pos_rate_pct = float(y.mean() * 100.0)

    hosp_agg = (
        train.groupby("hospital_id")["diabetes_mellitus"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    hosp_tab = pd.crosstab(train["hospital_id"], train["diabetes_mellitus"])
    hosp_chi2_p = float(stats.chi2_contingency(hosp_tab)[1])
    hosp_30 = hosp_agg[hosp_agg["count"] >= 30].copy()
    hosp_rates = hosp_30["mean"] * 100.0
    hosp_iqr = float(hosp_rates.quantile(0.75) - hosp_rates.quantile(0.25))
    hosp_p10 = float(hosp_rates.quantile(0.10))
    hosp_p90 = float(hosp_rates.quantile(0.90))

    tr_strata = add_strata_cols(train)
    te_strata = add_strata_cols(test)
    core_4 = ["age_bin", "ethnicity", "icu_type", "elective_surgery"]
    sec_4 = ["bmi_bin", "gender", "icu_admit_source", "arf_apache"]

    rows = []
    k_sig_core = 0
    for s_name in core_4 + sec_4:
        tab = pd.crosstab(tr_strata[s_name], y)
        chi2_p = float(stats.chi2_contingency(tab)[1])
        te_props = te_strata[s_name].value_counts(normalize=True)
        grp_stats = []
        for lvl, grp_idx in tr_strata.groupby(s_name).groups.items():
            n_lvl = len(grp_idx)
            k_lvl = int(y[grp_idx].sum())
            rate = k_lvl / n_lvl
            lo, hi = wilson_ci(k_lvl, n_lvl)
            grp_stats.append((lvl, n_lvl, k_lvl, rate * 100.0, lo * 100.0, hi * 100.0))
        rates_30 = [r[3] for r in grp_stats if r[1] >= 30]
        spread = float(max(rates_30) - min(rates_30)) if len(rates_30) >= 2 else 0.0
        if s_name in core_4 and chi2_p < 0.001 and spread >= 3.0:
            k_sig_core += 1
        for lvl, n_lvl, k_lvl, rate_pct, lo_pct, hi_pct in grp_stats:
            rows.append(
                {
                    "stratum": s_name,
                    "level": lvl,
                    "n_train": n_lvl,
                    "pos_count": k_lvl,
                    "prev_pct": round(rate_pct, 4),
                    "wilson_lo_pct": round(lo_pct, 4),
                    "wilson_hi_pct": round(hi_pct, 4),
                    "train_share_pct": round(100.0 * n_lvl / len(train), 4),
                    "test_share_pct": round(100.0 * float(te_props.get(lvl, 0.0)), 4),
                    "stratum_chi2_p": chi2_p,
                    "stratum_range_pp_n30": round(spread, 4),
                }
            )

    # Also append hospital summary rows into stage2_target_prevalence.csv
    for _, hr in hosp_agg.iterrows():
        lo, hi = wilson_ci(int(hr["sum"]), int(hr["count"]))
        rows.append(
            {
                "stratum": "hospital_id",
                "level": str(int(hr["hospital_id"])),
                "n_train": int(hr["count"]),
                "pos_count": int(hr["sum"]),
                "prev_pct": round(float(hr["mean"] * 100.0), 4),
                "wilson_lo_pct": round(lo * 100.0, 4),
                "wilson_hi_pct": round(hi * 100.0, 4),
                "train_share_pct": round(100.0 * int(hr["count"]) / len(train), 4),
                "test_share_pct": np.nan,
                "stratum_chi2_p": hosp_chi2_p,
                "stratum_range_pp_n30": round(hosp_iqr, 4),
            }
        )

    prev_df = pd.DataFrame(rows)
    prev_df.to_csv(REPORTS_DIR / "stage2_target_prevalence.csv", index=False)

    i6_out = (
        "TRUE"
        if (hosp_chi2_p < 0.001 and hosp_iqr >= 4.0 and k_sig_core >= 3)
        else (
            "FALSE"
            if ((hosp_chi2_p >= 0.05 or hosp_iqr < 2.0) and k_sig_core == 0)
            else "INCONCLUSIVE"
        )
    )

    # I6.1: Missing-stratum bias & direct standardization prior shift
    miss_bias_details = []
    k_miss_bias = 0
    for col in ["age", "bmi", "weight", "ethnicity"]:
        is_na = train[col].isna().to_numpy()
        n_na = int(is_na.sum())
        if n_na >= 30:
            p_na = float(y[is_na].mean() * 100.0)
            p_ok = float(y[~is_na].mean() * 100.0)
            diff_pp = abs(p_na - p_ok)
            p_val = float(stats.chi2_contingency(pd.crosstab(is_na, y))[1])
            if p_val < 0.01 and diff_pp >= 2.0:
                k_miss_bias += 1
            miss_bias_details.append(
                {
                    "col": col,
                    "n_na": n_na,
                    "prev_na_pct": round(p_na, 3),
                    "prev_present_pct": round(p_ok, 3),
                    "abs_diff_pp": round(diff_pp, 3),
                    "chi2_p": p_val,
                }
            )

    std_shifts = {}
    for s_col in ["age_bin", "bmi_bin", "ethnicity", "icu_type"]:
        tr_rates = pd.Series(y, index=tr_strata.index).groupby(tr_strata[s_col]).mean()
        te_weights = te_strata[s_col].value_counts(normalize=True)
        proj = float(sum(te_weights[k] * tr_rates.get(k, y.mean()) for k in te_weights.index))
        std_shifts[s_col] = round((proj * 100.0) - pos_rate_pct, 4)

    max_shift_pp = float(max(abs(v) for v in std_shifts.values()))
    i6_1_out = (
        "TRUE"
        if (k_miss_bias >= 1 and max_shift_pp >= 0.50)
        else ("FALSE" if (k_miss_bias == 0 and max_shift_pp < 0.20) else "INCONCLUSIVE")
    )

    return (
        {
            "outcome": i6_out,
            "pos_rate_pct": pos_rate_pct,
            "n_hosp_total": len(hosp_agg),
            "n_hosp_ge30": len(hosp_30),
            "hosp_chi2_p": hosp_chi2_p,
            "hosp_iqr_pp": hosp_iqr,
            "hosp_p10_pp": hosp_p10,
            "hosp_p90_pp": hosp_p90,
            "k_sig_core": k_sig_core,
        },
        {
            "outcome": i6_1_out,
            "k_miss_bias": k_miss_bias,
            "miss_bias_details": miss_bias_details,
            "std_shifts_pp": std_shifts,
            "max_shift_pp": max_shift_pp,
        },
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now_iso = iso_now_plus8()
    git_commit = current_git_commit()
    train_sha = sha256_file(TRAIN_PATH)
    test_sha = sha256_file(TEST_PATH)
    train, test, _ = load_raw_wids()

    r4, r4_1 = run_i4_and_i4_1(train, test)
    r5, r5_1 = run_i5_and_i5_1(train, test)
    r6, r6_1 = run_i6_and_i6_1(train, test)

    src_both = f"data/raw/TrainingWiDS2021.csv; data/raw/UnlabeledWiDS2021.csv (sha256={test_sha})"
    script_ref = "scripts/stage2_shift_and_target.py"
    entries = [
        make_inquiry(
            inq_id="I4",
            status=r4["outcome"],
            polarity="+",
            claim=(
                "Across TrainingWiDS2021.csv and UnlabeledWiDS2021.csv, icu_id is recycled across "
                "distinct hospital_id values (>= 50% combined and > 0% within-train collisions)."
            ),
            technique="Bipartite (hospital_id, icu_id) cardinality and NMI audit",
            parameters={"splits": ["train", "test", "combined"]},
            assumptions=["Each encounter maps to one hospital_id and icu_id."],
            missing_data_policy="Null IDs tallied and excluded from pair cardinality.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"comb_multi={r4['comb_multi'] * 100:.2f}%, tr_multi={r4['tr_multi'] * 100:.2f}%, "
                f"max_h_all={r4['max_h_all']}, max_h_tr={r4['max_h_tr']}, "
                f"nmi_tr={r4['nmi_tr']:.4f}, hosp_q={r4['hosp_q']}"
            ),
            formula="|{icu_id : nunique(hospital_id) >= 2}| / |unique(icu_id)|",
            timeframe="Static WiDS Datathon 2021 snapshot",
            filters=["All 130,157 Train and 10,234 Unlabeled rows"],
            outcome=r4["outcome"],
            on_true_cond="comb_multi >= 0.50 AND tr_multi > 0.0",
            on_false_cond="comb_multi == 0.0 AND tr_multi == 0.0",
            on_inconc_cond="Mixed collision across Train vs Combined",
            act_true=(
                "Lock hospital_id as CV group key; audit icu_id leakage in #I4.1 -> Ref to #I5."
            ),
            act_false="Re-audit facility ID parsing -> Ref to #I5.",
            act_inconc="Spawns #I4.1 to test icu_id proxy leakage -> Ref to #I5.",
            source_file=src_both,
            commit=git_commit,
            sha256=train_sha,
            columns=["hospital_id", "icu_id", "icu_type", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I4.1",
            status=r4_1["outcome"],
            polarity="+",
            claim=(
                "Subtends #I4. 5-fold StratifiedGroupKFold(groups=hospital_id) meets fold balance "
                "tolerances while icu_id target-encoding leaks under StratifiedKFold (>=0.55) "
                "and collapses under grouped CV (<0.52)."
            ),
            technique="5-fold StratifiedGroupKFold balance check and OOF smoothed icu_id TE test",
            parameters={"n_splits": 5, "smoothing_m": 10.0, "random_state": 42},
            assumptions=["Grouped CV isolates true out-of-hospital generalization."],
            missing_data_policy="Map null icu_id to '__MISSING__'.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"fold_share_rng={r4_1['fold_share_rng']:.3f}pp, "
                f"fold_rate_rng={r4_1['fold_rate_rng']:.3f}pp, "
                f"hosp_overlap={r4_1['max_h_overlap']}, dup167_folds={r4_1['dup_folds']}, "
                f"auc_rand={r4_1['auc_rand']:.4f}, auc_grp={r4_1['auc_grp']:.4f}"
            ),
            formula="OOF ROC-AUC of smoothed icu_id target encoding under SKF vs SGKF",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=r4_1["outcome"],
            on_true_cond="Balance passes AND auc_rand >= 0.55 AND auc_grp < 0.52",
            on_false_cond="Fold balance tolerances violated",
            on_inconc_cond="Partial balance or icu_id leakage signal",
            act_true=(
                "Lock 5-fold StratifiedGroupKFold(groups=hospital_id) and flag icu_id "
                "-> Ref to #I5."
            ),
            act_false="Adjust StratifiedGroupKFold splits before Stage 6 -> Ref to #I5.",
            act_inconc=(
                "Lock StratifiedGroupKFold and test icu_id ablation in Stage 5/6 -> Ref to #I5."
            ),
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=git_commit,
            sha256=train_sha,
            columns=["hospital_id", "icu_id", "encounter_id", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I5",
            status=r5["outcome"],
            polarity="+",
            claim=(
                "The 175 clinical predictors (F_clinical) exhibit material out-of-hospital "
                "covariate shift between Train and Unlabeled (grouped Adversarial AUC >= 0.65)."
            ),
            technique=(
                "5-fold StratifiedGroupKFold Adversarial Validation with HistGradientBoosting"
            ),
            parameters={"n_splits": 5, "max_iter": 100, "learning_rate": 0.08},
            assumptions=["Disjoint hospital_id prevents hospital memorization under SGKF."],
            missing_data_policy="Preserve NaN natively in HistGradientBoostingClassifier.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"adv_auc_raw_grouped={r5['auc_raw_grp']:.4f}, "
                f"adv_auc_raw_random={r5['auc_raw_rand']:.4f}, "
                f"top3={[d['column'] for d in r5['top_drivers'][:3]]}"
            ),
            formula="5-fold OOF ROC-AUC predicting is_test over all 140,391 rows",
            timeframe="Static WiDS Datathon 2021 combined cohort",
            filters=["130,157 Train (is_test=0) + 10,234 Unlabeled (is_test=1) on F_clinical"],
            outcome=r5["outcome"],
            on_true_cond="adv_auc_raw_grouped >= 0.65",
            on_false_cond="adv_auc_raw_grouped < 0.58",
            on_inconc_cond="0.58 <= adv_auc_raw_grouped < 0.65",
            act_true="Escalate top shift drivers; Spawns #I5.1; Ref to #I6.",
            act_false="Document low out-of-hospital shift; Spawns #I5.1; Ref to #I6.",
            act_inconc="Spawns #I5.1 to isolate missingness vs value shift -> Ref to #I6.",
            source_file=src_both,
            commit=git_commit,
            sha256=train_sha,
            columns=["F_clinical (175 predictors)", "hospital_id"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I5.1",
            status=r5_1["outcome"],
            polarity="+",
            claim=(
                "Subtends #I5. Adversarial discriminability is substantially driven by "
                "missingness-pattern shifts (bmi, weight, age) or icu_id collisions."
            ),
            technique="Nested Adversarial Validation ablation and univariate KS/PSI drift audit",
            parameters={"n_splits": 5, "psi_bins": 10},
            assumptions=["Comparing raw vs fold-imputed AUC isolates missingness-pattern signal."],
            missing_data_policy="Median/mode imputers fit strictly inside each training fold.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"auc_imp={r5_1['auc_imp_grp']:.4f}, auc_miss_only={r5_1['auc_miss_only']:.4f}, "
                f"auc_with_icu={r5_1['auc_with_icu']:.4f}, "
                f"auc_no_anthro={r5_1['auc_no_anthro']:.4f}, "
                f"delta_miss={r5_1['delta_miss']:.4f}, delta_icu={r5_1['delta_icu']:.4f}, "
                f"psi_ge_0.10={r5_1['n_psi_ge_01']}"
            ),
            formula="delta_miss = auc_raw - auc_imp; delta_icu = auc_with_icu - auc_raw",
            timeframe="Static WiDS Datathon 2021 combined cohort",
            filters=["All 140,391 combined rows grouped by hospital_id"],
            outcome=r5_1["outcome"],
            on_true_cond="(auc_miss_only >= 0.60 AND delta_miss >= 0.03) OR delta_icu >= 0.03",
            on_false_cond="auc_miss_only < 0.55 AND delta_miss < 0.01 AND delta_icu < 0.01",
            on_inconc_cond="Moderate partial missingness or icu_id shift",
            act_true=(
                "Flag shifted missingness/icu_id hazards for Stage 5/6 ablation -> Ref to #I6."
            ),
            act_false="Attribute shift to clinical distributions via PSI/KS -> Ref to #I6.",
            act_inconc="Track partial shift drivers and PSI>=0.10 cols for Stage 6 -> Ref to #I6.",
            source_file=src_both,
            commit=git_commit,
            sha256=train_sha,
            columns=["F_clinical", "icu_id", "age", "bmi", "weight"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I6",
            status=r6["outcome"],
            polarity="+",
            claim=(
                "Diabetes prevalence (~21.63%) varies substantially across hospitals "
                "(IQR >= 4.0 pp, chi2 p < 0.001) and across >= 3 of 4 core strata "
                "(age_bin, ethnicity, icu_type, elective_surgery; p < 0.001, range >= 3.0 pp)."
            ),
            technique="Hospital-level Wilson CI/IQR spread and subgroup Pearson chi-square tests",
            parameters={"min_hospital_rows": 30, "alpha": 0.001, "min_range_pp": 3.0},
            assumptions=["Hospitals with n >= 30 yield stable empirical prevalence estimates."],
            missing_data_policy="Retain nulls as explicit '__MISSING__' category.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"pos_rate={r6['pos_rate_pct']:.2f}%, hosp_chi2_p={r6['hosp_chi2_p']:.3e}, "
                f"hosp_iqr={r6['hosp_iqr_pp']:.2f}pp (P10={r6['hosp_p10_pp']:.2f}%, "
                f"P90={r6['hosp_p90_pp']:.2f}%, n_hosp30={r6['n_hosp_ge30']}), "
                f"k_sig_core={r6['k_sig_core']}/4"
            ),
            formula="IQR(prev_hosp | n>=30); count(core strata with chi2_p<0.001 & range>=3pp)",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 TrainingWiDS2021.csv rows"],
            outcome=r6["outcome"],
            on_true_cond="hosp_chi2_p < 0.001 AND hosp_iqr >= 4.0 AND k_sig_core >= 3",
            on_false_cond="(hosp_chi2_p >= 0.05 OR hosp_iqr < 2.0) AND k_sig_core == 0",
            on_inconc_cond="Partial hospital or subgroup heterogeneity",
            act_true="Require Log-Loss/Brier calibration audit + OOF aggregations; Spawns #I6.1.",
            act_false="Proceed with uniform prior assumption and Spawns #I6.1.",
            act_inconc="Inspect heterogeneity table in stage2_target_prevalence.csv; Spawns #I6.1.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=git_commit,
            sha256=train_sha,
            columns=["hospital_id", "age", "bmi", "ethnicity", "icu_type", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I6.1",
            status=r6_1["outcome"],
            polarity="+",
            claim=(
                "Subtends #I6 and #I3.1. Missing strata in age/bmi/weight/ethnicity differ by "
                ">= 2.0 pp in diabetes prevalence (p < 0.01), and Train-to-Unlabeled composition "
                "shifts imply >= 0.50 pp projected target prior shift."
            ),
            technique="Missing-vs-present chi-square tests and direct demographic standardization",
            parameters={"alpha": 0.01, "min_prev_diff_pp": 2.0, "min_prior_shift_pp": 0.50},
            assumptions=[
                "Train conditional prevalence within strata holds to first order in Test."
            ],
            missing_data_policy="Explicit '__MISSING__' stratum included in standardization.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"k_miss_bias={r6_1['k_miss_bias']}, max_shift_pp={r6_1['max_shift_pp']:.3f}pp, "
                f"std_shifts={r6_1['std_shifts_pp']}, details={r6_1['miss_bias_details']}"
            ),
            formula="sum_s P_test(s)*P_train(y=1|s) - P_train(y=1)",
            timeframe="Static WiDS Datathon 2021 combined cohort",
            filters=["All 130,157 Train and 10,234 Unlabeled rows"],
            outcome=r6_1["outcome"],
            on_true_cond="k_miss_bias >= 1 AND max_shift_pp >= 0.50",
            on_false_cond="k_miss_bias == 0 AND max_shift_pp < 0.20",
            on_inconc_cond="Mixed missing-stratum bias and composition prior shift",
            act_true=(
                "Escalate age/bmi/weight missingness bias to Stage 4 MNAR -> Proceeds Stage 3."
            ),
            act_false="Confirm minimal prior distortion -> Proceeds Stage 3.",
            act_inconc="Log exact missing-stratum bias for Stage 4 MNAR audit -> Proceeds Stage 3.",
            source_file=src_both,
            commit=git_commit,
            sha256=train_sha,
            columns=["age", "bmi", "weight", "ethnicity", "icu_type", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
    ]
    upsert_inquiries(entries)

    drivers_md = "\n".join(
        f"- `{d['column']}`: perm_imp={d['adv_perm_importance']:.4f}, "
        f"PSI={d['psi_10bin']}, KS={d['ks_stat']}"
        for d in r5["top_drivers"]
    )
    miss_md = "\n".join(
        f"- `{m['col']}` (`n_na={m['n_na']}`): null_prev={m['prev_na_pct']:.2f}% vs "
        f"present_prev={m['prev_present_pct']:.2f}% (|Δ|={m['abs_diff_pp']:.2f} pp, "
        f"p={m['chi2_p']:.3e})"
        for m in r6_1["miss_bias_details"]
    )
    md = f"""# Stage 2 Report: Facility Collisions, Covariate Shift & Target Prevalence

- **Generated**: `{now_iso}` (Base Commit: `{git_commit}`)

## Key Findings

1. **[I4 = {r4["outcome"]}, I4.1 = {r4_1["outcome"]}] Facility Collision & CV Lock**:
   - Combined multi-hospital `icu_id` collisions: **`{r4["comb_multi"] * 100:.2f}%`**
     (`max={r4["max_h_all"]}` hospitals/icu_id across {r4["n_icu_all"]} unique IDs).
   - Within-train multi-hospital `icu_id` collisions: **`{r4["tr_multi"] * 100:.2f}%`**
     (`max={r4["max_h_tr"]}`, NMI=`{r4["nmi_tr"]:.4f}`).
   - 5-fold `StratifiedGroupKFold(groups=hospital_id)`: fold share range =
     `{r4_1["fold_share_rng"]:.3f} pp`, target rate range = `{r4_1["fold_rate_rng"]:.3f} pp`,
     hospital overlap = `{r4_1["max_h_overlap"]}`, dup cluster folds = `{r4_1["dup_folds"]}`.
   - `icu_id` target-encoding AUC: Random SKF = **`{r4_1["auc_rand"]:.4f}`** vs.
     Grouped SGKF = **`{r4_1["auc_grp"]:.4f}`**.

2. **[I5 = {r5["outcome"]}, I5.1 = {r5_1["outcome"]}] Out-of-Hospital Adversarial Validation**:
   - `adv_auc_raw_grouped` (SGKF): **`{r5["auc_raw_grp"]:.4f}`** vs.
     `adv_auc_raw_random` (SKF): **`{r5["auc_raw_rand"]:.4f}`**.
   - Ablations (SGKF): imputed=`{r5_1["auc_imp_grp"]:.4f}` (`Δ_miss={r5_1["delta_miss"]:+.4f}`),
     missingness-only=`{r5_1["auc_miss_only"]:.4f}`, with `icu_id`=`{r5_1["auc_with_icu"]:.4f}`
     (`Δ_icu={r5_1["delta_icu"]:+.4f}`), no-anthro-miss=`{r5_1["auc_no_anthro"]:.4f}`.
   - Features with `PSI >= 0.10`: `{r5_1["n_psi_ge_01"]}`.

### Top 10 Adversarial Shift Drivers
{drivers_md}

3. **[I6 = {r6["outcome"]}, I6.1 = {r6_1["outcome"]}] Target Prevalence & Prior Shift**:
   - Overall `diabetes_mellitus=1`: **`{r6["pos_rate_pct"]:.2f}%`**.
   - Hospital spread (`n >= 30`, {r6["n_hosp_ge30"]}/{r6["n_hosp_total"]} hospitals):
     chi2 `p={r6["hosp_chi2_p"]:.3e}`, IQR = **`{r6["hosp_iqr_pp"]:.2f} pp`**
     (P10=`{r6["hosp_p10_pp"]:.2f}%`, P90=`{r6["hosp_p90_pp"]:.2f}%`).
   - Significant core strata (`p < 0.001` & range `>= 3.0 pp`): **`{r6["k_sig_core"]}/4`**.
   - Standardized Train->Unlabeled prior shifts (pp): `{r6_1["std_shifts_pp"]}`.

### Missingness-Stratum Target Bias (`k_miss_bias = {r6_1["k_miss_bias"]}`)
{miss_md}
"""
    (REPORTS_DIR / "stage2_shift_and_target.md").write_text(md, encoding="utf-8")
    print(
        f"Stage 2 done: I4={r4['outcome']}, I4.1={r4_1['outcome']}, "
        f"I5={r5['outcome']}, I5.1={r5_1['outcome']}, "
        f"I6={r6['outcome']}, I6.1={r6_1['outcome']}"
    )


if __name__ == "__main__":
    main()
