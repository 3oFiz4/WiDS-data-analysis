"""Stage 3 EDA: Normality Audit (P1), Linear vs Non-Linear Ranking & Feature Triage."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

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
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "stage3"
DROP_COLS = {"Unnamed: 0", "encounter_id", "readmission_status", "hospital_id", "icu_id"}
ABLATE_GLUC_DIAG = {
    "d1_glucose_max",
    "d1_glucose_min",
    "h1_glucose_max",
    "h1_glucose_min",
    "glucose_apache",
    "apache_2_diagnosis",
    "apache_3j_diagnosis",
}


def cramers_v(s: pd.Series, y: np.ndarray) -> float:
    """Compute bias-corrected Cramer's V on non-null categorical values."""
    mask = s.notna().to_numpy()
    if mask.sum() < 10:
        return 0.0
    tab = pd.crosstab(s[mask].astype(str), y[mask])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = float(stats.chi2_contingency(tab)[0])
    n = float(tab.to_numpy().sum())
    r, k = tab.shape
    return float(np.sqrt(max(0.0, chi2 / (n * max(1, min(r - 1, k - 1))))))


def compute_univariate_mi(col: str, vals: np.ndarray, is_num: bool, y: np.ndarray) -> float:
    """Compute non-null mutual information for a single feature."""
    if is_num:
        mask = ~np.isnan(vals)
        if mask.sum() < 50:
            return 0.0
        x_nn = vals[mask]
        y_nn = y[mask]
        is_disc = len(np.unique(x_nn)) <= 2
        res = mutual_info_classif(
            x_nn.reshape(-1, 1),
            y_nn,
            discrete_features=[is_disc],
            n_neighbors=5,
            random_state=42,
        )
        return float(res[0])
    s = pd.Series(vals)
    mask = s.notna().to_numpy()
    if mask.sum() < 50:
        return 0.0
    codes, _ = pd.factorize(s[mask].astype(str))
    res = mutual_info_classif(
        codes.reshape(-1, 1),
        y[mask],
        discrete_features=[True],
        n_neighbors=5,
        random_state=42,
    )
    return float(res[0])


def impossible_mask(col: str, s: pd.Series) -> tuple[pd.Series, str]:
    """Return boolean mask of impossible non-null clinical values and rule text."""
    if not pd.api.types.is_numeric_dtype(s):
        return pd.Series(False, index=s.index), "none (categorical)"
    v = s.astype(float)
    nn = v.notna()
    rule = "none"
    bad = pd.Series(False, index=s.index)
    if col == "pre_icu_los_days":
        bad, rule = nn & (v < 0), "val < 0"
    elif col == "age":
        bad, rule = nn & ((v <= 0) | (v > 110)), "val <= 0 OR val > 110"
    elif col == "height":
        bad, rule = nn & ((v <= 50) | (v > 250)), "val <= 50 OR val > 250 cm"
    elif col == "weight":
        bad, rule = nn & ((v <= 20) | (v > 400)), "val <= 20 OR val > 400 kg"
    elif col == "bmi":
        bad, rule = nn & ((v <= 10) | (v > 100)), "val <= 10 OR val > 100"
    elif "_spo2_" in col:
        bad, rule = nn & ((v < 0) | (v > 100)), "val < 0 OR val > 100%"
    elif col == "fio2_apache":
        bad, rule = nn & ((v <= 0) | (v > 1.0)), "val <= 0 OR val > 1.0"
    elif "hematocrit" in col:
        bad, rule = nn & ((v <= 0) | (v > 1.0)), "val <= 0 OR val > 1.0"
    elif "arterial_ph" in col or col == "ph_apache":
        bad, rule = nn & ((v < 6.5) | (v > 8.0)), "val < 6.5 OR val > 8.0"
    elif "_temp_" in col or col == "temp_apache":
        bad, rule = nn & ((v < 25.0) | (v > 45.0)), "val < 25 OR val > 45 C"
    elif col == "gcs_eyes_apache":
        bad, rule = nn & ((v < 1) | (v > 4)), "val < 1 OR val > 4"
    elif col == "gcs_motor_apache":
        bad, rule = nn & ((v < 1) | (v > 6)), "val < 1 OR val > 6"
    elif col == "gcs_verbal_apache":
        bad, rule = nn & ((v < 1) | (v > 5)), "val < 1 OR val > 5"
    elif col == "urineoutput_apache":
        bad, rule = nn & (v < 0), "val < 0 mL"
    else:
        pos_tokens = (
            "heartrate",
            "heart_rate",
            "sysbp",
            "diasbp",
            "mbp",
            "map_apache",
            "resprate",
            "glucose",
            "bun",
            "creatinine",
            "sodium",
            "potassium",
            "calcium",
            "bilirubin",
            "albumin",
            "hemaglobin",
            "platelets",
            "wbc",
            "lactate",
            "pao2",
            "paco2",
            "pao2fio2ratio",
            "hco3",
            "inr",
        )
        if any(tok in col for tok in pos_tokens):
            bad, rule = nn & (v <= 0), "val <= 0"
    return bad, rule


def eval_1d_grouped_auc(
    col: str,
    s: pd.Series,
    is_num: bool,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """Return (auc_tree_nonnull, auc_hgb_full_grouped) for a single feature."""
    oof_hgb = np.zeros(len(s), dtype=float)
    nn_mask = s.notna().to_numpy()
    oof_tree = np.full(len(s), np.nan, dtype=float)

    if is_num:
        vals = s.to_numpy(dtype=float).reshape(-1, 1)
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
            oof_hgb[va_i] = clf.predict_proba(vals[va_i])[:, 1]

            tr_nn = tr_i[nn_mask[tr_i]]
            va_nn = va_i[nn_mask[va_i]]
            if len(tr_nn) >= 100 and len(va_nn) >= 10 and len(np.unique(y[tr_nn])) == 2:
                dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=42)
                dt.fit(vals[tr_nn], y[tr_nn])
                oof_tree[va_nn] = dt.predict_proba(vals[va_nn])[:, 1]
    else:
        s_obj = s.astype(object).to_numpy().reshape(-1, 1)
        for tr_i, va_i in splits:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=np.nan,
            )
            x_tr = enc.fit_transform(s_obj[tr_i])
            x_va = enc.transform(s_obj[va_i])
            clf = HistGradientBoostingClassifier(
                max_iter=25,
                learning_rate=0.1,
                max_leaf_nodes=15,
                min_samples_leaf=100,
                categorical_features=[0],
                early_stopping=False,
                random_state=42,
            )
            clf.fit(x_tr, y[tr_i])
            oof_hgb[va_i] = clf.predict_proba(x_va)[:, 1]

    auc_hgb = float(roc_auc_score(y, oof_hgb))
    valid_t = ~np.isnan(oof_tree)
    auc_tree = (
        float(roc_auc_score(y[valid_t], oof_tree[valid_t]))
        if (valid_t.sum() >= 50 and len(np.unique(y[valid_t])) == 2)
        else np.nan
    )
    return auc_tree, auc_hgb


def eval_multivariate_subset(
    df: pd.DataFrame,
    cols: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, list[float]]:
    """Run 5-fold grouped HGB on a feature subset and return OOF and per-fold AUC."""
    oof = np.zeros(len(df), dtype=float)
    fold_aucs = []
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    for tr_i, va_i in splits:
        x_tr = df.iloc[tr_i][cols].copy()
        x_va = df.iloc[va_i][cols].copy()
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
        preds = clf.predict_proba(x_va)[:, 1]
        oof[va_i] = preds
        fold_aucs.append(float(roc_auc_score(y[va_i], preds)))
    return float(roc_auc_score(y, oof)), fold_aucs


def render_stage3_charts(
    triage: pd.DataFrame,
    corr_mat: pd.DataFrame,
    train: pd.DataFrame,
    y: np.ndarray,
    subset_results: list[dict],
) -> None:
    """Save 4 readable Stage 3 summary charts into reports/figures/stage3/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Linear vs Non-Linear ranking comparison (Top 25 by grouped HGB AUC)
    top25 = triage.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top25))
    ax.barh(y_pos - 0.18, top25["auc_hgb_full_grouped"], height=0.35, label="HGB Grouped AUC")
    mon_vals = top25["auc_monotonic_nonnull"].fillna(0.5)
    ax.barh(y_pos + 0.18, mon_vals, height=0.35, label="Monotonic Rank AUC (non-null)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top25["column"], fontsize=9)
    ax.set_xlim(0.48, max(0.75, float(top25["auc_hgb_full_grouped"].max()) + 0.03))
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Top 25 Predictors: Non-Linear Grouped HGB AUC vs. Monotonic AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage3_linear_vs_nonlinear_ranking.png", dpi=150)
    plt.close(fig)

    # 2. Top-25 pairwise Spearman correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_mat.abs(),
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        annot=False,
        square=True,
        ax=ax,
    )
    ax.set_title("Top-25 Predictors: Pairwise |Spearman rho| Collinearity Matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage3_top_collinearity_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. Top non-redundant numeric analytes: decile prevalence curves
    top_reps = triage.loc[
        triage["is_cluster_rep"] & triage["is_numeric"] & (triage["nunique_train"] >= 10),
        "column",
    ].head(6)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, top_reps, strict=False):
        s = train[col]
        valid = s.notna()
        q = pd.qcut(s[valid], q=10, duplicates="drop")
        df_q = pd.DataFrame({"bin": q, "y": y[valid]})
        grp = df_q.groupby("bin", observed=True)["y"].agg(["mean", "count"])
        x_lbls = [f"{iv.mid:.1f}" for iv in grp.index]
        ax.plot(range(len(x_lbls)), grp["mean"] * 100.0, marker="o", linewidth=2)
        ax.set_xticks(range(len(x_lbls)))
        ax.set_xticklabels(x_lbls, rotation=45, fontsize=8)
        ax.set_ylabel("Diabetes %")
        ax.set_title(
            f"{col} (AUC={triage.set_index('column').loc[col, 'auc_hgb_full_grouped']:.3f})"
        )
    fig.suptitle("Top Non-Redundant Analytes: Decile Diabetes Prevalence Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage3_top_analytes_distributions.png", dpi=150)
    plt.close(fig)

    # 4. Subset AUC scaling curve
    sub_df = pd.DataFrame(subset_results)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sub_df["subset"], sub_df["oof_auc"], marker="o", linewidth=2.5, color="#1f77b4")
    for idx, row in sub_df.iterrows():
        ax.annotate(
            f"{row['oof_auc']:.4f}",
            (idx, row["oof_auc"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("5-Fold StratifiedGroupKFold ROC-AUC")
    ax.set_title("Multivariate Feature Saturation & Glucose/Diagnosis Ablation (I8.1 / I9.1)")
    ax.set_ylim(min(sub_df["oof_auc"]) - 0.02, max(sub_df["oof_auc"]) + 0.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage3_subset_auc_scaling.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
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

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(train, y, g))
    rng = np.random.default_rng(42)

    # Compute MI in parallel across 4 workers
    is_num_map = {c: bool(pd.api.types.is_numeric_dtype(train[c])) for c in f_clin}
    mi_vals = Parallel(n_jobs=4)(
        delayed(compute_univariate_mi)(c, train[c].to_numpy(), is_num_map[c], y) for c in f_clin
    )
    mi_map = dict(zip(f_clin, mi_vals, strict=True))

    # Build full 175-feature triage records (I7, I7.1, I9 impossible check)
    records = []
    imp_details = []
    for c in f_clin:
        s_tr = train[c]
        s_te = test[c]
        is_num = is_num_map[c]
        n_nn = int(s_tr.notna().sum())
        nuq = int(s_tr.nunique(dropna=True))
        miss_tr = float(s_tr.isna().mean() * 100.0)
        miss_te = float(s_te.isna().mean() * 100.0)

        skew_v, kurt_v, shap_w, non_norm = np.nan, np.nan, np.nan, np.nan
        pear_r, spear_r, auc_mon, cv = np.nan, np.nan, np.nan, np.nan

        if is_num:
            nn_mask = s_tr.notna().to_numpy()
            vals = s_tr.to_numpy(dtype=float)[nn_mask]
            y_nn = y[nn_mask]
            if nuq >= 10 and len(vals) >= 20:
                skew_v = float(stats.skew(vals))
                kurt_v = float(stats.kurtosis(vals))
                samp = vals if len(vals) <= 5000 else rng.choice(vals, size=5000, replace=False)
                shap_w = float(stats.shapiro(samp).statistic)
                non_norm = int((abs(skew_v) > 1.0) or (abs(kurt_v) > 2.0) or (shap_w < 0.95))
            if len(vals) >= 10 and np.std(vals) > 0 and len(np.unique(y_nn)) == 2:
                pear_r = float(stats.pearsonr(vals, y_nn).statistic)
                spear_r = float(stats.spearmanr(vals, y_nn).statistic)
                raw_auc = float(roc_auc_score(y_nn, vals))
                auc_mon = max(raw_auc, 1.0 - raw_auc)
            else:
                pear_r, spear_r, auc_mon = 0.0, 0.0, 0.5
        else:
            cv = cramers_v(s_tr, y)

        auc_tree, auc_hgb = eval_1d_grouped_auc(c, s_tr, is_num, y, splits)

        # Impossible values audit in Train and Test
        bad_tr, rule = impossible_mask(c, s_tr)
        bad_te, _ = impossible_mask(c, s_te)
        n_bad_tr, n_bad_te = int(bad_tr.sum()), int(bad_te.sum())
        if n_bad_tr > 0 or n_bad_te > 0:
            imp_details.append(
                {
                    "column": c,
                    "rule": rule,
                    "train_impossible_n": n_bad_tr,
                    "train_impossible_pct": round(100.0 * n_bad_tr / max(1, n_nn), 4),
                    "train_min": float(s_tr.min()) if is_num else np.nan,
                    "train_max": float(s_tr.max()) if is_num else np.nan,
                    "test_impossible_n": n_bad_te,
                    "test_impossible_pct": round(
                        100.0 * n_bad_te / max(1, int(s_te.notna().sum())), 4
                    ),
                }
            )

        nl_lift = (
            float(auc_tree - auc_mon) if (pd.notna(auc_tree) and pd.notna(auc_mon)) else np.nan
        )
        rank_lift = (
            float(abs(spear_r) - abs(pear_r))
            if (pd.notna(spear_r) and pd.notna(pear_r))
            else np.nan
        )
        leak_flag = (
            "HARD_LEAK_RISK"
            if auc_hgb >= 0.90
            else (
                "CLINICAL_PROXY_AUDIT"
                if (
                    "glucose" in c
                    or c
                    in {"apache_2_diagnosis", "apache_3j_diagnosis", "arf_apache", "age", "bmi"}
                )
                else "PASS"
            )
        )
        records.append(
            {
                "column": c,
                "dd_category": dd_map.get(c, "unknown"),
                "is_numeric": is_num,
                "nunique_train": nuq,
                "miss_pct_train": round(miss_tr, 4),
                "miss_pct_test": round(miss_te, 4),
                "skewness": round(skew_v, 4) if pd.notna(skew_v) else np.nan,
                "excess_kurtosis": round(kurt_v, 4) if pd.notna(kurt_v) else np.nan,
                "shapiro_w_5000": round(shap_w, 5) if pd.notna(shap_w) else np.nan,
                "non_normal_p1": non_norm,
                "pearson_r": round(pear_r, 5) if pd.notna(pear_r) else np.nan,
                "abs_pearson_r": round(abs(pear_r), 5) if pd.notna(pear_r) else np.nan,
                "spearman_rho": round(spear_r, 5) if pd.notna(spear_r) else np.nan,
                "abs_spearman_rho": round(abs(spear_r), 5) if pd.notna(spear_r) else np.nan,
                "cramers_v": round(cv, 5) if pd.notna(cv) else np.nan,
                "mi_non_null": round(mi_map[c], 6),
                "auc_monotonic_nonnull": round(auc_mon, 5) if pd.notna(auc_mon) else np.nan,
                "auc_tree_nonnull": round(auc_tree, 5) if pd.notna(auc_tree) else np.nan,
                "auc_hgb_full_grouped": round(auc_hgb, 5),
                "nonlinear_tree_lift": round(nl_lift, 5) if pd.notna(nl_lift) else np.nan,
                "rank_corr_lift": round(rank_lift, 5) if pd.notna(rank_lift) else np.nan,
                "impossible_train_n": n_bad_tr,
                "impossible_test_n": n_bad_te,
                "impossible_rule": rule,
                "leakage_audit": leak_flag,
            }
        )

    triage = (
        pd.DataFrame(records)
        .sort_values(["auc_hgb_full_grouped", "mi_non_null"], ascending=[False, False])
        .reset_index(drop=True)
    )
    triage["rank_hgb_auc"] = np.arange(1, len(triage) + 1)
    triage["rank_pearson"] = triage["abs_pearson_r"].rank(ascending=False, method="min")
    triage["rank_mi"] = triage["mi_non_null"].rank(ascending=False, method="min")

    # I7 metrics
    cont_df = triage[triage["non_normal_p1"].notna()]
    n_cont = len(cont_df)
    n_non_norm = int((cont_df["non_normal_p1"] == 1).sum())
    frac_non_norm = float(n_non_norm / max(1, n_cont))
    near_gauss_cols = cont_df.loc[cont_df["non_normal_p1"] == 0, "column"].tolist()
    i7_out = (
        "TRUE" if frac_non_norm <= 0.90 else ("FALSE" if frac_non_norm == 1.0 else "INCONCLUSIVE")
    )

    # I7.1 metrics
    top20_r = set(
        triage.dropna(subset=["abs_pearson_r"])
        .sort_values("abs_pearson_r", ascending=False)
        .head(20)["column"]
    )
    top20_mi = set(triage.sort_values("mi_non_null", ascending=False).head(20)["column"])
    jacc_20 = float(len(top20_r & top20_mi) / max(1, len(top20_r | top20_mi)))
    nl_lift_mask = (triage["nonlinear_tree_lift"] >= 0.015) | (triage["rank_corr_lift"] >= 0.03)
    k_nl_lift = int(nl_lift_mask.fillna(False).sum())
    i7_1_out = (
        "TRUE"
        if (jacc_20 < 0.80 and k_nl_lift >= 5)
        else ("FALSE" if (jacc_20 >= 0.90 and k_nl_lift == 0) else "INCONCLUSIVE")
    )

    # I8: Top-25 pairwise Spearman collinearity clustering
    top25_cols = triage.head(25)["column"].tolist()
    corr_mat = pd.DataFrame(np.eye(len(top25_cols)), index=top25_cols, columns=top25_cols)
    for idx_a, ca in enumerate(top25_cols):
        if not is_num_map[ca]:
            continue
        for idx_b in range(idx_a + 1, len(top25_cols)):
            cb = top25_cols[idx_b]
            if not is_num_map[cb]:
                continue
            pair = train[[ca, cb]].dropna()
            if len(pair) >= 1000:
                rho = float(stats.spearmanr(pair[ca], pair[cb]).statistic)
            else:
                rho = 0.0
            corr_mat.loc[ca, cb] = rho
            corr_mat.loc[cb, ca] = rho

    adj: dict[str, list[str]] = {c: [] for c in top25_cols}
    for idx_a, ca in enumerate(top25_cols):
        for idx_b in range(idx_a + 1, len(top25_cols)):
            cb = top25_cols[idx_b]
            if abs(float(corr_mat.loc[ca, cb])) >= 0.80:
                adj[ca].append(cb)
                adj[cb].append(ca)

    comp_id: dict[str, int] = {}
    clusters: list[list[str]] = []
    for c in top25_cols:
        if c in comp_id:
            continue
        cid = len(clusters) + 1
        q = deque([c])
        comp_id[c] = cid
        members = []
        while q:
            u = q.popleft()
            members.append(u)
            for v in adj[u]:
                if v not in comp_id:
                    comp_id[v] = cid
                    q.append(v)
        clusters.append(members)

    c_top1 = top25_cols[0]
    top1_twins = adj[c_top1]
    n_clusters_top25 = len(clusters)
    top25_reps = [cl[0] for cl in clusters]  # top25_cols already ordered by auc_hgb_full_grouped
    triage["top25_cluster_id"] = triage["column"].map(comp_id)
    triage["is_cluster_rep"] = triage["column"].isin(set(top25_reps)) | (
        triage["rank_hgb_auc"] > 25
    )

    cl_rows = []
    for cid, members in enumerate(clusters, start=1):
        for m in members:
            row_t = triage.set_index("column").loc[m]
            cl_rows.append(
                {
                    "cluster_id": cid,
                    "column": m,
                    "is_cluster_rep": m == members[0],
                    "rank_hgb_auc": int(row_t["rank_hgb_auc"]),
                    "auc_hgb_full_grouped": float(row_t["auc_hgb_full_grouped"]),
                    "degree_rho_ge_080": len(adj[m]),
                    "twins_rho_ge_080": ";".join(adj[m]),
                }
            )
    pd.DataFrame(cl_rows).to_csv(
        REPORTS_DIR / "stage3_top25_collinearity_clusters.csv", index=False
    )

    i8_out = (
        "TRUE"
        if (n_clusters_top25 <= 14 and len(top1_twins) >= 1)
        else ("FALSE" if (n_clusters_top25 >= 20 and len(top1_twins) == 0) else "INCONCLUSIVE")
    )

    # I8.1 & I9.1: Multivariate subset AUC scaling + Glucose/Diag proxy ablation
    all_ranked = triage["column"].tolist()
    no_gluc_diag = [c for c in all_ranked if c not in ABLATE_GLUC_DIAG]
    subsets_spec = [
        ("Top-1", all_ranked[:1]),
        ("Top-10", all_ranked[:10]),
        ("Top-25", all_ranked[:25]),
        ("Top-25-dedup", top25_reps),
        ("Top-50", all_ranked[:50]),
        ("All-175", all_ranked),
        ("No-Glucose-Diag", no_gluc_diag),
    ]
    subset_results = []
    fold_auc_map = {}
    for s_name, s_cols in subsets_spec:
        oof_auc, f_aucs = eval_multivariate_subset(train, s_cols, y, splits)
        fold_auc_map[s_name] = (oof_auc, f_aucs)
        subset_results.append(
            {
                "subset": s_name,
                "n_features": len(s_cols),
                "oof_auc": round(oof_auc, 5),
                "fold_mean_auc": round(float(np.mean(f_aucs)), 5),
                "fold_std_auc": round(float(np.std(f_aucs)), 5),
            }
        )

    auc_25, f_25 = fold_auc_map["Top-25"]
    auc_175, f_175 = fold_auc_map["All-175"]
    delta_175_25 = float(auc_175 - auc_25)
    k_folds_175_beats_25 = sum(a > b for a, b in zip(f_175, f_25, strict=True))
    i8_1_out = (
        "TRUE"
        if (delta_175_25 >= 0.015 and k_folds_175_beats_25 >= 4)
        else ("FALSE" if delta_175_25 < 0.005 else "INCONCLUSIVE")
    )

    # I9 & I9.1
    triage.to_csv(REPORTS_DIR / "feature_triage_table.csv", index=False)
    imp_df = pd.DataFrame(imp_details).sort_values("train_impossible_n", ascending=False)
    imp_df.to_csv(REPORTS_DIR / "stage3_impossible_values_detail.csv", index=False)

    k_hard_leak = int((triage.head(40)["auc_hgb_full_grouped"] >= 0.90).sum())
    k_imp_cols = int((triage["impossible_train_n"] >= 1).sum())
    i9_out = (
        "TRUE"
        if (k_hard_leak == 0 and k_imp_cols >= 3)
        else ("FALSE" if (k_hard_leak >= 1 or k_imp_cols == 0) else "INCONCLUSIVE")
    )

    # High-prevalence APACHE diagnosis codes (n >= 30, prev >= 85%)
    diag_high = []
    for dcol in ["apache_2_diagnosis", "apache_3j_diagnosis"]:
        s_d = train[dcol].fillna("__MISSING__").astype(str)
        grp = pd.DataFrame({"code": s_d, "y": y}).groupby("code")["y"].agg(["count", "sum", "mean"])
        hi = grp[(grp["count"] >= 30) & (grp["mean"] >= 0.85)]
        for code, r in hi.iterrows():
            diag_high.append(
                {
                    "col": dcol,
                    "code": code,
                    "n": int(r["count"]),
                    "pos": int(r["sum"]),
                    "prev_pct": round(float(r["mean"] * 100.0), 2),
                }
            )
    k_diag_85 = len(diag_high)
    pos_in_85 = sum(d["pos"] for d in diag_high)
    share_pos_85 = float(100.0 * pos_in_85 / max(1, int(y.sum())))
    auc_no_gd, _ = fold_auc_map["No-Glucose-Diag"]
    delta_abl = float(auc_175 - auc_no_gd)
    i9_1_out = (
        "TRUE"
        if (k_diag_85 >= 1 and delta_abl >= 0.04 and auc_no_gd >= 0.72)
        else ("FALSE" if (k_diag_85 == 0 and delta_abl < 0.02) else "INCONCLUSIVE")
    )

    render_stage3_charts(triage, corr_mat, train, y, subset_results)

    # Upsert HMECVAS units I7..I9.1
    src_all = (
        f"data/raw/TrainingWiDS2021.csv; data/raw/UnlabeledWiDS2021.csv (sha256={test_sha}); "
        f"data/raw/DataDictionaryWiDS2021.csv (sha256={dict_sha})"
    )
    script_ref = "scripts/stage3_signal_and_triage.py"
    entries = [
        make_inquiry(
            inq_id="I7",
            status=i7_out,
            polarity="-",
            claim=(
                "Rivals P1 ('All variables are non-linear distribution'). Fewer than 100% of "
                "continuous predictors (nunique>=10) violate normality, with >=10% near-Gaussian."
            ),
            technique="Skewness, excess kurtosis, and 5000-row seeded Shapiro-Wilk W audit",
            parameters={"shapiro_sample_n": 5000, "seed": 42},
            assumptions=["Substantive non-normality requires |skew|>1, |kurt|>2, or W<0.95."],
            missing_data_policy="Non-null values per column.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"n_cont={n_cont}, n_non_normal={n_non_norm} "
                f"({frac_non_norm * 100:.2f}%), near_gaussian={near_gauss_cols}"
            ),
            formula="count(|skew|>1 OR |kurt|>2 OR W<0.95) / n_cont",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["Numeric predictors in F_clinical with nunique_train >= 10"],
            outcome=i7_out,
            on_true_cond="frac_non_normal <= 0.90",
            on_false_cond="frac_non_normal == 1.0",
            on_inconc_cond="0.90 < frac_non_normal < 1.0",
            act_true="Refute blanket P1; tag near_gaussian cols in triage table; Spawns #I7.1.",
            act_false="Confirm P1 across 100% of continuous vars; Spawns #I7.1.",
            act_inconc=(
                "Document near-universal non-normality and small Gaussian list; Spawns #I7.1."
            ),
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["Continuous numeric predictors in F_clinical"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I7.1",
            status=i7_1_out,
            polarity="+",
            claim=(
                "Subtends #I7. Non-linear grouped HGB AUC and MI surface predictive signals missed "
                "by Pearson |r| (Top-20 Jaccard < 0.80 and >= 5 numeric features with lift)."
            ),
            technique="175-feature Pearson |r|, Spearman |rho|, MI, DecisionTree, and HGB AUC",
            parameters={"cv": "StratifiedGroupKFold(5)", "tree_depth": 4, "mi_neighbors": 5},
            assumptions=["Comparing depth-4 tree vs rank AUC on non-null rows isolates U-shapes."],
            missing_data_policy=(
                "Linear/MI/Tree on non-null rows; HGB natively over all 130,157 rows."
            ),
            script_ref=script_ref,
            ev_type="N",
            metric=f"top20_jaccard={jacc_20:.4f}, k_nonlinear_lift={k_nl_lift}",
            formula="Jaccard(Top20(|r|), Top20(MI)); count(tree_lift>=0.015 OR rank_lift>=0.03)",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 175 F_clinical predictors"],
            outcome=i7_1_out,
            on_true_cond="top20_jaccard < 0.80 AND k_nonlinear_lift >= 5",
            on_false_cond="top20_jaccard >= 0.90 AND k_nonlinear_lift == 0",
            on_inconc_cond="Partial rank divergence or non-monotonic lift",
            act_true="Adopt auc_hgb_full_grouped + MI as canonical triage ranking -> Ref to #I8.",
            act_false="Document linear sufficiency while retaining grouped HGB -> Ref to #I8.",
            act_inconc="Retain dual linear + non-linear triage rankings -> Ref to #I8.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["F_clinical (175 predictors)", "hospital_id", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I8",
            status=i8_out,
            polarity="+",
            claim=(
                "Within Top-25 predictors by grouped HGB AUC, |Spearman rho| >= 0.80 collapses the "
                "25 features into <= 14 clusters with c_top1 having >= 1 Top-25 twin."
            ),
            technique="Top-25 pairwise Spearman matrix and connected-component clustering",
            parameters={"top_k": 25, "rho_threshold": 0.80, "min_overlap_n": 1000},
            assumptions=["Cross-window twins of one analyte cluster with |rho| >= 0.80."],
            missing_data_policy="Pairwise complete non-null observations.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"c_top1={c_top1} (auc={triage.loc[0, 'auc_hgb_full_grouped']:.4f}, "
                f"r={triage.loc[0, 'pearson_r']}, rho={triage.loc[0, 'spearman_rho']}), "
                f"n_clusters={n_clusters_top25}, top1_twins={top1_twins}"
            ),
            formula="Connected components of G(Top25, |Spearman rho| >= 0.80)",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["Top-25 predictors in F_clinical ranked by auc_hgb_full_grouped"],
            outcome=i8_out,
            on_true_cond="n_clusters <= 14 AND top1_twin_count >= 1",
            on_false_cond="n_clusters >= 20 AND top1_twin_count == 0",
            on_inconc_cond="Moderate cluster count (15-19) or singleton c_top1",
            act_true="Tag redundant Top-25 twins for Stage 5; Spawns #I8.1; Ref to #I9.",
            act_false="Retain Top-25 without proxy merging; Spawns #I8.1; Ref to #I9.",
            act_inconc=(
                "Review moderate clustering in stage3_top25_collinearity_clusters.csv; "
                "Spawns #I8.1."
            ),
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=top25_cols,
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I8.1",
            status=i8_1_out,
            polarity="-",
            claim=(
                "Subtends #I8 and rivals pre-assumptive low-R discard claim. Adding features "
                "ranked 26..175 increases 5-fold grouped HGB AUC by >= +0.015 (>= 4/5 folds)."
            ),
            technique=("Nested multivariate subset scaling across Top-1..All-175 on locked SGKF"),
            parameters={
                "subsets": ["Top-1", "Top-10", "Top-25", "Top-25-dedup", "Top-50", "All-175"]
            },
            assumptions=["Identical CV folds isolate incremental predictive signal beyond Top-25."],
            missing_data_policy="Native NaN handling inside HistGradientBoostingClassifier.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"auc_top25={auc_25:.5f}, auc_all175={auc_175:.5f}, "
                f"delta={delta_175_25:+.5f}, folds_improved={k_folds_175_beats_25}/5, "
                f"curve={subset_results}"
            ),
            formula="OOF_AUC(All-175) - OOF_AUC(Top-25)",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=i8_1_out,
            on_true_cond="delta >= 0.015 AND folds_improved >= 4",
            on_false_cond="delta < 0.005",
            on_inconc_cond="0.005 <= delta < 0.015 or < 4 folds improved",
            act_true=(
                "Reject pre-assumptive low-R discard; mandate All-175 retention -> Ref to #I9."
            ),
            act_false="Confirm Top-25 saturation -> Ref to #I9.",
            act_inconc="Keep Top-50 vs All-175 benchmark in Stage 6 -> Ref to #I9.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["F_clinical (175 predictors)", "hospital_id", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I9",
            status=i9_out,
            polarity="+",
            claim=(
                "Zero Top-40 features cross deterministic single-feature leakage (AUC >= 0.90), "
                "while >= 3 F_clinical columns contain hard physiological/unit impossible values."
            ),
            technique=("175-feature triage table compilation, single-feature AUC, and range scan"),
            parameters={"top_leak_n": 40, "leak_auc_min": 0.90},
            assumptions=["DataDictionary units define valid physiological intervals."],
            missing_data_policy="Evaluate bounds strictly on non-null Train and Unlabeled values.",
            script_ref=script_ref,
            ev_type="N",
            metric=f"k_hard_leak_top40={k_hard_leak}, k_impossible_cols={k_imp_cols}",
            formula="count(Top40 AUC>=0.90); count(impossible_train_n >= 1)",
            timeframe="Static WiDS Datathon 2021 snapshot",
            filters=["All 130,157 Train and 10,234 Unlabeled rows across 175 predictors"],
            outcome=i9_out,
            on_true_cond="k_hard_leak_top40 == 0 AND k_impossible_cols >= 3",
            on_false_cond="k_hard_leak_top40 >= 1 OR k_impossible_cols == 0",
            on_inconc_cond="k_hard_leak_top40 == 0 AND 1 <= k_impossible_cols <= 2",
            act_true=(
                "Publish feature_triage_table.csv and escalate impossible cols -> Spawns #I9.1."
            ),
            act_false="Quarantine leaky cols or continue Stage 4 consistency checks; Spawns #I9.1.",
            act_inconc="Audit the 1-2 flagged cols in Stage 4; Spawns #I9.1.",
            source_file=src_all,
            commit=commit,
            sha256=train_sha,
            columns=["F_clinical (175 predictors)", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I9.1",
            status=i9_1_out,
            polarity="+",
            claim=(
                "Subtends #I9. At least 1 APACHE diagnosis code (n>=30) has >= 85% diabetes "
                "prevalence, and dropping glucose + diagnosis lowers grouped AUC by >= 0.04 "
                "while retaining non-glucose AUC >= 0.72."
            ),
            technique="High-prevalence diagnosis code screen and grouped HGB ablation",
            parameters={"min_code_n": 30, "prev_min_pct": 85.0},
            assumptions=["AUC >= 0.72 without glucose/diagnosis confirms broad systemic signal."],
            missing_data_policy="Missing diagnosis codes mapped to '__MISSING__'.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"k_diag_ge85={k_diag_85} (pos_share={share_pos_85:.2f}%), "
                f"auc_all175={auc_175:.5f}, auc_no_gluc_diag={auc_no_gd:.5f}, "
                f"delta={delta_abl:.5f}, codes={diag_high}"
            ),
            formula="OOF_AUC(All-175) - OOF_AUC(All-175 \\ (glucose U diagnosis))",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=i9_1_out,
            on_true_cond="k_diag_ge85 >= 1 AND delta >= 0.04 AND auc_no_gluc_diag >= 0.72",
            on_false_cond="k_diag_ge85 == 0 AND delta < 0.02",
            on_inconc_cond="Partial diagnosis code tail or glucose reliance",
            act_true=("Document valid admission DKA codes and strong non-glucose -> Stage 4."),
            act_false="Document low diagnosis/glucose dependency -> Proceeds Stage 4.",
            act_inconc=("Inspect diagnosis codes and non-glucose baseline in Stage 5 -> Stage 4."),
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=sorted(ABLATE_GLUC_DIAG | {"hospital_id", "diabetes_mellitus"}),
            timestamp=now_iso,
        ),
    ]
    upsert_inquiries(entries)

    # Write Stage 3 summary markdown
    top25_md_rows = []
    for _, r in triage.head(25).iterrows():
        top25_md_rows.append(
            f"| {int(r['rank_hgb_auc'])} | `{r['column']}` | `{r['dd_category']}` | "
            f"{r['miss_pct_train']:.1f}% | {r['abs_pearson_r']} | {r['abs_spearman_rho']} | "
            f"{r['mi_non_null']:.4f} | {r['auc_hgb_full_grouped']:.4f} | "
            f"{int(r['impossible_train_n'])} | `{r['leakage_audit']}` |"
        )
    sub_md_rows = [
        f"| `{s['subset']}` | {s['n_features']} | {s['oof_auc']:.5f} | "
        f"{s['fold_mean_auc']:.5f} +/- {s['fold_std_auc']:.5f} |"
        for s in subset_results
    ]
    imp_md_rows = [
        f"- `{r['column']}` (`{r['rule']}`): Train={int(r['train_impossible_n'])} "
        f"({r['train_impossible_pct']:.3f}%), Test={int(r['test_impossible_n'])} "
        f"({r['test_impossible_pct']:.3f}%)"
        for _, r in imp_df.head(15).iterrows()
    ]
    md = f"""# Stage 3 Report: Non-Linear Distribution Audit, Signal Ranking & Feature Triage

- **Generated**: `{now_iso}` (Base Commit: `{commit}`)

## Inquiry Outcomes

- **[I7 = {i7_out}] Test of Pre-Assumptive P1**: `{n_non_norm}/{n_cont}` continuous features
  (`{frac_non_norm * 100:.2f}%`) violate normality (`|skew|>1`, `|kurt|>2`, or `W<0.95`).
  Near-Gaussian (`{len(near_gauss_cols)}`): `{near_gauss_cols}`.
- **[I7.1 = {i7_1_out}] Linear vs Non-Linear Signal**: Top-20 Jaccard(`|r|`, `MI`) =
  **`{jacc_20:.4f}`**; features with non-linear/rank lift = **`{k_nl_lift}`**.
- **[I8 = {i8_out}] Top-25 Collinearity Clusters (`|rho| >= 0.80`)**: Top-most feature
  `c_top1 = {c_top1}` (`auc={triage.loc[0, "auc_hgb_full_grouped"]:.4f}`), twins = `{top1_twins}`,
  collapsing Top-25 into **`{n_clusters_top25}`** connected clusters.
- **[I8.1 = {i8_1_out}] Multivariate Subset Saturation**: `Top-25 = {auc_25:.5f}` ->
  `All-175 = {auc_175:.5f}` (`Δ = {delta_175_25:+.5f}`, `{k_folds_175_beats_25}/5` folds improved).
- **[I9 = {i9_out}] Triage Leakage & Impossible Values**: `k_hard_leak_top40 = {k_hard_leak}`,
  columns with impossible values = **`{k_imp_cols}`**.
- **[I9.1 = {i9_1_out}] Glucose & Diagnosis Proxy Ablation**: `k_diag>=85% = {k_diag_85}`
  (`{share_pos_85:.2f}%` of positives, `{diag_high}`); `All-175 = {auc_175:.5f}` ->
  `No-Glucose-Diag = {auc_no_gd:.5f}` (`Δ = -{delta_abl:.5f}`).

## Subset Scaling & Ablation Table
| Subset | Features | OOF ROC-AUC | Mean +/- Std |
| :--- | ---: | ---: | ---: |
{chr(10).join(sub_md_rows)}

## Top 25 Predictors in `<Feature Triage Summary Table>`
| Rank | Column | Category | Null % | |r| | |rho| | MI | HGB AUC | Imp N | Audit |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
{chr(10).join(top25_md_rows)}

## Top Impossible-Value Columns (`{k_imp_cols}` Total)
{chr(10).join(imp_md_rows)}
"""
    (REPORTS_DIR / "stage3_feature_triage.md").write_text(md, encoding="utf-8")
    print(
        f"Stage 3 done: I7={i7_out}, I7.1={i7_1_out}, I8={i8_out}, "
        f"I8.1={i8_1_out}, I9={i9_out}, I9.1={i9_1_out}"
    )


if __name__ == "__main__":
    main()
