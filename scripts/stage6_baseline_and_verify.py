"""Stage 6 Verification: Grouped CV Optimism, Baseline Comparison & HMECVAS Linter."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from project_name.config import PROJECT_ROOT
from project_name.features import (
    DROP_STRUCTURAL,
    add_clinical_ratios,
    add_glucose_features,
    add_hierarchy_and_miss_counts,
    compute_fold_safe_features,
    sanitize_clinical_df,
)
from project_name.hmecvas import (
    INQUIRIES_PATH,
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
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "stage6"
DUP_ENCOUNTERS_167 = {
    244601,
    150518,
    264918,
    225543,
    147789,
    234525,
    189191,
    197188,
}


def compute_ece10(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute 10-bin Expected Calibration Error."""
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    n = len(y_true)
    for idx in range(10):
        lo, hi = bins[idx], bins[idx + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if idx == 9 else y_prob < hi)
        if mask.any():
            ece += (mask.sum() / n) * abs(float(y_true[mask].mean() - y_prob[mask].mean()))
    return float(ece)


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute discrimination and calibration suite."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece_10": compute_ece10(y_true, y_prob),
    }


def build_fe_abcd(
    df: pd.DataFrame, f_clin: list[str], dd_map: dict[str, str]
) -> tuple[pd.DataFrame, list[str]]:
    """Apply Stage 4 sanitization and Stage 5 Families A-D feature engineering."""
    san = sanitize_clinical_df(df, f_clin)
    fe_b = add_glucose_features(san)
    fe_c = add_clinical_ratios(san)
    fe_d = add_hierarchy_and_miss_counts(san, f_clin, dd_map)
    full = pd.concat([san, fe_b, fe_c, fe_d], axis=1)
    cols = (
        f_clin
        + ["n_min_max_swapped"]
        + list(fe_b.columns)
        + list(fe_c.columns)
        + list(fe_d.columns)
    )
    return full, cols


def run_hgb_cv(
    df: pd.DataFrame,
    cols: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    use_te: bool = False,
    test_df: pd.DataFrame | None = None,
    drop_train_mask: np.ndarray | None = None,
    compute_perm: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
    """Run 5-fold HGB CV returning OOF probs, optional test ensemble, and perm imp."""
    oof = np.zeros(len(df), dtype=float)
    test_preds = np.zeros(len(test_df), dtype=float) if test_df is not None else None
    perm_accum: dict[str, list[float]] = {}
    base_cats = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    for tr_i, va_i in splits:
        if drop_train_mask is not None:
            tr_i = tr_i[~drop_train_mask[tr_i]]

        x_tr = df.iloc[tr_i][cols].copy()
        x_va = df.iloc[va_i][cols].copy()
        x_te = test_df[cols].copy() if test_df is not None else None

        if use_te:
            tr_te, va_te = compute_fold_safe_features(
                df.iloc[tr_i], df.iloc[va_i], y[tr_i], use_te=True
            )
            x_tr = pd.concat([x_tr, tr_te], axis=1)
            x_va = pd.concat([x_va, va_te], axis=1)
            if x_te is not None and test_df is not None:
                _, te_te = compute_fold_safe_features(df.iloc[tr_i], test_df, y[tr_i], use_te=True)
                x_te = pd.concat([x_te, te_te], axis=1)

        if base_cats:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=np.nan,
            )
            x_tr[base_cats] = enc.fit_transform(x_tr[base_cats].astype(object))
            x_va[base_cats] = enc.transform(x_va[base_cats].astype(object))
            if x_te is not None:
                x_te[base_cats] = enc.transform(x_te[base_cats].astype(object))

        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            categorical_features=base_cats if base_cats else None,
            early_stopping=True,
            random_state=42,
        )
        clf.fit(x_tr, y[tr_i])
        oof[va_i] = clf.predict_proba(x_va)[:, 1]
        if test_preds is not None and x_te is not None:
            test_preds += clf.predict_proba(x_te)[:, 1] / len(splits)

        if compute_perm:
            res = permutation_importance(
                clf, x_va, y[va_i], scoring="roc_auc", n_repeats=2, random_state=42, n_jobs=1
            )
            for c_name, val in zip(list(x_tr.columns), res.importances_mean, strict=True):
                perm_accum.setdefault(c_name, []).append(float(val))

    perm_mean = {k: float(np.mean(v)) for k, v in perm_accum.items()}
    return oof, test_preds, perm_mean


def run_logreg_fe_cv(
    df: pd.DataFrame,
    cols: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Run 5-fold SGKF regularized LogisticRegression on collinearity-pruned fe_ABCD."""
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in cols if c not in set(cat_cols)]

    # Greedy |rho| >= 0.90 pruning over numeric cols
    corr = df[num_cols].corr(method="spearman").abs()
    kept_num: list[str] = []
    for c in num_cols:
        if not any(pd.notna(corr.loc[c, k]) and float(corr.loc[c, k]) >= 0.90 for k in kept_num):
            kept_num.append(c)

    oof = np.zeros(len(df), dtype=float)
    for tr_i, va_i in splits:
        tr_te, va_te = compute_fold_safe_features(
            df.iloc[tr_i], df.iloc[va_i], y[tr_i], use_te=True
        )
        x_tr_num = pd.concat([df.iloc[tr_i][kept_num], tr_te], axis=1)
        x_va_num = pd.concat([df.iloc[va_i][kept_num], va_te], axis=1)

        imp = SimpleImputer(strategy="median", add_indicator=True)
        scl = StandardScaler()
        x_tr_n = scl.fit_transform(imp.fit_transform(x_tr_num))
        x_va_n = scl.transform(imp.transform(x_va_num))

        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=30)
        x_tr_c = ohe.fit_transform(df.iloc[tr_i][cat_cols].fillna("__MISSING__").astype(str))
        x_va_c = ohe.transform(df.iloc[va_i][cat_cols].fillna("__MISSING__").astype(str))

        x_tr_mat = hstack([x_tr_n, x_tr_c]).tocsr()
        x_va_mat = hstack([x_va_n, x_va_c]).tocsr()

        lr = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
        lr.fit(x_tr_mat, y[tr_i])
        oof[va_i] = lr.predict_proba(x_va_mat)[:, 1]
    return oof


def render_stage6_charts(
    comp_df: pd.DataFrame,
    y: np.ndarray,
    oof_a: np.ndarray,
    oof_b: np.ndarray,
    oof_lr: np.ndarray,
    test_b: np.ndarray,
    hosp_aucs: list[float],
) -> None:
    """Save 3 publication-quality Stage 6 verification plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. CV Leakage & Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sub = comp_df.iloc[::-1]
    axes[0].barh(sub["model_regime"], sub["roc_auc"], color="#1f77b4")
    axes[0].set_xlim(
        max(0.75, float(sub["roc_auc"].min()) - 0.02), float(sub["roc_auc"].max()) + 0.01
    )
    axes[0].set_title("5-Fold ROC-AUC Across CV Regimes & Models")
    axes[1].barh(sub["model_regime"], sub["log_loss"], color="#ff7f0e")
    axes[1].set_xlim(float(sub["log_loss"].min()) - 0.01, float(sub["log_loss"].max()) + 0.01)
    axes[1].set_title("5-Fold Log-Loss Across CV Regimes & Models")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage6_cv_and_model_comparison.png", dpi=150)
    plt.close(fig)

    # 2. Calibration Curve + Train-OOF vs Unlabeled Prediction Overlay
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0.0, 1.0, 11)
    for label, probs in [("Raw Model A", oof_a), ("Model B (fe_ABCD)", oof_b), ("LogReg", oof_lr)]:
        xs, ys = [], []
        for idx in range(10):
            m = (probs >= bins[idx]) & (
                probs <= bins[idx + 1] if idx == 9 else probs < bins[idx + 1]
            )
            if m.sum() >= 10:
                xs.append(float(probs[m].mean()))
                ys.append(float(y[m].mean()))
        axes[0].plot(xs, ys, marker="o", label=label)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Observed Diabetes Fraction")
    axes[0].set_title("10-Bin Reliability Diagram (SGKF OOF)")
    axes[0].legend()

    axes[1].hist(oof_b, bins=40, density=True, alpha=0.5, label="Train OOF (130,157)")
    axes[1].hist(test_b, bins=40, density=True, alpha=0.5, label="Unlabeled 5-Fold Ens (10,234)")
    axes[1].set_xlabel("Predicted Diabetes Probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Model B Predicted Probability Alignment (Train vs Unlabeled)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage6_calibration_and_alignment.png", dpi=150)
    plt.close(fig)

    # 3. Per-Hospital AUC Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(hosp_aucs, bins=25, color="#2ca02c", edgecolor="white")
    ax.axvline(
        float(np.median(hosp_aucs)),
        color="red",
        linestyle="--",
        label=f"Median={np.median(hosp_aucs):.4f}",
    )
    ax.axvline(0.75, color="black", linestyle=":", label="Threshold=0.7500")
    ax.set_xlabel("Out-of-Hospital ROC-AUC")
    ax.set_ylabel("Hospital Count")
    ax.set_title(f"Per-Hospital Model B ROC-AUC Distribution (n={len(hosp_aucs)} Hospitals)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "stage6_hospital_auc_distribution.png", dpi=150)
    plt.close(fig)


def lint_hmecvas_registry() -> dict[str, int]:
    """Validate inquiries.yaml against all 5 INQUIRY.md REGEXP rules & export master CSV."""
    text = INQUIRIES_PATH.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    inqs = doc.get("inquiries", [])

    re_ref = re.compile(r"#(?:PRB|I\d+(?:\.\d+)*)(?::[a-z_]+)*")
    re_edge = re.compile(
        r"\b(?:Resolves|Spawns|Subtends|Audits|Rivals|Reverts|Ref\s+(?:to|by))\s+"
        r"#(?:PRB|I\d+(?:\.\d+)*)(?::[a-z_]+)*"
    )
    re_outcome = re.compile(r'^\s*outcome:\s*"?(TRUE|FALSE|INCONCLUSIVE)"?\s*(?:#.*)?$', re.M)
    re_status = re.compile(
        r'^\s*status:\s*"?(OPEN|TRUE|FALSE|INCONCLUSIVE|SKIPPED)"?\s*(?:#.*)?$', re.M
    )
    re_sha = re.compile(r'sha256:\s*"?[a-fA-F0-9]{64}"?')
    re_commit = re.compile(r'commit:\s*"?[a-fA-F0-9]{7,40}"?')
    re_ts = re.compile(r'timestamp:\s*"?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})"?')
    re_placeholder = re.compile(r"#ID(?::[a-z_]+)?\b|<[A-Za-z_][^>]*>|PENDING_EXECUTION_")

    n_inqs = len(inqs)
    open_cnt = sum(1 for item in inqs if item.get("status") == "OPEN")
    valid_outcomes = len(re_outcome.findall(text))
    valid_statuses = len(re_status.findall(text))
    valid_shas = len(re_sha.findall(text))
    valid_commits = len(re_commit.findall(text))
    valid_ts = len(re_ts.findall(text))
    placeholders = re_placeholder.findall(text)

    master_rows = [
        {
            "id": item["id"],
            "status": item["status"],
            "outcome": item["verdict"]["outcome"],
            "polarity": item["hypothesis"]["polarity"],
            "script": item["methodology"]["script_ref"],
            "sha256": item["source"]["version"]["sha256"],
            "commit": item["source"]["version"]["commit"],
            "timestamp": item["source"]["extraction"]["timestamp"],
            "claim": item["hypothesis"]["claim"],
        }
        for item in inqs
    ]
    pd.DataFrame(master_rows).to_csv(REPORTS_DIR / "hmecvas_master_registry.csv", index=False)

    return {
        "total_inquiries": n_inqs,
        "open_count": open_cnt,
        "placeholder_count": len(placeholders),
        "invalid_outcome_count": n_inqs - valid_outcomes,
        "invalid_status_count": n_inqs - valid_statuses,
        "invalid_sha256_count": n_inqs - valid_shas,
        "invalid_commit_count": n_inqs - valid_commits,
        "invalid_timestamp_count": n_inqs - valid_ts,
        "ref_matches": len(re_ref.findall(text)),
        "edge_matches": len(re_edge.findall(text)),
    }


def finalize_hmecvas_lint(
    i18_2_entry: dict[str, object] | None = None,
) -> tuple[str, dict[str, int]]:
    """Run two-pass HMECVAS lint update on I18.2 and return (outcome, lint_stats)."""
    if i18_2_entry is None:
        doc = yaml.safe_load(INQUIRIES_PATH.read_text(encoding="utf-8")) or {}
        matches = [item for item in doc.get("inquiries", []) if item.get("id") == "I18.2"]
        if not matches:
            raise ValueError("I18.2 entry not found in inquiries.yaml")
        i18_2_entry = matches[0]
        i18_2_entry["source"]["extraction"]["timestamp"] = iso_now_plus8()
        i18_2_entry["source"]["version"]["sha256"] = sha256_file(INQUIRIES_PATH)
        upsert_inquiries([i18_2_entry])

    lint = lint_hmecvas_registry()
    lint_ok = (
        lint["placeholder_count"] == 0
        and lint["invalid_sha256_count"] == 0
        and lint["invalid_commit_count"] == 0
        and lint["invalid_timestamp_count"] == 0
        and lint["invalid_outcome_count"] == 0
        and lint["invalid_status_count"] == 0
        and lint["open_count"] == 0
    )
    i18_2_out = "TRUE" if lint_ok else "FALSE"
    i18_2_entry["status"] = i18_2_out
    i18_2_entry["verdict"]["outcome"] = i18_2_out  # type: ignore[index]
    i18_2_entry["evidence"]["metric"] = str(lint)  # type: ignore[index]
    upsert_inquiries([i18_2_entry])
    lint = lint_hmecvas_registry()
    return i18_2_out, lint


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    now_iso = iso_now_plus8()
    commit = current_git_commit()
    train_sha = sha256_file(TRAIN_PATH)
    test_sha = sha256_file(TEST_PATH)

    train, test, dd = load_raw_wids()
    dd_map = dd.set_index("Variable Name")["Category"].to_dict()
    y = train["diabetes_mellitus"].to_numpy(dtype=int)
    g = train["hospital_id"].to_numpy()
    f_clin = [c for c in test.columns if c not in DROP_STRUCTURAL]

    sgkf_splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(train, y, g)
    )
    skf_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, y))

    df_fe_tr, cols_fe = build_fe_abcd(train, f_clin, dd_map)
    df_fe_te, _ = build_fe_abcd(test, f_clin, dd_map)

    # I17: SKF_with_IDs vs SKF_no_IDs vs SGKF_no_IDs (Model B)
    oof_skf_ids, _, _ = run_hgb_cv(
        df_fe_tr, cols_fe + ["hospital_id", "icu_id"], y, skf_splits, use_te=True
    )
    oof_skf_noids, _, _ = run_hgb_cv(df_fe_tr, cols_fe, y, skf_splits, use_te=True)
    oof_model_b, test_model_b, perm_b = run_hgb_cv(
        df_fe_tr, cols_fe, y, sgkf_splits, use_te=True, test_df=df_fe_te, compute_perm=True
    )
    assert test_model_b is not None

    m_skf_ids = calc_metrics(y, oof_skf_ids)
    m_skf_noids = calc_metrics(y, oof_skf_noids)
    m_b = calc_metrics(y, oof_model_b)

    delta_naive = m_skf_ids["roc_auc"] - m_b["roc_auc"]
    delta_split = m_skf_noids["roc_auc"] - m_b["roc_auc"]
    i17_out = (
        "TRUE"
        if (delta_naive >= 0.0040 and delta_split >= 0.0015)
        else ("FALSE" if (delta_naive <= 0.0 and delta_split <= 0.0) else "INCONCLUSIVE")
    )

    # I17.1: LogReg vs HGB + Blend
    oof_lr = run_logreg_fe_cv(df_fe_tr, cols_fe, y, sgkf_splits)
    oof_blend = 0.85 * oof_model_b + 0.15 * oof_lr
    m_lr = calc_metrics(y, oof_lr)
    m_blend = calc_metrics(y, oof_blend)
    delta_hgb_lr = m_b["roc_auc"] - m_lr["roc_auc"]
    i17_1_out = (
        "TRUE"
        if (delta_hgb_lr >= 0.0250 and m_lr["roc_auc"] >= 0.7800)
        else ("FALSE" if delta_hgb_lr <= 0.0050 else "INCONCLUSIVE")
    )

    # I18: Raw Model A vs Model B vs Model B-dedup
    oof_a, _, _ = run_hgb_cv(train, f_clin, y, sgkf_splits, use_te=False)
    m_a = calc_metrics(y, oof_a)

    dup_mask = train["encounter_id"].isin(DUP_ENCOUNTERS_167).to_numpy()
    oof_b_dedup, _, _ = run_hgb_cv(
        df_fe_tr, cols_fe, y, sgkf_splits, use_te=True, drop_train_mask=dup_mask
    )
    m_b_dedup_all = calc_metrics(y, oof_b_dedup)
    m_b_dedup_uniq = calc_metrics(y[~dup_mask], oof_b_dedup[~dup_mask])

    auc_gain = m_b["roc_auc"] - m_a["roc_auc"]
    prauc_gain = m_b["pr_auc"] - m_a["pr_auc"]
    ll_drop = m_a["log_loss"] - m_b["log_loss"]
    brier_drop = m_a["brier"] - m_b["brier"]
    dedup_delta = abs(m_b_dedup_all["roc_auc"] - m_b["roc_auc"])
    i18_out = (
        "TRUE"
        if (
            auc_gain >= 0.0030
            and prauc_gain >= 0.0040
            and ll_drop >= 0.0030
            and brier_drop >= 0.0005
            and m_b["ece_10"] <= 0.0150
            and dedup_delta < 0.0010
        )
        else (
            "FALSE"
            if (auc_gain <= 0.0 or prauc_gain <= 0.0 or ll_drop <= 0.0 or brier_drop <= 0.0)
            else "INCONCLUSIVE"
        )
    )

    # Save baseline comparison & feature importance CSVs
    comp_rows = [
        {"model_regime": "SKF_with_IDs_ModelB", "cv": "StratifiedKFold", **m_skf_ids},
        {"model_regime": "SKF_no_IDs_ModelB", "cv": "StratifiedKFold", **m_skf_noids},
        {"model_regime": "SGKF_Raw_ModelA", "cv": "StratifiedGroupKFold", **m_a},
        {"model_regime": "SGKF_LogReg_fe_ABCD", "cv": "StratifiedGroupKFold", **m_lr},
        {"model_regime": "SGKF_ModelB_fe_ABCD", "cv": "StratifiedGroupKFold", **m_b},
        {"model_regime": "SGKF_ModelB_dedup_all", "cv": "StratifiedGroupKFold", **m_b_dedup_all},
        {"model_regime": "SGKF_ModelB_dedup_uniq", "cv": "StratifiedGroupKFold", **m_b_dedup_uniq},
        {"model_regime": "SGKF_Blend_85HGB_15LR", "cv": "StratifiedGroupKFold", **m_blend},
    ]
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(REPORTS_DIR / "stage6_baseline_comparison.csv", index=False)

    pd.DataFrame(
        sorted(perm_b.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "mean_5fold_perm_auc_importance"],
    ).to_csv(REPORTS_DIR / "stage6_feature_importance.csv", index=False)

    # I18.1: Per-hospital AUC, subgroup calibration, Platt/Isotonic & Train-vs-Test KS
    hosp_aucs = []
    sub_cal_rows = []
    for h_id, idx in train.groupby("hospital_id").groups.items():
        idx_arr = np.array(list(idx))
        n_h = len(idx_arr)
        n_pos = int(y[idx_arr].sum())
        n_neg = n_h - n_pos
        if n_h >= 30 and n_pos >= 5 and n_neg >= 5:
            h_auc = float(roc_auc_score(y[idx_arr], oof_model_b[idx_arr]))
            hosp_aucs.append(h_auc)
            sub_cal_rows.append(
                {
                    "group_type": "hospital_id",
                    "level": str(h_id),
                    "n": n_h,
                    "pos_rate": round(float(y[idx_arr].mean()), 5),
                    "mean_pred": round(float(oof_model_b[idx_arr].mean()), 5),
                    **calc_metrics(y[idx_arr], oof_model_b[idx_arr]),
                }
            )

    age_bins = pd.cut(
        train["age"],
        bins=[-np.inf, 45, 65, 80, np.inf],
        labels=["<45", "45-64", "65-79", "80+"],
        right=False,
    )
    bmi_bins = pd.cut(
        train["bmi"],
        bins=[-np.inf, 18.5, 25, 30, 40, np.inf],
        labels=["<18.5", "18.5-24.9", "25-29.9", "30-39.9", "40+"],
        right=False,
    )
    strata_df = pd.DataFrame(
        {
            "age_bin": age_bins.astype(object).fillna("__MISSING__").astype(str),
            "bmi_bin": bmi_bins.astype(object).fillna("__MISSING__").astype(str),
            "ethnicity": train["ethnicity"].fillna("__MISSING__").astype(str),
            "icu_type": train["icu_type"].fillna("__MISSING__").astype(str),
        },
        index=train.index,
    )
    for g_type in ["age_bin", "bmi_bin", "ethnicity", "icu_type"]:
        for lvl, idx in strata_df.groupby(g_type).groups.items():
            idx_arr = np.array(list(idx))
            if len(np.unique(y[idx_arr])) == 2:
                sub_cal_rows.append(
                    {
                        "group_type": g_type,
                        "level": str(lvl),
                        "n": len(idx_arr),
                        "pos_rate": round(float(y[idx_arr].mean()), 5),
                        "mean_pred": round(float(oof_model_b[idx_arr].mean()), 5),
                        **calc_metrics(y[idx_arr], oof_model_b[idx_arr]),
                    }
                )
    pd.DataFrame(sub_cal_rows).to_csv(
        REPORTS_DIR / "stage6_hospital_subgroup_calibration.csv", index=False
    )

    # Nested OOF Platt & Isotonic recalibration check
    oof_platt = np.zeros(len(y), dtype=float)
    oof_iso = np.zeros(len(y), dtype=float)
    eps_p = 1e-6
    logits_b = np.log(
        np.clip(oof_model_b, eps_p, 1.0 - eps_p) / (1.0 - np.clip(oof_model_b, eps_p, 1.0 - eps_p))
    )
    for tr_i, va_i in sgkf_splits:
        platt = LogisticRegression(C=1e6)
        platt.fit(logits_b[tr_i].reshape(-1, 1), y[tr_i])
        oof_platt[va_i] = platt.predict_proba(logits_b[va_i].reshape(-1, 1))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_model_b[tr_i], y[tr_i])
        oof_iso[va_i] = iso.predict(oof_model_b[va_i])
    m_platt = calc_metrics(y, oof_platt)
    m_iso = calc_metrics(y, oof_iso)

    med_h_auc = float(np.median(hosp_aucs))
    iqr_h_auc = float(np.quantile(hosp_aucs, 0.75) - np.quantile(hosp_aucs, 0.25))
    frac_h_75 = float(np.mean(np.array(hosp_aucs) >= 0.75))
    ks_preds = float(stats.ks_2samp(oof_model_b, test_model_b).statistic)
    i18_1_out = (
        "TRUE"
        if (med_h_auc >= 0.8200 and frac_h_75 >= 0.85 and ks_preds < 0.05)
        else (
            "FALSE"
            if (med_h_auc < 0.7800 or frac_h_75 < 0.65 or ks_preds >= 0.10)
            else "INCONCLUSIVE"
        )
    )

    render_stage6_charts(comp_df, y, oof_a, oof_model_b, oof_lr, test_model_b, hosp_aucs)

    src_both = f"data/raw/TrainingWiDS2021.csv; data/raw/UnlabeledWiDS2021.csv (sha256={test_sha})"
    script_ref = "scripts/stage6_baseline_and_verify.py"
    entries = [
        make_inquiry(
            inq_id="I17",
            status=i17_out,
            polarity="+",
            claim=(
                "Random 5-fold StratifiedKFold artificially inflates fe_ABCD ROC-AUC vs. "
                "StratifiedGroupKFold(groups=hospital_id) by >= +0.0040 with facility IDs "
                "and >= +0.0015 even after dropping IDs."
            ),
            technique="3-arm CV-regime comparison on identical fe_ABCD HGB configuration",
            parameters={"n_splits": 5, "max_iter": 200},
            assumptions=["Train/Test disjointness makes hospital-spanning folds optimistic."],
            missing_data_policy="Native NaN preserved; fold-safe OOF TE per fold.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"skf_ids={m_skf_ids['roc_auc']:.5f}, skf_noids={m_skf_noids['roc_auc']:.5f}, "
                f"sgkf_noids={m_b['roc_auc']:.5f}, delta_naive={delta_naive:+.5f}, "
                f"delta_split={delta_split:+.5f}"
            ),
            formula="auc_skf_ids - auc_sgkf; auc_skf_noids - auc_sgkf",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows of TrainingWiDS2021.csv"],
            outcome=i17_out,
            on_true_cond="delta_naive >= 0.0040 AND delta_split >= 0.0015",
            on_false_cond="delta_naive <= 0.0 AND delta_split <= 0.0",
            on_inconc_cond="Partial random-CV optimism",
            act_true="Lock StratifiedGroupKFold(groups=hospital_id); Spawns #I17.1; Ref to #I18.",
            act_false="Document zero inflation while keeping SGKF; Spawns #I17.1; Ref to #I18.",
            act_inconc=(
                "Log partial ID/split optimism in comparison CSV; Spawns #I17.1; Ref to #I18."
            ),
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["fe_ABCD (209 cols)", "hospital_id", "icu_id", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I17.1",
            status=i17_1_out,
            polarity="+",
            claim=(
                "Subtends #I17 and #I7.1. Under 5-fold SGKF, non-linear fe_ABCD HGB outperforms "
                "preprocessed LogisticRegression by >= +0.0250 AUC while LogReg hits >= 0.7800."
            ),
            technique="SGKF model-family benchmark: HGB vs regularized LogisticRegression + blend",
            parameters={"logreg_C": 0.1, "prune_rho": 0.90, "blend": "0.85*HGB + 0.15*LR"},
            assumptions=[
                "Non-monotonic features and ratios favor tree splits over linear log-odds."
            ],
            missing_data_policy="Median+indicator+scaler inside fold for LogReg; native NaN HGB.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"hgb={m_b['roc_auc']:.5f}, logreg={m_lr['roc_auc']:.5f} "
                f"(ll={m_lr['log_loss']:.5f}), delta={delta_hgb_lr:+.5f}, "
                f"blend_auc={m_blend['roc_auc']:.5f} (ll={m_blend['log_loss']:.5f})"
            ),
            formula="auc_model_B_hgb - auc_logreg_fe",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=i17_1_out,
            on_true_cond="delta_hgb_lr >= 0.0250 AND auc_logreg >= 0.7800",
            on_false_cond="delta_hgb_lr <= 0.0050",
            on_inconc_cond="Intermediate tree-vs-linear gap",
            act_true="Confirm GBDT superiority and record linear/blend benchmark -> Ref to #I18.",
            act_false="Elevate LogReg/blend as co-primary baseline -> Ref to #I18.",
            act_inconc="Document linear-vs-tree gap and blend lift -> Ref to #I18.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["fe_ABCD (209 cols)", "hospital_id", "diabetes_mellitus"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I18",
            status=i18_out,
            polarity="+",
            claim=(
                "Under 5-fold SGKF, cleaned+engineered Model B outperforms Raw Model A across "
                "ROC-AUC (>=+0.0030), PR-AUC (>=+0.0040), Log-Loss (>=-0.0030), Brier "
                "(>=-0.0005), ECE_10 <= 0.0150, and dedup |ΔAUC| < 0.0010."
            ),
            technique="Multi-metric baseline comparison and dedup sensitivity under 5-fold SGKF",
            parameters={"ece_bins": 10, "dedup_rows_dropped": len(DUP_ENCOUNTERS_167)},
            assumptions=["All 9 identical hospital_id=167 rows stay inside one SGKF fold."],
            missing_data_policy="Native NaN preserved across Model A, Model B, and Model B-dedup.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"A(auc={m_a['roc_auc']:.5f}, pr={m_a['pr_auc']:.5f}, ll={m_a['log_loss']:.5f}, "
                f"brier={m_a['brier']:.5f}, ece={m_a['ece_10']:.5f}) -> "
                f"B(auc={m_b['roc_auc']:.5f}, pr={m_b['pr_auc']:.5f}, ll={m_b['log_loss']:.5f}, "
                f"brier={m_b['brier']:.5f}, ece={m_b['ece_10']:.5f}); "
                f"dedup_auc={m_b_dedup_all['roc_auc']:.5f} (|Δ|={dedup_delta:.5f})"
            ),
            formula="Model B minus Model A across AUC, PR-AUC, LogLoss, Brier",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows grouped by hospital_id"],
            outcome=i18_out,
            on_true_cond="All 6 discrimination, calibration, and dedup thresholds met",
            on_false_cond="Any primary metric fails to improve over Model A",
            on_inconc_cond="Mixed calibration or dedup sensitivity",
            act_true="Promote Model B as verified benchmark; Spawns #I18.1; Ref to #I18.2.",
            act_false="Roll back failing Stage 4-5 transforms; Spawns #I18.1; Ref to #I18.2.",
            act_inconc="Document metric nuances in stage6 report; Spawns #I18.1; Ref to #I18.2.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=commit,
            sha256=train_sha,
            columns=["Raw 175 cols vs fe_ABCD (209 cols)"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I18.1",
            status=i18_1_out,
            polarity="+",
            claim=(
                "Subtends #I18 and #I6. Model B achieves median per-hospital AUC >= 0.8200 (>=85% "
                "of evaluable hospitals >= 0.7500) while Train OOF vs Unlabeled predictions "
                "align with KS < 0.05."
            ),
            technique="Per-hospital OOF AUC audit, subgroup calibration, Platt/Isotonic, and KS",
            parameters={"min_hosp_n": 30, "min_pos": 5, "min_neg": 5},
            assumptions=[
                "Aligned Train/Unlabeled prediction distributions confirm low shift distortion."
            ],
            missing_data_policy="Identical sanitization + fe_ABCD applied to Train and Unlabeled.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"n_hospitals={len(hosp_aucs)}, med_auc={med_h_auc:.5f} (iqr={iqr_h_auc:.5f}), "
                f"frac_ge75={frac_h_75 * 100:.2f}%, ks_train_vs_test={ks_preds:.5f}, "
                f"train_mean_p={oof_model_b.mean():.5f}, test_mean_p={test_model_b.mean():.5f}, "
                f"platt_ll={m_platt['log_loss']:.5f}, iso_ll={m_iso['log_loss']:.5f}"
            ),
            formula="median(hosp_auc); KS(Train_OOF_p, Unlabeled_5fold_p)",
            timeframe="Static WiDS Datathon 2021 Train and Unlabeled snapshots",
            filters=["130,157 Train OOF and 10,234 Unlabeled 5-fold ensemble predictions"],
            outcome=i18_1_out,
            on_true_cond="med_auc >= 0.8200 AND frac_ge75 >= 0.85 AND ks < 0.05",
            on_false_cond="med_auc < 0.7800 OR frac_ge75 < 0.65 OR ks >= 0.10",
            on_inconc_cond="Moderate hospital spread or KS in [0.05, 0.10)",
            act_true="Certify cross-hospital robustness & alignment -> Ref to #I18.2.",
            act_false="Audit low-AUC hospital outliers before submission -> Ref to #I18.2.",
            act_inconc="Document moderate hospital spread in stage6 summary -> Ref to #I18.2.",
            source_file=src_both,
            commit=commit,
            sha256=train_sha,
            columns=[
                "fe_ABCD Train & Unlabeled",
                "hospital_id",
                "age",
                "bmi",
                "ethnicity",
                "icu_type",
            ],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I18.2",
            status="TRUE",
            polarity="+",
            claim=(
                "All Stage 1-6 Inquiry Units in inquiries.yaml pass all 5 INQUIRY.md schema and "
                "REGEXP checks with zero OPEN inquiries, placeholders, or invalid hashes."
            ),
            technique="Automated INQUIRY.md REGEXP check across reports/findings/inquiries.yaml",
            parameters={"regex_rules": 5},
            assumptions=[
                "Every inquiry unit records commit, sha256, timestamp, and final outcome."
            ],
            missing_data_policy="Missing required fields count directly as schema violations.",
            script_ref=script_ref,
            ev_type="L",
            metric="Pending post-upsert regex check",
            formula="count(placeholders + bad_sha + bad_commit + bad_ts + open) == 0",
            timeframe="Post-Stage-6 registry freeze",
            filters=["All inquiry units in reports/findings/inquiries.yaml"],
            outcome="TRUE",
            on_true_cond="All 5 regex checks pass with 0 OPEN and 0 placeholders",
            on_false_cond="Any regex violation or unfilled placeholder found",
            on_inconc_cond="Valid syntax with >= 1 OPEN inquiry",
            act_true="Resolves #PRB: Close WiDS 2021 HMECVAS audit and finalize todo/jo-todo.md.",
            act_false="Repair failing provenance/placeholders and re-run linter.",
            act_inconc="Resolve remaining OPEN inquiries and re-run linter.",
            source_file="reports/findings/inquiries.yaml",
            commit=commit,
            sha256=sha256_file(INQUIRIES_PATH),
            columns=[
                "id",
                "status",
                "hypothesis",
                "methodology",
                "evidence",
                "condition",
                "verdict",
                "action",
                "source",
            ],
            timestamp=now_iso,
        ),
    ]
    upsert_inquiries(entries)
    i18_2_out, lint = finalize_hmecvas_lint(entries[-1])

    top15_perm = sorted(perm_b.items(), key=lambda x: x[1], reverse=True)[:15]
    top15_md = "\n".join(f"- `{k}`: `{v:.5f}`" for k, v in top15_perm)
    comp_md = "\n".join(
        f"| `{r['model_regime']}` | `{r['cv']}` | {r['roc_auc']:.5f} | {r['pr_auc']:.5f} | "
        f"{r['log_loss']:.5f} | {r['brier']:.5f} | {r['ece_10']:.5f} |"
        for _, r in comp_df.iterrows()
    )

    md = f"""# Stage 6 Report: Baseline Verification, Calibration & HMECVAS Registry Audit

- **Generated**: `{now_iso}` (Base Commit: `{commit}`)

## Inquiry Outcomes

- **[I17 = {i17_out}] CV Leakage Optimism**: `SKF_with_IDs={m_skf_ids["roc_auc"]:.5f}`,
  `SKF_no_IDs={m_skf_noids["roc_auc"]:.5f}`, canonical `SGKF_no_IDs={m_b["roc_auc"]:.5f}`
  (`Δ_naive={delta_naive:+.5f}`, `Δ_split={delta_split:+.5f}`).
- **[I17.1 = {i17_1_out}] HGB vs LogReg Benchmark**: `HGB={m_b["roc_auc"]:.5f}`,
  `LogReg={m_lr["roc_auc"]:.5f}` (`Δ={delta_hgb_lr:+.5f}`), `Blend(85/15)={m_blend["roc_auc"]:.5f}`
  (Log-Loss `{m_blend["log_loss"]:.5f}`).
- **[I18 = {i18_out}] Raw Model A vs Engineered Model B (`fe_ABCD`)**:
  - ROC-AUC: `{m_a["roc_auc"]:.5f} -> {m_b["roc_auc"]:.5f}` (**`{auc_gain:+.5f}`**)
  - PR-AUC: `{m_a["pr_auc"]:.5f} -> {m_b["pr_auc"]:.5f}` (**`{prauc_gain:+.5f}`**)
  - Log-Loss: `{m_a["log_loss"]:.5f} -> {m_b["log_loss"]:.5f}` (**`{-ll_drop:+.5f}`**)
  - Brier: `{m_a["brier"]:.5f} -> {m_b["brier"]:.5f}` (**`{-brier_drop:+.5f}`**),
    `ECE_10={m_b["ece_10"]:.5f}`
  - Dedup sensitivity (`-8` redundant rows): `AUC={m_b_dedup_all["roc_auc"]:.5f}`
    (`|Δ|={dedup_delta:.5f}`).
- **[I18.1 = {i18_1_out}] Cross-Hospital Robustness & Unlabeled Alignment**:
  - Across `{len(hosp_aucs)}` evaluable hospitals (`n>=30, pos>=5, neg>=5`): median AUC =
    **`{med_h_auc:.5f}`** (IQR `{iqr_h_auc:.5f}`), `{frac_h_75 * 100:.2f}%` `>= 0.7500`.
  - Train OOF (`mean={oof_model_b.mean():.5f}`) vs Unlabeled 5-fold ensemble
    (`mean={test_model_b.mean():.5f}`): `KS = {ks_preds:.5f}`.
  - Nested recalibration Log-Loss: raw HGB `{m_b["log_loss"]:.5f}`,
    Platt `{m_platt["log_loss"]:.5f}`, Isotonic `{m_iso["log_loss"]:.5f}`.
- **[I18.2 = {i18_2_out}] HMECVAS Registry Regex Linter**: `{lint}`.

## Model & CV Regime Comparison
| Model / CV Regime | Splitter | ROC-AUC | PR-AUC | Log-Loss | Brier | ECE_10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
{comp_md}

## Top 15 Model B Permutation Feature Importances (5-Fold Mean)
{top15_md}
"""
    (REPORTS_DIR / "stage6_baseline_validation.md").write_text(md, encoding="utf-8")
    print(
        f"Stage 6 done: I17={i17_out}, I17.1={i17_1_out}, I18={i18_out}, "
        f"I18.1={i18_1_out}, I18.2={i18_2_out}"
    )
    print("HMECVAS lint:", lint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6 verification & HMECVAS registry lint.")
    parser.add_argument(
        "--lint-only",
        action="store_true",
        help="Re-run I18.2 HMECVAS regex linter and update report/CSV without retraining models.",
    )
    args = parser.parse_args()
    if args.lint_only:
        out, stats_dict = finalize_hmecvas_lint(None)
        md_path = REPORTS_DIR / "stage6_baseline_validation.md"
        if md_path.exists():
            old_md = md_path.read_text(encoding="utf-8")
            new_line = f"- **[I18.2 = {out}] HMECVAS Registry Regex Linter**: `{stats_dict}`."
            patched = re.sub(
                r"^- \*\*\[I18\.2 = [A-Z]+\] HMECVAS Registry Regex Linter\*\*: .*$",
                new_line,
                old_md,
                flags=re.M,
            )
            md_path.write_text(patched, encoding="utf-8")
        print(f"Stage 6 lint-only done: I18.2={out}")
        print("HMECVAS lint:", stats_dict)
    else:
        main()
