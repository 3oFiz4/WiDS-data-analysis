"""Shared sanitization and domain feature engineering helpers for WiDS 2021."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DROP_STRUCTURAL = {"Unnamed: 0", "encounter_id", "readmission_status", "hospital_id", "icu_id"}
DEMO_COLS = {"age", "bmi", "height", "weight", "ethnicity", "gender"}
TE_CAT_COLS = [
    "apache_3j_diagnosis",
    "apache_3j_major",
    "apache_2_diagnosis",
    "icu_type",
    "ethnicity",
    "hospital_admit_source",
]


def safe_div(num: pd.Series, den: pd.Series, eps: float = 1e-6) -> pd.Series:
    """Divide num by den returning NaN when den <= eps or inputs are missing."""
    valid = num.notna() & den.notna() & (den > eps)
    out = pd.Series(np.nan, index=num.index, dtype=float)
    out.loc[valid] = num.loc[valid].astype(float) / den.loc[valid].astype(float)
    return out


def sanitize_clinical_df(df: pd.DataFrame, f_clin: list[str]) -> pd.DataFrame:
    """Apply Stage 4 min>max swap and sentinel/impossible-bound sanitization."""
    out = df.copy()
    swaps = np.zeros(len(out), dtype=np.int16)
    eps = 1e-6

    for c_min in sorted(c for c in f_clin if c.endswith("_min")):
        c_max = c_min[:-4] + "_max"
        if c_max not in out.columns:
            continue
        m = out[c_min].notna() & out[c_max].notna() & (out[c_min] > out[c_max] + eps)
        if m.any():
            tmp_min = out.loc[m, c_min].copy()
            out.loc[m, c_min] = out.loc[m, c_max]
            out.loc[m, c_max] = tmp_min
            swaps += m.to_numpy(dtype=np.int16)

    out["n_min_max_swapped"] = swaps

    for c in f_clin:
        if not pd.api.types.is_numeric_dtype(out[c]):
            continue
        v = out[c].astype(float)
        nn = v.notna()
        bad = pd.Series(False, index=out.index)
        if c == "pre_icu_los_days":
            bad = nn & (v < 0)
        elif c == "age":
            bad = nn & ((v <= 0) | (v > 110))
        elif "resprate" in c:
            bad = nn & ((v <= 0) | (v > 100))
        elif "heartrate" in c or c == "heart_rate_apache":
            bad = nn & ((v <= 0) | (v > 250))
        elif "sysbp" in c:
            bad = nn & ((v <= 0) | (v > 300))
        elif "diasbp" in c:
            bad = nn & ((v <= 0) | (v > 200))
        elif "_spo2_" in c:
            bad = nn & ((v < 30) | (v > 100))
        elif "glucose" in c:
            bad = nn & ((v < 10) | (v > 1500))
        elif "_temp_" in c or c == "temp_apache":
            bad = nn & ((v < 28) | (v > 43))
        if bad.any():
            out.loc[bad, c] = np.nan
    return out


def add_glucose_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Family B glycemic variability and cross-window glucose deltas."""
    return pd.DataFrame(
        {
            "d1_glucose_range": df["d1_glucose_max"] - df["d1_glucose_min"],
            "d1_glucose_ratio": safe_div(df["d1_glucose_max"], df["d1_glucose_min"]),
            "d1_glucose_mid": 0.5 * (df["d1_glucose_max"] + df["d1_glucose_min"]),
            "h1_glucose_range": df["h1_glucose_max"] - df["h1_glucose_min"],
            "glucose_h1_vs_d1_max_delta": df["h1_glucose_max"] - df["d1_glucose_max"],
            "glucose_apache_vs_d1_min": df["glucose_apache"] - df["d1_glucose_min"],
            "glucose_apache_vs_d1_max": df["glucose_apache"] - df["d1_glucose_max"],
        },
        index=df.index,
    )


def add_clinical_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add Family C renal, metabolic, hemodynamic, and BMI ratios."""
    bmi_calc = safe_div(df["weight"], (df["height"] / 100.0) ** 2)
    inv_bp_cols = [c for c in df.columns if "_invasive_" in c]
    return pd.DataFrame(
        {
            "d1_bun_creatinine_ratio": safe_div(df["d1_bun_max"], df["d1_creatinine_max"]),
            "apache_bun_creatinine_ratio": safe_div(df["bun_apache"], df["creatinine_apache"]),
            "bmi_calc": bmi_calc,
            "bmi_discrepancy": df["bmi"] - bmi_calc,
            "d1_shock_index_max": safe_div(df["d1_heartrate_max"], df["d1_sysbp_min"]),
            "d1_pulse_pressure_max": df["d1_sysbp_max"] - df["d1_diasbp_min"],
            "d1_anion_gap_proxy": df["d1_sodium_max"] - df["d1_hco3_min"],
            "d1_temp_range": df["d1_temp_max"] - df["d1_temp_min"],
            "d1_sysbp_range": df["d1_sysbp_max"] - df["d1_sysbp_min"],
            "d1_heartrate_range": df["d1_heartrate_max"] - df["d1_heartrate_min"],
            "d1_hb_hct_ratio": safe_div(df["d1_hemaglobin_max"], df["d1_hematocrit_max"]),
            "has_invasive_bp": df[inv_bp_cols].notna().any(axis=1).astype(np.int8),
        },
        index=df.index,
    )


def add_hierarchy_and_miss_counts(
    df: pd.DataFrame, f_clin: list[str], dd_map: dict[str, str]
) -> pd.DataFrame:
    """Add APACHE III-J hierarchy components and non-demographic missingness counts."""
    a3 = df["apache_3j_diagnosis"]
    non_demo = [c for c in f_clin if c not in DEMO_COLS]
    h1_labs = [c for c in non_demo if c.startswith("h1_") and dd_map.get(c) == "labs"]
    d1_labs = [c for c in non_demo if c.startswith("d1_") and dd_map.get(c) == "labs"]
    bg_cols = [c for c in non_demo if dd_map.get(c) == "labs blood gas"]
    ap_cols = [c for c in non_demo if dd_map.get(c) == "APACHE covariate"]
    return pd.DataFrame(
        {
            "apache_3j_major": np.floor(a3),
            "apache_3j_system": np.floor(a3 / 100.0),
            "apache_3j_sub": np.round(a3 - np.floor(a3), 2),
            "n_miss_total_nondemo": df[non_demo].isna().sum(axis=1).astype(np.int16),
            "n_miss_h1_labs": df[h1_labs].isna().sum(axis=1).astype(np.int16),
            "n_miss_d1_labs": df[d1_labs].isna().sum(axis=1).astype(np.int16),
            "n_miss_blood_gas": df[bg_cols].isna().sum(axis=1).astype(np.int16),
            "n_miss_apache": df[ap_cols].isna().sum(axis=1).astype(np.int16),
        },
        index=df.index,
    )


def add_hosp_covariate_rel(df: pd.DataFrame) -> pd.DataFrame:
    """Add strictly target-free within-hospital covariate centering and order rates."""
    out = pd.DataFrame(index=df.index)
    h = df["hospital_id"]
    h_cnt = h.map(h.value_counts())
    valid_h = h_cnt >= 5
    for c in ["d1_glucose_max", "bmi", "age"]:
        h_mean = df.groupby("hospital_id")[c].transform("mean")
        diff = df[c] - h_mean
        out[f"{c}_hosp_centered"] = diff.where(valid_h, np.nan)
    gluc_rate = df["d1_glucose_max"].notna().astype(float).groupby(h).transform("mean")
    out["hosp_glucose_order_rate"] = gluc_rate.where(valid_h, np.nan)
    return out


def compute_fold_safe_features(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    y_tr: np.ndarray,
    *,
    use_te: bool = False,
    use_stratum_z: bool = False,
    te_smoothing_m: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute fold-safe OOF target encodings and clinical-stratum z-scores."""
    tr_extra: dict[str, np.ndarray] = {}
    va_extra: dict[str, np.ndarray] = {}

    if use_te:
        prior = float(y_tr.mean())
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_splits = list(inner_skf.split(tr_df, y_tr))
        for col in TE_CAT_COLS:
            k_tr = tr_df[col].fillna("__MISSING__").astype(str)
            k_va = va_df[col].fillna("__MISSING__").astype(str)
            tr_arr = np.full(len(tr_df), prior, dtype=float)
            for itr, iva in inner_splits:
                iprior = float(y_tr[itr].mean())
                iagg = (
                    pd.DataFrame({"k": k_tr.iloc[itr].to_numpy(), "y": y_tr[itr]})
                    .groupby("k")["y"]
                    .agg(["sum", "count"])
                )
                imp = (iagg["sum"] + te_smoothing_m * iprior) / (iagg["count"] + te_smoothing_m)
                tr_arr[iva] = k_tr.iloc[iva].map(imp).fillna(iprior).to_numpy()
            agg = (
                pd.DataFrame({"k": k_tr.to_numpy(), "y": y_tr})
                .groupby("k")["y"]
                .agg(["sum", "count"])
            )
            mp = (agg["sum"] + te_smoothing_m * prior) / (agg["count"] + te_smoothing_m)
            tr_extra[f"{col}_te"] = tr_arr
            va_extra[f"{col}_te"] = k_va.map(mp).fillna(prior).to_numpy()

    if use_stratum_z:

        def make_stratum(d: pd.DataFrame) -> pd.Series:
            ab = pd.cut(
                d["age"],
                bins=[-np.inf, 45, 65, 80, np.inf],
                labels=["<45", "45-64", "65-79", "80+"],
                right=False,
            )
            icu = d["icu_type"].fillna("__NA__").astype(str)
            return ab.astype(object).fillna("__NA__").astype(str) + "|" + icu

        s_tr = make_stratum(tr_df)
        s_va = make_stratum(va_df)
        for col in ["d1_glucose_max", "glucose_apache", "d1_creatinine_max", "bmi"]:
            g_df = pd.DataFrame({"s": s_tr.to_numpy(), "v": tr_df[col].to_numpy(dtype=float)})
            med_s = g_df.groupby("s")["v"].median()
            q75_s = g_df.groupby("s")["v"].quantile(0.75)
            q25_s = g_df.groupby("s")["v"].quantile(0.25)
            iqr_s = (q75_s - q25_s).clip(lower=1e-3)
            g_med = float(tr_df[col].median())
            g_iqr = max(1e-3, float(tr_df[col].quantile(0.75) - tr_df[col].quantile(0.25)))

            tr_med = s_tr.map(med_s).fillna(g_med).to_numpy()
            tr_iqr = s_tr.map(iqr_s).fillna(g_iqr).to_numpy()
            va_med = s_va.map(med_s).fillna(g_med).to_numpy()
            va_iqr = s_va.map(iqr_s).fillna(g_iqr).to_numpy()

            tr_extra[f"{col}_stratum_z"] = (tr_df[col].to_numpy(dtype=float) - tr_med) / tr_iqr
            va_extra[f"{col}_stratum_z"] = (va_df[col].to_numpy(dtype=float) - va_med) / va_iqr

    return pd.DataFrame(tr_extra, index=tr_df.index), pd.DataFrame(va_extra, index=va_df.index)
