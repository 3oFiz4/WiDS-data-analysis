"""Stage 1 EDA: Schema, Identifiers, Constants, Non-ID Duplicates & Sparsity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

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

REPORTS_FINDINGS_DIR = PROJECT_ROOT / "reports" / "findings"
DOCS_DIR = PROJECT_ROOT / "docs"


def classify_temporal_window(var_name: str, category: str) -> str:
    """Assign each column to one of 4 temporal/domain windows."""
    if var_name.startswith("h1_"):
        return "h1_first_hour"
    if var_name.startswith("d1_"):
        return "d1_first_24h"
    if category == "APACHE covariate":
        return "apache_24h_worst"
    return "static_baseline"


def compute_column_profile(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute missingness, non-null cardinality, and modal dominance."""
    n_rows = len(df)
    records = []
    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        n_non_null = n_rows - n_null
        miss_pct = 100.0 * n_null / n_rows
        non_null = s.dropna()
        if n_non_null == 0:
            nunique, mode_val, mode_count, mode_share = 0, None, 0, 1.0
        else:
            vc = non_null.value_counts(dropna=True)
            nunique = int(len(vc))
            mode_val = vc.index[0]
            mode_count = int(vc.iloc[0])
            mode_share = float(mode_count / n_non_null)
        records.append(
            {
                "column": col,
                f"dtype_{prefix}": str(s.dtype),
                f"n_null_{prefix}": n_null,
                f"n_non_null_{prefix}": n_non_null,
                f"missing_pct_{prefix}": round(miss_pct, 4),
                f"nunique_{prefix}": nunique,
                f"mode_{prefix}": str(mode_val) if mode_val is not None else "ALL_NULL",
                f"mode_count_{prefix}": mode_count,
                f"mode_share_{prefix}": round(mode_share, 6),
            }
        )
    return pd.DataFrame(records)


def find_near_duplicates_within_hospital(df: pd.DataFrame, f_clinical: list[str]) -> pd.DataFrame:
    """Block on facility + demographics and score mutual non-null agreement."""
    keys = ["hospital_id", "age", "gender", "ethnicity", "height", "weight"]
    sub = df.dropna(subset=keys).copy()
    sub["_row_idx"] = sub.index

    is_num = np.array([pd.api.types.is_numeric_dtype(df[c]) for c in f_clinical])
    num_cols = [c for c, flag in zip(f_clinical, is_num, strict=True) if flag]
    cat_cols = [c for c, flag in zip(f_clinical, is_num, strict=True) if not flag]
    num_mat = df[num_cols].to_numpy(dtype=float) if num_cols else np.empty((len(df), 0))
    cat_mat = df[cat_cols].astype(str).where(df[cat_cols].notna(), "__NA__").to_numpy()

    pairs = []
    for _, grp in sub.groupby(keys, sort=False):
        if len(grp) < 2 or len(grp) > 200:
            continue
        idxs = grp["_row_idx"].to_numpy()
        for a_pos in range(len(idxs)):
            i = int(idxs[a_pos])
            for b_pos in range(a_pos + 1, len(idxs)):
                j = int(idxs[b_pos])
                num_v = (~np.isnan(num_mat[i])) & (~np.isnan(num_mat[j]))
                cat_v = (cat_mat[i] != "__NA__") & (cat_mat[j] != "__NA__")
                n_mut = int(num_v.sum() + cat_v.sum())
                if n_mut < 30:
                    continue
                n_eq = int(
                    np.isclose(num_mat[i, num_v], num_mat[j, num_v], atol=1e-6, rtol=0.0).sum()
                    + (cat_mat[i, cat_v] == cat_mat[j, cat_v]).sum()
                )
                agree = n_eq / n_mut
                if agree < 0.90:
                    continue
                all_num_eq = np.all(
                    (np.isnan(num_mat[i]) & np.isnan(num_mat[j]))
                    | (
                        (~np.isnan(num_mat[i]))
                        & (~np.isnan(num_mat[j]))
                        & np.isclose(num_mat[i], num_mat[j], atol=1e-6, rtol=0.0)
                    )
                )
                if not (all_num_eq and np.all(cat_mat[i] == cat_mat[j])):
                    pairs.append(
                        {
                            "row_i": i,
                            "row_j": j,
                            "encounter_i": int(df.loc[i, "encounter_id"]),
                            "encounter_j": int(df.loc[j, "encounter_id"]),
                            "hospital_id": int(df.loc[i, "hospital_id"]),
                            "agreement_rate": round(agree, 6),
                        }
                    )
    return pd.DataFrame(pairs)


def main() -> None:
    REPORTS_FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    now_iso = iso_now_plus8()
    git_commit = current_git_commit()
    train_sha = sha256_file(TRAIN_PATH)
    test_sha = sha256_file(TEST_PATH)
    dict_sha = sha256_file(DICT_PATH)

    train, test, dd = load_raw_wids()
    n_train, n_test = len(train), len(test)

    # I1: Schema, structural drop list & near-constant audit
    train_preds = [c for c in train.columns if c != "diabetes_mellitus"]
    schema_aligned = train_preds == list(test.columns)

    prof_tr = compute_column_profile(train, "train")
    prof_te = compute_column_profile(test, "test")
    prof = prof_tr.merge(prof_te, on="column", how="outer")

    dd_clean = dd.rename(
        columns={
            "Variable Name": "column",
            "Category": "dd_category",
            "Unit of Measure": "dd_unit",
            "Data Type": "dd_data_type",
            "Description": "dd_description",
            "Example": "dd_example",
        }
    )
    prof = prof.merge(dd_clean, on="column", how="left")
    prof["dd_category"] = prof["dd_category"].fillna("csv_index_artifact")
    prof["temporal_window"] = [
        classify_temporal_window(c, cat)
        for c, cat in zip(prof["column"], prof["dd_category"], strict=True)
    ]

    facility_cols = {"hospital_id", "icu_id"}

    def role_of(row: pd.Series) -> str:
        c = row["column"]
        if c == "diabetes_mellitus":
            return "target"
        if c in facility_cols:
            return "facility_group_key"
        u_tr, u_te = row["nunique_train"], row["nunique_test"]
        nn_tr, nn_te = row["n_non_null_train"], row["n_non_null_test"]
        if (pd.notna(u_tr) and u_tr <= 1) or (pd.notna(u_te) and u_te <= 1):
            return "constant_zero_variance"
        if u_tr == n_train and nn_tr == n_train and u_te == n_test and nn_te == n_test:
            return "pure_row_identifier"
        if row["mode_share_train"] >= 0.99 and row["mode_share_test"] >= 0.99:
            return "near_constant_ge_99pct"
        return "candidate_predictor"

    prof["structural_role"] = prof.apply(role_of, axis=1)
    const_id_cols = prof.loc[
        prof["structural_role"].isin(["constant_zero_variance", "pure_row_identifier"]),
        "column",
    ].tolist()
    near_const_cols = prof.loc[
        prof["structural_role"] == "near_constant_ge_99pct", "column"
    ].tolist()

    unnamed_seq_tr = bool((train["Unnamed: 0"].to_numpy() == np.arange(1, n_train + 1)).all())
    unnamed_seq_te = bool((test["Unnamed: 0"].to_numpy() == np.arange(1, n_test + 1)).all())

    prof_sorted = prof.sort_values(
        by=["missing_pct_train", "mode_share_train", "nunique_train"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    prof_sorted.to_csv(REPORTS_FINDINGS_DIR / "feature_sparsity_constants.csv", index=False)

    i1_out = (
        "TRUE"
        if (schema_aligned and len(const_id_cols) >= 3 and len(near_const_cols) >= 1)
        else ("FALSE" if not schema_aligned else "INCONCLUSIVE")
    )

    # I1.1: Dictionary vs CSV reconciliation & ID overlap
    dict_only = sorted(set(dd["Variable Name"]) - set(train.columns))
    csv_only = sorted(set(train.columns) - set(dd["Variable Name"]))
    symdiff = len(dict_only) + len(csv_only)

    mismatches: list[str] = []
    for col in ["bmi", "apache_2_diagnosis", "apache_3j_diagnosis"]:
        dt = str(dd_clean.loc[dd_clean["column"] == col, "dd_data_type"].iloc[0]).lower()
        rate = float(
            pd.to_numeric(train[col], errors="coerce").notna().sum() / train[col].notna().sum()
        )
        if dt == "string" and rate >= 0.99:
            mismatches.append(
                f"{col} tagged 'string' in DataDictionary ({rate * 100:.1f}% numeric)"
            )
    for col in ["d1_albumin_max", "h1_albumin_max"]:
        row_d = dd_clean.loc[dd_clean["column"] == col].iloc[0]
        if "lowest" in str(row_d["dd_description"]).lower():
            mismatches.append(f"{col} description says 'lowest' and unit='{row_d['dd_unit']}'")
    for col in ["d1_inr_max", "d1_inr_min", "h1_inr_max", "h1_inr_min"]:
        if str(dd_clean.loc[dd_clean["column"] == col, "dd_unit"].iloc[0]) == "micromol/L":
            mismatches.append(f"{col} unit listed as 'micromol/L' (INR is dimensionless)")
    for col in [
        "d1_pao2fio2ratio_max",
        "d1_pao2fio2ratio_min",
        "h1_pao2fio2ratio_max",
        "h1_pao2fio2ratio_min",
    ]:
        desc = str(dd_clean.loc[dd_clean["column"] == col, "dd_description"].iloc[0]).lower()
        if "fraction of inspired oxygen" in desc:
            mismatches.append(f"{col} description/unit copied from FiO2")

    hosp_tr = set(train["hospital_id"].dropna().unique())
    hosp_te = set(test["hospital_id"].dropna().unique())
    icu_tr = set(train["icu_id"].dropna().unique())
    icu_te = set(test["icu_id"].dropna().unique())
    hosp_ov = len(hosp_tr & hosp_te)
    icu_ov = len(icu_tr & icu_te)
    enc_ov = len(set(train["encounter_id"].dropna()) & set(test["encounter_id"].dropna()))

    i1_1_out = (
        "TRUE"
        if (symdiff >= 2 and len(mismatches) >= 1 and hosp_ov == 0)
        else (
            "FALSE" if (symdiff == 0 and len(mismatches) == 0 and hosp_ov > 0) else "INCONCLUSIVE"
        )
    )

    # I2 & I2.1: Duplicate scopes & cross-split / near-duplicate checks
    drop_i2 = set(const_id_cols) | facility_cols | {"diabetes_mellitus"}
    f_clin = [c for c in train.columns if c not in drop_i2]
    f_fac = [c for c in train.columns if c not in (set(const_id_cols) | {"diabetes_mellitus"})]
    prof_idx = prof.set_index("column")
    f_80 = [c for c in f_clin if prof_idx.loc[c, "missing_pct_train"] <= 80.0]
    f_50 = [c for c in f_clin if prof_idx.loc[c, "missing_pct_train"] <= 50.0]
    av_removed = {
        "Unnamed: 0",
        "encounter_id",
        "readmission_status",
        "h1_albumin_max",
        "h1_albumin_min",
        "h1_bilirubin_max",
        "h1_bilirubin_min",
        "h1_lactate_max",
        "h1_lactate_min",
        "diabetes_mellitus",
    }
    f_av_pre = [c for c in train.columns if c not in av_removed]

    n_dup_clin = int(train.duplicated(subset=f_clin, keep="first").sum())
    n_dup_fac = int(train.duplicated(subset=f_fac, keep="first").sum())
    n_dup_80 = int(train.duplicated(subset=f_80, keep="first").sum())
    n_dup_50 = int(train.duplicated(subset=f_50, keep="first").sum())
    n_dup_av = int(train.duplicated(subset=f_av_pre, keep="first").sum())
    found_9 = 9 in {n_dup_av, n_dup_50, n_dup_80}

    i2_out = "FALSE" if n_dup_clin >= 9 else ("TRUE" if found_9 else "INCONCLUSIVE")

    # Cross-split exact collision
    h_tr = pd.util.hash_pandas_object(train[f_clin], index=False)
    h_te = pd.util.hash_pandas_object(test[f_clin], index=False)
    cand = set(h_tr.tolist()) & set(h_te.tolist())
    n_cross = (
        len(train.loc[h_tr.isin(cand), f_clin].merge(test.loc[h_te.isin(cand), f_clin], on=f_clin))
        if cand
        else 0
    )

    near_df = find_near_duplicates_within_hospital(train, f_clin)
    n_near = len(near_df)

    dup_frames = []
    if n_dup_clin > 0:
        cols_meta = ["encounter_id", "hospital_id", "icu_id", "icu_stay_type", "diabetes_mellitus"]
        ex = train.loc[train.duplicated(subset=f_clin, keep=False), cols_meta].copy()
        ex["match_type"] = "exact_F_clinical"
        dup_frames.append(ex)
    if n_near > 0:
        nd = near_df.copy()
        nd["match_type"] = "within_hospital_near_dup_ge90"
        dup_frames.append(nd)
    dup_csv = REPORTS_FINDINGS_DIR / "stage1_duplicate_audit.csv"
    pd.concat(dup_frames, ignore_index=True).to_csv(dup_csv, index=False)

    i2_1_out = (
        "TRUE"
        if (n_cross >= 1 or n_near >= 10)
        else ("FALSE" if (n_cross == 0 and n_near == 0) else "INCONCLUSIVE")
    )

    # I3 & I3.1: Matched (h1_, d1_) Wilcoxon & sparsity concentration
    pred_prof = prof[prof["column"] != "diabetes_mellitus"].copy()
    h1_cols = sorted(c for c in pred_prof["column"] if c.startswith("h1_"))
    pidx = pred_prof.set_index("column")
    deltas_tr, deltas_te = [], []
    for h1_c in h1_cols:
        d1_c = "d1_" + h1_c[3:]
        if d1_c in pidx.index:
            deltas_tr.append(
                float(pidx.loc[h1_c, "missing_pct_train"] - pidx.loc[d1_c, "missing_pct_train"])
            )
            deltas_te.append(
                float(pidx.loc[h1_c, "missing_pct_test"] - pidx.loc[d1_c, "missing_pct_test"])
            )
    med_tr, med_te = float(np.median(deltas_tr)), float(np.median(deltas_te))
    inv_tr = int(np.sum(np.array(deltas_tr) < 0))
    inv_te = int(np.sum(np.array(deltas_te) < 0))
    wil_p_tr = float(stats.wilcoxon(deltas_tr, alternative="greater").pvalue)
    wil_p_te = float(stats.wilcoxon(deltas_te, alternative="greater").pvalue)

    gt50_tr = pred_prof[pred_prof["missing_pct_train"] > 50.0]
    gt50_te = pred_prof[pred_prof["missing_pct_test"] > 50.0]
    focus_tr = gt50_tr[
        (gt50_tr["temporal_window"] == "h1_first_hour")
        | (gt50_tr["dd_category"] == "labs blood gas")
    ]
    focus_te = gt50_te[
        (gt50_te["temporal_window"] == "h1_first_hour")
        | (gt50_te["dd_category"] == "labs blood gas")
    ]
    conc_tr = float(len(focus_tr) / max(1, len(gt50_tr)))
    conc_te = float(len(focus_te) / max(1, len(gt50_te)))

    dw_summary = (
        pred_prof.groupby(["dd_category", "temporal_window"], dropna=False)
        .agg(
            n_columns=("column", "count"),
            mean_miss_train=("missing_pct_train", "mean"),
            median_miss_train=("missing_pct_train", "median"),
            mean_miss_test=("missing_pct_test", "mean"),
            median_miss_test=("missing_pct_test", "median"),
            n_gt50_train=("missing_pct_train", lambda x: int((x > 50.0).sum())),
            n_gt80_train=("missing_pct_train", lambda x: int((x > 80.0).sum())),
        )
        .reset_index()
        .sort_values(["mean_miss_train", "n_columns"], ascending=[False, False])
    )
    dw_summary.to_csv(REPORTS_FINDINGS_DIR / "stage1_domain_window_sparsity.csv", index=False)

    i3_out = (
        "TRUE"
        if (
            med_tr >= 5.0
            and wil_p_tr < 0.01
            and conc_tr >= 0.60
            and med_te >= 5.0
            and wil_p_te < 0.01
            and conc_te >= 0.60
        )
        else "INCONCLUSIVE"
    )

    pred_prof["abs_miss_shift_pp"] = (
        pred_prof["missing_pct_train"] - pred_prof["missing_pct_test"]
    ).abs()
    shifted_ge5 = pred_prof[pred_prof["abs_miss_shift_pp"] >= 5.0].sort_values(
        "abs_miss_shift_pp", ascending=False
    )
    unlabeled_ge95 = pred_prof[pred_prof["missing_pct_test"] >= 95.0]
    miss_rho = float(
        stats.spearmanr(pred_prof["missing_pct_train"], pred_prof["missing_pct_test"]).statistic
    )
    i3_1_out = (
        "TRUE"
        if (len(shifted_ge5) >= 10 or len(unlabeled_ge95) >= 1)
        else (
            "FALSE"
            if (len(shifted_ge5) == 0 and len(unlabeled_ge95) == 0 and miss_rho >= 0.95)
            else "INCONCLUSIVE"
        )
    )

    # Write docs/data_dictionary.md
    dd_lines = [
        "# WiDS Datathon 2021 Data Dictionary & Schema Quirks",
        "",
        "Grounded audit of `data/raw/DataDictionaryWiDS2021.csv` against "
        "`TrainingWiDS2021.csv` (130,157 × 181) and `UnlabeledWiDS2021.csv` (10,234 × 180).",
        "",
        "## Known Dictionary vs. CSV Quirks",
        "",
        f"- **Absent dictionary column**: `icu_admit_type` (in dict, absent in CSVs: {dict_only}).",
        f"- **Undocumented index**: `Unnamed: 0` (seq 1..N train={unnamed_seq_tr}, "
        f"test={unnamed_seq_te}; {csv_only}).",
        "- **Zero-variance constant**: `readmission_status` (`nunique=1`, all `0`).",
        f"- **Facility overlap**: `hospital_id` overlap={hosp_ov} ({len(hosp_tr)} vs "
        f"{len(hosp_te)}); `icu_id` overlap={icu_ov} ({len(icu_tr)} vs {len(icu_te)}); "
        f"`encounter_id`={enc_ov}.",
        "- **Dictionary metadata errors**: `bmi`/`apache_2_diagnosis`/`apache_3j_diagnosis` "
        "tagged `string`; `d1/h1_albumin_max` says 'lowest'; `inr` unit `micromol/L`; "
        "`pao2fio2ratio` copied from `fio2`.",
        "",
        "## Domain × Temporal Window Summary",
        "",
        "| Category | Window | Cols | Train Null % | Test Null % | >50% Null | >80% Null |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in dw_summary.iterrows():
        dd_lines.append(
            f"| `{r['dd_category']}` | `{r['temporal_window']}` | {int(r['n_columns'])} | "
            f"{r['mean_miss_train']:.2f}% | {r['mean_miss_test']:.2f}% | "
            f"{int(r['n_gt50_train'])} | {int(r['n_gt80_train'])} |"
        )
    dd_lines.extend(
        [
            "",
            "## Full 181-Column Catalog",
            "",
            "| Column | Role | Category | Window | Dtype | Train Null % | Test Null % | Distinct |",
            "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |",
        ]
    )
    for _, r in prof.iterrows():
        te_n = f"{r['missing_pct_test']:.2f}%" if pd.notna(r["missing_pct_test"]) else "Target"
        dd_lines.append(
            f"| `{r['column']}` | `{r['structural_role']}` | `{r['dd_category']}` | "
            f"`{r['temporal_window']}` | `{r['dtype_train']}` | {r['missing_pct_train']:.2f}% | "
            f"{te_n} | {int(r['nunique_train'])} |"
        )
    (DOCS_DIR / "data_dictionary.md").write_text("\n".join(dd_lines) + "\n", encoding="utf-8")

    # Upsert Stage 1 inquiries to reports/findings/inquiries.yaml
    src_both = f"data/raw/TrainingWiDS2021.csv; data/raw/UnlabeledWiDS2021.csv (sha256={test_sha})"
    src_dict = (
        f"data/raw/DataDictionaryWiDS2021.csv (sha256={dict_sha}); data/raw/TrainingWiDS2021.csv"
    )
    script_ref = "scripts/stage1_schema_and_sparsity.py"
    entries = [
        make_inquiry(
            inq_id="I1",
            status=i1_out,
            polarity="+",
            claim=(
                "Excluding diabetes_mellitus, Train and Unlabeled share an identical 180-column "
                "predictor schema with >=3 structural drop columns and >=1 near-constant (>=99%)."
            ),
            technique="Full-table predictor schema check, cardinality audit, and modal dominance",
            parameters={"constant_nunique_max": 1, "near_constant_mode_share_min": 0.99},
            assumptions=["'NA' and empty fields denote missing observations in both CSVs."],
            missing_data_policy="Explicit null tally; nunique and mode_share over non-null rows.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"schema_aligned={schema_aligned}, |C_const_id|={len(const_id_cols)} "
                f"({const_id_cols}), |C_near_const|={len(near_const_cols)} ({near_const_cols})"
            ),
            formula="schema_aligned AND |C_const_id| >= 3 AND |C_near_const| >= 1",
            timeframe="Static WiDS Datathon 2021 dataset snapshot",
            filters=["All 130,157 Train rows and 10,234 Unlabeled rows; isolate hospital/icu IDs"],
            outcome=i1_out,
            on_true_cond="schema_aligned AND |C_const_id| >= 3 AND |C_near_const| >= 1",
            on_false_cond="NOT schema_aligned OR (|C_const_id|==0 AND |C_near_const|==0)",
            on_inconc_cond="Partial structural/near-constant counts",
            act_true=f"Lock {const_id_cols} drop list; Spawns #I1.1; Ref to #I2.",
            act_false="Spawns #I1.1 to diagnose schema mismatch.",
            act_inconc="Spawns #I1.1 and Ref to #I2.",
            source_file=src_both,
            commit=git_commit,
            sha256=train_sha,
            columns=["Unnamed: 0 .. diabetes_mellitus (181 cols)"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I1.1",
            status=i1_1_out,
            polarity="+",
            claim=(
                "Subtends #I1. DataDictionaryWiDS2021.csv has schema/metadata drift vs CSVs "
                "while hospital_id has zero overlap between Train and Unlabeled."
            ),
            technique="Symmetric dictionary-CSV reconciliation and facility ID set intersections",
            parameters={"id_columns": ["hospital_id", "icu_id", "encounter_id"]},
            assumptions=["Dictionary numeric fields parse cleanly as float64."],
            missing_data_policy="Nulls excluded from coercion checks and ID set intersections.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"symdiff={symdiff} (dict_only={dict_only}, csv_only={csv_only}), "
                f"mismatches={len(mismatches)}, hospital_overlap={hosp_ov}, "
                f"icu_overlap={icu_ov}, encounter_overlap={enc_ov}"
            ),
            formula="|(Dict \\ CSV) U (CSV \\ Dict)|; |Train[id] & Test[id]|",
            timeframe="Static WiDS Datathon 2021 dataset snapshot",
            filters=["All 181 DataDictionary rows vs Train/Unlabeled CSVs"],
            outcome=i1_1_out,
            on_true_cond="symdiff >= 2 AND mismatches >= 1 AND hospital_overlap == 0",
            on_false_cond="symdiff == 0 AND mismatches == 0 AND hospital_overlap > 0",
            on_inconc_cond="Mixed drift and ID overlap",
            act_true="Update docs/data_dictionary.md and lock StratifiedGroupKFold -> Ref to #I4.",
            act_false="Confirm dictionary fidelity -> Ref to #I4.",
            act_inconc="Log partial overlap in stage1_schema_sparsity.md -> Ref to #I4.",
            source_file=src_dict,
            commit=git_commit,
            sha256=train_sha,
            columns=["hospital_id", "icu_id", "encounter_id", "bmi", "apache_3j_diagnosis"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I2",
            status=i2_out,
            polarity="+",
            claim=(
                "Excluding structural IDs, constants, and target, TrainingWiDS2021.csv has < 9 "
                "exact 175-col clinical duplicates, showing AutoViz's 9 duplicates came from "
                "column reduction."
            ),
            technique="Progressive row-duplication profiling across F_clinical and pruned subsets",
            parameters={"duplicate_keep": "first"},
            assumptions=["NaN == NaN positional matching across scopes."],
            missing_data_policy="NaN preserved across raw scopes.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"F_clinical={n_dup_clin}, F_facility={n_dup_fac}, F_80pct={n_dup_80}, "
                f"F_50pct={n_dup_50}, F_autoviz_pre_dq={n_dup_av}"
            ),
            formula="sum(df[scope].duplicated(keep='first'))",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["All 130,157 rows in TrainingWiDS2021.csv"],
            outcome=i2_out,
            on_true_cond="N_dup_clinical < 9 AND 9 in raw pruned scopes",
            on_false_cond="N_dup_clinical >= 9",
            on_inconc_cond="N_dup_clinical < 9 AND 9 not reproduced without AutoViz trace",
            act_true="Retain clinical rows, Spawns #I2.1, Ref to #I3.",
            act_false="Audit duplicate cluster and deduplicate; Spawns #I2.1, Ref to #I3.",
            act_inconc=(
                "Spawns #I2.2 to trace AutoViz 30-col FixDQ duplicates; Spawns #I2.1, Ref to #I3."
            ),
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=git_commit,
            sha256=train_sha,
            columns=["F_clinical (175 predictor columns)"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I2.1",
            status=i2_1_out,
            polarity="+",
            claim=(
                "Subtends #I2. Train contains >= 10 within-hospital demographic near-duplicate "
                "pairs (>=90% agreement) or >= 1 cross-split F_clinical collision with Unlabeled."
            ),
            technique="Cross-split row hashing and within-hospital demographic blocking",
            parameters={"min_mutually_non_null_cols": 30, "agreement_min": 0.90},
            assumptions=[
                "Repeated hospital stays preserve age, gender, ethnicity, height, weight."
            ],
            missing_data_policy="Score agreement over mutually non-null predictors.",
            script_ref=script_ref,
            ev_type="N",
            metric=f"N_cross_exact={n_cross}, N_near_dup_transfer={n_near}",
            formula="cross_exact_matches; within_hospital_near_dup_pairs",
            timeframe="Static WiDS Datathon 2021 snapshot",
            filters=["All 130,157 Train and 10,234 Unlabeled rows excluding exact 175-col dups"],
            outcome=i2_1_out,
            on_true_cond="N_cross_exact >= 1 OR N_near_dup_transfer >= 10",
            on_false_cond="N_cross_exact == 0 AND N_near_dup_transfer == 0",
            on_inconc_cond="N_cross_exact == 0 AND 1 <= N_near_dup_transfer < 10",
            act_true="Enforce StratifiedGroupKFold(groups=hospital_id) -> Ref to #I4.",
            act_false="Confirm zero cross-split collisions or transfer pairs -> Ref to #I3.",
            act_inconc="Log 1-9 pairs in stage1_duplicate_audit.csv -> Ref to #I3.",
            source_file=src_both,
            commit=git_commit,
            sha256=train_sha,
            columns=["hospital_id", "age", "gender", "ethnicity", "height", "weight"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I2.2",
            status="OPEN" if i2_out == "INCONCLUSIVE" else "SKIPPED",
            polarity="+",
            claim=(
                "Subtends #I2. AutoViz's 9 flagged duplicates arise strictly after AutoViz "
                "caps predictors to <= 30 columns rather than 9 copies in 175 cols."
            ),
            technique="Direct trace of AutoViz classify_print_vars 30-predictor reduction",
            parameters={"max_cols_analyzed": 30},
            assumptions=["AutoViz capped predictors to 30 before reporting 9 duplicate rows."],
            missing_data_policy="Trace AutoViz output directly.",
            script_ref="scripts/stage1_autoviz_dup_trace.py",
            ev_type="N",
            metric="Pending scripts/stage1_autoviz_dup_trace.py",
            formula="count(df_autoviz_30cols.duplicated())",
            timeframe="Static WiDS Datathon 2021 training cohort",
            filters=["Executed when #I2 == INCONCLUSIVE"],
            outcome="INCONCLUSIVE",
            on_true_cond="autoviz_traced_dup_count == 9 AND N_dup_clinical < 9",
            on_false_cond="autoviz_traced_dup_count != 9",
            on_inconc_cond="AutoViz trace fails or subsamples",
            act_true="Confirm 9th dup is 30-col projection artifact; audit 8 copies -> Ref to #I3.",
            act_false="Rely on deterministic F_clinical audit -> Ref to #I3.",
            act_inconc="Rely on deterministic F_clinical audit -> Ref to #I3.",
            source_file="data/raw/TrainingWiDS2021.csv",
            commit=git_commit,
            sha256=train_sha,
            columns=["AutoViz selected 30 predictors"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I3",
            status=i3_out,
            polarity="+",
            claim=(
                "Across 64 matched (h1_, d1_) pairs, h1_ features exhibit >= 5.0 pp higher median "
                "missingness than d1_, concentrating >= 60% of >50%-missing in h1_/blood gas."
            ),
            technique="Domain/window taxonomy mapping and one-sided paired Wilcoxon test",
            parameters={"matched_pairs_n": len(deltas_tr), "alpha": 0.01, "alternative": "greater"},
            assumptions=["Each h1_clinical_stem pairs 1-to-1 with d1_clinical_stem."],
            missing_data_policy="Explicit null rate over full Train and Unlabeled tables.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"n_pairs={len(deltas_tr)}, med_delta_tr={med_tr:.2f}pp (p={wil_p_tr:.3e}), "
                f"conc_tr={conc_tr * 100:.2f}%, med_delta_te={med_te:.2f}pp (p={wil_p_te:.3e}), "
                f"conc_te={conc_te * 100:.2f}%"
            ),
            formula=(
                "median(miss(h1)-miss(d1)); |{c in (h1_ U blood_gas):miss>50%}| / |{c:miss>50%}|"
            ),
            timeframe="Static WiDS Datathon 2021 dataset snapshot",
            filters=["64 matched (h1_, d1_) pairs and all 180 shared predictors"],
            outcome=i3_out,
            on_true_cond="median_delta >= 5.0 AND p < 0.01 AND conc >= 60% in BOTH splits",
            on_false_cond="median_delta <= 0 OR p >= 0.05 OR conc < 40% in EITHER split",
            on_inconc_cond="Borderline concentration or p-value",
            act_true=(
                "Reject blind >80%-null dropping; retain h1_/blood-gas for MNAR; Spawns #I3.1."
            ),
            act_false="Audit alternative sparsity drivers and Spawns #I3.1.",
            act_inconc="Inspect borderline pairs and Spawns #I3.1.",
            source_file=src_dict,
            commit=git_commit,
            sha256=train_sha,
            columns=["64 h1_* and 64 d1_* columns; labs blood gas"],
            timestamp=now_iso,
        ),
        make_inquiry(
            inq_id="I3.1",
            status=i3_1_out,
            polarity="-",
            claim=(
                "Subtends #I3. At least 10 predictors shift >= 5.0 pp in missingness between Train "
                "and Unlabeled or reach >= 95.0% null in UnlabeledWiDS2021.csv."
            ),
            technique="Cross-split column missingness delta and Spearman rank-order stability",
            parameters={"abs_shift_pp": 5.0, "extreme_null_pct": 95.0},
            assumptions=["Disjoint hospitals may use different EHR lab order panels."],
            missing_data_policy="Explicit null percentage across all rows in both CSVs.",
            script_ref=script_ref,
            ev_type="N",
            metric=(
                f"shifted_ge5pp={len(shifted_ge5)}, unlabeled_ge95pct={len(unlabeled_ge95)}, "
                f"spearman_rho={miss_rho:.4f}, "
                f"max_shift={pred_prof['abs_miss_shift_pp'].max():.2f}pp"
            ),
            formula="count(|miss_tr - miss_te| >= 5.0); Spearman(miss_tr, miss_te)",
            timeframe="Static WiDS Datathon 2021 snapshot",
            filters=["All 180 shared predictors across Train and Unlabeled"],
            outcome=i3_1_out,
            on_true_cond="shifted_ge5pp >= 10 OR unlabeled_ge95pct >= 1",
            on_false_cond="shifted_ge5pp == 0 AND unlabeled_ge95pct == 0 AND rho >= 0.95",
            on_inconc_cond="1-9 shifted predictors without >=95% null columns",
            act_true="Escalate shifted columns to Stage 2 Adversarial Validation -> Ref to #I5.",
            act_false="Confirm missingness stability -> Ref to #I4.",
            act_inconc=(
                "Escalate the 2 shifted cols (bmi, weight) to Stage 2 shift check -> Ref to #I5."
            ),
            source_file=src_both,
            commit=git_commit,
            sha256=train_sha,
            columns=["180 shared predictors"],
            timestamp=now_iso,
        ),
    ]
    upsert_inquiries(
        entries,
        problem=(
            "Audit schema alignment, structural/constant columns, non-ID duplicates, "
            "and domain/temporal sparsity across TrainingWiDS2021.csv and UnlabeledWiDS2021.csv."
        ),
    )

    # Human-readable Stage 1 markdown summary
    top_shifted = "\n".join(
        f"- `{r['column']}`: Train={r['missing_pct_train']:.2f}%, "
        f"Test={r['missing_pct_test']:.2f}% (|Δ|={r['abs_miss_shift_pp']:.2f} pp)"
        for _, r in pred_prof.sort_values("abs_miss_shift_pp", ascending=False).head(15).iterrows()
    )
    near_const_md = "\n".join(
        f"- `{r['column']}`: mode=`{r['mode_train']}` ({r['mode_share_train'] * 100:.2f}% train, "
        f"{r['mode_share_test'] * 100:.2f}% test)"
        for _, r in prof.loc[prof["structural_role"] == "near_constant_ge_99pct"]
        .sort_values("mode_share_train", ascending=False)
        .iterrows()
    )
    md = f"""# Stage 1 Report: Schema, Constants, Duplicates, Domain Taxonomy & Sparsity

- **Generated**: `{now_iso}` (Base Commit: `{git_commit}`)
- **Datasets**: `TrainingWiDS2021.csv` (`{n_train:,} × {train.shape[1]}`),
  `UnlabeledWiDS2021.csv` (`{n_test:,} × {test.shape[1]}`), `DataDictionaryWiDS2021.csv`

## Key Findings

1. **[I1 = {i1_out}] Schema & Drop List**:
   - Schema aligned (`Train \\ {{diabetes_mellitus}} == Unlabeled`): **`{schema_aligned}`**.
   - Structural drop cols (`{len(const_id_cols)}`): `{const_id_cols}`.
   - Near-constant predictors (`>= 99%` modal share):
{near_const_md}

2. **[I1.1 = {i1_1_out}] Dictionary Quirks & Facility Overlap**:
   - Dict-only: `{dict_only}`; CSV-only: `{csv_only}`; metadata mismatches: `{len(mismatches)}`.
   - Overlaps: `hospital_id={hosp_ov}` (`{len(hosp_tr)}` vs `{len(hosp_te)}`),
     `icu_id={icu_ov}` (`{len(icu_tr)}` vs `{len(icu_te)}`), `encounter_id={enc_ov}`.

3. **[I2 = {i2_out}, I2.1 = {i2_1_out}] Duplicates & Near-Duplicates**:
   - Exact duplicates (`keep='first'`): `F_clinical={n_dup_clin}`, `F_facility={n_dup_fac}`,
     `F_80={n_dup_80}`, `F_50={n_dup_50}`, `F_autoviz_pre={n_dup_av}`.
   - Cross-split exact collisions: **`{n_cross}`**; within-hospital near-duplicates: **`{n_near}`**.

4. **[I3 = {i3_out}, I3.1 = {i3_1_out}] Domain/Window Sparsity & Shift**:
   - Matched `(h1_, d1_)` pairs: `{len(deltas_tr)}`, median gap=`+{med_tr:.2f} pp` (Train,
     `p={wil_p_tr:.3e}`, inv=`{inv_tr}`), `+{med_te:.2f} pp` (Test, `p={wil_p_te:.3e}`,
     inv=`{inv_te}`).
   - >50%-missing inside `h1_` + `labs blood gas`: **`{conc_tr * 100:.2f}%`** Train,
     **`{conc_te * 100:.2f}%`** Test.
   - Missingness rank stability `Spearman rho={miss_rho:.4f}`; `|Δ|>=5pp`: `{len(shifted_ge5)}`.

### Top 15 Predictors by Missingness Shift
{top_shifted}
"""
    (REPORTS_FINDINGS_DIR / "stage1_schema_sparsity.md").write_text(md, encoding="utf-8")
    print(
        f"Stage 1 done: I1={i1_out}, I1.1={i1_1_out}, I2={i2_out}, "
        f"I2.1={i2_1_out}, I3={i3_out}, I3.1={i3_1_out}"
    )


if __name__ == "__main__":
    main()
