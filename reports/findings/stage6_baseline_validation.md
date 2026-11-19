# Stage 6 Report: Baseline Verification, Calibration & HMECVAS Registry Audit

- **Generated**: `2026-11-19T07:44:30+08:00` (Base Commit: `3d36a20`)

## Inquiry Outcomes

- **[I17 = TRUE] CV Leakage Optimism**: `SKF_with_IDs=0.87144`,
  `SKF_no_IDs=0.86127`, canonical `SGKF_no_IDs=0.85520`
  (`Δ_naive=+0.01624`, `Δ_split=+0.00607`).
- **[I17.1 = TRUE] HGB vs LogReg Benchmark**: `HGB=0.85520`,
  `LogReg=0.82983` (`Δ=+0.02537`), `Blend(85/15)=0.85561`
  (Log-Loss `0.37107`).
- **[I18 = TRUE] Raw Model A vs Engineered Model B (`fe_ABCD`)**:
  - ROC-AUC: `0.85139 -> 0.85520` (**`+0.00381`**)
  - PR-AUC: `0.60380 -> 0.61053` (**`+0.00672`**)
  - Log-Loss: `0.37458 -> 0.37072` (**`-0.00386`**)
  - Brier: `0.11918 -> 0.11803` (**`-0.00115`**),
    `ECE_10=0.00650`
  - Dedup sensitivity (`-8` redundant rows): `AUC=0.85521`
    (`|Δ|=0.00001`).
- **[I18.1 = TRUE] Cross-Hospital Robustness & Unlabeled Alignment**:
  - Across `181` evaluable hospitals (`n>=30, pos>=5, neg>=5`): median AUC =
    **`0.86789`** (IQR `0.05473`), `95.03%` `>= 0.7500`.
  - Train OOF (`mean=0.21669`) vs Unlabeled 5-fold ensemble
    (`mean=0.20900`): `KS = 0.01638`.
  - Nested recalibration Log-Loss: raw HGB `0.37072`,
    Platt `0.37091`, Isotonic `0.37123`.
- **[I18.2 = TRUE] HMECVAS Registry Regex Linter**: `{'total_inquiries': 38, 'open_count': 0, 'placeholder_count': 0, 'invalid_outcome_count': 0, 'invalid_status_count': 0, 'invalid_sha256_count': 0, 'invalid_commit_count': 0, 'invalid_timestamp_count': 0, 'ref_matches': 152, 'edge_matches': 144}`.

## Model & CV Regime Comparison
| Model / CV Regime | Splitter | ROC-AUC | PR-AUC | Log-Loss | Brier | ECE_10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `SKF_with_IDs_ModelB` | `StratifiedKFold` | 0.87144 | 0.65322 | 0.35286 | 0.11166 | 0.00907 |
| `SKF_no_IDs_ModelB` | `StratifiedKFold` | 0.86127 | 0.62618 | 0.36431 | 0.11577 | 0.00779 |
| `SGKF_Raw_ModelA` | `StratifiedGroupKFold` | 0.85139 | 0.60380 | 0.37458 | 0.11918 | 0.00593 |
| `SGKF_LogReg_fe_ABCD` | `StratifiedGroupKFold` | 0.82983 | 0.55673 | 0.40472 | 0.12810 | 0.02895 |
| `SGKF_ModelB_fe_ABCD` | `StratifiedGroupKFold` | 0.85520 | 0.61053 | 0.37072 | 0.11803 | 0.00650 |
| `SGKF_ModelB_dedup_all` | `StratifiedGroupKFold` | 0.85521 | 0.61073 | 0.37071 | 0.11804 | 0.00658 |
| `SGKF_ModelB_dedup_uniq` | `StratifiedGroupKFold` | 0.85520 | 0.61073 | 0.37073 | 0.11805 | 0.00658 |
| `SGKF_Blend_85HGB_15LR` | `StratifiedGroupKFold` | 0.85561 | 0.61007 | 0.37107 | 0.11802 | 0.01057 |

## Top 15 Model B Permutation Feature Importances (5-Fold Mean)
- `d1_glucose_max`: `0.06809`
- `d1_glucose_range`: `0.02060`
- `age`: `0.01100`
- `apache_3j_diagnosis_te`: `0.00682`
- `glucose_apache`: `0.00627`
- `bmi`: `0.00326`
- `weight`: `0.00238`
- `bmi_calc`: `0.00201`
- `arf_apache`: `0.00197`
- `ethnicity`: `0.00179`
- `d1_creatinine_min`: `0.00158`
- `glucose_apache_vs_d1_max`: `0.00157`
- `d1_hemaglobin_max`: `0.00152`
- `d1_wbc_max`: `0.00152`
- `d1_calcium_min`: `0.00116`
