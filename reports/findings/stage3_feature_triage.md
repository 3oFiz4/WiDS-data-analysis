# Stage 3 Report: Non-Linear Distribution Audit, Signal Ranking & Feature Triage

- **Generated**: `2026-11-19T06:53:14+08:00` (Base Commit: `4c0ca25`)

## Inquiry Outcomes

- **[I7 = TRUE] Test of Pre-Assumptive P1**: `82/153` continuous features
  (`53.59%`) violate normality (`|skew|>1`, `|kurt|>2`, or `W<0.95`).
  Near-Gaussian (`71`): `['age', 'd1_hemaglobin_max', 'd1_hemaglobin_min', 'd1_hematocrit_max', 'd1_sysbp_noninvasive_max', 'd1_sysbp_max', 'd1_hematocrit_min', 'd1_potassium_min', 'hematocrit_apache', 'd1_hco3_min', 'd1_calcium_max', 'h1_diasbp_min', 'h1_diasbp_noninvasive_min', 'heart_rate_apache', 'h1_diasbp_max', 'd1_diasbp_min', 'd1_sodium_max', 'd1_calcium_min', 'h1_diasbp_noninvasive_max', 'd1_diasbp_noninvasive_min', 'd1_hco3_max', 'h1_sysbp_noninvasive_max', 'h1_potassium_max', 'h1_potassium_min', 'd1_diasbp_invasive_min', 'h1_sysbp_max', 'd1_diasbp_max', 'h1_sysbp_noninvasive_min', 'd1_diasbp_noninvasive_max', 'd1_heartrate_min', 'd1_albumin_min', 'h1_calcium_min', 'h1_calcium_max', 'd1_albumin_max', 'h1_diasbp_invasive_min', 'albumin_apache', 'h1_sysbp_min', 'h1_diasbp_invasive_max', 'h1_hemaglobin_min', 'h1_hemaglobin_max', 'd1_sysbp_min', 'd1_sysbp_noninvasive_min', 'h1_hematocrit_min', 'd1_platelets_min', 'h1_hematocrit_max', 'd1_heartrate_max', 'h1_mbp_min', 'h1_mbp_invasive_min', 'h1_hco3_min', 'h1_hco3_max', 'h1_mbp_max', 'd1_pao2fio2ratio_max', 'h1_mbp_noninvasive_min', 'd1_pao2fio2ratio_min', 'h1_mbp_noninvasive_max', 'd1_mbp_max', 'h1_heartrate_min', 'd1_mbp_noninvasive_max', 'd1_arterial_ph_max', 'd1_mbp_min', 'h1_pao2fio2ratio_min', 'height', 'h1_pao2fio2ratio_max', 'ph_apache', 'd1_mbp_noninvasive_min', 'h1_albumin_min', 'h1_heartrate_max', 'h1_albumin_max', 'h1_sysbp_invasive_max', 'h1_sysbp_invasive_min', 'd1_sysbp_invasive_min']`.
- **[I7.1 = TRUE] Linear vs Non-Linear Signal**: Top-20 Jaccard(`|r|`, `MI`) =
  **`0.7391`**; features with non-linear/rank lift = **`15`**.
- **[I8 = TRUE] Top-25 Collinearity Clusters (`|rho| >= 0.80`)**: Top-most feature
  `c_top1 = d1_glucose_max` (`auc=0.7839`), twins = `['glucose_apache']`,
  collapsing Top-25 into **`13`** connected clusters.
- **[I8.1 = INCONCLUSIVE] Multivariate Subset Saturation**: `Top-25 = 0.84083` ->
  `All-175 = 0.84963` (`Δ = +0.00880`, `5/5` folds improved).
- **[I9 = TRUE] Triage Leakage & Impossible Values**: `k_hard_leak_top40 = 0`,
  columns with impossible values = **`10`**.
- **[I9.1 = INCONCLUSIVE] Glucose & Diagnosis Proxy Ablation**: `k_diag>=85% = 0`
  (`0.00%` of positives, `[]`); `All-175 = 0.84963` ->
  `No-Glucose-Diag = 0.74388` (`Δ = -0.10575`).

## Subset Scaling & Ablation Table
| Subset | Features | OOF ROC-AUC | Mean +/- Std |
| :--- | ---: | ---: | ---: |
| `Top-1` | 1 | 0.78358 | 0.78428 +/- 0.00341 |
| `Top-10` | 10 | 0.82745 | 0.82807 +/- 0.00246 |
| `Top-25` | 25 | 0.84083 | 0.84130 +/- 0.00260 |
| `Top-25-dedup` | 13 | 0.83767 | 0.83811 +/- 0.00287 |
| `Top-50` | 50 | 0.84575 | 0.84633 +/- 0.00238 |
| `All-175` | 175 | 0.84963 | 0.85002 +/- 0.00232 |
| `No-Glucose-Diag` | 168 | 0.74388 | 0.74386 +/- 0.00603 |

## Top 25 Predictors in `<Feature Triage Summary Table>`
| Rank | Column | Category | Null % | |r| | |rho| | MI | HGB AUC | Imp N | Audit |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | `d1_glucose_max` | `labs` | 6.3% | 0.40074 | 0.40843 | 0.0933 | 0.7839 | 0 | `CLINICAL_PROXY_AUDIT` |
| 2 | `glucose_apache` | `APACHE covariate` | 11.3% | 0.35436 | 0.32342 | 0.0806 | 0.7610 | 0 | `CLINICAL_PROXY_AUDIT` |
| 3 | `d1_glucose_min` | `labs` | 6.3% | 0.13585 | 0.09931 | 0.0217 | 0.6442 | 0 | `CLINICAL_PROXY_AUDIT` |
| 4 | `h1_glucose_max` | `labs` | 57.7% | 0.31685 | 0.32047 | 0.0606 | 0.6376 | 0 | `CLINICAL_PROXY_AUDIT` |
| 5 | `h1_glucose_min` | `labs` | 57.7% | 0.30452 | 0.3015 | 0.0590 | 0.6363 | 0 | `CLINICAL_PROXY_AUDIT` |
| 6 | `apache_3j_diagnosis` | `APACHE covariate` | 0.7% | 0.00954 | 0.00791 | 0.0296 | 0.6304 | 0 | `CLINICAL_PROXY_AUDIT` |
| 7 | `apache_2_diagnosis` | `APACHE covariate` | 1.3% | 0.0038 | 0.00569 | 0.0248 | 0.6241 | 0 | `CLINICAL_PROXY_AUDIT` |
| 8 | `bmi` | `demographic` | 3.4% | 0.16904 | 0.17475 | 0.0160 | 0.6186 | 0 | `CLINICAL_PROXY_AUDIT` |
| 9 | `d1_bun_max` | `labs` | 10.6% | 0.14699 | 0.17373 | 0.0143 | 0.6170 | 0 | `PASS` |
| 10 | `d1_creatinine_max` | `labs` | 10.2% | 0.12793 | 0.16748 | 0.0145 | 0.6120 | 0 | `PASS` |
| 11 | `bun_apache` | `APACHE covariate` | 19.5% | 0.14524 | 0.17221 | 0.0150 | 0.6072 | 0 | `PASS` |
| 12 | `weight` | `demographic` | 2.7% | 0.15552 | 0.15385 | 0.0127 | 0.6050 | 0 | `PASS` |
| 13 | `d1_bun_min` | `labs` | 10.6% | 0.1373 | 0.15592 | 0.0114 | 0.6047 | 0 | `PASS` |
| 14 | `d1_creatinine_min` | `labs` | 10.2% | 0.12583 | 0.15324 | 0.0129 | 0.6008 | 0 | `PASS` |
| 15 | `creatinine_apache` | `APACHE covariate` | 19.1% | 0.12489 | 0.15672 | 0.0150 | 0.5963 | 0 | `PASS` |
| 16 | `age` | `demographic` | 3.8% | 0.08102 | 0.0699 | 0.0071 | 0.5824 | 30 | `CLINICAL_PROXY_AUDIT` |
| 17 | `d1_potassium_max` | `labs` | 9.6% | 0.09674 | 0.0934 | 0.0035 | 0.5679 | 0 | `PASS` |
| 18 | `d1_sodium_min` | `labs` | 10.2% | 0.05055 | 0.06515 | 0.0031 | 0.5544 | 0 | `PASS` |
| 19 | `d1_hemaglobin_max` | `labs` | 12.5% | 0.0788 | 0.08136 | 0.0039 | 0.5538 | 0 | `PASS` |
| 20 | `d1_hemaglobin_min` | `labs` | 12.5% | 0.06233 | 0.06696 | 0.0040 | 0.5495 | 0 | `PASS` |
| 21 | `d1_hematocrit_max` | `labs` | 12.0% | 0.06727 | 0.07118 | 0.0043 | 0.5474 | 114569 | `PASS` |
| 22 | `d1_sysbp_noninvasive_max` | `vitals` | 1.2% | 0.07231 | 0.0705 | 0.0041 | 0.5468 | 0 | `PASS` |
| 23 | `sodium_apache` | `APACHE covariate` | 18.8% | 0.04221 | 0.05984 | 0.0031 | 0.5467 | 0 | `PASS` |
| 24 | `d1_sysbp_max` | `vitals` | 0.2% | 0.07204 | 0.07028 | 0.0044 | 0.5465 | 0 | `PASS` |
| 25 | `d1_hematocrit_min` | `labs` | 12.0% | 0.05313 | 0.05894 | 0.0021 | 0.5449 | 114569 | `PASS` |

## Top Impossible-Value Columns (`10` Total)
- `d1_hematocrit_max` (`val <= 0 OR val > 1.0`): Train=114569 (100.000%), Test=9005 (100.000%)
- `d1_hematocrit_min` (`val <= 0 OR val > 1.0`): Train=114569 (100.000%), Test=9005 (100.000%)
- `hematocrit_apache` (`val <= 0 OR val > 1.0`): Train=103399 (100.000%), Test=8120 (100.000%)
- `h1_hematocrit_max` (`val <= 0 OR val > 1.0`): Train=27201 (100.000%), Test=2195 (100.000%)
- `h1_hematocrit_min` (`val <= 0 OR val > 1.0`): Train=27201 (100.000%), Test=2195 (100.000%)
- `d1_resprate_min` (`val <= 0`): Train=5868 (4.532%), Test=516 (5.069%)
- `h1_resprate_min` (`val <= 0`): Train=905 (0.732%), Test=87 (0.915%)
- `d1_heartrate_min` (`val <= 0`): Train=901 (0.694%), Test=70 (0.686%)
- `pre_icu_los_days` (`val < 0`): Train=471 (0.362%), Test=40 (0.391%)
- `age` (`val <= 0 OR val > 110`): Train=30 (0.024%), Test=0 (0.000%)
