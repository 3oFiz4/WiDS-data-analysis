- [x] Pull dataset
- [x] Yield surface-level EDA analysis
- [x] Export common charts

### Pre-assumptive Action

I only assume these methods below from a single glimpse, and these methods originates from my inference from small data.

- [x] Sort features by most empty, most constants (low distinct value)

> **#I1 (TRUE) & #I3 (TRUE)**: Sorted all 181 columns in [feature_sparsity_constants.csv](../reports/findings/feature_sparsity_constants.csv) by `missing_pct_train`, `mode_share_train`, and `nunique_train`. Top missingness reaches `92.09%` (`h1_bilirubin_min/max`), `73` columns exceed `50%` missingness, `readmission_status` has `nunique=1` (`0`), and 3 near-constants exceed `99%` modal share (`aids=99.90%`, `lymphoma=99.58%`, `leukemia=99.27%`).

- [x] Yield a non-linear correlation analysis over all variables

> **#I7 (TRUE) & #I7.1 (TRUE)**: Completed across all `175` `F_clinical` features. Pre-assumptive `P1` (*all variables are non-linear/non-normal*) is refuted (`82/153` = `53.59%` violate normality while `71/153` = `46.41%` are near-Gaussian), yet non-linear target signal is pervasive (`Top-20 Jaccard(|r|, MI) = 0.7391`, `15` numeric features show non-monotonic/rank lift, e.g. `d1_glucose_min` `|r|=0.1359` vs `HGB AUC=0.6442` and `apache_3j_diagnosis` `|r|=0.0095` vs `HGB AUC=0.6304`).

  - [no] Pre-assumptive P1: All continuous variables follow non-linear / non-normal distributions

  > **#I7 (TRUE)**: Refuted (`- [no]`)—`71/153` (`46.41%`) continuous numeric predictors (`age`, `height`, `ph_apache`, `d1_hemaglobin_*`, `d1_hematocrit_*`, `d1_sysbp_*`, `d1_albumin_*`) satisfy `|skew| <= 1.0`, `|kurt| <= 2.0`, and `Shapiro-Wilk W >= 0.95`.

  - [x] **[I7] Sub-Analysis: Distributional Normality Audit (Testing P1 Across 153 Continuous Predictors)**

  > **#I7 (TRUE)**: `82/153` (`53.59%`) fail normality thresholds while `71/153` (`46.41%`) are near-Gaussian ([feature_triage_table.csv](../reports/findings/feature_triage_table.csv)).

  - [x] **[I7.1] Sub-Analysis: Linear vs. Non-Linear & Non-Monotonic Target-Signal Ranking (All 175 Features)**

  > **#I7.1 (TRUE)**: Compared `|Pearson r|`, `|Spearman rho|`, `MI_non_null`, `DecisionTree(depth=4)`, and 5-fold grouped `HGB AUC` ([stage3_linear_vs_nonlinear_ranking.png](../reports/figures/stage3/stage3_linear_vs_nonlinear_ranking.png)).

- [x] Yield @..\reports\figures\diabetes_mellitus\Pair_Scatter_Plots.png and assay directly the top-most correlated column

> **#I8 (TRUE) & #I8.1 (INCONCLUSIVE)**: Replaced the unreadable AutoViz strip with quantitative Spearman clustering ([stage3_top_collinearity_heatmap.png](../reports/figures/stage3/stage3_top_collinearity_heatmap.png)) and decile risk plots ([stage3_top_analytes_distributions.png](../reports/figures/stage3/stage3_top_analytes_distributions.png)). Top-most feature is `c_top1 = d1_glucose_max` (`HGB AUC=0.7839`, `r=0.4007`, `rho=0.4084`, twin `glucose_apache`).

  - [x] **[I8] Sub-Analysis: Top-25 Collinear Proxy Clustering (`|Spearman rho| >= 0.80`) & `c_top1` Clinical Assay**

  > **#I8 (TRUE)**: `|Spearman rho| >= 0.80` collapses the Top-25 predictors into `13` connected clinical clusters ([stage3_top25_collinearity_clusters.csv](../reports/findings/stage3_top25_collinearity_clusters.csv)).

  - [x] **[I8.1] Sub-Analysis: Multivariate Signal Saturation Test (Top-25 vs. All-175 `StratifiedGroupKFold` AUC)**

  > **#I8.1 (INCONCLUSIVE)**: `Top-25 = 0.84083` (`Top-25-dedup = 0.83767`) -> `Top-50 = 0.84575` -> `All-175 = 0.84963` (`Δ = +0.00880`, improving **`5/5` folds**, [stage3_subset_auc_scaling.png](../reports/figures/stage3/stage3_subset_auc_scaling.png)).

  - [no] Discard lower-ranked features (ranks 26..175) as useless

  > **#I8.1 (INCONCLUSIVE)**: Rejected (`- [no]`)—retaining features 26..175 boosts out-of-hospital CV ROC-AUC in `5/5` folds (`+0.00880` overall).

- [no] Blindly drop >80%-missing columns (`h1_*` / `labs blood gas`) prior to MNAR testing

> **#I3 (TRUE)**: Rejected (`- [no]`) blind deletion of the `32` >80%-missing columns because `71.23%` (`52/73`) of >50%-missing predictors concentrate in first-hour (`h1_`) and `labs blood gas` clinical orders (`+17.77 pp` median `h1_ - d1_` gap, Wilcoxon `p=3.772e-12`), which represent informative clinical-suspicion missingness to test in Stage 4.

- [x] Rank each columns by Linearity and Non-Linearity Signal

> **#I7.1 (TRUE)**: All `175` `F_clinical` features ranked by `auc_hgb_full_grouped`, `mi_non_null`, `|Pearson r|`, `|Spearman rho|`, `nonlinear_tree_lift`, and `rank_corr_lift` in [feature_triage_table.csv](../reports/findings/feature_triage_table.csv).

- [x] Yield a Feature Triage Summary Table

> **#I9 (TRUE)**: Exported the full 175-feature triage table to [feature_triage_table.csv](../reports/findings/feature_triage_table.csv) and Top-25 summary in [stage3_feature_triage.md](../reports/findings/stage3_feature_triage.md).

- [x] In <Feature Triage Summary Table>, analysis each variable ((as much as needed) top-most ranked) and audit them if they justify any of these questions below: Is this post-diagnosis leakage?; Does impossible values exist?

> **#I9 (TRUE) & #I9.1 (INCONCLUSIVE)**: Zero Top-40 features exceed `AUC >= 0.90` (`k_hard_leak_top40 = 0`; max `0.7839`), while `10` columns fail physiological/unit bounds ([stage3_impossible_values_detail.csv](../reports/findings/stage3_impossible_values_detail.csv))—most strikingly all `5` `hematocrit` columns are `100%` in percentage scale (`15.5–51.7`) rather than `[0,1]` fraction, `d1_resprate_min == 0` (`4.53%`), `pre_icu_los_days < 0` (`0.36%`), and `age == 0` (`30` train rows).

  - [x] **[I9] Sub-Analysis: 175-Feature Single-Feature Leakage & Hard Impossible-Value Screen**

  > **#I9 (TRUE)**: Flagged `10` impossible-value columns across Train/Test (`5` `hematocrit` unit mismatch, `d1/h1_resprate_min == 0`, `d1_heartrate_min == 0`, `pre_icu_los_days < 0`, `age == 0`).

  - [x] **[I9.1] Sub-Analysis: APACHE Diagnosis Code (`>=85%`) & Glucose-Proxy Phenotype Ablation**

  > **#I9.1 (INCONCLUSIVE)**: `0` APACHE diagnosis codes (`n >= 30`) reach `>= 85%` prevalence (`k_diag_ge85 = 0`); dropping all 5 glucose + 2 diagnosis features lowers grouped HGB AUC from `0.84963` to `0.74388` (`Δ = -0.10575`), confirming strong non-glucose systemic signal (`0.74388`).

- [x] **Schema, Identifier & Constant Column Audit** (`Unnamed: 0`, `encounter_id`, `hospital_id`, `icu_id`, `readmission_status`)

> **#I1 (TRUE)**: `TrainingWiDS2021.csv` (`130,157 × 181`) and `UnlabeledWiDS2021.csv` (`10,234 × 180`) share an identical 180-predictor schema. Locked `['Unnamed: 0', 'encounter_id', 'readmission_status']` onto the modeling drop list (`Unnamed: 0` is sequential `1..N`, `encounter_id` is 100% unique, `readmission_status` is constant `0`), while isolating `hospital_id` and `icu_id` as facility keys.

- [x] **[I1.1] Sub-Analysis: Data Dictionary vs. CSV Reconciliation & Facility-ID Overlap Audit**

> **#I1.1 (TRUE)**: Found 13 metadata discrepancies in `DataDictionaryWiDS2021.csv` (absent `icu_admit_type`, undocumented `Unnamed: 0`, `bmi`/`apache_2_diagnosis`/`apache_3j_diagnosis` mistyped `string`, `albumin_max` `"lowest"` typo, `inr` unit `micromol/L`, `pao2fio2ratio` copied from `fio2`) documented in [data_dictionary.md](../docs/data_dictionary.md). `hospital_id` has `0` overlap (`204` train vs `190` test), while `icu_id` unexpectedly overlaps on `264` IDs (`328` train vs `302` test).

- [x] **Train vs. Unlabeled Test Distribution Shift & Group Overlap Check**

> **#I4 (INCONCLUSIVE -> resolved via #I4.1=TRUE), #I5 (INCONCLUSIVE), #I5.1 (FALSE)**: Confirmed `0` hospital overlap (`204` train vs `190` test) and `adv_auc_raw_grouped = 0.5913` under `StratifiedGroupKFold(groups=hospital_id)` (vs. inflated `0.7536` under random `StratifiedKFold`). Zero features exceed `PSI >= 0.10`; locked 5-fold `StratifiedGroupKFold(groups=hospital_id)` as mandatory CV splitter.

  - [x] **[I4] Sub-Analysis: Facility-Key Collision (`icu_id` across `hospital_id`) & Grouping Hierarchy Audit**

  > **#I4 (INCONCLUSIVE)**: `within_train_multi_hosp_icu_share = 0.00%` (`NMI = 0.9293`, 1-to-1 within Train), yet `72.13%` of combined `icu_id` values (`max=2` hospitals across `366` IDs) collide across Train and Unlabeled hospitals ([stage2_icu_hospital_mapping.csv](../reports/findings/stage2_icu_hospital_mapping.csv)).

  - [x] **[I4.1] Sub-Analysis: `StratifiedGroupKFold(groups=hospital_id)` Fold Balance & `icu_id` Proxy-Leakage Test**

  > **#I4.1 (TRUE)**: 5-fold `StratifiedGroupKFold` satisfies strict tolerances (`fold_share_range=0.104 pp`, `fold_target_range=0.150 pp`, `hospital_overlap=0`, all 9 duplicate rows of `hospital_id=167` in `1` fold). `icu_id` target-encoding AUC collapses from `0.6027` (random `StratifiedKFold`) to `0.4993` (grouped `StratifiedGroupKFold`).

  - [no] Retain `icu_id` as a model feature or use random `StratifiedKFold` CV

  > **#I4.1 (TRUE) & #I5.1 (FALSE)**: Rejected (`- [no]`) because `icu_id` leaks `hospital_id` inside Train (`0.6027` -> `0.4993`) and actively degrades out-of-hospital adversarial generalization (`adv_auc_with_icu_grouped = 0.3831`, `Δ_icu = -0.2082`).

  - [x] **[I5] Sub-Analysis: Out-of-Hospital Adversarial Validation (`StratifiedKFold` vs. `StratifiedGroupKFold`)**

  > **#I5 (INCONCLUSIVE)**: Moderate out-of-hospital shift (`0.5913`, vs `0.7536` random SKF); top driver is `age` (`perm_imp=0.0418`, `KS=0.0376`, `PSI=0.0136`), followed by `h1_heartrate_min` (`0.0062`) and `d1_potassium_max` (`0.0060`) ([stage2_adversarial_validation.csv](../reports/findings/stage2_adversarial_validation.csv)).

  - [no] **[I5.1] Sub-Analysis: Missingness-Shift (`age`/`bmi`/`weight`) vs. Value-Shift & `icu_id` Decomposition**

  > **#I5.1 (FALSE)**: Rejected (`- [no]`) hypothesis that missingness patterns alone drive test shift: `adv_auc_miss_only = 0.4677` and fold-imputing all values yields `0.5931` (`Δ_miss = -0.0017`), though imputing `age/bmi/weight` slightly lowers shift to `0.5787`.

- [x] **Duplicate & Near-Duplicate Row Investigation** (excluding IDs)

> **#I2 (INCONCLUSIVE -> resolved via #I2.2)**: Across all 175 non-facility clinical predictors (`F_clinical`) and 177 facility+clinical predictors, there are `8` exact redundant copies (`9` identical rows total in `hospital_id=167, icu_id=903, diabetes_mellitus=0`, [stage1_duplicate_audit.csv](../reports/findings/stage1_duplicate_audit.csv)).

- [no] **[I2.1] Sub-Analysis: Cross-Split Collision & Within-Hospital Near-Duplicate Transfer Hypothesis**

> **#I2.1 (FALSE)**: Hypothesis rejected (`- [no]`)—found `0` exact 175-feature matches between Train and Unlabeled and `0` within-hospital demographic near-duplicate pairs (`>= 90%` agreement across `>= 30` mutually non-null predictors).

- [x] **[I2.2] Sub-Analysis: AutoViz 9-Duplicate Projection & FixDQ Source Trace**

> **#I2.2 (TRUE)**: Verified via `classify_print_vars` that AutoViz projected the dataset to `(130157, 31)`, creating 1 extra projection collision on top of the 8 redundant copies (`duplicated(keep='first') = 9`, `keep=False = 10`), whereas full 175-feature space has `8` redundant copies (`9` total rows).

- [x] **Target Variable (`diabetes_mellitus`) Class Balance & Subgroup Prevalence Analysis**

> **#I6 (TRUE)**: Confirmed strong inter-hospital (`188` hospitals with `n >= 30`, `chi2 p=0.0`, `IQR = 8.82 pp`, `P10=8.96%`, `P90=29.17%`) and subgroup heterogeneity across `3/4` core strata (`age_bin` `13.59 pp`, `ethnicity` `18.14 pp`, `icu_type` `11.86 pp`, vs `elective_surgery` `0.52 pp`, `p=0.078`), requiring probability calibration (Log-Loss/Brier) alongside ROC-AUC ([stage2_target_prevalence.csv](../reports/findings/stage2_target_prevalence.csv)).

  - [x] **[I6] Sub-Analysis: Inter-Hospital Target Variance & Clinical/Demographic Subgroup Prevalence Audit**

  > **#I6 (TRUE)**: `diabetes_mellitus=1` is `21.63%` overall, spanning `11.22%` (`bmi < 18.5`) to `36.31%` (`bmi 40+`, `25.09 pp` range) and surging from `20.87%` (`arf_apache=0`) to `47.94%` (`arf_apache=1`).

  - [x] **[I6.1] Sub-Analysis: Missing-Stratum Target Bias (`age`/`bmi`/`weight`/`ethnicity`) & Projected Unlabeled Prior Shift**

  > **#I6.1 (INCONCLUSIVE)**: Detected strong target-rate separation (`k_miss_bias = 2`) in `age` missingness (`13.23%` vs `21.96%`, `|Δ|=8.73 pp`, `p=9.94e-49`) and `ethnicity` (`13.99%` vs `21.72%`, `|Δ|=7.73 pp`, `p=1.29e-13`), though standardized Train->Unlabeled base-rate shift remains small (`max = +0.197 pp` via `age_bin`).

- [x] **Data Dictionary Grouping & Clinical Domain Mapping** (`DataDictionaryWiDS2021.csv`)

> **#I3 (TRUE)**: Mapped all 181 columns into 8 clinical categories × 4 temporal windows in [stage1_domain_window_sparsity.csv](../reports/findings/stage1_domain_window_sparsity.csv) and [data_dictionary.md](../docs/data_dictionary.md); `h1_` has `+17.77 pp` higher missingness than `d1_` in train (`+18.82 pp` in test, Wilcoxon `p=3.772e-12`).

- [x] **[I3.1] Sub-Analysis: Cross-Split Missingness Stability & Extreme Sparsity Shift Audit**

> **#I3.1 (INCONCLUSIVE)**: Missingness ranks are highly stable (`Spearman rho = 0.9955`, `0` test columns `>= 95%` missing), but `2` key anthropometric predictors shifted `>= 5.0 pp`: `bmi` (`3.45%` train vs `9.92%` test, `|Δ|=6.47 pp`) and `weight` (`2.66%` vs `8.87%`, `|Δ|=6.21 pp`), escalated to Stage 2 Adversarial Validation (`#I5`).

- [ ] **Missingness Pattern & Informative Missingness (MNAR/MAR) Audit**

> `73` columns are >50% missing and `32` are >80% missing (e.g., `h1_bilirubin_min/max` ~92.1%, `h1_albumin_min/max` ~91.4%, `h1_lactate_min/max` ~91.0%), with only `21` columns 0% missing. In ICU data, missingness is rarely random—ordering a blood gas or `h1_` lactate reflects clinical suspicion. Testing correlation between missingness indicators (`isna()`) and `diabetes_mellitus` tells us whether to engineer missingness flags/counts before imputing.

- [ ] **Measurement-Method & Paired-Column Consistency Check** (`invasive` vs. `noninvasive`, `min` vs. `max`, `height`/`weight` vs. `bmi`)

> Many vitals appear in both invasive and noninvasive pairs (e.g., `d1_sysbp_invasive_max` vs. `d1_sysbp_noninvasive_max`) and `min`/`max` pairs. Auditing `min > max` violations, `bmi` vs. `weight / (height/100)^2` mismatches, and whether combining invasive/noninvasive into a unified reading preserves signal reduces collinearity and catches sensor/entry errors.

- [ ] **Physiological Range & Impossible Value Audit (Outliers & Unit Errors)**

> ICU EHR exports often contain sentinel values, negative/zero vitals, extreme lab spikes, or unit mix-ups (e.g., `height` in cm vs. m, `temp` in °F vs. °C, impossible `gcs_*` scores, negative `pre_icu_los_days`). Auditing min/max quantiles against clinical bounds from `DataDictionaryWiDS2021.csv` prevents extreme outliers from distorting linear/neural models or summary statistics.

- [ ] **Categorical Cardinality, Rare Levels & Encoding Strategy Audit** (`ethnicity`, `hospital_admit_source`, `icu_admit_source`, `icu_stay_type`, `icu_type`, `gender`, `apache_2_diagnosis`, `apache_3j_diagnosis`, `hospital_id`, `icu_id`)

> String categoricals, numeric-coded diagnosis IDs (`apache_2_diagnosis`, `apache_3j_diagnosis`), and high-cardinality facility IDs (`hospital_id`, `icu_id`) require different handling. Checking unseen categories in `UnlabeledWiDS2021.csv`, rare-category frequencies, and target-encoding leakage risk ensures clean encoding without test-time failures.

- [ ] **Collinearity, Redundancy & Cross-Window Signal Analysis** (`h1_` vs. `d1_` vs. `_apache`)

> AutoViz's SULO step already reduced `150` numeric predictors down to `73` due to high correlation. Quantifying Pearson/Spearman/Phik/mutual-information redundancy across `h1_`, `d1_`, and `_apache` versions of the same analyte (especially `glucose_apache`, `d1_glucose_max/min`, `h1_glucose_max/min`) identifies which window carries unique predictive signal vs. noise.

- [ ] **Target Leakage & Clinical Proxy Audit**

> Glucose-related labs (`d1_glucose_max`, `glucose_apache`, `h1_glucose_max`), `arf_apache`, `creatinine_apache`, `bmi`, and `age` are clinically tied to diabetes, whereas some APACHE/comorbidity or admission fields could act as post-hoc or coding proxies. Auditing top mutual-information features against clinical timeline (first-24h ICU data) ensures all features are legitimately available at prediction time.

- [ ] **Feature Engineering & Domain Ratio Exploration**

> Beyond raw `min`/`max` columns, clinical ratios and deltas—such as `d1_glucose_max - d1_glucose_min` (glycemic variability), `h1_` vs. `d1_` deltas, `bun / creatinine` ratio, `invasive` monitoring indicator, missingness count per patient, and hospital-level/ICU-level aggregations (computed strictly out-of-fold)—can capture non-linear physiological stress patterns missed by single-column splits.

- [ ] **Baseline Model & Validation Pipeline Verification**

> Before heavy feature selection or tuning, run a fast baseline (e.g., LightGBM/XGBoost with `StratifiedGroupKFold(groups=hospital_id)`) to establish a leak-free CV ROC-AUC benchmark, inspect permutation/SHAP feature importances, and verify that every data cleaning and engineering step actually improves out-of-hospital generalization.
