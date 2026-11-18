- [x] Pull dataset
- [x] Yield surface-level EDA analysis
- [x] Export common charts

### Pre-assumptive Action

I only assume these methods below from a single glimpse, and these methods originates from my inference from small data.

- [x] Sort features by most empty, most constants (low distinct value)

> **#I1 (TRUE) & #I3 (TRUE)**: Sorted all 181 columns in [feature_sparsity_constants.csv](../reports/findings/feature_sparsity_constants.csv) by `missing_pct_train`, `mode_share_train`, and `nunique_train`. Top missingness reaches `92.09%` (`h1_bilirubin_min/max`), `73` columns exceed `50%` missingness, `readmission_status` has `nunique=1` (`0`), and 3 near-constants exceed `99%` modal share (`aids=99.90%`, `lymphoma=99.58%`, `leukemia=99.27%`).

- [ ] Yield a non-linear correlation analysis over all variables

> According to @..\reports\figures\diabetes_mellitus\Dist_Plots_Numberics.png, almost all of the variables have non-linear distribution, we will assume for now that P1: All variables are non-linear distribution

- [ ] Yield @..\reports\figures\diabetes_mellitus\Pair_Scatter_Plots.png and assay directly the top-most correlated column

> This triage workflow might be useful. Most columns are useless, not only that but 100+ features to be EDA'ed sounds impossible, unless you are some kind of speciallized obtuse folk. We only seek to yield those whom seemingly useful, except those who proxies for the diagnosis itself (e.g., `bmi_level` vs `is_bmi`, both explain the same thing). We will start with these features, EDA these features, and then we will slowly scale to the lower R value features.

- [no] Blindly drop >80%-missing columns (`h1_*` / `labs blood gas`) prior to MNAR testing

> **#I3 (TRUE)**: Rejected (`- [no]`) blind deletion of the `32` >80%-missing columns because `71.23%` (`52/73`) of >50%-missing predictors concentrate in first-hour (`h1_`) and `labs blood gas` clinical orders (`+17.77 pp` median `h1_ - d1_` gap, Wilcoxon `p=3.772e-12`), which represent informative clinical-suspicion missingness to test in Stage 4.

- [ ] Rank each columns by Linearity and Non-Linearity Signal
- [ ] Yield a Feature Triage Summary Table
- [ ] In <Feature Triage Summary Table>, analysis each variable ((as much as needed) top-most ranked) and audit them if they justify any of these questions below: Is this post-diagnosis leakage?; Does impossible values exist?

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
