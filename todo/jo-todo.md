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

- [x] **Missingness Pattern & Informative Missingness (MNAR/MAR) Audit**

> **#I10 (TRUE) & #I10.1 (INCONCLUSIVE)**: Deduplicating `89` unique missingness masks yielded **`37`** significant MNAR/MAR masks (`BH q < 0.01`, `|Δ| >= 2.0 pp`), led by `glucose_apache` (`7.09%` missing vs `23.48%` present, `|Δ|=16.39 pp`) and `d1_glucose_max/min` (`6.47%` vs `22.65%`, `|Δ|=16.19 pp`), with `n_miss_total` vs DM `r = -0.0426` (`p=1.94e-53`). Native GBDT `NaN` splits capture nearly all MNAR signal (`native_raw=0.84963`, `native_counts=0.84967`, `imp_no_flags=0.84739`, `imp_flags=0.84881`, [stage4_missingness_mechanisms.csv](../reports/findings/stage4_missingness_mechanisms.csv)).

  - [x] **[I10] Sub-Analysis: Deduplicated Missingness-Mask MNAR/MAR Contingency & Burden Audit**

  > **#I10 (TRUE)**: `37/89` deduplicated masks show significant target separation (`BH q < 0.01`, `|Δ| >= 2.0 pp`).

  - [x] **[I10.1] Sub-Analysis: Missingness-Flag / Panel-Order Count Grouped-CV Ablation**

  > **#I10.1 (INCONCLUSIVE)**: Adding non-demographic panel missingness counts yields `+0.00004` over `native_raw` (`0.84967` vs `0.84963`) while explicit flags recover `+0.00142` under fold imputation (`0.84881` vs `0.84739`).

- [x] **Measurement-Method & Paired-Column Consistency Check** (`invasive` vs. `noninvasive`, `min` vs. `max`, `height`/`weight` vs. `bmi`)

> **#I11 (TRUE) & #I11.1 (FALSE)**: Found **`1,683`** `min > max` inversions (`d1_hco3` `171` train + `14` test, `d1_albumin` `117+12`, `d1_temp` `95+0`), **`80,081`** `h1_` outside `d1_` instances, **`250,511`** `_apache` outside `d1_` instances, and **`2,274`** `bmi` >2% formula discrepancies (`0` recoverable null BMIs; [stage4_paired_consistency.csv](../reports/findings/stage4_paired_consistency.csv)).

  - [x] **[I11] Sub-Analysis: 64 Min/Max Pairs, 64 `(h1_, d1_)` Bounds & BMI Formula Integrity Audit**

  > **#I11 (TRUE)**: Confirmed `1,683` `min > max` violations and `80,081` / `250,511` cross-window deviations proving `h1_` and `_apache` carry non-redundant information beyond `d1_`.

  - [no] **[I11.1] Sub-Analysis: Hypothesis That General BP Absorbs >=99% of Invasive/Non-Invasive BP Triplets**

  > **#I11.1 (FALSE)**: Rejected (`- [no]`)—up to `17.52%` of invasive/noninvasive readings are unabsorbed by general BP and invasive BP averages `+3.17 mmHg` higher, though pruning the 24 sub-columns + `has_invasive_bp` kept grouped AUC nearly unchanged (`0.84962` vs `0.84963`).

- [x] **Physiological Range & Impossible Value Audit (Outliers & Unit Errors)**

> **#I12 (TRUE) & #I12.1 (TRUE)**: Confirmed **`9`** DataDictionary unit mismatches (`5` `hematocrit` `%` vs `Fraction` and `4` `pao2fio2ratio` ratio vs `Fraction`) and **`6`** extreme-tail columns `>=0.1%` ([stage4_physiological_outliers.csv](../reports/findings/stage4_physiological_outliers.csv)); sanitizing sentinels/extremes to `NaN` improved grouped CV (`0.84963 -> 0.84974`, `Δ = +0.00011`).

  - [x] **[I12] Sub-Analysis: Full-Table Quantile Tail, Unit-Scale Mismatch & Sanitized-CV Audit**

  > **#I12 (TRUE)**: Verified `hematocrit` must stay in `%` (`15.5–51.7`) and sanitizing hard sentinel/bound errors preserves/improves CV AUC (`+0.00011`).

  - [x] **[I12.1] Sub-Analysis: Hospital-Sentinel Clustering Test on Zero-Coded Vitals & `age==0`**

  > **#I12.1 (TRUE)**: `4/4` zero-coded columns show hospital clustering: `age==0` (`n=30`) is `80.0%` concentrated in top-5 hospitals (`10.0%` vs `21.97%` DM), and `d1_resprate_min==0` reaches `35.3%` in a single `n>=30` hospital (`h1_resprate_min` `29.9%`, `d1_heartrate_min` `25.0%`).

- [x] **Categorical Cardinality, Rare Levels & Encoding Strategy Audit** (`ethnicity`, `gender`, `hospital_admit_source`, `icu_admit_source`, `icu_stay_type`, `icu_type`, `apache_2_diagnosis`, `apache_3j_diagnosis`)

> **#I13 (TRUE) & #I13.1 (INCONCLUSIVE)**: Verified `apache_2_bodysystem` and `apache_3j_bodysystem` do **not** exist in WiDS 2021 (WiDS 2020 columns). Across the `8` realized categorical/diagnosis columns, found **`126`** rare levels (`n < 30`: `124` in `apache_3j_diagnosis`, `2` in `hospital_admit_source`) and **`2`** unseen test levels (`0.0195%` rows in `apache_3j_diagnosis`), with `0` whitespace/case collisions ([stage4_categorical_audit.csv](../reports/findings/stage4_categorical_audit.csv)).

  - [x] **[I13] Sub-Analysis: 8-Column Categorical/Diagnosis Cardinality, Rare Levels & Unseen Test-Level Audit**

  > **#I13 (TRUE)**: Confirmed unknown-safe encoding + rare-level handling is required for `apache_3j_diagnosis` (`401` train levels, `124` rare, `2` unseen in test).

  - [x] **[I13.1] Sub-Analysis: APACHE III-J Decimal Hierarchy Decomposition & OOF Target-Encoding Benchmark**

  > **#I13.1 (INCONCLUSIVE)**: Splitting `apache_3j_diagnosis` into `major + system + sub` + OOF smoothed TE lifted diagnosis-only AUC from `0.63872 -> 0.64564` (`Δ = +0.00692`) and lifted full 175-col grouped HGB AUC from `0.84963 -> 0.85165` (`Δ = +0.00202`).

- [x] **Collinearity, Redundancy & Cross-Window Signal Analysis** (`h1_` vs. `d1_` vs. `_apache`)

> **#I14 (INCONCLUSIVE) & #I14.1 (INCONCLUSIVE)**: Across `16` tri-window analytes, median `|Spearman rho(d1_max, apache)| = 0.8369` (`bun=0.9916`, `bilirubin=0.9896`, down to `temp=0.3538`, `mbp=0.4866`). Dropping `h1_` (`-0.00011`) or `15` `_apache` twins (`-0.00056`) hurts CV slightly, and aggressive AutoViz-style SULO pruning at `|rho| >= 0.80` (`93` cols) loses `-0.00167` ROC-AUC ([stage5_cross_window_collinearity.csv](../reports/findings/stage5_cross_window_collinearity.csv)).

  - [x] **[I14] Sub-Analysis: Cross-Window Analyte Collinearity & Window-Ablation CV Audit**

  > **#I14 (INCONCLUSIVE)**: Verified multi-window retention—dropping `h1_` (`0.84946`) or `_apache` twins (`0.84901`) slightly degrades `sanitized_base_A` (`0.84957`).

  - [no] **[I14.1] Sub-Analysis: Apply Aggressive `|rho| >= 0.80` SULO Pruning for GBDT Models**

  > **#I14.1 (INCONCLUSIVE)**: Rejected (`- [no]`) `|rho| >= 0.80` pruning (`93` cols, `0.84790`, `Δ=-0.00167`) for gradient boosting; `|rho| >= 0.90` (`119` cols, `0.84948`, `Δ=-0.00009`) remains a viable parsimonious benchmark for linear models.

- [x] **Target Leakage & Clinical Proxy Audit**

> **#I15 (TRUE) & #I15.1 (INCONCLUSIVE)**: Audited Top-25 + `8` watchlist proxies (`pre_icu_los_days`, `elective_surgery`, `icu_stay_type`, `apache_3j_diagnosis`, `glucose_*`, `arf_apache`). Found `0` post-24h predictors, `0` single-feature `AUC >= 0.90` leaks, and `0` fold-unstable predictors (all fold stds `<= 0.00972`; `arf_apache=1` median creatinine `5.80` vs `0.94`; [stage5_leakage_proxy_audit.csv](../reports/findings/stage5_leakage_proxy_audit.csv)).

  - [x] **[I15] Sub-Analysis: Prediction-Time Timeline & Cross-Hospital Fold Stability Audit**

  > **#I15 (TRUE)**: All audited predictors generalize stably across all 5 disjoint hospital folds (`std <= 0.00972`, `min fold AUC >= 0.54` when `mean >= 0.55`).

  - [x] **[I15.1] Sub-Analysis: Temporal Access Horizon Ablation (`T0 Door` -> `1h` -> `24h No-APACHE` -> `All 24h`)**

  > **#I15.1 (INCONCLUSIVE)**: Grouped SGKF AUC progresses `T0_door=0.66578` -> `T0+T1_1h=0.75777` -> `T0+T1+T2_24h=0.84453` -> `All_24h=0.84957` (`Δ_apache_t3 = +0.00504`).

- [x] **Feature Engineering & Domain Ratio Exploration** *(Note: `bodysystem_unified` omitted because `apache_*_bodysystem` does not exist in WiDS 2021)*

> **#I16 (TRUE) & #I16.1 (FALSE)**: Cumulative domain FE lifted 5-fold SGKF ROC-AUC from `sanitized_base_A = 0.84957` (`LogLoss=0.37667`) -> `fe_AB_glucose = 0.85114` -> `fe_ABC_ratios = 0.85143` -> **`fe_ABCD_hierarchy_te_miss = 0.85344`** (`Δ_AUC = +0.00387`, `Δ_LL = -0.00400`, **`5/5` folds**), led by `d1_glucose_range` (`perm_imp=0.02551`), `apache_3j_diagnosis_te` (`0.00546`), and `bmi_calc` (`0.00277`) ([stage5_feature_engineering_cv.csv](../reports/findings/stage5_feature_engineering_cv.csv)).

  - [x] **[I16] Sub-Analysis: Stepwise Domain Feature Engineering CV Lift (`Families A-D`)**

  > **#I16 (TRUE)**: Every engineering family (`A` cleaning/swap, `B` glycemic dynamics, `C` clinical ratios, `D` APACHE3j hierarchy + nested OOF TE + panel miss counts) contributed monotonic CV AUC and Log-Loss gains.

  - [no] **[I16.1] Sub-Analysis: Clinical-Stratum (`age_bin x icu_type`) & Hospital-Covariate Relative Aggregations**

  > **#I16.1 (FALSE)**: Rejected (`- [no]`)—adding `stratum_z` (`0.85321`, `Δ=-0.00023`) or unsupervised `hosp_cov_rel` (`0.85266`, `Δ=-0.00077`) degraded `fe_ABCD` (`0.85344`).

- [ ] **Baseline Model & Validation Pipeline Verification**

> Before heavy feature selection or tuning, run a fast baseline (e.g., LightGBM/XGBoost with `StratifiedGroupKFold(groups=hospital_id)`) to establish a leak-free CV ROC-AUC benchmark, inspect permutation/SHAP feature importances, and verify that every data cleaning and engineering step actually improves out-of-hospital generalization.
