- [x] Pull dataset
- [x] Yield surface-level EDA analysis
- [x] Export common charts

### Pre-assumptive Action

I only assume these methods below from a single glimpse, and these methods originates from my inference from small data.

- [ ] Sort features by most empty, most constants (low distinct value)
- [ ] Yield a non-linear correlation analysis over all variables

> According to @..\reports\figures\diabetes_mellitus\Dist_Plots_Numberics.png, almost all of the variables have non-linear distribution, we will assume for now that P1: All variables are non-linear distribution

- [ ] Yield @..\reports\figures\diabetes_mellitus\Pair_Scatter_Plots.png and assay directly the top-most correlated column

> This triage workflow might be useful. Most columns are useless, not only that but 100+ features to be EDA'ed sounds impossible, unless you are some kind of speciallized obtuse folk. We only seek to yield those whom seemingly useful, except those who proxies for the diagnosis itself (e.g., `bmi_level` vs `is_bmi`, both explain the same thing). We will start with these features, EDA these features, and then we will slowly scale to the lower R value features.

- [ ] Rank each columns by Linearity and Non-Linearity Signal
- [ ] Yield a Feature Triage Summary Table
- [ ] In <Feature Triage Summary Table>, analysis each variable ((as much as needed) top-most ranked) and audit them if they justify any of these questions below: Is this post-diagnosis leakage?; Does impossible values exist?

- [ ] **Schema, Identifier & Constant Column Audit** (`Unnamed: 0`, `encounter_id`, `hospital_id`, `icu_id`, `readmission_status`)

> `TrainingWiDS2021.csv` has 181 columns (`130,157` rows) while `UnlabeledWiDS2021.csv` has 180 (`10,234` rows, missing `diabetes_mellitus`). `readmission_status` is constant (1 unique value), `Unnamed: 0` and `encounter_id` are pure row/encounter IDs (`0` duplicate `encounter_id`s), and `hospital_id`/`icu_id` are facility identifiers. Dropping non-predictive IDs and zero-variance columns prevents trees from memorizing spurious splits, while separating IDs early keeps `hospital_id` available for grouped validation.

- [ ] **Train vs. Unlabeled Test Distribution Shift & Group Overlap Check**

> `hospital_id` has `204` unique hospitals in train and `190` in test with **0 overlap** between them. If we use a random train/val split, models can memorize hospital-specific coding/lab practices and overfit; checking covariate shift (adversarial validation AUC) and enforcing `StratifiedGroupKFold(groups=hospital_id)` ensures our local CV reflects true out-of-hospital generalization on the test set.

- [ ] **Duplicate & Near-Duplicate Row Investigation** (excluding IDs)

> While `encounter_id` is unique, AutoViz flagged `9` duplicate rows once ID and low-information columns were removed. Auditing whether these are true clinical duplicates (e.g., transferred patients or repeated encounters) vs. coincidental identical vitals prevents subtle train/val leakage if identical feature vectors land in different folds.

- [ ] **Target Variable (`diabetes_mellitus`) Class Balance & Subgroup Prevalence Analysis**

> The target is imbalanced (`78.37%` negative vs. `21.63%` positive). Checking whether diabetes prevalence varies systematically across `hospital_id`, `icu_type`, `age`, `ethnicity`, or `elective_surgery` reveals baseline risk priors and determines whether class weighting, stratified grouping, and probability calibration (for ROC-AUC / log-loss) are needed.

- [ ] **Data Dictionary Grouping & Clinical Domain Mapping** (`DataDictionaryWiDS2021.csv`)

> The 181 columns fall into 8 domain groups: `labs` (60), `vitals` (52), `APACHE covariate` (28), `labs blood gas` (16), `demographic` (15), `APACHE comorbidity` (7), `identifier` (2), and `Target Variable` (1). Analyzing features by clinical group (and time window: `h1_` first hour vs. `d1_` first 24 hours vs. `_apache` worst 24h) makes 180 predictors tractable and exposes structural redundancy.

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
