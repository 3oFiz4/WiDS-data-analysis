# Stage 4 Report: Missingness Mechanisms, Consistency, Ranges & Categoricals

- **Generated**: `2026-11-19T07:04:13+08:00` (Base Commit: `33f771a`)
- **Schema Correction**: Confirmed `apache_2_bodysystem` and `apache_3j_bodysystem` do **not**
  exist in WiDS Datathon 2021 (WiDS 2020 columns).

## Key Findings

1. **[I10 = TRUE, I10.1 = INCONCLUSIVE] Informative Missingness**:
   - Unique non-trivial masks: `89`, MNAR/MAR masks (`BH q<0.01`, `|Δ|>=2pp`):
     **`37`**; `n_miss_total` vs DM Spearman `r=-0.0426` (`p=1.943e-53`).
   - 4-arm SGKF AUC: `native_raw=0.84963`,
     `native_counts=0.84967` (`Δ=+0.00004`),
     `imp_no_flags=0.84739`,
     `imp_flags=0.84881` (`Δ=+0.00142`).

2. **[I11 = TRUE, I11.1 = FALSE] Paired & BP Consistency**:
   - `min > max` violations: **`1683`**; `h1_` outside `d1_` violations:
     **`80081`**; `_apache` outside `d1_` violations: **`250511`**.
   - `bmi` >2% formula mismatches: **`2274`**; recoverable null BMIs:
     Train=`0`, Test=`0`.
   - BP triplets: max unabsorbed=`17.5223%`, mean `(inv - noninv)`=
     `+3.17 mmHg`, pruned BP AUC=`0.84962`
     (`Δ=-0.00001`).

3. **[I12 = TRUE, I12.1 = TRUE] Physiological Bounds & Zero Sentinels**:
   - Dictionary unit mismatches (`9`): `['hematocrit_apache', 'd1_hematocrit_max', 'd1_hematocrit_min', 'h1_hematocrit_max', 'h1_hematocrit_min', 'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min', 'h1_pao2fio2ratio_max', 'h1_pao2fio2ratio_min']`.
   - Extreme tail cols (`>=0.1%`): `['pre_icu_los_days', 'd1_heartrate_min', 'd1_resprate_min', 'd1_spo2_min', 'h1_resprate_min', 'h1_spo2_min']`; sanitized AUC=`0.84974`
     (`Δ=+0.00011`).
   - Hospital-concentrated zero columns (`k_hosp_conc=4`):
- `d1_resprate_min`: `n_zero=5868`, top-5 hosp share=`19.4%`, max hosp rate (`n>=30`)=`35.3%`, DM% (zero vs non-zero)=`24.42% vs 21.51%`
- `h1_resprate_min`: `n_zero=905`, top-5 hosp share=`21.0%`, max hosp rate (`n>=30`)=`29.9%`, DM% (zero vs non-zero)=`25.75% vs 21.62%`
- `d1_heartrate_min`: `n_zero=901`, top-5 hosp share=`18.9%`, max hosp rate (`n>=30`)=`25.0%`, DM% (zero vs non-zero)=`19.76% vs 21.66%`
- `age`: `n_zero=30`, top-5 hosp share=`80.0%`, max hosp rate (`n>=30`)=`3.8%`, DM% (zero vs non-zero)=`10.0% vs 21.97%`

4. **[I13 = TRUE, I13.1 = INCONCLUSIVE] Categoricals & APACHE III-J Hierarchy**:
   - Total rare levels (`n<30`): **`126`**, unseen test levels:
     **`2`** (max row share `0.0195%`).
   - Diagnosis-only AUC: raw=`0.63872` -> hier+TE=
     `0.64564` (`Δ=+0.00692`);
     Full AUC: raw=`0.84963` -> hier+TE=
     `0.85165` (`Δ=+0.00202`).

| Column | Train Nunique | Test Nunique | Rare (<30) | Unseen Test (Rows %) | Collisions |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `ethnicity` | 7 | 7 | 0 | 0 (0.000%) | 0 |
| `gender` | 3 | 3 | 0 | 0 (0.000%) | 0 |
| `hospital_admit_source` | 16 | 11 | 2 | 0 (0.000%) | 0 |
| `icu_admit_source` | 6 | 6 | 0 | 0 (0.000%) | 0 |
| `icu_stay_type` | 3 | 3 | 0 | 0 (0.000%) | 0 |
| `icu_type` | 8 | 8 | 0 | 0 (0.000%) | 0 |
| `apache_2_diagnosis` | 45 | 45 | 0 | 0 (0.000%) | 0 |
| `apache_3j_diagnosis` | 401 | 344 | 124 | 2 (0.019%) | 0 |
