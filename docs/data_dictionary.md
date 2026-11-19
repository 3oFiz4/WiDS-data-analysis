# WiDS Datathon 2021 Data Dictionary & Schema Quirks

Grounded audit of `data/raw/DataDictionaryWiDS2021.csv` against `TrainingWiDS2021.csv` (130,157 × 181) and `UnlabeledWiDS2021.csv` (10,234 × 180).

## Known Dictionary vs. CSV Quirks

- **Absent dictionary column**: `icu_admit_type` (in dict, absent in CSVs: ['icu_admit_type']).
- **Undocumented index**: `Unnamed: 0` (seq 1..N train=True, test=True; ['Unnamed: 0']).
- **Zero-variance constant**: `readmission_status` (`nunique=1`, all `0`).
- **Facility overlap**: `hospital_id` overlap=0 (204 vs 190); `icu_id` overlap=264 (328 vs 302); `encounter_id`=0.
- **Dictionary metadata errors**: `bmi`/`apache_2_diagnosis`/`apache_3j_diagnosis` tagged `string`; `d1/h1_albumin_max` says 'lowest'; `inr` unit `micromol/L`; `pao2fio2ratio` copied from `fio2`.

## Domain × Temporal Window Summary

| Category | Window | Cols | Train Mean Null % | Test Mean Null % | >50% Null | >80% Null |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `labs blood gas` | `h1_first_hour` | 8 | 83.81% | 82.73% | 8 | 8 |
| `labs` | `h1_first_hour` | 30 | 79.69% | 79.28% | 30 | 18 |
| `labs blood gas` | `d1_first_24h` | 8 | 66.58% | 64.37% | 8 | 0 |
| `labs` | `d1_first_24h` | 30 | 25.12% | 25.04% | 8 | 0 |
| `vitals` | `h1_first_hour` | 26 | 24.49% | 26.30% | 6 | 6 |
| `APACHE covariate` | `apache_24h_worst` | 28 | 24.32% | 24.18% | 7 | 0 |
| `vitals` | `d1_first_24h` | 26 | 17.57% | 17.38% | 6 | 0 |
| `demographic` | `static_baseline` | 14 | 2.75% | 3.62% | 0 | 0 |
| `APACHE comorbidity` | `static_baseline` | 7 | 0.00% | 0.00% | 0 | 0 |
| `identifier` | `static_baseline` | 2 | 0.00% | 0.00% | 0 | 0 |
| `csv_index_artifact` | `static_baseline` | 1 | 0.00% | 0.00% | 0 | 0 |

## Full 181-Column Catalog

| Column | Role | Category | Window | Dtype | Train Null % | Test Null % | Distinct |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: |
| `Unnamed: 0` | `pure_row_identifier` | `csv_index_artifact` | `static_baseline` | `int64` | 0.00% | 0.00% | 130157 |
| `age` | `candidate_predictor` | `demographic` | `static_baseline` | `float64` | 3.83% | 0.00% | 75 |
| `aids` | `near_constant_ge_99pct` | `APACHE comorbidity` | `static_baseline` | `int64` | 0.00% | 0.00% | 2 |
| `albumin_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 60.05% | 59.17% | 36 |
| `apache_2_diagnosis` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 1.29% | 1.66% | 44 |
| `apache_3j_diagnosis` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 0.66% | 1.06% | 400 |
| `apache_post_operative` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `int64` | 0.00% | 0.00% | 2 |
| `arf_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `int64` | 0.00% | 0.00% | 2 |
| `bilirubin_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 63.43% | 61.66% | 585 |
| `bmi` | `candidate_predictor` | `demographic` | `static_baseline` | `float64` | 3.45% | 9.92% | 41453 |
| `bun_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 19.52% | 20.14% | 476 |
| `cirrhosis` | `candidate_predictor` | `APACHE comorbidity` | `static_baseline` | `int64` | 0.00% | 0.00% | 2 |
| `creatinine_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 19.12% | 19.96% | 1198 |
| `d1_albumin_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 54.86% | 54.13% | 40 |
| `d1_albumin_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 54.86% | 54.13% | 38 |
| `d1_arterial_pco2_max` | `candidate_predictor` | `labs blood gas` | `d1_first_24h` | `float64` | 64.89% | 62.64% | 861 |
| `d1_arterial_pco2_min` | `candidate_predictor` | `labs blood gas` | `d1_first_24h` | `float64` | 64.89% | 62.64% | 696 |
| `d1_arterial_ph_max` | `candidate_predictor` | `labs blood gas` | `d1_first_24h` | `float64` | 65.16% | 63.37% | 542 |
| `d1_arterial_ph_min` | `candidate_predictor` | `labs blood gas` | `d1_first_24h` | `float64` | 65.16% | 63.37% | 635 |
| `d1_arterial_po2_max` | `candidate_predictor` | `labs blood gas` | `d1_first_24h` | `float64` | 64.55% | 61.97% | 2546 |
| `d1_arterial_po2_min` | `candidate_predictor` | `labs blood gas` | `d1_first_24h` | `float64` | 64.55% | 61.97% | 1883 |
| `d1_bilirubin_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 58.96% | 57.26% | 593 |
| `d1_bilirubin_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 58.96% | 57.26% | 559 |
| `d1_bun_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 10.55% | 10.83% | 494 |
| `d1_bun_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 10.55% | 10.83% | 471 |
| `d1_calcium_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 12.82% | 13.70% | 47 |
| `d1_calcium_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 12.82% | 13.70% | 49 |
| `d1_creatinine_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 10.20% | 10.66% | 1247 |
| `d1_creatinine_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 10.20% | 10.66% | 1147 |
| `d1_diasbp_invasive_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 73.04% | 71.58% | 145 |
| `d1_diasbp_invasive_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 73.04% | 71.58% | 85 |
| `d1_diasbp_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.21% | 0.22% | 120 |
| `d1_diasbp_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.21% | 0.22% | 78 |
| `d1_diasbp_noninvasive_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 1.26% | 1.46% | 120 |
| `d1_diasbp_noninvasive_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 1.26% | 1.46% | 78 |
| `d1_glucose_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 6.33% | 6.65% | 538 |
| `d1_glucose_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 6.33% | 6.65% | 256 |
| `d1_hco3_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 15.40% | 15.58% | 242 |
| `d1_hco3_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 15.40% | 15.58% | 273 |
| `d1_heartrate_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.20% | 0.23% | 120 |
| `d1_heartrate_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.20% | 0.23% | 156 |
| `d1_hemaglobin_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 12.47% | 12.20% | 105 |
| `d1_hemaglobin_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 12.47% | 12.20% | 115 |
| `d1_hematocrit_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 11.98% | 12.01% | 312 |
| `d1_hematocrit_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 11.98% | 12.01% | 340 |
| `d1_inr_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 62.40% | 58.98% | 517 |
| `d1_inr_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 62.40% | 58.98% | 415 |
| `d1_lactate_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 73.38% | 76.11% | 783 |
| `d1_lactate_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 73.38% | 76.11% | 546 |
| `d1_mbp_invasive_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 72.89% | 71.48% | 285 |
| `d1_mbp_invasive_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 72.89% | 71.48% | 118 |
| `d1_mbp_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.25% | 0.36% | 158 |
| `d1_mbp_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.25% | 0.36% | 128 |
| `d1_mbp_noninvasive_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 1.71% | 2.07% | 155 |
| `d1_mbp_noninvasive_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 1.71% | 2.07% | 128 |
| `d1_pao2fio2ratio_max` | `candidate_predictor` | `labs blood gas` | `d1_first_24h` | `float64` | 71.71% | 69.49% | 5602 |
| `d1_pao2fio2ratio_min` | `candidate_predictor` | `labs blood gas` | `d1_first_24h` | `float64` | 71.71% | 69.49% | 5320 |
| `d1_platelets_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 14.26% | 13.65% | 559 |
| `d1_platelets_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 14.26% | 13.65% | 541 |
| `d1_potassium_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 9.64% | 9.48% | 104 |
| `d1_potassium_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 9.64% | 9.48% | 118 |
| `d1_resprate_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.52% | 0.54% | 79 |
| `d1_resprate_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.52% | 0.54% | 55 |
| `d1_sodium_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 10.20% | 10.42% | 71 |
| `d1_sodium_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 10.20% | 10.42% | 138 |
| `d1_spo2_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.41% | 0.61% | 47 |
| `d1_spo2_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.41% | 0.61% | 101 |
| `d1_sysbp_invasive_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 73.02% | 71.55% | 225 |
| `d1_sysbp_invasive_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 73.02% | 71.55% | 163 |
| `d1_sysbp_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.21% | 0.22% | 143 |
| `d1_sysbp_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 0.21% | 0.22% | 120 |
| `d1_sysbp_noninvasive_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 1.25% | 1.45% | 143 |
| `d1_sysbp_noninvasive_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 1.25% | 1.45% | 120 |
| `d1_temp_max` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 3.45% | 4.20% | 261 |
| `d1_temp_min` | `candidate_predictor` | `vitals` | `d1_first_24h` | `float64` | 3.45% | 4.20% | 309 |
| `d1_wbc_max` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 13.39% | 13.89% | 3141 |
| `d1_wbc_min` | `candidate_predictor` | `labs` | `d1_first_24h` | `float64` | 13.39% | 13.89% | 2810 |
| `diabetes_mellitus` | `target` | `Target Variable` | `static_baseline` | `int64` | 0.00% | Target | 2 |
| `elective_surgery` | `candidate_predictor` | `demographic` | `static_baseline` | `int64` | 0.00% | 0.00% | 2 |
| `encounter_id` | `pure_row_identifier` | `identifier` | `static_baseline` | `int64` | 0.00% | 0.00% | 130157 |
| `ethnicity` | `candidate_predictor` | `demographic` | `static_baseline` | `object` | 1.22% | 1.99% | 6 |
| `fio2_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 76.62% | 74.60% | 87 |
| `gcs_eyes_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 1.68% | 2.00% | 4 |
| `gcs_motor_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 1.68% | 2.00% | 6 |
| `gcs_unable_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 0.54% | 0.60% | 2 |
| `gcs_verbal_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 1.68% | 2.00% | 5 |
| `gender` | `candidate_predictor` | `demographic` | `static_baseline` | `object` | 0.05% | 0.05% | 2 |
| `glucose_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 11.29% | 11.82% | 566 |
| `h1_albumin_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 91.43% | 91.51% | 38 |
| `h1_albumin_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 91.43% | 91.51% | 37 |
| `h1_arterial_pco2_max` | `candidate_predictor` | `labs blood gas` | `h1_first_hour` | `float64` | 82.72% | 81.56% | 826 |
| `h1_arterial_pco2_min` | `candidate_predictor` | `labs blood gas` | `h1_first_hour` | `float64` | 82.72% | 81.56% | 811 |
| `h1_arterial_ph_max` | `candidate_predictor` | `labs blood gas` | `h1_first_hour` | `float64` | 82.86% | 82.09% | 580 |
| `h1_arterial_ph_min` | `candidate_predictor` | `labs blood gas` | `h1_first_hour` | `float64` | 82.86% | 82.09% | 601 |
| `h1_arterial_po2_max` | `candidate_predictor` | `labs blood gas` | `h1_first_hour` | `float64` | 82.55% | 81.16% | 1844 |
| `h1_arterial_po2_min` | `candidate_predictor` | `labs blood gas` | `h1_first_hour` | `float64` | 82.55% | 81.16% | 1843 |
| `h1_bilirubin_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 92.09% | 91.92% | 314 |
| `h1_bilirubin_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 92.09% | 91.92% | 314 |
| `h1_bun_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 80.66% | 80.42% | 259 |
| `h1_bun_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 80.66% | 80.42% | 259 |
| `h1_calcium_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 81.38% | 81.16% | 59 |
| `h1_calcium_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 81.38% | 81.16% | 62 |
| `h1_creatinine_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 80.51% | 80.32% | 879 |
| `h1_creatinine_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 80.51% | 80.32% | 876 |
| `h1_diasbp_invasive_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 80.54% | 79.86% | 103 |
| `h1_diasbp_invasive_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 80.54% | 79.86% | 86 |
| `h1_diasbp_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 4.25% | 6.37% | 107 |
| `h1_diasbp_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 4.25% | 6.37% | 92 |
| `h1_diasbp_noninvasive_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 8.71% | 11.30% | 108 |
| `h1_diasbp_noninvasive_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 8.71% | 11.30% | 93 |
| `h1_glucose_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 57.68% | 58.53% | 612 |
| `h1_glucose_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 57.68% | 58.53% | 604 |
| `h1_hco3_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 81.74% | 81.47% | 224 |
| `h1_hco3_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 81.74% | 81.47% | 211 |
| `h1_heartrate_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 3.13% | 5.38% | 119 |
| `h1_heartrate_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 3.13% | 5.38% | 109 |
| `h1_hemaglobin_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 78.97% | 78.61% | 124 |
| `h1_hemaglobin_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 78.97% | 78.61% | 124 |
| `h1_hematocrit_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 79.10% | 78.55% | 358 |
| `h1_hematocrit_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 79.10% | 78.55% | 361 |
| `h1_inr_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 62.40% | 58.98% | 517 |
| `h1_inr_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 62.40% | 58.98% | 415 |
| `h1_lactate_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 91.02% | 92.06% | 599 |
| `h1_lactate_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 91.02% | 92.06% | 591 |
| `h1_mbp_invasive_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 80.49% | 79.99% | 255 |
| `h1_mbp_invasive_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 80.49% | 79.99% | 133 |
| `h1_mbp_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 5.02% | 7.79% | 155 |
| `h1_mbp_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 5.02% | 7.79% | 137 |
| `h1_mbp_noninvasive_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 10.22% | 13.57% | 153 |
| `h1_mbp_noninvasive_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 10.22% | 13.57% | 137 |
| `h1_pao2fio2ratio_max` | `candidate_predictor` | `labs blood gas` | `h1_first_hour` | `float64` | 87.12% | 86.11% | 3425 |
| `h1_pao2fio2ratio_min` | `candidate_predictor` | `labs blood gas` | `h1_first_hour` | `float64` | 87.12% | 86.11% | 3404 |
| `h1_platelets_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 81.23% | 80.09% | 555 |
| `h1_platelets_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 81.23% | 80.09% | 555 |
| `h1_potassium_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 77.46% | 76.99% | 115 |
| `h1_potassium_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 77.46% | 76.99% | 113 |
| `h1_resprate_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 4.96% | 7.07% | 50 |
| `h1_resprate_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 4.96% | 7.07% | 94 |
| `h1_sodium_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 78.20% | 77.83% | 99 |
| `h1_sodium_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 78.20% | 77.83% | 113 |
| `h1_spo2_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 4.80% | 7.09% | 79 |
| `h1_spo2_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 4.80% | 7.09% | 101 |
| `h1_sysbp_invasive_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 80.52% | 79.83% | 182 |
| `h1_sysbp_invasive_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 80.52% | 79.83% | 168 |
| `h1_sysbp_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 4.24% | 6.36% | 149 |
| `h1_sysbp_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 4.24% | 6.36% | 142 |
| `h1_sysbp_noninvasive_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 8.70% | 11.29% | 149 |
| `h1_sysbp_noninvasive_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 8.70% | 11.29% | 143 |
| `h1_temp_max` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 22.82% | 25.93% | 383 |
| `h1_temp_min` | `candidate_predictor` | `vitals` | `h1_first_hour` | `float64` | 22.82% | 25.93% | 392 |
| `h1_wbc_max` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 81.43% | 80.75% | 2132 |
| `h1_wbc_min` | `candidate_predictor` | `labs` | `h1_first_hour` | `float64` | 81.43% | 80.75% | 2128 |
| `heart_rate_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 0.24% | 0.29% | 149 |
| `height` | `candidate_predictor` | `demographic` | `static_baseline` | `float64` | 1.60% | 2.94% | 443 |
| `hematocrit_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 20.56% | 20.66% | 353 |
| `hepatic_failure` | `candidate_predictor` | `APACHE comorbidity` | `static_baseline` | `int64` | 0.00% | 0.00% | 2 |
| `hospital_admit_source` | `candidate_predictor` | `demographic` | `static_baseline` | `object` | 25.51% | 26.71% | 15 |
| `hospital_id` | `facility_group_key` | `identifier` | `static_baseline` | `int64` | 0.00% | 0.00% | 204 |
| `icu_admit_source` | `candidate_predictor` | `demographic` | `static_baseline` | `object` | 0.18% | 0.24% | 5 |
| `icu_id` | `facility_group_key` | `demographic` | `static_baseline` | `int64` | 0.00% | 0.00% | 328 |
| `icu_stay_type` | `candidate_predictor` | `demographic` | `static_baseline` | `object` | 0.00% | 0.00% | 3 |
| `icu_type` | `candidate_predictor` | `demographic` | `static_baseline` | `object` | 0.00% | 0.00% | 8 |
| `immunosuppression` | `candidate_predictor` | `APACHE comorbidity` | `static_baseline` | `int64` | 0.00% | 0.00% | 2 |
| `intubated_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `int64` | 0.00% | 0.00% | 2 |
| `leukemia` | `near_constant_ge_99pct` | `APACHE comorbidity` | `static_baseline` | `int64` | 0.00% | 0.00% | 2 |
| `lymphoma` | `near_constant_ge_99pct` | `APACHE comorbidity` | `static_baseline` | `int64` | 0.00% | 0.00% | 2 |
| `map_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 0.32% | 0.51% | 172 |
| `paco2_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 76.62% | 74.60% | 731 |
| `paco2_for_ph_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 76.62% | 74.60% | 731 |
| `pao2_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 76.62% | 74.60% | 2043 |
| `ph_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 76.62% | 74.60% | 576 |
| `pre_icu_los_days` | `candidate_predictor` | `demographic` | `static_baseline` | `float64` | 0.00% | 0.00% | 10912 |
| `readmission_status` | `constant_zero_variance` | `demographic` | `static_baseline` | `int64` | 0.00% | 0.00% | 1 |
| `resprate_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 0.62% | 0.63% | 77 |
| `sodium_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 18.84% | 19.47% | 119 |
| `solid_tumor_with_metastasis` | `candidate_predictor` | `APACHE comorbidity` | `static_baseline` | `int64` | 0.00% | 0.00% | 2 |
| `temp_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 5.08% | 6.56% | 207 |
| `urineoutput_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 48.53% | 50.71% | 32958 |
| `ventilated_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `int64` | 0.00% | 0.00% | 2 |
| `wbc_apache` | `candidate_predictor` | `APACHE covariate` | `apache_24h_worst` | `float64` | 22.65% | 23.22% | 3124 |
| `weight` | `candidate_predictor` | `demographic` | `static_baseline` | `float64` | 2.66% | 8.87% | 3636 |
