# Stage 2 Report: Facility Collisions, Covariate Shift & Target Prevalence

- **Generated**: `2026-11-19T06:40:30+08:00` (Base Commit: `174dd17`)

## Key Findings

1. **[I4 = INCONCLUSIVE, I4.1 = TRUE] Facility Collision & CV Lock**:
   - Combined multi-hospital `icu_id` collisions: **`72.13%`**
     (`max=2` hospitals/icu_id across 366 unique IDs).
   - Within-train multi-hospital `icu_id` collisions: **`0.00%`**
     (`max=1`, NMI=`0.9293`).
   - 5-fold `StratifiedGroupKFold(groups=hospital_id)`: fold share range =
     `0.104 pp`, target rate range = `0.150 pp`,
     hospital overlap = `0`, dup cluster folds = `1`.
   - `icu_id` target-encoding AUC: Random SKF = **`0.6027`** vs.
     Grouped SGKF = **`0.4993`**.

2. **[I5 = INCONCLUSIVE, I5.1 = FALSE] Out-of-Hospital Adversarial Validation**:
   - `adv_auc_raw_grouped` (SGKF): **`0.5913`** vs.
     `adv_auc_raw_random` (SKF): **`0.7536`**.
   - Ablations (SGKF): imputed=`0.5931` (`Δ_miss=-0.0017`),
     missingness-only=`0.4677`, with `icu_id`=`0.3831`
     (`Δ_icu=-0.2082`), no-anthro-miss=`0.5787`.
   - Features with `PSI >= 0.10`: `0`.

### Top 10 Adversarial Shift Drivers
- `age`: perm_imp=0.0418, PSI=0.0136, KS=0.037635
- `h1_heartrate_min`: perm_imp=0.0062, PSI=0.000612, KS=0.006456
- `d1_potassium_max`: perm_imp=0.0060, PSI=0.000741, KS=0.010348
- `h1_sysbp_max`: perm_imp=0.0057, PSI=0.002024, KS=0.017461
- `d1_calcium_min`: perm_imp=0.0051, PSI=0.013966, KS=0.051252
- `d1_sysbp_max`: perm_imp=0.0045, PSI=0.007034, KS=0.033491
- `d1_resprate_max`: perm_imp=0.0041, PSI=0.005543, KS=0.030814
- `d1_hemaglobin_min`: perm_imp=0.0039, PSI=0.001066, KS=0.00894
- `heart_rate_apache`: perm_imp=0.0039, PSI=0.000937, KS=0.016916
- `d1_heartrate_max`: perm_imp=0.0035, PSI=0.002177, KS=0.015527

3. **[I6 = TRUE, I6.1 = INCONCLUSIVE] Target Prevalence & Prior Shift**:
   - Overall `diabetes_mellitus=1`: **`21.63%`**.
   - Hospital spread (`n >= 30`, 188/204 hospitals):
     chi2 `p=0.000e+00`, IQR = **`8.82 pp`**
     (P10=`8.96%`, P90=`29.17%`).
   - Significant core strata (`p < 0.001` & range `>= 3.0 pp`): **`3/4`**.
   - Standardized Train->Unlabeled prior shifts (pp): `{'age_bin': 0.1974, 'bmi_bin': -0.0837, 'ethnicity': -0.128, 'icu_type': 0.0143}`.

### Missingness-Stratum Target Bias (`k_miss_bias = 2`)
- `age` (`n_na=4988`): null_prev=13.23% vs present_prev=21.96% (|Δ|=8.73 pp, p=9.944e-49)
- `bmi` (`n_na=4490`): null_prev=20.38% vs present_prev=21.67% (|Δ|=1.29 pp, p=4.019e-02)
- `weight` (`n_na=3463`): null_prev=20.99% vs present_prev=21.65% (|Δ|=0.65 pp, p=3.685e-01)
- `ethnicity` (`n_na=1587`): null_prev=13.99% vs present_prev=21.72% (|Δ|=7.73 pp, p=1.291e-13)
