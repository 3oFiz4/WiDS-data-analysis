# Stage 5 Report: Cross-Window Collinearity, Leakage Audit & Domain FE

- **Generated**: `2026-11-19T07:17:22+08:00` (Base Commit: `bad8295`)

## Inquiry Outcomes

- **[I14 = INCONCLUSIVE] Cross-Window Collinearity**: Median `|rho(d1_max, apache)| =
  0.8369`; dropping `h1_` shifts AUC `+0.00011`, dropping 15
  `_apache` twins shifts AUC `+0.00056`.
- **[I14.1 = INCONCLUSIVE] SULO Correlation Pruning**: `|rho|>=0.90` keeps `118`
  (`Δ=-0.00009`); `|rho|>=0.80` keeps `92` (`Δ=-0.00167`).
- **[I15 = TRUE] Timeline & Proxy Stability**: `n_post_24h=0`,
  `n_leak_ge90=0`, `n_unstable=0`, `arf1_creat_med=5.80 vs arf0_creat_med=0.94`.
- **[I15.1 = INCONCLUSIVE] Temporal Access Horizons**: `T0=0.66578`,
  `T0+T1=0.75777`, `T0+T1+T2=0.84453`,
  `All_24h=0.84957` (`Δ_apache_t3=+0.00504`).
- **[I16 = TRUE] Stepwise Domain FE**: `sanitized_A=0.84957` ->
  `ABCD=0.85344` (`Δ_AUC=+0.00387`, `Δ_LL=-0.00400`,
  `5/5` folds). Top engineered perm importances: `[('d1_glucose_range', 0.02551), ('apache_3j_diagnosis_te', 0.00546), ('bmi_calc', 0.00277), ('d1_glucose_mid', 0.00112), ('glucose_apache_vs_d1_max', 0.00112), ('d1_pulse_pressure_max', 0.00063)]`.
- **[I16.1 = FALSE] Contextual Aggregations**: `stratum_z` `Δ=-0.00023`,
  `hosp_cov_rel` `Δ=-0.00077`, `both` `0.85248`.

## Stepwise Experiment Benchmark (`StratifiedGroupKFold`, 5-Fold)
| Experiment | Cols | OOF ROC-AUC | OOF Log-Loss | Δ AUC vs Sanitized |
| :--- | ---: | ---: | ---: | ---: |
| `sanitized_base_A` | 176 | 0.84957 | 0.37667 | +0.00000 |
| `no_h1_window` | 112 | 0.84946 | 0.37675 | -0.00011 |
| `no_apache_twins` | 161 | 0.84901 | 0.37714 | -0.00056 |
| `sulo_rho_0.90` | 119 | 0.84948 | 0.37676 | -0.00009 |
| `sulo_rho_0.80` | 93 | 0.84790 | 0.37823 | -0.00167 |
| `horizon_T0_door` | 19 | 0.66578 | 0.49273 | -0.18379 |
| `horizon_T0_T1_1h` | 83 | 0.75777 | 0.44599 | -0.09180 |
| `horizon_T0_T1_T2_no_apache` | 147 | 0.84453 | 0.38178 | -0.00504 |
| `fe_AB_glucose` | 183 | 0.85114 | 0.37505 | +0.00157 |
| `fe_ABC_ratios` | 195 | 0.85143 | 0.37475 | +0.00185 |
| `fe_ABCD_hierarchy_te_miss` | 209 | 0.85344 | 0.37267 | +0.00387 |
| `fe_ABCD_plus_stratum_z` | 213 | 0.85321 | 0.37289 | +0.00364 |
| `fe_ABCD_plus_hosp_cov_rel` | 213 | 0.85266 | 0.37363 | +0.00309 |
| `fe_ABCD_plus_both_contextual` | 217 | 0.85248 | 0.37388 | +0.00291 |
