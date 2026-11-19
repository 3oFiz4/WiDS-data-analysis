# Stage 1 Report: Schema, Constants, Duplicates, Domain Taxonomy & Sparsity

- **Generated**: `2026-11-19T06:36:10+08:00` (Base Commit: `9faf954`)
- **Datasets**: `TrainingWiDS2021.csv` (`130,157 × 181`),
  `UnlabeledWiDS2021.csv` (`10,234 × 180`), `DataDictionaryWiDS2021.csv` (`181`)

## Key Findings

1. **[I1 = TRUE] Schema & Drop List**:
   - Schema aligned (`Train \ {diabetes_mellitus} == Unlabeled`): **`True`** (180 cols).
   - Structural drop cols (`3`): `['Unnamed: 0', 'encounter_id', 'readmission_status']`.
   - Near-constant predictors (`>= 99%` modal share):
- `aids`: mode=`0` (99.90% train, 99.90% test)
- `lymphoma`: mode=`0` (99.58% train, 99.59% test)
- `leukemia`: mode=`0` (99.27% train, 99.35% test)

2. **[I1.1 = TRUE] Dictionary Quirks & Facility Overlap**:
   - Dict-only: `['icu_admit_type']`; CSV-only: `['Unnamed: 0']`; metadata mismatches: `13`.
   - Overlaps: `hospital_id=0` (`204` vs `190`),
     `icu_id=264` (`328` vs `302`), `encounter_id=0`.

3. **[I2 = INCONCLUSIVE, I2.1 = FALSE] Duplicates & Near-Duplicates**:
   - Exact duplicates (`keep='first'`): `F_clinical=8`, `F_facility=8`,
     `F_80=8`, `F_50=8`, `F_autoviz_pre=8`.
   - Cross-split exact collisions: **`0`**; within-hospital near-duplicates: **`0`**.

4. **[I3 = TRUE, I3.1 = INCONCLUSIVE] Domain/Window Sparsity & Shift**:
   - Matched `(h1_, d1_)` pairs: `64`, median gap=`+17.77 pp` (Train,
     `p=3.772e-12`, inv=`0`), `+18.82 pp` (Test, `p=3.772e-12`, inv=`0`).
   - >50%-missing inside `h1_` + `labs blood gas`: **`71.23%`** Train,
     **`70.27%`** Test.
   - Missingness rank stability `Spearman rho=0.9955`; `|Δ|>=5pp`: `2`.

### Top 15 Predictors by Missingness Shift
- `bmi`: Train=3.45%, Test=9.92% (|Δ|=6.47 pp)
- `weight`: Train=2.66%, Test=8.87% (|Δ|=6.21 pp)
- `age`: Train=3.83%, Test=0.00% (|Δ|=3.83 pp)
- `d1_inr_min`: Train=62.40%, Test=58.98% (|Δ|=3.42 pp)
- `h1_inr_max`: Train=62.40%, Test=58.98% (|Δ|=3.42 pp)
- `h1_inr_min`: Train=62.40%, Test=58.98% (|Δ|=3.42 pp)
- `d1_inr_max`: Train=62.40%, Test=58.98% (|Δ|=3.42 pp)
- `h1_mbp_noninvasive_min`: Train=10.22%, Test=13.57% (|Δ|=3.36 pp)
- `h1_mbp_noninvasive_max`: Train=10.22%, Test=13.57% (|Δ|=3.36 pp)
- `h1_temp_max`: Train=22.82%, Test=25.93% (|Δ|=3.11 pp)
- `h1_temp_min`: Train=22.82%, Test=25.93% (|Δ|=3.11 pp)
- `h1_mbp_max`: Train=5.02%, Test=7.79% (|Δ|=2.77 pp)
- `h1_mbp_min`: Train=5.02%, Test=7.79% (|Δ|=2.77 pp)
- `d1_lactate_min`: Train=73.38%, Test=76.11% (|Δ|=2.73 pp)
- `d1_lactate_max`: Train=73.38%, Test=76.11% (|Δ|=2.73 pp)

## Sub-Analysis I2.2 Trace (`AutoViz` 9-Duplicate Warning)
- AutoViz `classify_print_vars` reduced predictors to `dft.shape = (130157, 31)`.
- On `dft`: `duplicated(keep='first') = 9`, `duplicated(keep=False) = 10`.
- On full 175 clinical predictors: `duplicated(keep='first') = 8`, `duplicated(keep=False) = 9`.
- Exact mechanism: 9 identical rows (`encounter_ids=[159417, 244601, 150518, 264918, 225543, 147789, 234525, 232838, 189191, 197188]`, indices `[89589, 89608, 89616, 89629, 89637, 89659, 89662, 89664, 89665, 89697]`) exist in `hospital_id=167, icu_id=903, icu_stay_type='admit', diabetes_mellitus=0`, plus 1 extra row collided only under AutoViz's 30-col reduction.
