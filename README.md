## Setup

1. Copy `.env.example` to `.env` and fill in your Kaggle credentials:
   ```bash
   cp .env.example .env
   ```
2. Install dependencies from `uv.lock` and enable local pre-commit hooks (auto-strips notebook outputs and runs `ruff` on commit):
   ```bash
   uv sync
   uvx pre-commit install
   ```
3. Fetch and verify the dataset into `data/raw/`:
   ```bash
   uv run python scripts/fetch_data.py
   ```

## File Architecture

```
.
├── .github/
│   ├── ISSUE_TEMPLATE/hypothesis.md   # Template for proposing new hypotheses
│   ├── pull_request_template.md       # PR checklist (takeaway, README update)
│   └── workflows/ci.yml               # Runs gitleaks, >5MB guard, ruff, and nbstripout
├── data/                              # Git-ignored local data
│   ├── raw/                           # Downloaded Kaggle files
│   └── processed/                     # Cleaned/intermediate outputs
├── docs/
│   └── data_dictionary.md             # Column definitions, quirks, and edge cases
├── notebooks/
│   ├── 01-eda/                        # Exploratory data analysis
│   ├── 02-hypotheses/                 # Hypothesis tests (h01-..., h02-...)
│   └── 03-modeling/                   # Occasional prediction experiments
├── reports/
│   ├── figures/                       # Exported charts for slides/docs
│   └── findings/                      # Short markdown takeaways & exported HTML/PDFs
├── scripts/
│   └── fetch_data.py                  # One-command Kaggle downloader & checksum verifier
├── src/project_name/
│   ├── config.py                      # Shared dataset slugs, date cutoffs, & thresholds
│   ├── loaders.py                     # Kaggle download (.env) & shared cleaning helpers
│   └── stats.py                       # Shared hypothesis test helpers
├── .env.example                       # KAGGLE_USERNAME and KAGGLE_KEY placeholders
├── .gitignore                         # Ignores .env, data/, .ipynb_checkpoints/
├── .pre-commit-config.yaml            # Local nbstripout + ruff hooks
├── pyproject.toml                     # Dependencies + ruff & nbstripout settings
└── uv.lock                            # Locked environment for reproducible runs
```

## Creating Notebooks ("One Person, One Notebook")

To prevent Git merge conflicts on `.ipynb` files, scope each notebook to a single owner and topic using:

```
<prefix>-<owner>-<topic>.ipynb
```

Examples in `notebooks/02-hypotheses/`:
- `h01-alex-pricing-impact.ipynb`
- `h02-sam-retention-cohorts.ipynb`

Workflow:
1. Create a new notebook with your name and the next hypothesis number (`h03-<you>-<topic>.ipynb`).
2. To build on a teammate's work, copy their notebook to a new file (`h03-...`) or move shared cleaning/test logic into `src/project_name/`.
3. Keep `notebooks/` clean—commit with `pre-commit` active so outputs are stripped. To share rendered charts with stakeholders, export to `reports/findings/` via `jupyter nbconvert --to html`.

## Hypothesis Status

Full 38-unit HMECVAS ledger (0 `OPEN`, 0 REGEXP violations): [`reports/findings/inquiries.yaml`](reports/findings/inquiries.yaml) · [`reports/findings/hmecvas_master_registry.csv`](reports/findings/hmecvas_master_registry.csv). Note: Stage 5 ablation runs used `HGB(max_iter=120, lr=0.06)` (`0.84957 -> 0.85344`, `+0.00387`), while Stage 6 final verification used `HGB(max_iter=200, lr=0.05)` (`0.85139 -> 0.85520`, `+0.00381`).

| ID | Hypothesis | Script / Report | Status | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| `I1`–`I1.1` | Schema alignment, structural/near-const drops, dict drift & zero hospital overlap | [`stage1_schema_sparsity.md`](reports/findings/stage1_schema_sparsity.md) | `TRUE` / `TRUE` | `180` shared predictors; drop `['Unnamed: 0', 'encounter_id', 'readmission_status']`; `204` vs `190` disjoint hospitals (`0` overlap); `13` dict errors documented in [`docs/data_dictionary.md`](docs/data_dictionary.md). |
| `I2`–`I2.2` | Exact 175-col clinical duplicates, near-dups & AutoViz 30-col projection | [`stage1_schema_sparsity.md`](reports/findings/stage1_schema_sparsity.md) | `INCONCLUSIVE` / `FALSE` / `TRUE` | `8` excess copies (`9` identical rows in `hospital_id=167`); `0` Train-Test collisions; AutoViz `9` dups trace to `<=30` column cap. |
| `I3`–`I3.1` | Paired `h1_` vs `d1_` missingness gap & Train-Unlabeled missingness drift | [`stage1_schema_sparsity.md`](reports/findings/stage1_schema_sparsity.md) | `TRUE` / `INCONCLUSIVE` | `h1_` `+17.77 pp` train / `+18.82 pp` test higher median nullness (`p=3.772e-12`); ranks stable (`rho=0.9955`), `bmi` (`+6.47 pp`) & `weight` (`+6.21 pp`) shifted. |
| `I4`–`I4.1` | Cross-split `icu_id` collisions, `StratifiedGroupKFold` & TE leakage | [`stage2_shift_and_target.md`](reports/findings/stage2_shift_and_target.md) | `INCONCLUSIVE` / `TRUE` | `72.13%` cross-split vs `0.00%` within-train `icu_id` collisions (`NMI=0.9293`); `icu_id` TE drops `0.6027 -> 0.4993` under `SGKF(hospital_id)`. |
| `I5`–`I5.1` | Grouped adversarial shift & missingness vs `icu_id` drivers | [`stage2_shift_and_target.md`](reports/findings/stage2_shift_and_target.md) | `INCONCLUSIVE` / `FALSE` | Grouped Adversarial AUC=`0.5913` (`0` cols `PSI>=0.10`; imputed=`0.5931`, miss-only=`0.4677`, `with_icu=0.3831`). |
| `I6`–`I6.1` | Hospital/demographic prevalence & missingness bias prior shift | [`stage2_shift_and_target.md`](reports/findings/stage2_shift_and_target.md) | `TRUE` / `INCONCLUSIVE` | Base rate `21.63%`, hospital IQR `8.82 pp` (`p=0.0`); `age`/`ethnicity` nulls biased (`p<=1.29e-13`), projected Test prior shift `<=0.197 pp`. |
| `I7`–`I7.1` | Normality violations & non-linear vs Pearson signal discovery | [`stage3_feature_triage.md`](reports/findings/stage3_feature_triage.md) | `TRUE` / `TRUE` | Refutes `P1`: `71/153` (`46.41%`) near-Gaussian; Top-20 Jaccard=`0.7391` with `15` non-linear lift features (`d1_glucose_min`, `apache_3j_diagnosis`). |
| `I8`–`I8.1` | Top-25 collinearity clustering & Top-25 vs Top-50 vs All-175 curve | [`stage3_feature_triage.md`](reports/findings/stage3_feature_triage.md) | `TRUE` / `INCONCLUSIVE` | Top-25 collapses to `13` clusters; adding `26..175` lifts `SGKF` AUC `0.84083 -> 0.84963` (`+0.00880`, `5/5` folds). |
| `I9`–`I9.1` | Deterministic leak check, impossible values & glucose/diag ablation | [`stage3_feature_triage.md`](reports/findings/stage3_feature_triage.md) | `TRUE` / `INCONCLUSIVE` | `0` Top-40 hard leaks (`>=0.90`); `10` sentinel/unit cols; dropping glucose+diag cuts AUC `0.84963 -> 0.74388`. |
| `I10`–`I10.1` | MNAR/MAR missingness masks & native-NaN vs indicator imputation | [`stage4_integrity_missingness.md`](reports/findings/stage4_integrity_missingness.md) | `TRUE` / `INCONCLUSIVE` | `37/89` deduplicated masks `BH q<0.01 & |Δ|>=2pp`; `native_raw=0.84963`, `native_counts=0.84967`, `imp_flags=0.84881`, `imp_no_flags=0.84739`. |
| `I11`–`I11.1` | Paired structural inversions, BMI errors & BP redundancy test | [`stage4_integrity_missingness.md`](reports/findings/stage4_integrity_missingness.md) | `TRUE` / `FALSE` | `1,683` `min>max`, `80,081` `h1>d1`, `2,274` BMI mismatches; `17.52%` max unabsorbed BP so keep all BP sub-columns. |
| `I12`–`I12.1` | Unit mismatches, sentinel sanitization & hospital-concentrated zeros | [`stage4_integrity_missingness.md`](reports/findings/stage4_integrity_missingness.md) | `TRUE` / `TRUE` | `9` dict unit mismatches, `6` tail cols `>=0.1%`, sanitized CV `0.84974`; `4/4` zero vital/age cols clustered in specific hospitals. |
| `I13`–`I13.1` | Rare/unseen categoricals & `apache_3j_diagnosis` hierarchy + OOF TE | [`stage4_integrity_missingness.md`](reports/findings/stage4_integrity_missingness.md) | `TRUE` / `INCONCLUSIVE` | `126` rare levels (`<30`), `2` unseen test levels (`0.0195%`); hierarchy+OOF TE lifts full AUC `0.84963 -> 0.85165` (`+0.00202`). |
| `I14`–`I14.1` | Tri-window (`h1`/`d1`/`apache`) ablation & SULO correlation pruning | [`stage5_collinearity_and_fe.md`](reports/findings/stage5_collinearity_and_fe.md) | `INCONCLUSIVE` / `INCONCLUSIVE` | `h1_` (`-0.00011`) & `_apache` (`-0.00056`) mildly dispensable for HGB; SULO `\|rho\|>=0.90` (`-0.00009`) neutral, `\|rho\|>=0.80` (`-0.00167`) hurts trees slightly. |
| `I15`–`I15.1` | Post-24h leak proxy audit & temporal horizon (`T0 -> 1h -> 24h`) ablation | [`stage5_collinearity_and_fe.md`](reports/findings/stage5_collinearity_and_fe.md) | `TRUE` / `INCONCLUSIVE` | `0` post-24h or unstable predictors; horizon curve `T0=0.66578 -> 1h=0.75777 -> 24h_labs=0.84453 -> All=0.84957`. |
| `I16`–`I16.1` | Cumulative domain FE (`Families A–D`) vs contextual stratum/hosp normalization | [`stage5_collinearity_and_fe.md`](reports/findings/stage5_collinearity_and_fe.md) | `TRUE` / `FALSE` | `fe_ABCD` lifts AUC `0.84957 -> 0.85344` (`+0.00387`, `5/5` folds, `-0.00400` LL); `stratum_z` (`-0.00023`) & `hosp_cov_rel` (`-0.00077`) rejected. |
| `I17`–`I17.1` | Random `SKF` optimism vs `SGKF(hospital_id)` & HGB vs LogReg/Blend | [`stage6_baseline_validation.md`](reports/findings/stage6_baseline_validation.md) | `TRUE` / `TRUE` | `SKF_with_IDs=0.87144`, `SKF_no_IDs=0.86127` vs `SGKF=0.85520` (`Δ_naive=+0.01624`); HGB beats LogReg (`0.82983`, `+0.02537`), 85/15 blend=`0.85561`. |
| `I18`–`I18.2` | Raw Model A vs `fe_ABCD` Model B, hospital generalization & HMECVAS lint | [`stage6_baseline_validation.md`](reports/findings/stage6_baseline_validation.md) | `TRUE` / `TRUE` / `TRUE` | Model B hits `0.85520` AUC (`+0.00381`), `0.37072` LL (`-0.00386`), `0.00650` ECE; `181` hospitals median AUC=`0.86789` (`95.03% >= 0.75`); Train-Test `KS=0.01638`; `38/38` HMECVAS lint clean. |
