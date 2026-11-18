"""Run automated EDA (ydata-profiling and AutoViz) on TrainingWiDS2021.csv."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class
from ydata_profiling import ProfileReport

from project_name.config import PROJECT_ROOT, RAW_DATA_DIR

REPORTS_FINDINGS_DIR = PROJECT_ROOT / "reports" / "findings"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main() -> None:
    data_path = RAW_DATA_DIR / "TrainingWiDS2021.csv"
    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    print(f"Loaded shape: {df.shape}")

    REPORTS_FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pandas Profiling (ydata-profiling)
    profile_path = REPORTS_FINDINGS_DIR / "wids2021_pandas_profiling.html"
    print(f"Generating Pandas Profiling report -> {profile_path} ...")
    profile = ProfileReport(
        df,
        title="WiDS Datathon 2021 - Automated EDA (Pandas Profiling)",
        minimal=True,
    )
    profile.to_file(profile_path)
    print(f"Saved Pandas Profiling report to {profile_path}")

    # 2. AutoViz
    print(f"Running AutoViz (target='diabetes_mellitus') -> {REPORTS_FIGURES_DIR} ...")
    av = AutoViz_Class()
    av.AutoViz(
        filename="",
        sep=",",
        depVar="diabetes_mellitus",
        dfte=df,
        header=0,
        verbose=2,
        lowess=False,
        chart_format="png",
        max_rows_analyzed=150000,
        max_cols_analyzed=30,
        save_plot_dir=str(REPORTS_FIGURES_DIR),
    )
    print(f"Saved AutoViz charts under {REPORTS_FIGURES_DIR}")


if __name__ == "__main__":
    main()
