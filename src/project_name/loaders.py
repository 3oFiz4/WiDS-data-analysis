"""Data fetching (Kaggle API via .env) and shared cleaning helpers."""

import hashlib
import os
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from project_name.config import EXPECTED_CHECKSUMS, RAW_DATA_DIR


def verify_checksums(data_dir: Path = RAW_DATA_DIR) -> None:
    """Verify SHA-256 checksums for files listed in EXPECTED_CHECKSUMS."""
    for filename, expected_hash in EXPECTED_CHECKSUMS.items():
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Expected data file not found: {file_path}")
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )


@contextmanager
def _env_kaggle_auth_scope(temp_home: Path):
    """Isolate HOME/USERPROFILE and map KGAT_ keys to KAGGLE_API_TOKEN when using .env."""
    load_dotenv()
    kaggle_key = os.getenv("KAGGLE_KEY", "")
    old_token = os.environ.get("KAGGLE_API_TOKEN")
    if kaggle_key.startswith("KGAT_") and not old_token:
        os.environ["KAGGLE_API_TOKEN"] = kaggle_key

    if os.getenv("KAGGLE_USERNAME") and kaggle_key:
        old_home = os.environ.get("HOME")
        old_userprofile = os.environ.get("USERPROFILE")
        os.environ["HOME"] = str(temp_home)
        os.environ["USERPROFILE"] = str(temp_home)
        try:
            yield
        finally:
            if old_token is None:
                os.environ.pop("KAGGLE_API_TOKEN", None)
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
            if old_userprofile is None:
                os.environ.pop("USERPROFILE", None)
            else:
                os.environ["USERPROFILE"] = old_userprofile
    else:
        yield


def download_kaggle_dataset(dataset_slug: str, dest_dir: Path = RAW_DATA_DIR) -> Path:
    """Authenticate via .env and download/unzip a Kaggle dataset into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with _env_kaggle_auth_scope(dest_dir):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True)
    verify_checksums(dest_dir)
    return dest_dir


def download_kaggle_competition(competition_slug: str, dest_dir: Path = RAW_DATA_DIR) -> Path:
    """Authenticate via .env and download/unzip a Kaggle competition dataset into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with _env_kaggle_auth_scope(dest_dir):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(competition_slug, path=str(dest_dir))
    zip_path = dest_dir / f"{competition_slug}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()
    verify_checksums(dest_dir)
    return dest_dir


def load_csv(filename: str, data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load a CSV from data_dir with basic whitespace/column normalization."""
    df = pd.read_csv(data_dir / filename)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df
