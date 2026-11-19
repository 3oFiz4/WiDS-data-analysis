"""Shared HMECVAS inquiry registry and data-provenance helpers."""

from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from project_name.config import PROJECT_ROOT, RAW_DATA_DIR

INQUIRIES_PATH = PROJECT_ROOT / "reports" / "findings" / "inquiries.yaml"
TRAIN_PATH = RAW_DATA_DIR / "TrainingWiDS2021.csv"
TEST_PATH = RAW_DATA_DIR / "UnlabeledWiDS2021.csv"
DICT_PATH = RAW_DATA_DIR / "DataDictionaryWiDS2021.csv"


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    """Return short git commit hash of HEAD."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT)
        .decode("utf-8")
        .strip()
    )


def iso_now_plus8() -> str:
    """Return current ISO-8601 timestamp with +08:00 timezone offset."""
    tz = timezone(offset=timedelta(hours=8))
    return datetime.now(tz).isoformat(timespec="seconds")


def load_raw_wids() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw TrainingWiDS2021, UnlabeledWiDS2021, and DataDictionaryWiDS2021."""
    read_kwargs = {"na_values": ["NA", ""], "keep_default_na": True, "low_memory": False}
    train = pd.read_csv(TRAIN_PATH, **read_kwargs)
    test = pd.read_csv(TEST_PATH, **read_kwargs)
    dd = pd.read_csv(DICT_PATH)
    return train, test, dd


def make_inquiry(
    *,
    inq_id: str,
    status: str,
    polarity: str,
    claim: str,
    technique: str,
    parameters: dict[str, Any],
    assumptions: list[str],
    missing_data_policy: str,
    script_ref: str,
    ev_type: str,
    metric: str,
    formula: str,
    timeframe: str,
    filters: list[str],
    outcome: str,
    on_true_cond: str,
    on_false_cond: str,
    on_inconc_cond: str,
    act_true: str,
    act_false: str,
    act_inconc: str,
    source_file: str,
    commit: str,
    sha256: str,
    columns: list[str],
    timestamp: str,
) -> dict[str, Any]:
    """Build one schema-compliant HMECVAS Inquiry Unit dictionary."""
    return {
        "id": inq_id,
        "status": status,
        "hypothesis": {"polarity": polarity, "claim": claim},
        "methodology": {
            "technique": technique,
            "parameters": parameters,
            "assumptions": assumptions,
            "missing_data_policy": missing_data_policy,
            "script_ref": script_ref,
        },
        "evidence": {"type": ev_type, "metric": metric, "formula": formula},
        "condition": {"timeframe": timeframe, "filters": filters},
        "verdict": {
            "outcome": outcome,
            "true": on_true_cond,
            "false": on_false_cond,
            "inconclusive": on_inconc_cond,
        },
        "action": {
            "on_true": act_true,
            "on_false": act_false,
            "on_inconclusive": act_inconc,
        },
        "source": {
            "file": source_file,
            "version": {"commit": commit, "sha256": sha256},
            "columns": columns,
            "extraction": {"script": script_ref, "timestamp": timestamp},
        },
    }


_DOUBLE_QUOTE_RE = re.compile(
    r"^(?:OPEN|TRUE|FALSE|INCONCLUSIVE|SKIPPED|"
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2}))$"
)


class _HmecvasSafeDumper(yaml.SafeDumper):
    """YAML dumper quoting status/outcome/timestamp strings with double quotes."""


def _represent_hmecvas_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Emit double quotes around HMECVAS status/outcome/timestamp scalars."""
    if _DOUBLE_QUOTE_RE.match(data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_HmecvasSafeDumper.add_representer(str, _represent_hmecvas_str)


def upsert_inquiries(
    entries: list[dict[str, Any]],
    *,
    problem: str | None = None,
    path: Path = INQUIRIES_PATH,
) -> None:
    """Insert or replace HMECVAS inquiry units in inquiries.yaml preserving ID order."""
    if path.exists():
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        doc = {}
    if problem is not None:
        doc["problem"] = problem
    existing: list[dict[str, Any]] = doc.get("inquiries", [])
    by_id = {item["id"]: item for item in existing}
    order = [item["id"] for item in existing]
    for entry in entries:
        eid = entry["id"]
        by_id[eid] = entry
        if eid not in order:
            order.append(eid)
    doc["inquiries"] = [by_id[eid] for eid in order]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(doc, Dumper=_HmecvasSafeDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
