from __future__ import annotations

from pathlib import Path


def data_dir(project_dir: Path) -> Path:
    return project_dir / "data"


def reports_dir(project_dir: Path) -> Path:
    return project_dir / "reports"


def regrets_path(project_dir: Path) -> Path:
    return data_dir(project_dir) / "regrets.jsonl"


def preferences_path(project_dir: Path) -> Path:
    return data_dir(project_dir) / "preferences.jsonl"


def sft_path(project_dir: Path) -> Path:
    return data_dir(project_dir) / "sft.jsonl"


def eval_path(project_dir: Path) -> Path:
    return reports_dir(project_dir) / "eval.json"


def report_path(project_dir: Path) -> Path:
    return reports_dir(project_dir) / "posttrain_report.md"
