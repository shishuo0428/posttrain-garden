from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .config import load_config
from .jsonl import read_jsonl
from .paths import eval_path, preferences_path, regrets_path, sft_path
from .schema import validate_preference, validate_regret, validate_sft


def run_eval(project_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    config = load_config(project_dir)
    regrets = read_jsonl(regrets_path(project_dir), validate_regret)
    preferences = read_jsonl(preferences_path(project_dir), validate_preference)
    sft_examples = read_jsonl(sft_path(project_dir), validate_sft)
    tag_counts = Counter(tag for record in regrets + preferences for tag in record.get("tags", []))
    scores = [float(record["score"]) for record in preferences]

    result: dict[str, Any] = {
        "project": str(project_dir),
        "base_model": config["base_model"],
        "dry_run": dry_run,
        "counts": {
            "regrets": len(regrets),
            "preferences": len(preferences),
            "sft_examples": len(sft_examples),
        },
        "preference_score_mean": round(mean(scores), 4) if scores else 0.0,
        "tag_counts": dict(sorted(tag_counts.items())),
        "readiness": readiness(regrets, preferences, sft_examples),
    }
    eval_file = eval_path(project_dir)
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def readiness(
    regrets: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    sft_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    issues: list[str] = []
    if not regrets:
        issues.append("No regrets collected yet.")
    if regrets and not preferences:
        issues.append("Regrets exist but no preference pairs were generated.")
    if preferences and not sft_examples:
        issues.append("Preference pairs exist but no SFT examples were distilled.")
    high_quality = [record for record in preferences if float(record.get("score", 0.0)) >= 0.5]
    if preferences and len(high_quality) < max(1, len(preferences) // 2):
        issues.append("Less than half of preference pairs pass the default score threshold.")
    return {
        "ok_for_dry_run": bool(regrets),
        "ok_for_sft": bool(sft_examples),
        "ok_for_dpo": bool(high_quality),
        "issues": issues,
    }
