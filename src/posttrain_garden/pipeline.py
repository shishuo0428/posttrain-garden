from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import save_config
from .jsonl import filter_by_tags, read_jsonl, upsert_by_id, write_jsonl
from .paths import data_dir, preferences_path, regrets_path, reports_dir, sft_path
from .samples import SAMPLE_REGRETS
from .schema import validate_preference, validate_regret, validate_sft


def init_project(project_dir: Path, force: bool = False) -> dict[str, Path]:
    project_dir.mkdir(parents=True, exist_ok=True)
    data_dir(project_dir).mkdir(parents=True, exist_ok=True)
    reports_dir(project_dir).mkdir(parents=True, exist_ok=True)
    config_file = project_dir / "garden.yaml"
    if force or not config_file.exists():
        save_config(project_dir)
    for path in [regrets_path(project_dir), preferences_path(project_dir), sft_path(project_dir)]:
        if force or not path.exists():
            write_jsonl(path, [])
    return {
        "project": project_dir,
        "config": config_file,
        "regrets": regrets_path(project_dir),
        "preferences": preferences_path(project_dir),
        "sft": sft_path(project_dir),
    }


def ingest_sample(project_dir: Path) -> int:
    init_project(project_dir)
    return upsert_by_id(regrets_path(project_dir), SAMPLE_REGRETS, validate_regret)


def ingest_file(project_dir: Path, input_path: Path, source: str = "file") -> int:
    init_project(project_dir)
    if input_path.suffix.lower() in {".jsonl", ".json"}:
        records = _read_structured_input(input_path)
    else:
        records = _read_markdown_input(input_path, source=source)
    return upsert_by_id(regrets_path(project_dir), records, validate_regret)


def create_preferences(
    project_dir: Path,
    strategy: str = "repair",
    tags: list[str] | None = None,
) -> int:
    init_project(project_dir)
    regrets = filter_by_tags(read_jsonl(regrets_path(project_dir), validate_regret), tags)
    preferences = [preference_from_regret(regret, strategy=strategy) for regret in regrets]
    return upsert_by_id(preferences_path(project_dir), preferences, validate_preference)


def preference_from_regret(regret: dict[str, Any], strategy: str = "repair") -> dict[str, Any]:
    prompt = regret["prompt"]
    rejected = regret["bad_response"]
    reason = regret["reason"]
    if strategy == "contrast":
        chosen = (
            "A better answer should directly address the prompt, avoid the recorded mistake, "
            f"and explicitly account for this failure reason: {reason}"
        )
    else:
        chosen = repair_response(prompt=prompt, bad_response=rejected, reason=reason)
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "judge": "duel:synthetic",
        "score": 1.0,
        "tags": regret.get("tags", []),
    }


def repair_response(prompt: str, bad_response: str, reason: str) -> str:
    return (
        f"For the prompt '{prompt}', avoid the previous failure: {reason}. "
        "Give a specific, grounded answer, state the key correction first, "
        f"and do not repeat this rejected response: '{bad_response}'."
    )


def judge_preferences(project_dir: Path, policy: str = "rules") -> int:
    init_project(project_dir)
    preferences = read_jsonl(preferences_path(project_dir), validate_preference)
    judged = [apply_rule_judge(record, policy=policy) for record in preferences]
    write_jsonl(preferences_path(project_dir), judged)
    return len(judged)


def apply_rule_judge(record: dict[str, Any], policy: str = "rules") -> dict[str, Any]:
    chosen = record["chosen"].strip()
    rejected = record["rejected"].strip()
    score = 0.0
    if chosen and chosen != rejected:
        score += 0.45
    if len(chosen) > len(rejected):
        score += 0.25
    if any(word in chosen.lower() for word in ["avoid", "because", "specific", "correct"]):
        score += 0.2
    if len(chosen.split()) >= 12:
        score += 0.1
    judged = dict(record)
    judged["judge"] = f"{policy}:v1"
    judged["score"] = round(min(score, 1.0), 3)
    return validate_preference(judged)


def distill_sft(project_dir: Path, min_score: float = 0.5, tags: list[str] | None = None) -> int:
    init_project(project_dir)
    preferences = filter_by_tags(read_jsonl(preferences_path(project_dir), validate_preference), tags)
    examples = [
        sft_from_preference(preference)
        for preference in preferences
        if float(preference.get("score", 0.0)) >= min_score
    ]
    normalized = [validate_sft(example) for example in examples]
    write_jsonl(sft_path(project_dir), normalized)
    return len(normalized)


def sft_from_preference(preference: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": preference["prompt"]},
            {"role": "assistant", "content": preference["chosen"]},
        ],
        "source_preference_id": preference["id"],
        "tags": preference.get("tags", []),
    }


def _read_structured_input(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix.lower() == ".jsonl":
        return read_jsonl(input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("JSON input must be an object or list of objects")


def _read_markdown_input(input_path: Path, source: str) -> list[dict[str, Any]]:
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    records: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        records.append(
            {
                "prompt": f"Review failed answer from {input_path.name} chunk {index}.",
                "bad_response": chunk,
                "reason": "Imported free-form failure note.",
                "tags": ["imported"],
                "source": source,
            }
        )
    return records
