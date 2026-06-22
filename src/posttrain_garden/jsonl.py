from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

Validator = Callable[[dict[str, Any]], dict[str, Any]]


def read_jsonl(path: Path, validator: Validator | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{lineno}: each JSONL line must be an object")
        records.append(validator(record) if validator else record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    path.write_text(body, encoding="utf-8")
    return path


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def upsert_by_id(path: Path, incoming: Iterable[dict[str, Any]], validator: Validator) -> int:
    existing = read_jsonl(path, validator)
    by_id = {str(record["id"]): record for record in existing}
    changed = 0
    for raw_record in incoming:
        record = validator(raw_record)
        record_id = str(record["id"])
        if by_id.get(record_id) != record:
            by_id[record_id] = record
            changed += 1
    write_jsonl(path, by_id.values())
    return changed


def filter_by_tags(records: Iterable[dict[str, Any]], tags: list[str] | None) -> list[dict[str, Any]]:
    if not tags:
        return list(records)
    wanted = set(tags)
    return [record for record in records if wanted.intersection(set(record.get("tags", [])))]
