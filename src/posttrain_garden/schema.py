from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any


REGRET_FIELDS = {"id", "prompt", "bad_response", "reason", "tags", "source", "created_at"}
PREFERENCE_FIELDS = {"id", "prompt", "chosen", "rejected", "judge", "score", "tags"}
SFT_FIELDS = {"messages", "source_preference_id", "tags"}


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
    material = "\n".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    raise ValueError("tags must be a list or comma-separated string")


def validate_regret(record: dict[str, Any]) -> dict[str, Any]:
    prompt = required_text(record, "prompt")
    bad_response = required_text(record, "bad_response")
    reason = str(record.get("reason", "")).strip() or "unspecified"
    source = str(record.get("source", "")).strip() or "manual"
    created_at = str(record.get("created_at", "")).strip() or now_iso()
    tags = normalize_tags(record.get("tags"))
    regret_id = str(record.get("id", "")).strip() or stable_id(
        "regret", prompt, bad_response, reason, source
    )
    return {
        "id": regret_id,
        "prompt": prompt,
        "bad_response": bad_response,
        "reason": reason,
        "tags": tags,
        "source": source,
        "created_at": created_at,
    }


def validate_preference(record: dict[str, Any]) -> dict[str, Any]:
    prompt = required_text(record, "prompt")
    chosen = required_text(record, "chosen")
    rejected = required_text(record, "rejected")
    judge = str(record.get("judge", "")).strip() or "manual"
    score = float(record.get("score", 1.0))
    tags = normalize_tags(record.get("tags"))
    pref_id = str(record.get("id", "")).strip() or stable_id("pref", prompt, chosen, rejected)
    return {
        "id": pref_id,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "judge": judge,
        "score": score,
        "tags": tags,
    }


def validate_sft(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = required_text(message, "role")
        content = required_text(message, "content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role}")
        normalized_messages.append({"role": role, "content": content})
    source_preference_id = str(record.get("source_preference_id", "")).strip()
    tags = normalize_tags(record.get("tags"))
    return {
        "messages": normalized_messages,
        "source_preference_id": source_preference_id,
        "tags": tags,
    }


def required_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
