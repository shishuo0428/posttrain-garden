from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "base_model": "Qwen/Qwen3-0.6B",
    "output_dir": "outputs/adapters/default",
    "train_method": "sft",
    "lora": {
        "enabled": True,
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    },
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
    },
    "eval_sets": ["data/regrets.jsonl"],
    "judge_policy": {
        "default": "rules",
        "min_score": 0.5,
    },
}


def default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def config_path(project_dir: Path) -> Path:
    return project_dir / "garden.yaml"


def load_config(project_dir: Path) -> dict[str, Any]:
    path = config_path(project_dir)
    if not path.exists():
        return default_config()
    loaded = loads_yaml(path.read_text(encoding="utf-8"))
    return merge_defaults(default_config(), loaded)


def save_config(project_dir: Path, config: dict[str, Any] | None = None) -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    path = config_path(project_dir)
    path.write_text(dumps_yaml(config or default_config()), encoding="utf-8")
    return path


def merge_defaults(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def dumps_yaml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    _dump_mapping(data, lines, indent=0)
    return "\n".join(lines) + "\n"


def _dump_mapping(data: dict[str, Any], lines: list[str], indent: int) -> None:
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            _dump_mapping(value, lines, indent + 2)
        else:
            lines.append(f"{pad}{key}: {_format_scalar(value)}")


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(str(value), ensure_ascii=False)


def loads_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"garden.yaml line {lineno}: indentation must use two spaces")
        stripped = line.strip()
        key, sep, value = stripped.partition(":")
        if not sep or not key:
            raise ValueError(f"garden.yaml line {lineno}: expected 'key: value'")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            raise ValueError(f"garden.yaml line {lineno}: invalid indentation")
        parent = stack[-1][1]
        value = value.strip()
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if raw.startswith("[") or raw.startswith('"'):
        return json.loads(raw)
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw
