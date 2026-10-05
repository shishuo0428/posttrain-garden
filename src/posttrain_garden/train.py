from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import load_config
from .paths import preferences_path, reports_dir, sft_path


def build_training_recipe(project_dir: Path, method: str) -> dict[str, Any]:
    config = load_config(project_dir)
    method = method.lower()
    if method not in {"sft", "dpo"}:
        raise ValueError("method must be 'sft' or 'dpo'")
    data_file = sft_path(project_dir) if method == "sft" else preferences_path(project_dir)
    output_dir = project_dir / str(config.get("output_dir", "outputs/adapters/default"))
    return {
        "method": method,
        "base_model": config["base_model"],
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "lora": config.get("lora", {}),
        "quantization": config.get("quantization", {}),
        "trainer": "trl.SFTTrainer" if method == "sft" else "trl.DPOTrainer",
        "notes": [
            "Dry-run recipes do not import ML dependencies.",
            "Install with: python -m pip install -e .[train]",
        ],
    }


def run_train(project_dir: Path, method: str, dry_run: bool = False) -> dict[str, Any]:
    recipe = build_training_recipe(project_dir, method)
    recipe_path = reports_dir(project_dir) / f"train_{method}_recipe.json"
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    if dry_run:
        return {"status": "dry-run", "recipe": recipe, "recipe_path": str(recipe_path)}

    missing = missing_training_dependencies()
    if missing:
        packages = ", ".join(missing)
        raise RuntimeError(
            "Training dependencies are missing: "
            f"{packages}. Install them with: python -m pip install -e .[train]"
        )
    if method.lower() == "sft":
        return _run_sft(project_dir, recipe)
    if method.lower() == "dpo":
        return _run_dpo(project_dir, recipe)
    raise ValueError("method must be 'sft' or 'dpo'")


def missing_training_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name in ["datasets", "peft", "transformers", "trl"]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


def _lora_config(lora: dict[str, Any]):
    from peft import LoraConfig

    if not lora.get("enabled", True):
        return None
    return LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
        task_type="CAUSAL_LM",
    )


def _training_args(method: str, output_dir: str):
    # These configs are required by the supported TRL versions. A generic
    # TrainingArguments fallback hides an incompatible/broken installation.
    if method not in {"sft", "dpo"}:
        raise ValueError("method must be 'sft' or 'dpo'")
    from trl import DPOConfig, SFTConfig

    config_class = SFTConfig if method == "sft" else DPOConfig
    return config_class(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=20,
        logging_steps=1,
        save_steps=20,
        # TRL defaults have changed over time; keep CPU/non-BF16 devices usable.
        bf16=False,
        fp16=False,
        report_to="none",
    )


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Training requires a tokenizer with a pad token or an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _run_sft(project_dir: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    from datasets import load_dataset
    from trl import SFTTrainer

    dataset = load_dataset("json", data_files=recipe["data_file"], split="train")
    trainer = SFTTrainer(
        model=recipe["base_model"],
        args=_training_args("sft", recipe["output_dir"]),
        train_dataset=dataset,
        processing_class=_load_tokenizer(recipe["base_model"]),
        peft_config=_lora_config(recipe["lora"]),
    )
    trainer.train()
    trainer.save_model(recipe["output_dir"])
    return {"status": "trained", "method": "sft", "output_dir": recipe["output_dir"]}


def _run_dpo(project_dir: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    from datasets import load_dataset
    from trl import DPOTrainer

    dataset = load_dataset("json", data_files=recipe["data_file"], split="train")
    trainer = DPOTrainer(
        model=recipe["base_model"],
        ref_model=None,
        args=_training_args("dpo", recipe["output_dir"]),
        train_dataset=dataset,
        processing_class=_load_tokenizer(recipe["base_model"]),
        peft_config=_lora_config(recipe["lora"]),
    )
    trainer.train()
    trainer.save_model(recipe["output_dir"])
    return {"status": "trained", "method": "dpo", "output_dir": recipe["output_dir"]}
