from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from posttrain_garden.config import default_config, save_config
from posttrain_garden.pipeline import init_project
from posttrain_garden.train import (
    _load_tokenizer,
    _training_args,
    build_training_recipe,
    run_train,
)


class TrainTests(unittest.TestCase):
    def test_training_recipes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            init_project(project)
            sft = build_training_recipe(project, "sft")
            dpo = build_training_recipe(project, "dpo")
            dry_run = run_train(project, "sft", dry_run=True)

        self.assertEqual(sft["trainer"], "trl.SFTTrainer")
        self.assertEqual(dpo["trainer"], "trl.DPOTrainer")
        self.assertEqual(dry_run["status"], "dry-run")

    def test_dry_run_never_imports_ml_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {name: None for name in ("torch", "datasets", "peft", "transformers", "trl")}
        ):
            for method in ("sft", "dpo"):
                self.assertEqual(run_train(Path(tmp), method, dry_run=True)["status"], "dry-run")

    def test_missing_dependencies_are_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"trl": None}), patch(
            "posttrain_garden.train.missing_training_dependencies", return_value=["trl"]
        ):
            with self.assertRaisesRegex(RuntimeError, r"trl.*pip install -e .\[train\]"):
                run_train(Path(tmp), "dpo")

    def test_configs_do_not_fall_back_to_training_arguments(self) -> None:
        with patch.dict(sys.modules, {"trl": SimpleNamespace(), "transformers": Mock()}):
            for method in ("sft", "dpo"):
                with self.assertRaises(ImportError):
                    _training_args(method, "unused")
        with self.assertRaises(ValueError):
            _training_args("unknown", "unused")

    def test_tokenizer_padding(self) -> None:
        for pad_id, eos_id in ((0, 2), (None, 2), (None, None)):
            with self.subTest(pad_id=pad_id, eos_id=eos_id):
                tokenizer = SimpleNamespace(
                    pad_token_id=pad_id,
                    pad_token="<pad>" if pad_id is not None else None,
                    eos_token_id=eos_id,
                    eos_token="<eos>" if eos_id is not None else None,
                )
                auto = Mock()
                auto.from_pretrained.return_value = tokenizer
                with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=auto)}):
                    if pad_id is None and eos_id is None:
                        with self.assertRaisesRegex(ValueError, "pad token or an EOS token"):
                            _load_tokenizer("local-model")
                    else:
                        self.assertIs(_load_tokenizer("local-model"), tokenizer)
                        self.assertEqual(tokenizer.pad_token, "<pad>" if pad_id is not None else "<eos>")
                auto.from_pretrained.assert_called_once_with("local-model")

    def test_training_entrypoints_wire_supported_api(self) -> None:
        for method in ("sft", "dpo"):
            for lora_enabled in (True, False):
                with self.subTest(method=method, lora=lora_enabled), tempfile.TemporaryDirectory() as tmp:
                    project = Path(tmp)
                    config = default_config()
                    config["lora"]["enabled"] = lora_enabled
                    save_config(project, config)
                    dataset, tokenizer, adapter = object(), object(), object()
                    trainer = Mock()
                    config_factory = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
                    load_dataset, lora = Mock(return_value=dataset), Mock(return_value=adapter)
                    modules = {
                        "datasets": SimpleNamespace(load_dataset=load_dataset),
                        "peft": SimpleNamespace(LoraConfig=lora),
                        "trl": SimpleNamespace(**{
                            method.upper() + "Trainer": trainer,
                            "SFTConfig": config_factory,
                            "DPOConfig": config_factory,
                        }),
                    }
                    with patch.dict(sys.modules, modules), patch(
                        "posttrain_garden.train.missing_training_dependencies", return_value=[]
                    ), patch("posttrain_garden.train._load_tokenizer", return_value=tokenizer) as load_tokenizer:
                        result = run_train(project, method)
                    kwargs = trainer.call_args.kwargs
                    self.assertEqual(kwargs["model"], "Qwen/Qwen3-0.6B")
                    self.assertIs(kwargs["processing_class"], tokenizer)
                    self.assertIs(kwargs["train_dataset"], dataset)
                    self.assertIs(kwargs["peft_config"], adapter if lora_enabled else None)
                    self.assertFalse(kwargs["args"].bf16)
                    self.assertFalse(kwargs["args"].fp16)
                    self.assertEqual(kwargs["args"].max_steps, 20)
                    self.assertEqual(kwargs["args"].report_to, "none")
                    if method == "dpo":
                        self.assertIsNone(kwargs["ref_model"])
                    load_tokenizer.assert_called_once_with("Qwen/Qwen3-0.6B")
                    filename = "sft.jsonl" if method == "sft" else "preferences.jsonl"
                    load_dataset.assert_called_once_with(
                        "json", data_files=str(project / "data" / filename), split="train"
                    )
                    trainer.return_value.train.assert_called_once_with()
                    trainer.return_value.save_model.assert_called_once_with(result["output_dir"])
                    self.assertEqual(result["status"], "trained")


if __name__ == "__main__":
    unittest.main()
