"""Opt-in, offline CPU tests against real Transformers/TRL (no Hub downloads)."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@unittest.skipUnless(os.environ.get("PTG_TEST_TRAIN") == "1", "set PTG_TEST_TRAIN=1 with .[train] installed")
class TrainingIntegrationTests(unittest.TestCase):
    def test_dependency_boundaries(self) -> None:
        import tomllib
        from packaging.requirements import Requirement

        requirements = {
            r.name: r.specifier
            for r in map(Requirement, tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["train"])
        }
        for name, accepted, rejected in (
            ("transformers", ("4.51.0", "4.57.0"), ("4.43.0", "4.50.3", "5.0.0", "5.0.0rc1")),
            ("trl", ("0.17.0", "0.26.0"), ("0.9.0", "0.16.1", "1.0.0", "1.15.0.dev0")),
        ):
            for version in accepted:
                self.assertIn(version, requirements[name])
            for version in rejected:
                self.assertNotIn(version, requirements[name])

    def test_qwen3_sft_and_dpo_train_and_save(self) -> None:
        import torch
        import trl
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config
        from posttrain_garden import train
        from posttrain_garden.config import default_config, save_config
        from posttrain_garden.pipeline import (
            init_project, ingest_sample, create_preferences, judge_preferences, distill_sft,
        )

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "tiny-qwen3"
            # Exercise the actual Qwen3 AutoModel mapping and PEFT target modules.
            model = AutoModelForCausalLM.from_config(Qwen3Config(
                vocab_size=16, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                head_dim=8, max_position_embeddings=256,
                bos_token_id=1, eos_token_id=2, pad_token_id=0,
            ))
            model.save_pretrained(model_dir)
            backend = Tokenizer(WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
                                           "user": 4, "assistant": 5}, unk_token="<unk>"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                               bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")
            tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + eos_token }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
            tokenizer.save_pretrained(model_dir)
            original_args = train._training_args

            def one_step(method, output_dir):
                args = original_args(method, output_dir)
                args.max_steps = 1
                args.gradient_accumulation_steps = 1
                args.use_cpu = True
                args.gradient_checkpointing = False
                args.max_length = 128
                args.save_strategy = "no"
                args.disable_tqdm = True
                return args

            for method in ("sft", "dpo"):
                for enabled in (True, False):
                    with self.subTest(method=method, lora=enabled):
                        project = root / f"{method}-{enabled}"
                        init_project(project)
                        config = default_config()
                        config["base_model"] = str(model_dir)
                        config["lora"]["enabled"] = enabled
                        save_config(project, config)
                        ingest_sample(project)
                        create_preferences(project)
                        judge_preferences(project)
                        distill_sft(project)
                        trainer_class = getattr(trl, method.upper() + "Trainer")
                        trainers = []

                        def capture_trainer(**kwargs):
                            instance = trainer_class(**kwargs)
                            trainers.append(instance)
                            return instance

                        with patch.object(train, "_training_args", side_effect=one_step), patch.object(
                            trl, method.upper() + "Trainer", side_effect=capture_trainer
                        ):
                            result = train.run_train(project, method)
                        self.assertEqual(result["status"], "trained")
                        self.assertEqual(trainers[0].state.global_step, 1)
                        losses = [entry["loss"] for entry in trainers[0].state.log_history if "loss" in entry]
                        self.assertTrue(losses)
                        self.assertTrue(all(math.isfinite(loss) for loss in losses))
                        output = Path(result["output_dir"])
                        self.assertTrue((output / ("adapter_model.safetensors" if enabled else "model.safetensors")).exists())
                        self.assertTrue((output / "tokenizer_config.json").exists())


if __name__ == "__main__":
    unittest.main()
