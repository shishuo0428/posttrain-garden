from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from posttrain_garden.pipeline import init_project
from posttrain_garden.train import build_training_recipe, run_train


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


if __name__ == "__main__":
    unittest.main()
