from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from posttrain_garden.jsonl import read_jsonl
from posttrain_garden.paths import preferences_path, regrets_path, sft_path
from posttrain_garden.pipeline import (
    create_preferences,
    distill_sft,
    ingest_sample,
    init_project,
    judge_preferences,
)
from posttrain_garden.schema import validate_preference, validate_regret, validate_sft


class PipelineTests(unittest.TestCase):
    def test_regret_to_sft_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            init_project(project)
            self.assertEqual(ingest_sample(project), 2)
            self.assertEqual(create_preferences(project), 2)
            self.assertEqual(judge_preferences(project), 2)
            self.assertEqual(distill_sft(project), 2)

            regrets = read_jsonl(regrets_path(project), validate_regret)
            preferences = read_jsonl(preferences_path(project), validate_preference)
            sft_examples = read_jsonl(sft_path(project), validate_sft)

        self.assertEqual(len(regrets), 2)
        self.assertEqual(len(preferences), 2)
        self.assertEqual(len(sft_examples), 2)
        self.assertEqual(sft_examples[0]["messages"][0]["role"], "user")
        self.assertEqual(sft_examples[0]["messages"][1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
