from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_ptg(self, *args: str) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        proc = subprocess.run(
            [sys.executable, "-m", "posttrain_garden", *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(proc.stdout)

    def test_cli_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "garden"
            init_result = self.run_ptg("init", str(project))
            self.assertIn("config", init_result)
            self.assertEqual(self.run_ptg("ingest", str(project), "--sample")["ingested_or_updated"], 2)
            self.assertEqual(self.run_ptg("duel", str(project))["preferences_created_or_updated"], 2)
            self.assertEqual(self.run_ptg("judge", str(project))["preferences_judged"], 2)
            self.assertEqual(self.run_ptg("distill", str(project))["sft_examples"], 2)
            eval_result = self.run_ptg("eval", str(project), "--dry-run")
            self.assertEqual(eval_result["counts"]["sft_examples"], 2)
            train_result = self.run_ptg("train", "sft", str(project), "--dry-run")
            self.assertEqual(train_result["status"], "dry-run")
            export_result = self.run_ptg("export", str(project))
            self.assertIn("report", export_result)


if __name__ == "__main__":
    unittest.main()
