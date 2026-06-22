from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from posttrain_garden.config import default_config, dumps_yaml, load_config, save_config


class ConfigTests(unittest.TestCase):
    def test_yaml_round_trip_and_defaults(self) -> None:
        config = default_config()
        text = dumps_yaml(config)
        self.assertIn('base_model: "Qwen/Qwen3-0.6B"', text)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            save_config(project, config)
            loaded = load_config(project)
        self.assertEqual(loaded["base_model"], "Qwen/Qwen3-0.6B")
        self.assertEqual(loaded["lora"]["r"], 16)
        self.assertEqual(loaded["quantization"]["load_in_4bit"], True)


if __name__ == "__main__":
    unittest.main()
