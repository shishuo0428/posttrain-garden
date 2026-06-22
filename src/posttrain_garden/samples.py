from __future__ import annotations

from .schema import stable_id


SAMPLE_REGRETS: list[dict[str, object]] = [
    {
        "id": stable_id(
            "regret",
            "Explain why LoRA can reduce fine-tuning memory.",
            "LoRA makes the model smaller by deleting layers.",
        ),
        "prompt": "Explain why LoRA can reduce fine-tuning memory.",
        "bad_response": "LoRA makes the model smaller by deleting layers.",
        "reason": "Confuses low-rank adapters with pruning.",
        "tags": ["lora", "accuracy"],
        "source": "sample",
        "created_at": "2026-06-22T00:00:00Z",
    },
    {
        "id": stable_id(
            "regret",
            "Turn this bug report into a regression test plan.",
            "Just run all tests and hope it is fixed.",
        ),
        "prompt": "Turn this bug report into a regression test plan.",
        "bad_response": "Just run all tests and hope it is fixed.",
        "reason": "Too vague; no acceptance criteria.",
        "tags": ["planning", "eval"],
        "source": "sample",
        "created_at": "2026-06-22T00:00:00Z",
    },
]
