# PostTrain Garden

PostTrain Garden is a local-first post-training workbench for small open models.
It turns bad model answers into a repeatable improvement loop: collect regrets,
create preference pairs, judge them, distill SFT examples, train LoRA adapters,
and compare the result against the baseline.

The project is intentionally usable before you install heavy ML dependencies.
Data processing, judging, dry-run evaluation, and report export use the Python
standard library. Training loads `trl`, `peft`, `datasets`, and `transformers`
only when you run a non-dry-run training command.

## Why this exists

Most post-training workflows start with a clean benchmark. Real teams usually
start with a messier thing: prompts where their model failed yesterday.
PostTrain Garden treats those failures as seeds. The `regret queue` becomes the
source of preference data, supervised fine-tuning examples, and regression
checks, so every model update has a memory of what went wrong before.

## Quick start

```powershell
cd D:\桌面文件夹\posttrain-garden
python -m pip install -e .
ptg init examples\demo-garden
ptg ingest examples\demo-garden --sample
ptg duel examples\demo-garden --strategy repair
ptg judge examples\demo-garden --policy rules
ptg distill examples\demo-garden
ptg eval examples\demo-garden --dry-run
ptg train sft examples\demo-garden --dry-run
ptg train dpo examples\demo-garden --dry-run
ptg export examples\demo-garden
```

The exported report will be written to:

```text
examples\demo-garden\reports\posttrain_report.md
```

Without installing, use `PYTHONPATH=src`:

```powershell
$env:PYTHONPATH = "src"
python -m posttrain_garden init examples\demo-garden
```

Training dependencies are optional:

```powershell
python -m pip install -e ".[train]"
ptg train sft examples\demo-garden
```

## Data files

`data/regrets.jsonl`

```json
{"id":"...","prompt":"...","bad_response":"...","reason":"...","tags":["math"],"source":"manual","created_at":"2026-06-22T12:00:00Z"}
```

`data/preferences.jsonl`

```json
{"id":"...","prompt":"...","chosen":"...","rejected":"...","judge":"rules:v1","score":1.0,"tags":["math"]}
```

`data/sft.jsonl`

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"source_preference_id":"...","tags":["math"]}
```

## Training design

The training entrypoints create a dataset, pass a base model id, an explicit
tokenizer (`processing_class`), and an `SFTConfig` or `DPOConfig` to the matching
TRL trainer, and optionally attach a PEFT `LoraConfig`. A missing padding token
falls back to EOS; an existing padding token is preserved. Mixed precision and
external experiment reporting are explicitly disabled so TRL default changes
do not require BF16 hardware or a tracking-service login.

The default model in `garden.yaml` is `Qwen/Qwen3-0.6B`, chosen because the TRL
SFT quickstart uses it as a compact open model for demonstration.

The supported dependency range is `transformers>=4.51.0,<5` and
`trl>=0.17.0,<1`. Transformers 4.51.0 introduced the Qwen3 architecture;
TRL 0.17.0 is our tested baseline for conversational SFT data and the explicit
`processing_class` trainer API. `datasets>=3.0` matches TRL's baseline requirement.
The major-version caps deliberately exclude unvalidated Transformers 5 / TRL 1
upgrades. They do not imply that every v1 SFT/DPO API is incompatible: TRL's
v1 migration also changes nested-null preprocessing, which needs a separate
data-compatibility review before widening the range. We do not silently fall
back to generic `TrainingArguments` on a broken or older TRL installation.

Upstream references:

- [Transformers 4.51.0 release](https://github.com/huggingface/transformers/releases/tag/v4.51.0)
- [TRL 0.17.0 DPO trainer](https://github.com/huggingface/trl/blob/v0.17.0/trl/trainer/dpo_trainer.py)
- [TRL v1 migration guide](https://github.com/huggingface/trl/blob/main/MIGRATION.md)

## Tests

The standard-library suite requires no training dependencies:

```sh
python -m unittest discover -s tests -v
```

After installing `.[train]`, enable the offline CPU integration suite:

```sh
PTG_TEST_TRAIN=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m unittest discover -s tests -v
```

CI runs both the minimum direct dependencies and the newest versions pip can
resolve within the declared ranges. Integration tests build a tiny random Qwen3
and a local tokenizer, train SFT and DPO for one step with and without LoRA,
check finite losses, and verify saved weights and tokenizer files. This checks
the real model/trainer API and project-generated data without downloading model
weights; it is not a full `Qwen/Qwen3-0.6B` checkpoint or GPU/quantization test.

## Repository layout

```text
src/posttrain_garden/   package and CLI
tests/                  stdlib unittest suite
examples/               small sample artifacts
docs/                   design notes
```

## License

Apache-2.0
