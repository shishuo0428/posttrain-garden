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

The training entrypoints follow the current Hugging Face TRL pattern: create a
dataset, pass a base model id and dataset to `SFTTrainer` or `DPOTrainer`, and
optionally attach a PEFT `LoraConfig`.

The default model in `garden.yaml` is `Qwen/Qwen3-0.6B`, chosen because the TRL
SFT quickstart uses it as a compact open model for demonstration.

## Repository layout

```text
src/posttrain_garden/   package and CLI
tests/                  stdlib unittest suite
examples/               small sample artifacts
docs/                   design notes
```

## License

Apache-2.0
