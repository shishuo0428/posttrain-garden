from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .eval import run_eval
from .export import export_project
from .pipeline import (
    create_preferences,
    distill_sft,
    ingest_file,
    ingest_sample,
    init_project,
    judge_preferences,
)
from .train import run_train


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    try:
        result = args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if result is not None:
        print_result(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptg",
        description="Local-first regret-to-preference post-training workbench.",
    )
    parser.add_argument("--version", action="version", version=f"posttrain-garden {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize a garden project.")
    init_parser.add_argument("project", type=Path)
    init_parser.add_argument("--force", action="store_true", help="Overwrite config and data files.")
    init_parser.set_defaults(func=cmd_init)

    ingest_parser = subparsers.add_parser("ingest", help="Import regrets.")
    ingest_parser.add_argument("project", type=Path)
    ingest_parser.add_argument("input", type=Path, nargs="?")
    ingest_parser.add_argument("--sample", action="store_true", help="Load bundled sample regrets.")
    ingest_parser.add_argument("--source", default="file", help="Source label for free-form imports.")
    ingest_parser.set_defaults(func=cmd_ingest)

    duel_parser = subparsers.add_parser("duel", help="Create preference pairs from regrets.")
    duel_parser.add_argument("project", type=Path)
    duel_parser.add_argument("--strategy", choices=["repair", "contrast"], default="repair")
    duel_parser.add_argument("--tag", action="append", dest="tags", help="Only process records with tag.")
    duel_parser.set_defaults(func=cmd_duel)

    judge_parser = subparsers.add_parser("judge", help="Score preference pairs.")
    judge_parser.add_argument("project", type=Path)
    judge_parser.add_argument("--policy", default="rules")
    judge_parser.set_defaults(func=cmd_judge)

    distill_parser = subparsers.add_parser("distill", help="Distill preference pairs into SFT examples.")
    distill_parser.add_argument("project", type=Path)
    distill_parser.add_argument("--min-score", type=float, default=0.5)
    distill_parser.add_argument("--tag", action="append", dest="tags", help="Only process records with tag.")
    distill_parser.set_defaults(func=cmd_distill)

    eval_parser = subparsers.add_parser("eval", help="Run lightweight evaluation.")
    eval_parser.add_argument("project", type=Path)
    eval_parser.add_argument("--dry-run", action="store_true")
    eval_parser.set_defaults(func=cmd_eval)

    train_parser = subparsers.add_parser("train", help="Train or dry-run a post-training recipe.")
    train_subparsers = train_parser.add_subparsers(dest="method", required=True)
    for method in ["sft", "dpo"]:
        method_parser = train_subparsers.add_parser(method, help=f"Run {method.upper()} training.")
        method_parser.add_argument("project", type=Path)
        method_parser.add_argument("--dry-run", action="store_true")
        method_parser.set_defaults(func=cmd_train)

    export_parser = subparsers.add_parser("export", help="Export cards and report.")
    export_parser.add_argument("project", type=Path)
    export_parser.set_defaults(func=cmd_export)

    return parser


def cmd_init(args: argparse.Namespace) -> dict[str, str]:
    result = init_project(args.project, force=args.force)
    return {key: str(value) for key, value in result.items()}


def cmd_ingest(args: argparse.Namespace) -> dict[str, Any]:
    if args.sample:
        count = ingest_sample(args.project)
    elif args.input:
        count = ingest_file(args.project, args.input, source=args.source)
    else:
        raise ValueError("provide an input file or use --sample")
    return {"ingested_or_updated": count, "project": str(args.project)}


def cmd_duel(args: argparse.Namespace) -> dict[str, Any]:
    count = create_preferences(args.project, strategy=args.strategy, tags=args.tags)
    return {"preferences_created_or_updated": count, "strategy": args.strategy}


def cmd_judge(args: argparse.Namespace) -> dict[str, Any]:
    count = judge_preferences(args.project, policy=args.policy)
    return {"preferences_judged": count, "policy": args.policy}


def cmd_distill(args: argparse.Namespace) -> dict[str, Any]:
    count = distill_sft(args.project, min_score=args.min_score, tags=args.tags)
    return {"sft_examples": count, "min_score": args.min_score}


def cmd_eval(args: argparse.Namespace) -> dict[str, Any]:
    return run_eval(args.project, dry_run=args.dry_run)


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    return run_train(args.project, args.method, dry_run=args.dry_run)


def cmd_export(args: argparse.Namespace) -> dict[str, str]:
    result = export_project(args.project)
    return {key: str(value) for key, value in result.items()}


def print_result(result: Any) -> None:
    if isinstance(result, (dict, list)):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)
