#!/usr/bin/env python3
"""Run one documented external RT recipe with explicitly supplied data paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "src/main/resources/py/v2/ai.py"


def build_command(args: argparse.Namespace, config: dict) -> list[str]:
    """Resolve one recipe without touching inputs or starting the trainer."""
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported recipe schema; expected schema_version 1")
    for key in ("recipes", "trainer_arguments"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Recipe configuration requires a {key} mapping")
    for key in ("targets", "seeds"):
        if not isinstance(config.get(key), list) or not config[key]:
            raise ValueError(f"Recipe configuration requires a nonempty {key} list")
    if args.recipe not in config["recipes"]:
        raise ValueError(f"Unknown recipe: {args.recipe}; see recipes in {args.config}")
    if args.target not in config["targets"] or args.seed not in config["seeds"]:
        raise ValueError("Target or seed is not part of this experiment configuration")
    recipe_target = config.get("recipe_targets", {}).get(args.recipe)
    if recipe_target and args.target != recipe_target:
        raise ValueError(f"Recipe {args.recipe} requires target {recipe_target}")
    scopes = config.get("target_scopes")
    if scopes and args.scope not in scopes[args.target]:
        raise ValueError(f"Select --scope from {scopes[args.target]} for {args.target}")
    if not scopes and args.scope:
        raise ValueError("--scope applies only to source-specific configurations")
    options = dict(config["trainer_arguments"])
    options.update(config["recipes"][args.recipe])
    options.update(
        in_dir=str(args.in_dir.resolve()),
        rt_validation_file=str(args.validation_file.resolve()),
        out_dir=str(args.out_dir.resolve()),
        device=args.device,
        seed=args.seed,
    )
    if args.rt_model:
        options["rt_model"] = str(args.rt_model.resolve())
    command = [sys.executable, str(TRAINER)]
    for key, value in options.items():
        if value is True:
            command.append(f"--{key}")
        elif value is not False and value is not None:
            command.extend(
                (
                    f"--{key}",
                    format(value, ".12g") if isinstance(value, float) else str(value),
                )
            )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument(
        "--target", required=True, help="Target coordinate of both supplied TSVs"
    )
    parser.add_argument(
        "--scope", help="Training source, or combined, for source-specific recipes"
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rt-model", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without accessing data or starting training",
    )
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Recipe configuration must be a JSON object")
        command = build_command(args, config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    required = [TRAINER, args.in_dir / "rt_train_data.tsv", args.validation_file]
    if args.rt_model:
        required.append(args.rt_model)
    for path in required:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if args.out_dir.exists() and not args.out_dir.is_dir():
        parser.error("Output path must be a directory")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(
            "Output directory is not empty; choose a separate directory for this run"
        )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
