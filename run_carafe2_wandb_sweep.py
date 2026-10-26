#!/usr/bin/env python3
"""Time-bounded random W&B sweep for prepared Carafe2 fine-tuning data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AI_SCRIPT = ROOT / "src" / "main" / "resources" / "py" / "v2" / "ai.py"
EVALUATE_SCRIPT = ROOT / "evaluate_organism_tests.py"
DEFAULT_SWEEP = ROOT / "configs" / "carafe2_finetune_sweep.json"
DEFAULT_WANDB = ROOT / "configs" / "wandb.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a random Carafe2 fine-tuning W&B sweep")
    parser.add_argument("--hours", type=float, required=True, help="Wall-time budget; a trial is never interrupted")
    parser.add_argument("--in-dir", type=Path, required=True, help="Prepared Carafe output containing psm_pdv.txt and rt_train_data.tsv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-dir", type=Path,
        help="Optional directory containing combined.tsv plus celegans/triplemix/human.tsv",
    )
    parser.add_argument(
        "--rt-alignment", type=Path,
        help="Training-only RT alignment JSON (default: <validation-dir>/rt_alignment.json)",
    )
    parser.add_argument("--python", default=sys.executable, help="Python from the Carafe/AlphaPeptDeep environment")
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--configuration-map", type=Path,
                        help="Optional JSON map from a sweep configuration name to fixed parameters")
    parser.add_argument("--wandb-config", type=Path, default=DEFAULT_WANDB)
    parser.add_argument("--sweep-id", help="Resume an existing W&B sweep")
    parser.add_argument("--max-runs", type=int, default=1000)
    parser.add_argument("--seeds", default="2024",
                        help="Comma-separated seeds per sampled configuration (default: 2024)")
    parser.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    parser.add_argument("--tf-type", default="all", choices=("all", "ms2", "rt", "ccs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_input(in_dir: Path, tf_type: str):
    required = ["rt_train_data.tsv"] if tf_type in {"all", "rt"} else []
    if tf_type in {"all", "ms2"}:
        required.extend(["psm_pdv.txt", "fragment_intensity_df.tsv"])
    missing = [str(in_dir / name) for name in required if not (in_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing prepared fine-tuning input:\n" + "\n".join(missing))


def main():
    args = parse_args()
    if args.hours <= 0 or args.max_runs < 1:
        raise SystemExit("--hours and --max-runs must be positive")
    require_input(args.in_dir, args.tf_type)
    validation_split = None
    rt_alignment = None
    if args.validation_dir:
        validation_split = args.validation_dir / "combined.tsv"
        validation_inputs = [
            args.validation_dir / f"{name}.tsv"
            for name in ("celegans", "triplemix", "human")
        ]
        rt_alignment = args.rt_alignment or args.validation_dir / "rt_alignment.json"
        missing_validation = [
            str(path)
            for path in (validation_split, *validation_inputs, rt_alignment)
            if not path.is_file()
        ]
        if missing_validation:
            raise SystemExit("Missing sweep validation input(s):\n" + "\n".join(missing_validation))
    elif args.rt_alignment:
        raise SystemExit("--rt-alignment requires --validation-dir")
    try:
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit("--seeds must be a comma-separated list of integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must contain at least one unique integer")
    if not AI_SCRIPT.is_file():
        raise SystemExit(f"Bundled fine-tuning script not found: {AI_SCRIPT}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # W&B otherwise creates ./wandb in the source checkout for the controller
    # and its agent.  This makes every sweep fully self-contained in its shared
    # output directory.
    wandb_controller_dir = args.output_dir / "wandb_controller"
    wandb_controller_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_controller_dir.resolve())
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("Install wandb in the Carafe Python environment first: python -m pip install 'wandb>=0.17,<1'") from exc
    try:
        from wandb.errors import CommError
    except ImportError:  # W&B < 0.19
        from wandb.errors.errors import CommError
    wandb_settings = json.loads(args.wandb_config.read_text(encoding="utf-8"))
    sweep_definition = json.loads(args.sweep_config.read_text(encoding="utf-8"))
    expected_metric = "final/rt/validation_composite_mae"
    if sweep_definition.get("metric", {}).get("name") == expected_metric and not args.validation_dir:
        raise SystemExit(f"Sweep metric {expected_metric} requires --validation-dir")
    configuration_map = {}
    if args.configuration_map:
        configuration_map = json.loads(args.configuration_map.read_text(encoding="utf-8"))
    if args.dry_run:
        print("Dry run complete; no W&B sweep was created.")
        return
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep_definition, project=wandb_settings["project"], entity=wandb_settings.get("entity"))
    (args.output_dir / "sweep_id.txt").write_text(sweep_id + "\n", encoding="utf-8")
    deadline = time.monotonic() + args.hours * 3600
    completed = 0

    def run_seed_trial(config, run_id, run_name, seed, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            args.python, "-u", str(AI_SCRIPT), "--in_dir", str(args.in_dir), "--out_dir", str(run_dir),
            "--device", args.device, "--tf_type", args.tf_type, "--seed", str(seed), "--verbose", "1",
            "--use_best_model",
            "--wandb-config", str(args.wandb_config), "--wandb-run-id", run_id,
            "--wandb-run-name", run_name, "--wandb-group", args.output_dir.name,
        ]
        if validation_split is not None:
            command.extend([
                "--rt_validation_file", str(validation_split), "--no_adjust_rt_batch_size",
            ])
        for option in (
            "rt_lr", "rt_batch_size", "rt_epoch", "rt_warmup_epoch",
            "rt_heavy_nterm_oversample", "rt_heavy_lysine_oversample",
            "rt_heavy_both_oversample", "rt_heavy_nterm_weight",
            "rt_heavy_lysine_weight", "rt_heavy_both_weight",
            "rt_loss", "rt_huber_delta",
            "rt_trainable", "rt_backbone_lr", "rt_weight_decay",
        ):
            if option in config:
                command.extend([f"--{option}", str(config[option])])
        if args.tf_type in {"all", "ms2"}:
            for option in ("ms2_lr", "ms2_batch_size", "ms2_epoch", "ms2_warmup_epoch", "nce"):
                if option in config:
                    command.extend([f"--{option}", str(config[option])])
        if "patience" in config:
            command.extend(["--patience", str(config["patience"])])
        if args.tf_type in {"all", "ms2"} and config.get("no_masking"):
            command.append("--no_masking")
        if config.get("early_stop"):
            command.append("--early_stop")
        (run_dir / "command.json").write_text(
            json.dumps({"wandb_config": config, "seed": seed, "command": command}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{completed + 1}] {run_dir.name}: {' '.join(command)}", flush=True)
        trial_environment = os.environ.copy()
        trial_environment["WANDB_DIR"] = str((run_dir / "wandb").resolve())
        subprocess.run(command, cwd=ROOT, env=trial_environment, check=True)
        metrics_path = run_dir / "model_evaluation_metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError(f"Missing final metrics: {metrics_path}")
        training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result = {**training_metrics.get("rt", {}).get("finetuned", {})}
        if args.validation_dir is None:
            return result
        subprocess.run([
            args.python, str(EVALUATE_SCRIPT), "--run-dir", str(run_dir),
            "--tests-dir", str(args.validation_dir), "--device", args.device,
            "--python", args.python, "--rt-alignment", str(rt_alignment),
        ], cwd=ROOT, env=trial_environment, check=True)
        shutil.copy2(run_dir / "organism_test_metrics.json", run_dir / "validation_metrics.json")
        organism_metrics = json.loads(
            (run_dir / "organism_test_metrics.json").read_text(encoding="utf-8")
        )
        components = []
        for organism in ("celegans", "triplemix", "human"):
            item = organism_metrics[organism]
            components.append(
                0.4 * float(item["mae"])
                + 0.3 * float(item["heavy_dimethyl_mae"])
                + 0.3 * float(item["heavy_nterm_dimethyl_mae"])
            )
        return {
            **result,
            "validation_composite_mae": sum(components) / len(components),
            "organism_metrics": organism_metrics,
        }

    def trial():
        nonlocal completed
        run = wandb.init()
        config = dict(run.config)
        selected_configuration = config.get("configuration")
        if selected_configuration:
            if selected_configuration not in configuration_map:
                raise RuntimeError(f"Unknown mapped configuration: {selected_configuration}")
            config.update(configuration_map[selected_configuration])
        if "rt_backbone_lr_ratio" in config and "rt_backbone_lr" not in config:
            config["rt_backbone_lr"] = float(config["rt_lr"]) * float(config["rt_backbone_lr_ratio"])
        parent_id = run.id
        parent_name = run.name or parent_id
        run.finish()
        try:
            seed_metrics = []
            for seed in seeds:
                child = wandb.init(
                    project=wandb_settings["project"], entity=wandb_settings.get("entity"),
                    name=f"{parent_name}.seed{seed}", group=args.output_dir.name,
                    config={**config, "seed": seed},
                )
                child_id = child.id
                child.finish()
                metrics = run_seed_trial(
                    config, child_id, f"{parent_name}.seed{seed}", seed,
                    args.output_dir / f"trial.{parent_id}" / f"seed.{seed}",
                )
                seed_metrics.append(metrics)

            heavy_metric_names = (
                "heavy_dimethyl_mae", "heavy_nterm_dimethyl_mae", "heavy_nterm_mae",
                "heavy_lysine_mae", "heavy_both_mae"
            )
            aggregate = {}
            for metric in heavy_metric_names + ("mae_normalized", "r2", "validation_composite_mae"):
                values = [float(item[metric]) for item in seed_metrics if metric in item]
                if values:
                    aggregate[f"final/rt/mean_seed_{metric}"] = sum(values) / len(values)
                    if len(values) > 1:
                        import statistics
                        aggregate[f"final/rt/std_seed_{metric}"] = statistics.stdev(values)
            composite_key = "final/rt/mean_seed_validation_composite_mae"
            if composite_key in aggregate:
                aggregate["final/rt/validation_composite_mae"] = aggregate[composite_key]
            if args.validation_dir:
                for organism in ("celegans", "triplemix", "human"):
                    for metric in ("mae", "heavy_dimethyl_mae", "heavy_nterm_dimethyl_mae"):
                        values = [
                            float(item["organism_metrics"][organism][metric])
                            for item in seed_metrics
                        ]
                        aggregate[f"final/rt/{organism}_{metric}"] = sum(values) / len(values)
            parent = wandb.init(
                id=parent_id, resume="allow", project=wandb_settings["project"],
                entity=wandb_settings.get("entity"), config=config,
            )
            if aggregate:
                parent.log(aggregate)
            parent.summary["seed_count"] = len(seeds)
            parent.finish()
        except subprocess.CalledProcessError:
            failed = wandb.init(id=parent_id, resume="allow", project=wandb_settings["project"], entity=wandb_settings.get("entity"))
            failed.finish(exit_code=1)
            raise
        completed += 1

    while completed < args.max_runs and time.monotonic() < deadline:
        try:
            wandb.agent(sweep_id, function=trial, count=1, project=wandb_settings["project"], entity=wandb_settings.get("entity"))
        except CommError as error:
            if "is not running" in str(error):
                print(
                    f"Sweep {sweep_id} is no longer running; "
                    f"stopping after {completed} completed trial(s)."
                )
                break
            raise
    print(f"Sweep complete: {args.output_dir} ({completed} completed trials; sweep {sweep_id})")


if __name__ == "__main__":
    main()
