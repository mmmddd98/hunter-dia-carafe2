#!/usr/bin/env python3
"""Evaluate one fine-tuned Carafe2 RT model on all organism hold-outs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    delta = predictions - targets
    denominator = float(np.sum((targets - targets.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "r2": float(1 - np.sum(delta * delta) / denominator) if denominator else float("nan"),
        "pearson": float(np.corrcoef(targets, predictions)[0, 1]) if len(targets) > 1 else float("nan"),
        "spearman": float(pd.Series(targets).corr(pd.Series(predictions), method="spearman")) if len(targets) > 1 else float("nan"),
    }


def heavy_dimethyl_metrics(frame: pd.DataFrame, targets: np.ndarray,
                           predictions: np.ndarray) -> dict[str, float]:
    """Return metrics for all heavy and N-terminal-heavy dimethyl peptides."""
    mods = frame["mods"].fillna("").astype(str)
    nterm = mods.str.contains("Dimethyl:2H(4)13C(2)@N-term", regex=False).to_numpy()
    lysine = mods.str.contains("Dimethyl:2H(4)13C(2)@K", regex=False).to_numpy()
    result = {}
    for name, mask in {
        "heavy_dimethyl": nterm | lysine,
        # Includes peptides carrying both N-terminal and lysine heavy dimethyl.
        "heavy_nterm_dimethyl": nterm,
    }.items():
        result[f"{name}_rows"] = int(mask.sum())
        if mask.any():
            result.update({
                f"{name}_{metric}": value
                for metric, value in regression_metrics(targets[mask], predictions[mask]).items()
            })
    return result


def observed_rt_metrics(frame: pd.DataFrame, predictions: np.ndarray, alignment: dict) -> dict:
    """Evaluate predictions in observed RT minutes across all target definitions."""
    alignment_module_dir = os.environ.get("CARAFE_DATA_COLLECT_DIR")
    if not alignment_module_dir:
        raise RuntimeError(
            "Set CARAFE_DATA_COLLECT_DIR to the MA data_collect directory; "
            "this study evaluator depends on its rt_alignment module."
        )
    alignment_module_dir = str(Path(alignment_module_dir).expanduser())
    if alignment_module_dir not in sys.path:
        sys.path.insert(0, alignment_module_dir)
    try:
        from rt_alignment import predictions_to_observed_rt
    except ImportError as exc:
        raise RuntimeError(
            f"Could not load rt_alignment.py from CARAFE_DATA_COLLECT_DIR={alignment_module_dir}"
        ) from exc
    missing = {"rt", "source_organism"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Observed-RT evaluation requires {sorted(missing)}; rerun "
            "prepare_organism_scaling_study.py to refresh legacy test inputs"
        )
    source = frame["source_organism"].astype(str)
    targets = pd.to_numeric(frame["rt"], errors="raise").to_numpy(float)
    predicted_rt = predictions_to_observed_rt(predictions, source, alignment)
    delta = predicted_rt - targets
    result = {
        "raw_mae_minutes": float(np.mean(np.abs(delta))),
        "raw_rmse_minutes": float(np.sqrt(np.mean(delta ** 2))),
        "raw_bias_minutes": float(np.mean(delta)),
    }
    heavy = heavy_dimethyl_metrics(frame, targets, predicted_rt)
    result.update({f"raw_{key}_minutes" if key.endswith(("mae", "rmse")) else f"raw_{key}": value
                   for key, value in heavy.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tests-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rt-alignment", type=Path, required=True,
                        help="Training-only target-to-observed RT calibration JSON")
    args = parser.parse_args()
    if not (args.run_dir / "rt_model.pt").is_file():
        raise SystemExit(f"Incomplete Carafe2 run: {args.run_dir}")
    organisms = ("celegans", "triplemix", "human")
    missing = [str(args.tests_dir / f"{organism}.tsv") for organism in organisms
               if not (args.tests_dir / f"{organism}.tsv").is_file()]
    if missing:
        raise SystemExit(
            "Missing organism test input(s):\n  " + "\n  ".join(missing) +
            "\nRun ma/data_collect/prepare_organism_scaling_study.py first, "
            "or pass the correct --tests-dir."
        )
    predictor = Path(__file__).parent / "src/main/resources/py/v2/ai_pred.py"
    all_metrics = {}
    alignment = json.loads(args.rt_alignment.read_text(encoding="utf-8"))
    for organism in organisms:
        prefix = f"test_predictions_{organism}"
        subprocess.run([
            args.python, str(predictor), "--model_dir", str(args.run_dir),
            "--in_file", str(args.tests_dir / f"{organism}.tsv"),
            "--out_dir", str(args.run_dir), "--out_prefix", prefix,
            "--device", args.device, "--tf_type", "rt",
        ], check=True)
        prediction_path = args.run_dir / f"{prefix}_rt_pred.tsv"
        frame = pd.read_csv(prediction_path, sep="\t")
        targets = pd.to_numeric(frame["rt_norm"], errors="raise").to_numpy(float)
        predictions = pd.to_numeric(frame["rt_pred"], errors="raise").to_numpy(float)
        metrics = regression_metrics(targets, predictions)
        metrics.update(heavy_dimethyl_metrics(frame, targets, predictions))
        metrics.update(observed_rt_metrics(frame, predictions, alignment))
        metrics["rows"] = len(frame)
        metrics["unique_sequences"] = int(frame["sequence"].nunique())
        all_metrics[organism] = metrics
    (args.run_dir / "organism_test_metrics.json").write_text(
        json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
