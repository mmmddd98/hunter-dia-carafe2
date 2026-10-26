import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
import sys
import re
import logging
import math
import json


# Optional W&B integration. W&B is imported only when --wandb-config is used,
# so the standard Carafe CLI remains independent of this optional dependency.
_wandb_run = None
_wandb_settings = {}


def start_wandb(config_path, run_name, group, run_config, run_id=None):
    global _wandb_run, _wandb_settings
    if not config_path:
        return
    with open(config_path, "r", encoding="utf-8") as handle:
        settings = json.load(handle)
    _wandb_settings = settings
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B tracking requested, but wandb is not installed in this Python environment") from exc
    options = {
        "project": settings.get("project", "carafe2-finetune"), "config": run_config,
        "name": run_name, "group": group, "mode": settings.get("mode", "online"),
        "tags": settings.get("tags", ["carafe2", "finetune"]),
    }
    if settings.get("entity"):
        options["entity"] = settings["entity"]
    if run_id:
        options["id"] = run_id
        options["resume"] = "allow"
    _wandb_run = wandb.init(**options)
    logging.info("W&B run started: %s", _wandb_run.url)


def log_wandb_epoch(model_name, record):
    """Send validation metrics live immediately after an epoch completes."""
    if _wandb_run is not None:
        _wandb_run.log({f"{model_name}/{key}": value for key, value in record.items()})


def finish_wandb(out_dir):
    """Upload final metrics and diagnostics before Carafe's explicit os._exit."""
    global _wandb_run
    if _wandb_run is None:
        return
    try:
        metrics = load_model_evaluation_metrics(out_dir)
        final = {}
        for model_name, model_metrics in metrics.items():
            for stage in ("pretrained", "finetuned"):
                for key, value in model_metrics.get(stage, {}).items():
                    if value is not None:
                        final[f"final/{model_name}/{stage}_{key}"] = value
        if final:
            _wandb_run.log(final)
        log_artifacts = _wandb_settings.get(
            "artifacts",
            _wandb_settings.get("log_artifacts", _wandb_settings.get("logArtifacts", True)),
        )
        if bool(log_artifacts):
            import wandb
            artifact = wandb.Artifact("carafe2-finetune-diagnostics", type="diagnostics")
            for relative in (
                "model_evaluation_metrics.json", "training_report.html", "training_diagnostics",
                "rt_test.tsv", "rt_validation_strata_metrics.csv",
            ):
                path = os.path.join(out_dir, relative)
                if os.path.isfile(path):
                    artifact.add_file(path, name=relative)
                elif os.path.isdir(path):
                    artifact.add_dir(path, name=relative)
            _wandb_run.log_artifact(artifact)
        else:
            logging.info("Skipping W&B diagnostic artifact upload because artifacts=false.")
        _wandb_run.finish(exit_code=0)
    except Exception as exc:
        logging.warning("Could not finish W&B run cleanly: %s", exc)
    finally:
        _wandb_run = None

# Set up peptdeep-style logging with timestamps
logging.basicConfig(
    format='%(asctime)s> %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)

from alphabase.peptide.mobility import mobility_to_ccs_for_df

# =========================
# ENV setup BEFORE numpy/pandas/torch
# =========================
def configure_env_for_device(device: str, n_threads: int):
    """
    Must run BEFORE importing numpy/pandas/torch.
    """
    device = (device or "gpu").lower()
    if device == "cpu":
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_PROC_BIND", "true")
    if os.name == "nt":
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    
    import torch
    try:
        if torch.cuda.is_available():
            pass
        else:
            # Optimization: Flush denormals to zero to prevent performance drops on CPU
            # when gradients or weights become extremely small.
            if torch.set_flush_denormals(True):
                logging.info("Denormals flushed to zero for CPU performance.")
    except:
        pass


def print_thread_state(tag: str = ""):
    """Print thread/BLAS configuration for debugging"""
    import torch
    from threadpoolctl import threadpool_info
    if tag:
        print(f"\n==== {tag} ====")
    print("Python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
    print("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS"))
    print("torch intra", torch.get_num_threads())
    print("torch interop", torch.get_num_interop_threads())
    try:
        print("mkldnn enabled:", bool(getattr(torch.backends, "mkldnn", None) and torch.backends.mkldnn.enabled))
    except Exception:
        pass
    try:
        import pprint
        pprint.pp(threadpool_info())
    except Exception:
        pass


def set_seed(seed):
    """Set random seed for reproducibility"""
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_steps_per_epoch(num_samples: int, batch_size: int) -> int:
    """Calculate steps per epoch."""
    if batch_size <= 0: return 0
    return math.ceil(num_samples / batch_size)


def adjust_batch_size(num_samples: int, current_batch_size: int, min_batch_size: int=32, min_steps_per_epoch: int=40) -> int:
    """Adjust batch size to ensure at least 40 steps per epoch."""
    steps = get_steps_per_epoch(num_samples, current_batch_size)
    if steps < min_steps_per_epoch:
        adjusted_batch_size = max(min_batch_size, math.ceil(num_samples / min_steps_per_epoch))
        # 2^floor(log2(adjusted_batch_size))
        adjusted_batch_size = 2 ** math.floor(math.log2(adjusted_batch_size))
        adjusted_batch_size = max(min_batch_size, adjusted_batch_size)
        logging.info(f"Adjusted batch size to {adjusted_batch_size} for at least {min_steps_per_epoch} steps/epoch.")
        return adjusted_batch_size
    return current_batch_size

def get_auto_tune_params(num_peptides: int, mode='ms2'):
    """
    Hybrid steps-based auto-tuning for consistent training effort.
    
    Instead of scaling epochs inversely with data size (which leads to 
    inconsistent gradient update counts), we fix a target training budget
    in steps and derive epochs accordingly.
    
    This ensures:
    - Consistent training effort (~2000 steps) across all dataset sizes.
    - Minimum epochs floor to prevent underfitting on small data.
    - Maximum epochs cap to prevent excessive training on tiny data.
    """
    if mode == 'ms2':
        target_steps = 2000  # Training budget in gradient updates
        batch_size = 512
        min_epochs = 15
        max_epochs = 50
        # LR: Start conservative, scale slightly with data size
        # Larger batches with more data benefit from slightly higher LR
        log_count = math.log10(max(1000, min(100000, num_peptides)))
        progress = (log_count - 3) / 2  # 0.0 at 1k, 1.0 at 100k
        lr = 1e-4 + (progress * 1e-4)  # 1e-4 -> 2e-4
    else:  # RT or CCS
        target_steps = 1000
        batch_size = 1024 # Conservative default for logic, but overridden below
        min_epochs = 20
        max_epochs = 60
        log_count = math.log10(max(1000, min(100000, num_peptides)))
        progress = (log_count - 3) / 2
        lr = 1e-4 + (progress * 1.5e-4) # 1e-4 -> 2.5e-4
    
    # Calculate epochs based on target budget
    # steps = (num_peptides / batch_size) * epochs
    # epochs = (steps * batch_size) / num_peptides
    if num_peptides > 0:
        calculated_epochs = int(math.ceil((target_steps * batch_size) / num_peptides))
    else:
        calculated_epochs = min_epochs
        
    final_epochs = max(min_epochs, min(max_epochs, calculated_epochs))
    
    return final_epochs, lr

def get_warmup_epochs(epochs):
    """Warmup: min 5, max 10, otherwise ~25% of epochs"""
    return min(10, max(5, epochs // 4))


def _to_serializable_metric(value):
    """Convert numpy scalars and non-finite values to JSON-safe Python values."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def load_model_evaluation_metrics(out_dir: str) -> dict:
    """Load persisted model evaluation metrics if present."""
    metrics_path = os.path.join(out_dir, "model_evaluation_metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logging.warning(f"Could not load model evaluation metrics from {metrics_path}: {e}")
        return {}


def save_model_evaluation_metrics(out_dir: str, metrics: dict) -> str:
    """Persist model evaluation metrics to JSON."""
    metrics_path = os.path.join(out_dir, "model_evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logging.info(f"Saved model evaluation metrics JSON to {metrics_path}")
    return metrics_path


def update_model_evaluation_metrics(out_dir: str,
                                    model_key: str,
                                    pretrained_metrics: dict = None,
                                    finetuned_metrics: dict = None,
                                    extra_metrics: dict = None) -> dict:
    """Merge metrics for a model and write them to the shared JSON manifest."""
    metrics = load_model_evaluation_metrics(out_dir)
    model_metrics = metrics.get(model_key, {})

    if pretrained_metrics is not None:
        model_metrics["pretrained"] = {
            key: _to_serializable_metric(value) for key, value in pretrained_metrics.items()
        }
    if finetuned_metrics is not None:
        model_metrics["finetuned"] = {
            key: _to_serializable_metric(value) for key, value in finetuned_metrics.items()
        }
    if extra_metrics:
        for key, value in extra_metrics.items():
            if isinstance(value, dict):
                model_metrics[key] = {
                    sub_key: _to_serializable_metric(sub_value) for sub_key, sub_value in value.items()
                }
            else:
                model_metrics[key] = value

    metrics[model_key] = model_metrics
    save_model_evaluation_metrics(out_dir, metrics)
    return model_metrics


def reset_model_evaluation_metrics(out_dir: str) -> str:
    """Start a run with a fresh model evaluation metrics JSON file."""
    return save_model_evaluation_metrics(out_dir, {})


def _finite_float(value):
    """Return a finite float or NaN; keeps CSV output rectangular and numeric."""
    try:
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def regression_epoch_metrics(y_true, y_pred, prefix="validation"):
    """Calculate regression metrics used for RT and CCS validation each epoch."""
    import numpy as np

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    n = len(y_true)
    result = {f"{prefix}_n": n}
    if n == 0:
        return result

    residual = y_pred - y_true
    result[f"{prefix}_mae"] = float(np.mean(np.abs(residual)))
    result[f"{prefix}_rmse"] = float(np.sqrt(np.mean(residual ** 2)))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    result[f"{prefix}_r2"] = float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else float("nan")
    if n > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        result[f"{prefix}_pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        try:
            from scipy.stats import spearmanr
            result[f"{prefix}_spearman_r"] = float(spearmanr(y_true, y_pred).statistic)
        except Exception:
            result[f"{prefix}_spearman_r"] = float("nan")
    else:
        result[f"{prefix}_pearson_r"] = float("nan")
        result[f"{prefix}_spearman_r"] = float("nan")
    return result


def rt_stratified_metrics(frame, predictions, prefix="validation_"):
    """Calculate RT metrics separately for normal and Heavy-Dimethyl rows."""
    import numpy as np
    from models import rt_heavy_strata

    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    targets = frame["rt_norm"].to_numpy(dtype=float).reshape(-1)
    strata = rt_heavy_strata(frame).to_numpy()
    result = {}
    for stratum in ("normal", "heavy_nterm", "heavy_lysine", "heavy_both"):
        mask = strata == stratum
        metrics = regression_epoch_metrics(targets[mask], predictions[mask], prefix="metric")
        for metric in ("n", "mae", "rmse", "r2", "pearson_r", "spearman_r"):
            key = f"metric_{metric}"
            if key in metrics:
                result[f"{prefix}{stratum}_{metric}"] = metrics[key]
    heavy_mask = strata != "normal"
    metrics = regression_epoch_metrics(targets[heavy_mask], predictions[heavy_mask], prefix="metric")
    for metric in ("n", "mae", "rmse", "r2", "pearson_r", "spearman_r"):
        key = f"metric_{metric}"
        if key in metrics:
            result[f"{prefix}heavy_dimethyl_{metric}"] = metrics[key]
    # Include heavy_both rows: they still carry the N-terminal heavy dimethyl.
    nterm_mask = np.isin(strata, ("heavy_nterm", "heavy_both"))
    metrics = regression_epoch_metrics(targets[nterm_mask], predictions[nterm_mask], prefix="metric")
    for metric in ("n", "mae", "rmse", "r2", "pearson_r", "spearman_r"):
        key = f"metric_{metric}"
        if key in metrics:
            result[f"{prefix}heavy_nterm_dimethyl_{metric}"] = metrics[key]
    return result


def rt_subset_loss_metrics(frame, predictions, *, loss_name, huber_delta, prefix="validation_"):
    """Calculate the unweighted RT objective for the two tracked heavy subsets."""
    from models import rt_heavy_strata

    targets = frame["rt_norm"].to_numpy(dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    absolute = np.abs(predictions - targets)
    if loss_name == "l1":
        values = absolute
    elif loss_name == "mse":
        values = absolute ** 2
    elif loss_name == "huber":
        values = np.where(
            absolute < huber_delta,
            0.5 * absolute ** 2,
            huber_delta * (absolute - 0.5 * huber_delta),
        )
    else:
        raise ValueError(f"Unknown RT loss: {loss_name}")
    strata = rt_heavy_strata(frame).to_numpy()
    masks = {
        "heavy_dimethyl": strata != "normal",
        "heavy_nterm_dimethyl": np.isin(strata, ("heavy_nterm", "heavy_both")),
    }
    return {
        f"{prefix}{name}_loss": float(np.mean(values[mask])) if mask.any() else float("nan")
        for name, mask in masks.items()
    }


def ms2_epoch_metrics(predictions, targets, valid_targets=None, prefix="validation"):
    """Summarise masked MS2 agreement for one validation epoch.

    The returned PCC, cosine similarity, and spectral angle are calculated per
    precursor and then averaged, preventing long peptides from dominating them.
    MAE and RMSE are calculated over all valid fragment intensities.
    """
    import numpy as np

    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if predictions.size == 0 or targets.size == 0:
        return {f"{prefix}_n": 0}
    predictions = predictions.reshape(predictions.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    if valid_targets is None:
        mask = np.ones_like(targets, dtype=bool)
    else:
        mask = np.asarray(valid_targets).reshape(targets.shape) <= 0

    masked_predictions = np.where(mask, predictions, 0.0)
    masked_targets = np.where(mask, targets, 0.0)
    valid_values = mask & np.isfinite(predictions) & np.isfinite(targets)
    errors = (predictions - targets)[valid_values]
    result = {
        f"{prefix}_n": int(predictions.shape[0]),
        f"{prefix}_fragment_n": int(valid_values.sum()),
        f"{prefix}_mae": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        f"{prefix}_rmse": float(np.sqrt(np.mean(errors ** 2))) if errors.size else float("nan"),
    }

    pccs, cosines, angles = [], [], []
    for pred, target, row_mask in zip(masked_predictions, masked_targets, mask):
        pred, target = pred[row_mask], target[row_mask]
        if len(pred) == 0:
            continue
        pred_norm, target_norm = np.linalg.norm(pred), np.linalg.norm(target)
        if pred_norm > 0 and target_norm > 0:
            cosine = float(np.clip(np.dot(pred, target) / (pred_norm * target_norm), -1.0, 1.0))
            cosines.append(cosine)
            angles.append(float(2.0 * np.arccos(cosine) / np.pi))
        if len(pred) > 1 and np.std(pred) > 0 and np.std(target) > 0:
            pccs.append(float(np.corrcoef(pred, target)[0, 1]))
    for name, values in (("pcc", pccs), ("cosine", cosines), ("spectral_angle", angles)):
        result[f"{prefix}_{name}_mean"] = float(np.mean(values)) if values else float("nan")
        result[f"{prefix}_{name}_median"] = float(np.median(values)) if values else float("nan")
    return result


def ms2_epoch_metrics_from_batches(batches, prefix="validation"):
    """Calculate exact MS2 epoch metrics for batches with different peptide lengths."""
    import numpy as np

    all_errors, pccs, cosines, angles = [], [], [], []
    precursor_count = 0
    fragment_count = 0
    for predictions, targets, valid_targets in batches:
        predictions = np.asarray(predictions, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if valid_targets is None:
            masks = np.ones_like(targets, dtype=bool)
        else:
            masks = np.asarray(valid_targets) <= 0
        for prediction, target, mask in zip(predictions, targets, masks):
            prediction, target, mask = prediction.reshape(-1), target.reshape(-1), mask.reshape(-1)
            valid = mask & np.isfinite(prediction) & np.isfinite(target)
            prediction, target = prediction[valid], target[valid]
            precursor_count += 1
            fragment_count += len(prediction)
            if len(prediction) == 0:
                continue
            all_errors.append(prediction - target)
            pred_norm, target_norm = np.linalg.norm(prediction), np.linalg.norm(target)
            if pred_norm > 0 and target_norm > 0:
                cosine = float(np.clip(np.dot(prediction, target) / (pred_norm * target_norm), -1.0, 1.0))
                cosines.append(cosine)
                angles.append(float(2.0 * np.arccos(cosine) / np.pi))
            if len(prediction) > 1 and np.std(prediction) > 0 and np.std(target) > 0:
                pccs.append(float(np.corrcoef(prediction, target)[0, 1]))

    errors = np.concatenate(all_errors) if all_errors else np.array([])
    result = {
        f"{prefix}_n": precursor_count,
        f"{prefix}_fragment_n": fragment_count,
        f"{prefix}_mae": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        f"{prefix}_rmse": float(np.sqrt(np.mean(errors ** 2))) if errors.size else float("nan"),
    }
    for name, values in (("pcc", pccs), ("cosine", cosines), ("spectral_angle", angles)):
        result[f"{prefix}_{name}_mean"] = float(np.mean(values)) if values else float("nan")
        result[f"{prefix}_{name}_median"] = float(np.median(values)) if values else float("nan")
    return result


def save_training_diagnostics(out_dir, model_name, epoch_records, metadata=None, write_plots=True):
    """Write per-epoch data and, optionally, one plot per metric.

    Layout:
      training_diagnostics/
        data/<model>_epoch_metrics.csv, all_epoch_metrics.csv, <model>_metadata.json
        plots/<model>/<metric>.png and <metric>.csv
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    root = os.path.join(out_dir, "training_diagnostics")
    data_dir = os.path.join(root, "data")
    plot_dir = os.path.join(root, "plots", model_name)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    frame = pd.DataFrame(epoch_records)
    if frame.empty:
        logging.warning("No epoch diagnostics available for %s", model_name)
        return
    frame.insert(0, "model", model_name)
    frame = frame.apply(lambda column: column.map(_finite_float)
                        if column.name not in ("model",) else column)
    model_csv = os.path.join(data_dir, f"{model_name}_epoch_metrics.csv")
    frame.to_csv(model_csv, index=False)

    if metadata is not None:
        with open(os.path.join(data_dir, f"{model_name}_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    frames = []
    for candidate in ("rt", "ccs", "ms2"):
        candidate_csv = os.path.join(data_dir, f"{candidate}_epoch_metrics.csv")
        if os.path.exists(candidate_csv):
            frames.append(pd.read_csv(candidate_csv))
    if frames:
        pd.concat(frames, ignore_index=True, sort=False).to_csv(
            os.path.join(data_dir, "all_epoch_metrics.csv"), index=False
        )

    metric_columns = [column for column in frame.columns
                      if column not in ("model", "epoch", "validation_n", "validation_fragment_n")
                      and not column.endswith(("_seconds", "_duration"))]
    if not write_plots:
        logging.info(
            "Saved %s numeric training diagnostics under %s (per-run plots disabled)",
            model_name.upper(), root,
        )
        return
    for metric in metric_columns:
        metric_frame = frame[["epoch", metric]].dropna()
        if metric_frame.empty:
            continue
        metric_frame.to_csv(os.path.join(plot_dir, f"{metric}.csv"), index=False)
        fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        axis.plot(metric_frame["epoch"], metric_frame[metric], marker="o", linewidth=1.8, markersize=4)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(f"{model_name.upper()} – {metric.replace('_', ' ').title()}")
        axis.grid(True, linestyle=":", alpha=0.55)
        axis.set_xticks(metric_frame["epoch"].astype(int).unique())
        fig.savefig(os.path.join(plot_dir, f"{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # A single common y-axis is only meaningful if loss/error metrics are made
    # directionally comparable to agreement/correlation metrics.  Therefore all
    # curves are expressed as a fold change relative to their first recorded
    # epoch: 1.0 is the initial value and values above 1.0 always mean better
    # performance.  Raw values remain available in the per-metric CSV/plots.
    relative_records = []
    lower_is_better_terms = ("loss", "error", "mae", "mse", "rmse", "l1")
    for metric in metric_columns:
        values = pd.to_numeric(frame[metric], errors="coerce")
        observed = values.dropna()
        if observed.empty:
            continue
        baseline = float(observed.iloc[0])
        if not np.isfinite(baseline) or abs(baseline) < 1e-12:
            logging.warning("Skipping relative overview for %s: first value is zero or non-finite", metric)
            continue
        lower_is_better = any(term in metric.lower() for term in lower_is_better_terms)
        relative_values = baseline / values if lower_is_better else values / baseline
        for epoch, raw_value, relative_value in zip(frame["epoch"], values, relative_values):
            if pd.notna(raw_value) and pd.notna(relative_value) and np.isfinite(relative_value):
                relative_records.append({
                    "epoch": int(epoch), "metric": metric, "raw_value": float(raw_value),
                    "relative_quality_vs_first_epoch": float(relative_value),
                    "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                })

    if relative_records:
        relative_frame = pd.DataFrame(relative_records)
        relative_frame.to_csv(os.path.join(plot_dir, "metrics_relative_to_first_epoch.csv"), index=False)
        fig, axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
        for metric, metric_frame in relative_frame.groupby("metric", sort=True):
            axis.plot(metric_frame["epoch"], metric_frame["relative_quality_vs_first_epoch"],
                      marker="o", linewidth=1.8, markersize=3.8, label=metric.replace("_", " "))
        axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.65, label="first epoch")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Relative quality vs first epoch (higher is better)")
        axis.set_title(f"{model_name.upper()} – All validation metrics on a common relative scale")
        axis.grid(True, linestyle=":", alpha=0.55)
        axis.set_xticks(sorted(relative_frame["epoch"].unique()))
        axis.legend(frameon=False, fontsize=8, ncol=2, loc="best")
        fig.savefig(os.path.join(plot_dir, "metrics_relative_to_first_epoch.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Complementary overview: every metric in one figure, retaining its native
    # unit and therefore avoiding any interpretation through normalization.
    available_metrics = [metric for metric in metric_columns if frame[metric].notna().any()]
    if available_metrics:
        n_columns = min(3, len(available_metrics))
        n_rows = int(math.ceil(len(available_metrics) / n_columns))
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(5.0 * n_columns, 3.6 * n_rows),
                                 squeeze=False, constrained_layout=True)
        for axis, metric in zip(axes.flat, available_metrics):
            metric_frame = frame[["epoch", metric]].dropna()
            axis.plot(metric_frame["epoch"], metric_frame[metric], marker="o", linewidth=1.7, markersize=3.6)
            axis.set_title(metric.replace("_", " ").title())
            axis.set_xlabel("Epoch")
            axis.set_ylabel(metric.replace("_", " ").title())
            axis.grid(True, linestyle=":", alpha=0.55)
        for axis in axes.flat[len(available_metrics):]:
            axis.set_visible(False)
        fig.suptitle(f"{model_name.upper()} – All training and validation metrics (native scales)", fontsize=14)
        fig.savefig(os.path.join(plot_dir, "metrics_overview_grid.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    if {"train_loss", "validation_loss"}.issubset(frame.columns):
        loss_frame = frame[["epoch", "train_loss", "validation_loss"]].dropna(how="all", subset=["train_loss", "validation_loss"])
        loss_frame.to_csv(os.path.join(plot_dir, "loss_comparison.csv"), index=False)
        fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        axis.plot(loss_frame["epoch"], loss_frame["train_loss"], marker="o", label="Training loss")
        axis.plot(loss_frame["epoch"], loss_frame["validation_loss"], marker="o", label="Validation loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("L1 loss")
        axis.set_title(f"{model_name.upper()} – Training and validation loss")
        axis.grid(True, linestyle=":", alpha=0.55)
        axis.legend(frameon=False)
        fig.savefig(os.path.join(plot_dir, "loss_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    logging.info("Saved %s training diagnostics under %s", model_name.upper(), root)


def _categorise_trypticity(frame):
    """Use explicit digestion metadata when available; never guess from sequence alone."""
    import pandas as pd

    for column in ("trypticity", "peptide_class", "digestion_class"):
        if column in frame.columns:
            values = frame[column].astype(str).str.lower()
            return values.map(lambda value: "semi_tryptic" if "semi" in value
                              else "fully_tryptic" if "full" in value or "tryptic" in value
                              else "other_or_unknown")
    if "is_semi_tryptic" in frame.columns:
        return frame["is_semi_tryptic"].map(
            lambda value: "semi_tryptic" if str(value).strip().lower() in {"1", "true", "yes"}
            else "fully_tryptic"
        )
    # The PSM training table has no protein position/flanking residues. A peptide
    # sequence alone cannot establish whether its N terminus is enzymatic.
    return pd.Series("not_available", index=frame.index, dtype="object")


def _categorise_protein_n_terminus(frame):
    """Read protein-N-terminal annotation if an upstream exporter supplied one."""
    import pandas as pd

    for column in ("is_protein_n_terminal", "protein_n_terminal", "protein_n_terminus"):
        if column in frame.columns:
            return frame[column].map(
                lambda value: "protein_n_terminal" if str(value).strip().lower() in {"1", "true", "yes"}
                else "internal"
            )
    return pd.Series("not_available", index=frame.index, dtype="object")


def add_ms2_stratification_columns(frame, fallback_nce):
    """Attach transparent peptide strata without discarding unclassifiable rows."""
    import numpy as np
    import pandas as pd

    frame = frame.copy()
    sequence = frame.get("sequence", frame.get("peptide", pd.Series("", index=frame.index))).fillna("").astype(str)
    mods = frame.get("mods", pd.Series("", index=frame.index)).fillna("").astype(str)
    mod_sites = frame.get("mod_sites", pd.Series("", index=frame.index)).fillna("").astype(str)
    mods_lower = mods.str.lower()

    frame["modified_sequence"] = sequence + " | " + mods + " | sites=" + mod_sites
    frame["peptide_length"] = sequence.str.len()
    frame["length_bin"] = np.select(
        [frame["peptide_length"] <= 10, frame["peptide_length"] <= 20],
        ["short_7_10", "medium_11_20"], default="long_21_plus"
    )
    frame["trypticity"] = _categorise_trypticity(frame)
    frame["protein_n_terminus"] = _categorise_protein_n_terminus(frame)
    n_term_modified = mod_sites.str.split(";").map(lambda sites: "0" in {site.strip() for site in sites}) | mods_lower.str.contains("n-term", regex=False)
    frame["n_terminal_modification"] = np.where(n_term_modified, "n_terminal_modified", "no_n_terminal_modification")

    is_dimethyl = mods_lower.str.contains("dimethyl", regex=False)
    is_heavy_dimethyl = is_dimethyl & (mods_lower.str.contains("2h", regex=False) | mods_lower.str.contains("13c", regex=False) | mods_lower.str.contains("heavy", regex=False))
    frame["dimethyl"] = np.select(
        [is_heavy_dimethyl, is_dimethyl], ["heavy_dimethyl", "light_dimethyl"], default="no_dimethyl"
    )
    frame["modification"] = np.where(mods.str.strip().eq(""), "unmodified", "modified")
    frame["oxidation_m"] = np.where(mods_lower.str.contains("oxidation@m", regex=False), "oxidized_methionine", "not_oxidized_methionine")

    charge_values = frame["charge"] if "charge" in frame.columns else pd.Series(np.nan, index=frame.index)
    nce_values = frame["nce"] if "nce" in frame.columns else pd.Series(float(fallback_nce), index=frame.index)
    frame["charge"] = pd.to_numeric(charge_values, errors="coerce").astype("Int64")
    nce = pd.to_numeric(nce_values, errors="coerce").fillna(float(fallback_nce))
    frame["collision_energy"] = nce
    frame["collision_energy_bin"] = np.select(
        [nce <= 25, nce <= 30], ["low_le_25", "medium_25_30"], default="high_gt_30"
    )
    return frame


def save_stratified_ms2_diagnostics(out_dir, psm_metrics, fallback_nce):
    """Save per-PSM and group-level MS2 validation diagnostics plus comparison plots."""
    import pandas as pd
    import matplotlib.pyplot as plt

    root = os.path.join(out_dir, "training_diagnostics", "stratified_ms2")
    data_dir = os.path.join(root, "data")
    plots_dir = os.path.join(root, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    annotated = add_ms2_stratification_columns(psm_metrics, fallback_nce)
    annotated.to_csv(os.path.join(data_dir, "validation_psm_metrics.csv"), index=False)

    dimensions = [
        "trypticity", "protein_n_terminus", "n_terminal_modification", "dimethyl",
        "modification", "oxidation_m", "charge", "length_bin", "collision_energy_bin",
    ]
    metrics = ["SA", "COS", "PCC", "SPC", "l1_loss", "rmse"]
    metric_labels = {
        "SA": "Spectral angle", "COS": "Cosine similarity", "PCC": "Pearson correlation",
        "SPC": "Spearman correlation", "l1_loss": "L1 loss", "rmse": "RMSE",
    }
    summaries = []
    for dimension in dimensions:
        for metric in metrics:
            if metric not in annotated.columns:
                continue
            grouped = annotated.groupby(dimension, dropna=False)[metric].agg(["count", "mean", "median", "std"]).reset_index()
            grouped.insert(0, "metric", metric)
            grouped.insert(0, "stratum", dimension)
            summaries.append(grouped)

            metric_dir = os.path.join(plots_dir, dimension)
            os.makedirs(metric_dir, exist_ok=True)
            grouped.to_csv(os.path.join(metric_dir, f"{metric}.csv"), index=False)
            plot_frame = grouped.dropna(subset=["median"])
            if plot_frame.empty:
                continue
            fig, axis = plt.subplots(figsize=(max(6.4, 1.15 * len(plot_frame)), 4.8), constrained_layout=True)
            bars = axis.bar(plot_frame[dimension].astype(str), plot_frame["median"], color="#377eb8", alpha=0.9)
            axis.set_title(f"MS2 validation: {metric_labels[metric]} by {dimension.replace('_', ' ')}")
            axis.set_ylabel(f"Median {metric_labels[metric]}")
            axis.set_xlabel(dimension.replace("_", " ").title())
            axis.grid(axis="y", linestyle=":", alpha=0.55)
            axis.tick_params(axis="x", labelrotation=30)
            for bar, count in zip(bars, plot_frame["count"]):
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={int(count)}",
                          ha="center", va="bottom", fontsize=8)
            fig.savefig(os.path.join(metric_dir, f"{metric}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary.to_csv(os.path.join(data_dir, "group_summary.csv"), index=False)
    coverage = pd.concat([
        annotated.groupby(dimension, dropna=False).size().rename("count").reset_index().assign(stratum=dimension)
        for dimension in dimensions
    ], ignore_index=True, sort=False)
    coverage.to_csv(os.path.join(data_dir, "stratum_coverage.csv"), index=False)
    metadata = {
        "evaluation_set": "sequence-disjoint validation hold-out used during MS2 training",
        "metrics": metric_labels,
        "length_bins": {"short": "7-10 residues", "medium": "11-20 residues", "long": "21+ residues"},
        "collision_energy_bins": {"low": "<=25", "medium": ">25 to <=30", "high": ">30"},
        "trypticity_note": "Only classified when explicit digestion metadata is present; otherwise recorded as not_available to avoid inference from insufficient sequence-only context.",
        "protein_n_terminus_note": "Only classified when an upstream exporter provides a protein-N-terminal flag.",
    }
    with open(os.path.join(data_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    logging.info("Saved stratified MS2 validation diagnostics under %s", root)
    return annotated, summary


def add_ms2_per_psm_error_metrics(psm_metrics, predicted_intensities, observed_intensities,
                                  valid_intensities, fragment_columns):
    """Add masked per-PSM L1 loss and RMSE to a similarity-metric dataframe."""
    import numpy as np

    result = psm_metrics.copy()
    l1_losses, rmses, valid_counts = [], [], []
    for _, row in result.iterrows():
        start, stop = int(row["frag_start_idx"]), int(row["frag_stop_idx"])
        predicted = predicted_intensities.iloc[start:stop][fragment_columns].to_numpy(dtype=float).reshape(-1)
        observed = observed_intensities.iloc[start:stop][fragment_columns].to_numpy(dtype=float).reshape(-1)
        if valid_intensities is None:
            mask = np.ones_like(observed, dtype=bool)
        else:
            valid = valid_intensities.iloc[start:stop][fragment_columns].to_numpy(dtype=float).reshape(-1)
            mask = valid <= 0
        finite = mask & np.isfinite(predicted) & np.isfinite(observed)
        errors = predicted[finite] - observed[finite]
        valid_counts.append(int(finite.sum()))
        l1_losses.append(float(np.mean(np.abs(errors))) if errors.size else float("nan"))
        rmses.append(float(np.sqrt(np.mean(errors ** 2))) if errors.size else float("nan"))
    result["l1_loss"] = l1_losses
    result["rmse"] = rmses
    result["valid_fragment_count"] = valid_counts
    return result


def save_validation_checkpoint(model, out_dir, model_name, epoch_record):
    """Persist a reproducible model snapshot immediately after validation."""
    checkpoint_dir = os.path.join(out_dir, "validation_checkpoints", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch = int(epoch_record["epoch"])
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
    model.save(checkpoint_path)
    metadata = {key: _to_serializable_metric(value) for key, value in epoch_record.items()}
    metadata["checkpoint_file"] = os.path.basename(checkpoint_path)
    with open(os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    logging.info("Saved %s validation checkpoint for epoch %d: %s", model_name.upper(), epoch, checkpoint_path)
    return checkpoint_path


def train_ms2(in_dir: str,
              use_valid=True,
              out_dir="./",
              out_prefix="ms2_pred",
              device='gpu',
              instrument='Eclipse',
              nce=27,
              mode_type="general",
              use_grid_nce_search=False,
              log_transform=False,
              threads=1,
              use_best_model=False,
              auto_tune=False,
              early_stop=False,
              patience=10,
              verbose=1,
              epoch_to_train_ms2=20,
              warmup_epoch_to_train_ms2=10,
              batch_size_to_train_ms2=512,
              lr_to_train_ms2=0.0001,
              ms2_max_test_samples=1000,
              torch_compile=False,
              pretrained_model: str = None,
              adjust_batch_size_for_steps=True,
              from_scratch=False,
              save_validation_checkpoints=False):
    """Train MS2 prediction model - replicates peptdeep exactly"""
    import torch
    from models import (ModelManager, psm_sampling_with_important_mods, 
                        normalize_fragment_intensities, calc_ms2_similarity_mask,
                        calc_ms2_similarity, settings)
    import alphabase.peptide.fragment as fragment
    import numpy as np
    import pandas as pd
    import math
    import time
    
    pd.options.mode.chained_assignment = None
    a = pd.read_csv(in_dir + "/psm_pdv.txt", sep="\t", dtype={'mod_sites': str, 'mods': str})
    if "peptide" in a.columns:
        a["sequence"] = a["peptide"]
    a["nAA"] = a["sequence"].str.len()
    a['mod_sites'] = a['mod_sites'].fillna("").astype(str)
    a['mods'] = a['mods'].fillna("").astype(str)

    # Calculate split sizes
    if mode_type == 'general':
        mask_modloss = True
    elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
        mask_modloss = False
    elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        mask_modloss = True
    else:
        print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
        mask_modloss = True
    model_mgr = ModelManager(mask_modloss=mask_modloss, device=device, out_dir=out_dir)
    
    if device == 'cpu':
        torch.set_num_threads(threads)
    # Load pretrained models
    if from_scratch:
        logging.info("Training from scratch (skipping all pretrained model loading)")
    else:
        try:
            if mode_type == 'general':
                model_mgr.load_installed_models('generic', model_list=['ms2'])
            elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
                model_mgr.load_installed_models('phos', model_list=['ms2'])
            elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
                model_mgr.load_installed_models('ubi', model_list=['ms2'])
            else:
                print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
                model_mgr.load_installed_models('generic', model_list=['ms2'])
        except Exception as e:
            print(f"Warning: Could not load pretrained models: {e}")

        # Load user-provided pretrained model if available
        if pretrained_model and os.path.exists(pretrained_model):
            try:
                print(f"Loading user-provided MS2 model from: {pretrained_model}")
                model_mgr.load_external_models(ms2_model_file=pretrained_model)
            except Exception as e:
                print(f"Error loading user-provided MS2 model: {e}")
    
    logging.info(f"MS2 Model Parameter Size: {model_mgr.ms2_model.get_parameter_num():,}")

    # Calculate split sizes
    n_test_ms2 = max(1, np.min([ms2_max_test_samples, int(math.ceil(a.shape[0] * 0.1) - 10)]))
    psm_num_to_test_ms2 = n_test_ms2
    psm_num_to_train_ms2 = a.shape[0] - n_test_ms2
    
    # Training parameters
    if auto_tune:
        epoch_to_train_ms2, lr_to_train_ms2 = get_auto_tune_params(psm_num_to_train_ms2, mode='ms2')
        warmup_epoch_to_train_ms2 = get_warmup_epochs(epoch_to_train_ms2)
        logging.info(f"[Auto-Tune] Scaling MS2 params for {psm_num_to_train_ms2} peptides: epochs={epoch_to_train_ms2}, lr={lr_to_train_ms2:.1e}, batch={batch_size_to_train_ms2}")
    elif adjust_batch_size_for_steps:
        batch_size_to_train_ms2 = adjust_batch_size(psm_num_to_train_ms2, batch_size_to_train_ms2)
    
    psm_num_per_mod_to_train_ms2 = 50
    top_n_mods_to_train = 10
    
    # EXACT PARITY: PeptDeep v1 Sequence-Unique splitting
    # 1. Sample training data with important modifications (uses library default seed 1337)
    tr_df = psm_sampling_with_important_mods(
        a, psm_num_to_train_ms2, top_n_mods_to_train, psm_num_per_mod_to_train_ms2
    ).copy()
    
    # 2. Split test set by excluding training sequences (peptdeep default)
    test_psm_df = a[~a.sequence.isin(set(tr_df.sequence))].copy()
    
    # 3. Sample test set to requested size using hardcoded seed 1337
    if len(test_psm_df) > psm_num_to_test_ms2:
        test_psm_df = test_psm_df.sample(psm_num_to_test_ms2, random_state=1337).copy()
    elif len(test_psm_df) == 0:
        test_psm_df = a.copy()
    
    print(f"The number of PSMs to train MS2 model: {len(tr_df)}")
    print(f"The number of PSMs to test MS2 model: {len(test_psm_df)}")
    steps = get_steps_per_epoch(len(tr_df), batch_size_to_train_ms2)
    print(f"Training steps per epoch: {steps}")
    
    model_mgr.nce = nce
    model_mgr.instrument = instrument

    if verbose >=2:
        print("Input PSM dtypes:")
        print(a.dtypes)
    
    b = pd.read_csv(in_dir + "/fragment_intensity_df.tsv", sep="\t", engine="pyarrow")
    
    if log_transform:
        print("log transform intensity data ...")
        b = b.apply(lambda x: np.log10(x + 1) / np.log10(2))
    
    valid = None
    if use_valid and os.path.exists(in_dir + "/fragment_intensity_valid.tsv"):
        print("Use valid intensity data ...")
        valid = pd.read_csv(in_dir + "/fragment_intensity_valid.tsv", sep="\t", engine="pyarrow")
    
    # Prepare training intensity dataframes
    tr_inten_df = pd.DataFrame()
    if valid is not None:
        tr_inten_valid_df = pd.DataFrame()
    else:
        tr_inten_valid_df = None

    for frag_type in model_mgr.ms2_model.charged_frag_types:
        if frag_type in b.columns:
            tr_inten_df[frag_type] = b[frag_type]
        else:
            tr_inten_df[frag_type] = 0.0
        
        if tr_inten_valid_df is not None:
            if frag_type in valid.columns:
                tr_inten_valid_df[frag_type] = valid[frag_type]
            else:
                tr_inten_valid_df[frag_type] = 0.0
    
    # Normalize every PSM before the sequence-disjoint train/test split is used
    # for loss calculation.  Normalizing only ``tr_df`` leaves hold-out targets
    # at their raw intensity scale, making validation loss incomparable to train
    # loss and unusable for checkpoint selection.
    normalize_fragment_intensities(a, tr_inten_df)
    
    # Advanced DIA Enhancements: Auto-Tune Scaling
    if auto_tune:
        logging.info("[Auto-Tune] Enhanced DIA optimization enabled (Auto-Scaling).")

    # Set NCE and instrument
    model_mgr_settings = settings['model_mgr']
    # use_grid_nce_search is already passed as a function parameter
    
    if use_grid_nce_search:
        print("Perform grid search for NCE and instrument ...")
        nce, instrument = model_mgr.ms2_model.grid_nce_search(
            tr_df.copy(), tr_inten_df,
            nce_first=model_mgr_settings['transfer']['grid_nce_first'],
            nce_last=model_mgr_settings['transfer']['grid_nce_last'],
            nce_step=model_mgr_settings['transfer']['grid_nce_step'],
            search_instruments=model_mgr_settings['transfer']['grid_instrument'],
        )
        tr_df['nce'] = int(nce)
        tr_df['instrument'] = instrument
        print(f"Optimal NCE: {nce}, Instrument: {instrument}")
    else:
        if 'nce' not in tr_df.columns:
            tr_df['nce'] = nce
        if 'instrument' not in tr_df.columns:
            tr_df['instrument'] = model_mgr.instrument
    
    # Prepare test set NCE/Instrument
    if 'nce' not in test_psm_df.columns:
        test_psm_df['nce'] = nce
    if 'instrument' not in test_psm_df.columns:
        test_psm_df['instrument'] = model_mgr.instrument
    
    # Pre-training evaluation
    pretrained_metrics = None
    if len(test_psm_df) > 0:
        print("Using masking in metrics calculation ...")
        pred_frag_df = model_mgr.ms2_model.predict(test_psm_df.copy(), reference_frag_df=tr_inten_df)
        test_res = calc_ms2_similarity_mask(
            test_psm_df.copy(), pred_frag_df, tr_inten_df, tr_inten_valid_df
        )
        out_test_res_to_file = os.path.join(out_dir, 'test_res_pretrained.csv')
        test_res[0].to_csv(out_test_res_to_file, index=False)
        # Determine output columns based on mask_modloss
        if mask_modloss:
            out_frag_cols = ['b_z1', 'b_z2', 'y_z1', 'y_z2']
        else:
            out_frag_cols = ['b_z1', 'b_z2', 'y_z1', 'y_z2',
                            'b_modloss_z1', 'b_modloss_z2', 'y_modloss_z1', 'y_modloss_z2']
        
        # Export predicted intensity matrix for pretrained model (test PSMs only)
        test_frag_rows = []
        truth_frag_rows = []
        valid_test_df = test_psm_df.dropna(subset=['frag_start_idx', 'frag_stop_idx'])
        for _, row in valid_test_df.iterrows():
            start, stop = int(row['frag_start_idx']), int(row['frag_stop_idx'])
            test_frag_rows.append(pred_frag_df.iloc[start:stop])
            truth_frag_rows.append(tr_inten_df.iloc[start:stop])
        if test_frag_rows:
            pd.concat(test_frag_rows, ignore_index=True)[out_frag_cols].to_csv(
                os.path.join(out_dir, 'test_pred_intensity_pretrained.tsv'), sep='\t', index=False)
            pd.concat(truth_frag_rows, ignore_index=True)[out_frag_cols].to_csv(
                os.path.join(out_dir, 'test_true_intensity.tsv'), sep='\t', index=False)
        
        # Export fragment m/z for test PSMs
        mz_df = fragment.create_fragment_mz_dataframe(
            valid_test_df[['sequence', 'mods', 'mod_sites', 'charge', 'nAA']],
            out_frag_cols,
            reference_fragment_df=None
        )
        mz_df.to_csv(os.path.join(out_dir, 'test_fragment_mz.tsv'), sep='\t', index=False)
        logging.info(f"Saved pretrained predicted intensity, ground truth, and fragment m/z for {len(valid_test_df)} test PSMs")
        # Store pretrained metrics for comparison
        pretrained_metrics = {
            'pcc': test_res[0]['PCC'].median(),
            'cos': test_res[0]['COS'].median(),
            'sa': test_res[0]['SA'].median(),
            'spc': test_res[0]['SPC'].median()
        }
        update_model_evaluation_metrics(out_dir, "ms2", pretrained_metrics=pretrained_metrics)
        logging.info("Testing pretrained MS2 model:\n" + str(test_res[-1]))
    
    # Training with per-epoch validation evaluation
    epoch_records = []
    if len(tr_df) > 0:
        logging.info(f"{len(tr_df)} PSMs for MS2 model training/transfer learning")
        logging.info(f"Training with fixed sequence length: 0")
        
        import torch
        # Prepare for training
        ms2_model = model_mgr.ms2_model
        if torch_compile:
            ms2_model.compile_model()
        ms2_model._prepare_training(tr_df, lr_to_train_ms2, 
                                    fragment_intensity_df=tr_inten_df,
                                    fragment_intensity_df_valid=tr_inten_valid_df)
        
        # EXACT PARITY: Re-enabling LR warmup (10 epochs) as in original script
        lr_scheduler = ms2_model._get_lr_schedule_with_warmup(warmup_epoch_to_train_ms2, epoch_to_train_ms2)
        
        best_test_loss = [float('inf')]
        epochs_without_improvement = [0]
        stopped_early = [False]
        def run_training_loop():
            for ep in range(epoch_to_train_ms2):
                epoch_started = time.monotonic()
                # Train one epoch
                batch_cost = ms2_model._train_one_epoch(
                    tr_df, ep, batch_size_to_train_ms2, False,
                    fragment_intensity_df=tr_inten_df,
                    fragment_intensity_df_valid=tr_inten_valid_df
                )
                lr_scheduler.step()
                train_loss = np.mean(batch_cost)
                
                # Evaluate on test data after each epoch
                test_loss_str = ""
                test_loss = 0
                test_losses = []
                validation_batches = []
                if len(test_psm_df) > 0:
                    # Calculate masked test loss (L1 loss consistent with training)
                    ms2_model.model.eval()
                    with torch.no_grad():
                        for nAA, df_group in test_psm_df.groupby('nAA', sort=False):
                            for i in range(0, len(df_group), batch_size_to_train_ms2):
                                batch_df = df_group.iloc[i:i + batch_size_to_train_ms2, :]
                                targets = ms2_model._get_targets_from_batch_df(
                                    batch_df, fragment_intensity_df=tr_inten_df
                                )
                                features = ms2_model._get_features_from_batch_df(batch_df)
                                predicts = ms2_model.model(*features)
                                
                                if tr_inten_valid_df is not None:
                                    # Apply masking consistent with training
                                    valid_targets = ms2_model._get_valid_targets_from_batch_df(
                                        batch_df, fragment_intensity_df_valid=tr_inten_valid_df
                                    )
                                    mask = torch.where(valid_targets <= 0, 1.0, 0.0)
                                    num_valid = torch.sum(mask)
                                    if num_valid > 0:
                                        loss = torch.nn.L1Loss(reduction='sum')(mask * predicts, mask * targets) / num_valid
                                        test_losses.append(loss.item())
                                    validation_mask = valid_targets.detach().cpu().numpy()
                                else:
                                    loss = torch.nn.L1Loss()(predicts, targets)
                                    test_losses.append(loss.item())
                                    validation_mask = None
                                validation_batches.append((
                                    predicts.detach().cpu().numpy(),
                                    targets.detach().cpu().numpy(),
                                    validation_mask,
                                ))
                    ms2_model.model.train()
                    
                    if test_losses:
                        test_loss = np.mean(test_losses)
                        test_loss_str = f", test_loss={test_loss}"
                
                ms2_model.history['epoch'].append(ep + 1)
                ms2_model.history['lr'].append(lr_scheduler.get_last_lr()[0])
                ms2_model.history['train_loss'].append(train_loss)
                ms2_model.history['test_loss'].append(test_loss if test_losses else 0)

                validation_metrics = ms2_epoch_metrics_from_batches(validation_batches)
                epoch_records.append({
                    "epoch": ep + 1,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "train_loss": train_loss,
                    "validation_loss": test_loss if test_losses else float("nan"),
                    "epoch_seconds": time.monotonic() - epoch_started,
                    **validation_metrics,
                })
                log_wandb_epoch("ms2", epoch_records[-1])
                if save_validation_checkpoints and test_losses:
                    save_validation_checkpoint(ms2_model, out_dir, "ms2", epoch_records[-1])

                # Checkpoint best model and track improvement
                improved = False
                if test_losses:
                    if test_loss < best_test_loss[0]:
                        best_test_loss[0] = test_loss
                        improved = True
                        epochs_without_improvement[0] = 0
                        if use_best_model:
                            ms2_model.save(out_dir + "/ms2_model.pt")
                            if verbose:
                                logging.info(f"New best MS2 model saved at epoch {ep + 1} with test_loss={test_loss:.4f}")
                    else:
                        epochs_without_improvement[0] += 1
                
                mask_str = " (Masked)" if tr_inten_valid_df is not None else ""
                logging.info(f"[Training] Epoch={ep + 1}{mask_str}, lr={lr_scheduler.get_last_lr()[0]:.1e}, train_loss={train_loss}{test_loss_str}")
                
                # Early stopping check
                if early_stop and epochs_without_improvement[0] >= patience:
                    logging.info(f"Early stopping triggered at epoch {ep + 1} (no improvement for {patience} epochs)")
                    stopped_early[0] = True
                    break
            
            torch.cuda.empty_cache()
        
        if device == 'cpu':
            from threadpoolctl import threadpool_limits
            # Allow BLAS to use the threads as well for matrix math performance
            with threadpool_limits(limits=threads, user_api="blas"):
                with threadpool_limits(limits=threads, user_api="openmp"):
                    if device == 'cpu':
                        torch.set_num_threads(threads)
                    run_training_loop()
        else:
            run_training_loop()
    
    # Pre-reload evaluation
    test_psm_df_clean = None
    if len(test_psm_df) > 0 and use_best_model:
        test_psm_df_clean = test_psm_df.copy()
        # Ensure indices remain integer to prevent IndexError in predict/normalize
        for col in ['frag_start_idx', 'frag_stop_idx', 'psm_id', 'ms2_scan']:
            if col in test_psm_df.columns:
                test_psm_df[col] = test_psm_df[col].astype(np.int64)

        pred_frag_df = model_mgr.ms2_model.predict(test_psm_df, reference_frag_df=tr_inten_df)
        test_res = calc_ms2_similarity_mask(
            test_psm_df, pred_frag_df, tr_inten_df, tr_inten_valid_df
        )
        logging.info("Performance of the model at the end of training (before reload):\n" + str(test_res[-1]))

    # Reload best model if saved and requested
    if use_best_model and os.path.exists(out_dir + "/ms2_model.pt"):
        model_mgr.ms2_model.load(out_dir + "/ms2_model.pt")
        logging.info("Reloaded best MS2 model from checkpoint.")

    # Post-training evaluation
    if len(test_psm_df) > 0:
        if test_psm_df_clean is not None:
            test_psm_df = test_psm_df_clean
        # verbose: 1=normal, 2=debug
        if verbose >=2:
            print("test_psm_df dtypes:")
            print(test_psm_df.dtypes)
        
        print("Using masking in metrics calculation ...")
        
        # Ensure indices remain integer to prevent IndexError in predict/normalize
        for col in ['frag_start_idx', 'frag_stop_idx', 'psm_id', 'ms2_scan']:
            if col in test_psm_df.columns:
                test_psm_df[col] = test_psm_df[col].astype(np.int64)

        pred_frag_df = model_mgr.ms2_model.predict(test_psm_df, reference_frag_df=tr_inten_df)
        test_res = calc_ms2_similarity_mask(
            test_psm_df, pred_frag_df, tr_inten_df, tr_inten_valid_df
        )
        out_test_res_to_file = os.path.join(out_dir, 'test_res_fine_tuned.csv')
        test_res[0].to_csv(out_test_res_to_file, index=False)
        try:
            stratified_psm_metrics = add_ms2_per_psm_error_metrics(
                test_res[0], pred_frag_df, tr_inten_df, tr_inten_valid_df,
                model_mgr.ms2_model.charged_frag_types,
            )
            save_stratified_ms2_diagnostics(out_dir, stratified_psm_metrics, nce)
        except Exception as e:
            # The core training result must remain available even if an unusual
            # third-party prediction dataframe cannot be stratified.
            logging.warning(f"Could not save stratified MS2 validation diagnostics: {e}")
        # Export predicted intensity matrix for fine-tuned model (test PSMs only)
        # Reuse out_frag_cols from pretrained export
        if mask_modloss:
            out_frag_cols = ['b_z1', 'b_z2', 'y_z1', 'y_z2']
        else:
            out_frag_cols = ['b_z1', 'b_z2', 'y_z1', 'y_z2',
                            'b_modloss_z1', 'b_modloss_z2', 'y_modloss_z1', 'y_modloss_z2']
        test_frag_rows = []
        valid_test_df = test_psm_df.dropna(subset=['frag_start_idx', 'frag_stop_idx'])
        for _, row in valid_test_df.iterrows():
            start, stop = int(row['frag_start_idx']), int(row['frag_stop_idx'])
            test_frag_rows.append(pred_frag_df.iloc[start:stop])
        if test_frag_rows:
            pd.concat(test_frag_rows, ignore_index=True)[out_frag_cols].to_csv(
                os.path.join(out_dir, 'test_pred_intensity_fine_tuned.tsv'), sep='\t', index=False)
        logging.info(f"Saved fine-tuned predicted intensity for {len(valid_test_df)} test PSMs")
        
        eval_label = "Final Evaluation (Best Model)" if use_best_model else "Final Evaluation (Last Model)"
        logging.info(f"{eval_label}:\n" + str(test_res[-1]))
        
        # Print improvement summary comparing pretrained vs fine-tuned
        if pretrained_metrics is not None:
            finetuned_metrics = {
                'pcc': test_res[0]['PCC'].median(),
                'cos': test_res[0]['COS'].median(),
                'sa': test_res[0]['SA'].median(),
                'spc': test_res[0]['SPC'].median()
            }
            use_finetuned_for_prediction = all(
                finetuned_metrics[metric] > pretrained_metrics[metric]
                for metric in ['pcc', 'cos', 'sa', 'spc']
            )
            update_model_evaluation_metrics(
                out_dir,
                "ms2",
                pretrained_metrics=pretrained_metrics,
                finetuned_metrics=finetuned_metrics,
                extra_metrics={"use_finetuned_for_prediction": use_finetuned_for_prediction}
            )
            logging.info("\n=== Improvement Summary (Median) ===")
            logging.info(f"{'Metric':<6} {'Pretrained':>12} {'Fine-tuned':>12} {'Improvement':>12}")
            logging.info("-" * 44)
            for metric in ['pcc', 'cos', 'sa', 'spc']:
                pre = pretrained_metrics[metric]
                post = finetuned_metrics[metric]
                delta = post - pre
                sign = "+" if delta >= 0 else ""
                logging.info(f"{metric.upper():<6} {pre:>12.4f} {post:>12.4f} {sign}{delta:>11.4f}")
            logging.info(f"MS2 prediction model selection: {'fine-tuned' if use_finetuned_for_prediction else 'pretrained'}")
        #elif pretrained_metrics is not None:
        #    update_model_evaluation_metrics(out_dir, "ms2", pretrained_metrics=pretrained_metrics)
    
    model_mgr.ms2_model.save(out_dir + "/ms2_model.pt")
    model_mgr.ms2_model.save_training_history(os.path.join(out_dir, "ms2_history.tsv"))
    model_mgr.ms2_model.plot_training_history(os.path.join(out_dir, "ms2_history.png"), mark_best=use_best_model)
    save_training_diagnostics(out_dir, "ms2", epoch_records, {
        "validation_split": "sequence-disjoint hold-out; used for per-epoch model selection",
        "training_psms": len(tr_df), "validation_psms": len(test_psm_df),
        "batch_size": batch_size_to_train_ms2, "epochs_requested": epoch_to_train_ms2,
        "warmup_epochs": warmup_epoch_to_train_ms2, "initial_learning_rate": lr_to_train_ms2,
        "masking_enabled": tr_inten_valid_df is not None,
    })
    return model_mgr



def train_rt(in_dir: str, out_dir: str, mode_type="general", device='gpu', threads=1, use_best_model=False, auto_tune=False, early_stop=False, patience=10, verbose=1,
             epoch_to_train_rt_ccs=40, warmup_epoch_to_train_rt_ccs=10, batch_size_to_train_rt_ccs=1024, lr_to_train_rt_ccs=0.0001, torch_compile=False,
             pretrained_model: str = None, adjust_batch_size_for_steps=True, from_scratch=False,
             save_validation_checkpoints=False, heavy_nterm_oversample=1.0,
             heavy_lysine_oversample=1.0, heavy_both_oversample=1.0,
             heavy_nterm_weight=1.0, heavy_lysine_weight=1.0, heavy_both_weight=1.0,
             rt_loss="l1", huber_delta=0.03, validation_file: str | None = None,
             validation_disjoint_by="sequence",
             trainable_strategy="all", backbone_learning_rate=None, weight_decay=0.0,
             organism_sampling=None, steps_per_epoch=None, sampling_seed=0,
             save_run_artifacts=True):
    """Train RT prediction model - replicates peptdeep exactly"""
    import pandas as pd
    import numpy as np
    import math
    import time
    from models import (ModelManager, ORGANISM_SAMPLING_STRATEGIES, psm_sampling_with_important_mods,
                        oversample_rt_heavy_strata, add_rt_heavy_loss_weights,
                        evaluate_linear_regression, rt_heavy_strata)
    
    import torch
    pd.options.mode.chained_assignment = None
    a = pd.read_csv(in_dir + "/rt_train_data.tsv", sep="\t", dtype={'mod_sites': str, 'mods': str})
    
    if "sequence" not in a.columns and "peptide" in a.columns:
        a["sequence"] = a["peptide"]
    
    a['mod_sites'] = a['mod_sites'].fillna("").astype(str)
    a['mods'] = a['mods'].fillna("").astype(str)
    if organism_sampling is not None:
        if organism_sampling not in ORGANISM_SAMPLING_STRATEGIES:
            raise ValueError(f"Unknown organism sampling strategy: {organism_sampling}")
        if steps_per_epoch is None or int(steps_per_epoch) < 1:
            raise ValueError("Organism sampling requires positive steps_per_epoch")
        if "source_organism" not in a.columns:
            raise ValueError("Organism sampling requires source_organism in rt_train_data.tsv")
    
    # Initialize ModelManager
    if mode_type == 'general':
        mask_modloss = True
    elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
        mask_modloss = False
    elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        mask_modloss = True
    else:
        print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
        mask_modloss = True
    model_mgr = ModelManager(mask_modloss=mask_modloss, device=device, out_dir=out_dir)
    
    if device == 'cpu':
        torch.set_num_threads(threads)
    # Load pretrained models
    if from_scratch:
        logging.info("Training from scratch (skipping all pretrained model loading)")
    else:
        try:
            if mode_type == 'general':
                model_mgr.load_installed_models('generic', model_list=['rt'])
            elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
                model_mgr.load_installed_models('phos', model_list=['rt'])
            elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
                model_mgr.load_installed_models('ubi', model_list=['rt'])
            else:
                print(f"Warning: Unknown mode_type: {mode_type}, loading generic model")
                model_mgr.load_installed_models('generic', model_list=['rt'])
        except Exception as e:
            print(f"Warning: Could not load pretrained models: {e}")

        # Load user-provided pretrained model if available
        if pretrained_model and os.path.exists(pretrained_model):
            try:
                print(f"Loading user-provided RT model from: {pretrained_model}")
                model_mgr.load_external_models(rt_model_file=pretrained_model)
            except Exception as e:
                print(f"Error loading user-provided RT model: {e}")

    logging.info(f"RT Model Parameter Size: {model_mgr.rt_model.get_parameter_num():,}")

    psm_num_per_mod_to_train_rt_ccs = 50
    top_n_mods_to_train = 10

    n_test_rt = max(1, np.min([1000, int(math.ceil(a.shape[0] * 0.1) - 10)]))
    psm_num_to_test_rt_ccs = n_test_rt
    psm_num_to_train_rt_ccs = a.shape[0] - n_test_rt
    if validation_file:
        psm_num_to_train_rt_ccs = a.shape[0]

    # Training parameters
    if auto_tune:
        epoch_to_train_rt_ccs, lr_to_train_rt_ccs = get_auto_tune_params(psm_num_to_train_rt_ccs, mode='rt_ccs')
        warmup_epoch_to_train_rt_ccs = get_warmup_epochs(epoch_to_train_rt_ccs)
        logging.info(f"[Auto-Tune] Scaling RT params for {psm_num_to_train_rt_ccs} peptides: epochs={epoch_to_train_rt_ccs}, lr={lr_to_train_rt_ccs:.1e}, batch={batch_size_to_train_rt_ccs}")
    elif adjust_batch_size_for_steps:
        batch_size_to_train_rt_ccs = adjust_batch_size(psm_num_to_train_rt_ccs, batch_size_to_train_rt_ccs)
        

    print(f"The number of peptides to train RT model: {psm_num_to_train_rt_ccs}")
    if not validation_file:
        print(f"The number of peptides to test RT model: {psm_num_to_test_rt_ccs}")

    # EXACT PARITY: PeptDeep v1 internal data preparation
    # 1. Median Grouping
    grouping_columns = ['sequence', 'mods', 'mod_sites']
    if organism_sampling is not None:
        grouping_columns.append('source_organism')
    psm_df = a.groupby(grouping_columns)[['rt_norm']].median().reset_index(drop=False)
    
    # 2. Sample (Seed 1337 default)
    base_tr_df = psm_sampling_with_important_mods(
        psm_df, psm_num_to_train_rt_ccs, top_n_mods_to_train, psm_num_per_mod_to_train_rt_ccs
    ).copy()
    
    # Add nAA metadata after sampling
    psm_df['nAA'] = psm_df['sequence'].str.len()
    tr_df, oversampling_counts = oversample_rt_heavy_strata(
        base_tr_df,
        nterm_factor=heavy_nterm_oversample,
        lysine_factor=heavy_lysine_oversample,
        both_factor=heavy_both_oversample,
        random_state=1337,
    )
    tr_df, loss_weights = add_rt_heavy_loss_weights(
        tr_df,
        nterm_weight=heavy_nterm_weight,
        lysine_weight=heavy_lysine_weight,
        both_weight=heavy_both_weight,
    )
    tr_df['nAA'] = tr_df['sequence'].str.len()
    tr_strata = rt_heavy_strata(tr_df)
    tr_df["heavy_dimethyl"] = tr_strata.ne("normal")
    tr_df["heavy_nterm_dimethyl"] = tr_strata.isin(("heavy_nterm", "heavy_both"))
    
    # 3. Use an externally prepared validation set with an explicit disjointness unit.
    if validation_file:
        test_psm_df = pd.read_csv(validation_file, sep="\t", dtype={'mod_sites': str, 'mods': str})
        test_psm_df['mods'] = test_psm_df['mods'].fillna("").astype(str)
        test_psm_df['mod_sites'] = test_psm_df['mod_sites'].fillna("").astype(str)
        test_psm_df = test_psm_df.groupby(['sequence','mods','mod_sites'])[['rt_norm']].median().reset_index(drop=False)
        test_psm_df['nAA'] = test_psm_df['sequence'].str.len()
        if validation_disjoint_by == "sequence":
            if set(test_psm_df.sequence) & set(tr_df.sequence):
                raise ValueError("External RT validation file shares sequences with training data")
        elif validation_disjoint_by == "peptidoform":
            key = ["sequence", "mods", "mod_sites"]
            training_forms = set(map(tuple, tr_df[key].to_numpy()))
            validation_forms = set(map(tuple, test_psm_df[key].to_numpy()))
            overlap = training_forms & validation_forms
            if overlap:
                raise ValueError(
                    f"External RT validation file shares {len(overlap)} exact peptidoforms "
                    "with training data"
                )
        else:
            raise ValueError(
                "validation_disjoint_by must be 'sequence' or 'peptidoform', got "
                f"{validation_disjoint_by!r}"
            )
    else:
        test_psm_df = psm_df[~psm_df.sequence.isin(set(tr_df.sequence))].copy()
        if len(test_psm_df) > psm_num_to_test_rt_ccs:
            test_psm_df = test_psm_df.sample(psm_num_to_test_rt_ccs, random_state=1337).copy()
        elif len(test_psm_df) == 0:
            test_psm_df = psm_df.copy()
    
    logging.info(f"{len(tr_df)} PSMs for RT model training/transfer learning")
    if validation_file:
        logging.info("%s peptidoforms in external RT validation set: %s", len(test_psm_df), validation_file)
    logging.info(
        "RT Heavy-Dimethyl oversampling factors: N-term=%.2f, K=%.2f, both=%.2f; added=%s",
        heavy_nterm_oversample, heavy_lysine_oversample, heavy_both_oversample, oversampling_counts,
    )
    logging.info(
        "RT Heavy-Dimethyl loss weights: normal=1.00, N-term=%.2f, K=%.2f, both=%.2f",
        heavy_nterm_weight, heavy_lysine_weight, heavy_both_weight,
    )
    steps = int(steps_per_epoch) if steps_per_epoch is not None else get_steps_per_epoch(len(tr_df), batch_size_to_train_rt_ccs)
    logging.info(f"Training steps per epoch: {steps}")
    if organism_sampling is not None:
        logging.info(
            "Organism sampling: strategy=%s, steps=%s, examples/epoch budget=%s, source rows=%s",
            organism_sampling, steps, steps * batch_size_to_train_rt_ccs,
            tr_df["source_organism"].astype(str).value_counts().sort_index().to_dict(),
        )
    
    rt_model = model_mgr.rt_model
    rt_model.set_loss(rt_loss, huber_delta=huber_delta)
    logging.info("RT training loss: %s%s", rt_loss, f" (delta={huber_delta})" if rt_loss == "huber" else "")
    if torch_compile:
        rt_model.compile_model()

    # Pre-training evaluation
    pretrained_metrics = None
    if len(test_psm_df) > 0:
        try:
            import scipy.stats
            from sklearn.metrics import mean_absolute_error,r2_score
            test_psm_df['rt_pred'] = rt_model.predict(test_psm_df.copy())['rt_pred']
            y_obs = test_psm_df['rt_norm'].values
            y_pred = test_psm_df['rt_pred'].values
            pre_r = scipy.stats.pearsonr(y_obs, y_pred)[0]
            pre_mae = mean_absolute_error(y_obs, y_pred)
            r2 = r2_score(y_obs, y_pred)
            pretrained_metrics = {'r2': r2, 'mae_normalized': pre_mae}
            pretrained_metrics.update(rt_stratified_metrics(test_psm_df, test_psm_df['rt_pred'], prefix=""))
            update_model_evaluation_metrics(out_dir, "rt", pretrained_metrics=pretrained_metrics)
            logging.info(f"Testing pretrained RT model: R2={r2:.4f}, MAE (normalized)={pre_mae:.4f}")
            
            if save_run_artifacts:
                # These visual artefacts are useful for a single experiment,
                # but are deliberately omitted in a large sweep.  The numeric
                # metrics above remain identical in both modes.
                test_psm_df.to_csv(os.path.join(out_dir, "rt_test_pretrained.tsv"), sep="\t", index=False)
                plot_rt(in_dir, out_dir, test_file="rt_test_pretrained.tsv",
                        output_plot="rt_performance_pretrained.png",
                        title="RT Prediction Performance (Pretrained)",
                        denormalize_rt=False)
        except Exception as e:
            logging.warning(f"Could not perform pre-training RT evaluation: {e}")
    
    rt_model._prepare_training(tr_df, lr_to_train_rt_ccs)
    finetuning_metadata = rt_model.configure_finetuning(
        strategy=trainable_strategy,
        learning_rate=lr_to_train_rt_ccs,
        backbone_learning_rate=backbone_learning_rate,
        weight_decay=weight_decay,
    )
    logging.info("RT fine-tuning policy: %s", json.dumps(finetuning_metadata, sort_keys=True))
    lr_scheduler = rt_model._get_lr_schedule_with_warmup(warmup_epoch_to_train_rt_ccs, epoch_to_train_rt_ccs)
    
    best_test_loss = [float('inf')]
    epochs_without_improvement = [0]
    stopped_early = [False]
    epoch_records = []

    def run_training_loop():
        for ep in range(epoch_to_train_rt_ccs):
            epoch_started = time.monotonic()
            batch_cost, train_subset_metrics = rt_model._train_one_epoch(
                tr_df, ep, batch_size_to_train_rt_ccs, False,
                loss_weight_column="_rt_loss_weight",
                loss_metric_columns=("heavy_dimethyl", "heavy_nterm_dimethyl"),
                organism_sampling=organism_sampling,
                steps_per_epoch=steps_per_epoch,
                sampling_seed=sampling_seed,
                sampling_epochs=epoch_to_train_rt_ccs,
            )
            lr_scheduler.step()
            train_loss = np.mean(batch_cost)
            
            test_loss_str = ""
            test_loss = 0
            test_losses = []
            validation_predictions = []
            validation_targets = []
            validation_frames = []
            if test_psm_df is not None and len(test_psm_df) > 0:
                rt_model.model.eval()
                with torch.no_grad():
                    for nAA, df_group in test_psm_df.groupby('nAA', sort=False):
                        for i in range(0, len(df_group), batch_size_to_train_rt_ccs):
                            batch_df = df_group.iloc[i:i + batch_size_to_train_rt_ccs, :]
                            targets = rt_model._get_targets_from_batch_df(batch_df)
                            features = rt_model._get_features_from_batch_df(batch_df)
                            predicts = rt_model.model(*features)
                            loss = torch.nn.L1Loss()(predicts, targets)
                            test_losses.append(loss.item())
                            validation_predictions.append(predicts.detach().cpu().numpy())
                            validation_targets.append(targets.detach().cpu().numpy())
                            validation_frames.append(batch_df.copy())
                rt_model.model.train()
                if test_losses:
                    test_loss = np.mean(test_losses)
                    test_loss_str = f", test_loss={test_loss}"
            
            rt_model.history['epoch'].append(ep + 1)
            rt_model.history['lr'].append(lr_scheduler.get_last_lr()[0])
            rt_model.history['train_loss'].append(train_loss)
            rt_model.history['test_loss'].append(test_loss if test_losses else 0)

            validation_metrics = regression_epoch_metrics(
                np.concatenate(validation_targets) if validation_targets else [],
                np.concatenate(validation_predictions) if validation_predictions else [],
            )
            if validation_frames and validation_predictions:
                validation_frame = pd.concat(validation_frames, ignore_index=True)
                validation_frame["rt_pred"] = np.concatenate(validation_predictions).reshape(-1)
                validation_metrics.update(
                    rt_stratified_metrics(validation_frame, validation_frame["rt_pred"], prefix="validation_")
                )
                validation_metrics.update(
                    rt_subset_loss_metrics(
                        validation_frame,
                        validation_frame["rt_pred"],
                        loss_name=rt_loss,
                        huber_delta=huber_delta,
                        prefix="validation_",
                    )
                )
            for name, value in {**train_subset_metrics, **validation_metrics}.items():
                if name.endswith("_loss"):
                    rt_model.history.setdefault(name, []).append(value)
            epoch_records.append({
                "epoch": ep + 1,
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "train_loss": train_loss,
                "validation_loss": test_loss if test_losses else float("nan"),
                "epoch_seconds": time.monotonic() - epoch_started,
                **train_subset_metrics,
                **validation_metrics,
            })
            log_wandb_epoch("rt", epoch_records[-1])
            if save_validation_checkpoints and test_losses:
                save_validation_checkpoint(rt_model, out_dir, "rt", epoch_records[-1])

            if test_losses:
                if test_loss < best_test_loss[0]:
                    best_test_loss[0] = test_loss
                    epochs_without_improvement[0] = 0
                    if use_best_model:
                        rt_model.save(out_dir + "/rt_model.pt")
                else:
                    epochs_without_improvement[0] += 1
            
            logging.info(f"[Training] Epoch={ep + 1}, lr={lr_scheduler.get_last_lr()[0]:.1e}, train_loss={train_loss}{test_loss_str}")
            
            if early_stop and epochs_without_improvement[0] >= patience:
                stopped_early[0] = True
                break
        torch.cuda.empty_cache()

    if device == 'cpu':
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=threads, user_api="blas"):
            with threadpool_limits(limits=threads, user_api="openmp"):
                torch.set_num_threads(threads)
                run_training_loop()
    else:
        run_training_loop()
    if use_best_model and os.path.exists(out_dir + "/rt_model.pt"):
        rt_model.load(out_dir + "/rt_model.pt")
    
    rt_model.save(out_dir + "/rt_model.pt")

    # Post-training evaluation
    if len(test_psm_df) > 0:
        try:
            import scipy.stats
            from sklearn.metrics import mean_absolute_error,r2_score
            test_psm_df['rt_pred'] = rt_model.predict(test_psm_df.copy())['rt_pred']
            y_obs = test_psm_df['rt_norm'].values
            y_pred = test_psm_df['rt_pred'].values
            post_r = scipy.stats.pearsonr(y_obs, y_pred)[0]
            post_mae = mean_absolute_error(y_obs, y_pred)
            post_r2 = r2_score(y_obs, y_pred)
            
            eval_label = "Final Evaluation (Best Model)" if use_best_model else "Final Evaluation (Last Model)"
            logging.info(f"{eval_label}: R2={post_r2:.4f}, MAE (normalized)={post_mae:.4f}")
            
            finetuned_metrics = {'r2': post_r2, 'mae_normalized': post_mae}
            finetuned_metrics.update(rt_stratified_metrics(test_psm_df, test_psm_df['rt_pred'], prefix=""))
            update_model_evaluation_metrics(
                out_dir,
                "rt",
                pretrained_metrics=pretrained_metrics,
                finetuned_metrics=finetuned_metrics
            )
            if save_run_artifacts:
                plot_rt_final_heavy_comparison(out_dir, pretrained_metrics, finetuned_metrics)

            if pretrained_metrics is not None:
                logging.info("\n=== Improvement Summary (RT) ===")
                logging.info(f"{'Metric':<20} {'Pretrained':>12} {'Fine-tuned':>12} {'Improvement':>12}")
                logging.info("-" * 60)
                for metric in ['r2', 'mae_normalized']:
                    pre = pretrained_metrics[metric]
                    if metric.startswith("r2"):
                        post = post_r2
                    elif metric.startswith("mae"):
                        post = post_mae
                    else:
                        print(f"Unknown metric: {metric}")
                        continue
                    delta = post - pre
                    # For R2, higher is better; for MAE, lower is better
                    if metric.startswith("r2"):
                        improved = "+" if delta >= 0 else ""
                    elif metric.startswith("mae"):
                        improved = "+" if delta <= 0 else ""
                    else:
                        print(f"Unknown metric: {metric}")
                        continue
                    logging.info(f"{metric:<20} {pre:>12.4f} {post:>12.4f} {improved}{delta:>11.4f}")
            
            if save_run_artifacts:
                test_psm_df.to_csv(os.path.join(out_dir, "rt_test.tsv"), sep="\t", index=False)
                from models import rt_heavy_strata
                strata_frame = test_psm_df.copy()
                strata_frame["heavy_stratum"] = rt_heavy_strata(strata_frame).values
                strata_rows = []
                for stratum, group in strata_frame.groupby("heavy_stratum", sort=False):
                    metrics = regression_epoch_metrics(group["rt_norm"], group["rt_pred"], prefix="metric")
                    strata_rows.append({"stratum": stratum, **{
                        key.removeprefix("metric_"): value for key, value in metrics.items()
                    }})
                pd.DataFrame(strata_rows).to_csv(
                    os.path.join(out_dir, "rt_validation_strata_metrics.csv"), index=False
                )
        except Exception as e:
            logging.warning(f"Could not perform post-training RT evaluation: {e}")

    rt_model.save_training_history(os.path.join(out_dir, "rt_history.tsv"))
    if save_run_artifacts:
        rt_model.plot_training_history(os.path.join(out_dir, "rt_history.png"), mark_best=use_best_model)
    save_training_diagnostics(out_dir, "rt", epoch_records, {
        "validation_split": (
            f"{validation_disjoint_by}-disjoint external hold-out; used for per-epoch model selection"
            if validation_file else "internal sequence-disjoint hold-out; used for per-epoch model selection"
        ),
        "training_psms": len(tr_df), "validation_psms": len(test_psm_df),
        "batch_size": batch_size_to_train_rt_ccs, "epochs_requested": epoch_to_train_rt_ccs,
        "warmup_epochs": warmup_epoch_to_train_rt_ccs, "initial_learning_rate": lr_to_train_rt_ccs,
        "heavy_nterm_oversample": heavy_nterm_oversample,
        "heavy_lysine_oversample": heavy_lysine_oversample,
        "heavy_both_oversample": heavy_both_oversample,
        "oversampling_added_rows": oversampling_counts,
        "heavy_nterm_loss_weight": heavy_nterm_weight,
        "heavy_lysine_loss_weight": heavy_lysine_weight,
        "heavy_both_loss_weight": heavy_both_weight,
        "loss_weights_by_stratum": loss_weights,
        "rt_loss": rt_loss,
        "huber_delta": huber_delta if rt_loss == "huber" else None,
        "finetuning": finetuning_metadata,
        "weight_decay": weight_decay,
        "organism_sampling": organism_sampling,
        "steps_per_epoch": steps_per_epoch,
        "examples_per_epoch_budget": (
            int(steps_per_epoch) * batch_size_to_train_rt_ccs
            if steps_per_epoch is not None else None
        ),
        "source_rows": (
            tr_df["source_organism"].astype(str).value_counts().sort_index().to_dict()
            if "source_organism" in tr_df.columns else None
        ),
    }, write_plots=save_run_artifacts)
    return model_mgr

def train_ccs(in_dir: str, out_dir: str, mode_type="general", device='gpu', threads=1, use_best_model=False, auto_tune=False, early_stop=False, patience=10, verbose=1,
              epoch_to_train_rt_ccs=40, warmup_epoch_to_train_rt_ccs=10, batch_size_to_train_rt_ccs=1024, lr_to_train_rt_ccs=0.0001, torch_compile=False,
              pretrained_model: str = None,
              adjust_batch_size_for_steps=True, from_scratch=False,
              save_validation_checkpoints=False):
    """Train CCS prediction model - replicates peptdeep exactly"""
    import pandas as pd
    import numpy as np
    import math
    import time
    from models import (ModelManager, psm_sampling_with_important_mods,
                        evaluate_linear_regression)
    import torch
    pd.options.mode.chained_assignment = None
    a = pd.read_csv(in_dir + "/ccs_train_data.tsv", sep="\t", dtype={'mod_sites': str, 'mods': str})
    
    if "sequence" not in a.columns and "peptide" in a.columns:
        a["sequence"] = a["peptide"]
    a['mod_sites'] = a['mod_sites'].fillna("").astype(str)
    a['mods'] = a['mods'].fillna("").astype(str)
    
    if mode_type == 'general':
        mask_modloss = True
    elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
        mask_modloss = False
    elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
        mask_modloss = True
    else:
        mask_modloss = True
    model_mgr = ModelManager(mask_modloss=mask_modloss, device=device, out_dir=out_dir)
    
    if device == 'cpu':
        torch.set_num_threads(threads)
    if from_scratch:
        logging.info("Training from scratch (skipping all pretrained model loading)")
    else:
        try:
            if mode_type == 'general':
                model_mgr.load_installed_models('generic', model_list=['ccs'])
            elif mode_type == 'phos' or mode_type == 'phospho' or mode_type == 'phosphorylation':
                model_mgr.load_installed_models('phos', model_list=['ccs'])
            elif mode_type == 'ubi' or mode_type == 'ubiquitination' or mode_type == 'kGG' or mode_type == 'kgg':
                model_mgr.load_installed_models('ubi', model_list=['ccs'])
            else:
                model_mgr.load_installed_models('generic', model_list=['ccs'])
        except Exception as e:
            print(f"Warning: Could not load pretrained models: {e}")

        # Load user-provided pretrained model if available
        if pretrained_model and os.path.exists(pretrained_model):
            try:
                print(f"Loading user-provided CCS model from: {pretrained_model}")
                model_mgr.load_external_models(ccs_model_file=pretrained_model)
            except Exception as e:
                print(f"Error loading user-provided CCS model: {e}")

    logging.info(f"CCS Model Parameter Size: {model_mgr.ccs_model.get_parameter_num():,}")

    n_test_ccs = max(1, np.min([1000, int(math.ceil(a.shape[0] * 0.1) - 10)]))
    psm_num_to_test_rt_ccs = n_test_ccs
    psm_num_to_train_rt_ccs = a.shape[0] - n_test_ccs

    if auto_tune:
        epoch_to_train_rt_ccs, lr_to_train_rt_ccs = get_auto_tune_params(psm_num_to_train_rt_ccs, mode='rt_ccs')
        warmup_epoch_to_train_rt_ccs = get_warmup_epochs(epoch_to_train_rt_ccs)
    elif adjust_batch_size_for_steps:
        batch_size_to_train_rt_ccs = adjust_batch_size(psm_num_to_train_rt_ccs, batch_size_to_train_rt_ccs)

    print(f"The number of peptides to train CCS model: {psm_num_to_train_rt_ccs}")
    print(f"The number of peptides to test CCS model: {psm_num_to_test_rt_ccs}")

    if 'ccs' not in a.columns and 'mobility' in a.columns:
        a['ccs'] = mobility_to_ccs_for_df(a, 'mobility')
    
    # 1. Median Grouping
    psm_df = a.groupby(['sequence','mods','mod_sites','charge'])[['mobility','ccs']].median().reset_index(drop=False)
    
    # 2. Sample
    tr_df = psm_sampling_with_important_mods(
        psm_df, psm_num_to_train_rt_ccs, 10, 50
    ).copy()
    
    # Add nAA metadata after sampling
    psm_df['nAA'] = psm_df['sequence'].str.len()
    tr_df['nAA'] = tr_df['sequence'].str.len()
    
    test_psm_df = psm_df[~psm_df.sequence.isin(set(tr_df.sequence))].copy()
    if len(test_psm_df) > psm_num_to_test_rt_ccs:
        test_psm_df = test_psm_df.sample(psm_num_to_test_rt_ccs, random_state=1337).copy()
    elif len(test_psm_df) == 0:
        test_psm_df = psm_df.copy()
    
    logging.info(f"{len(tr_df)} PSMs for CCS model training/transfer learning")
    steps = get_steps_per_epoch(len(tr_df), batch_size_to_train_rt_ccs)
    logging.info(f"Training steps per epoch: {steps}")

    ccs_model = model_mgr.ccs_model
    if torch_compile:
        ccs_model.compile_model()

    # Pre-training evaluation
    pretrained_metrics = None
    if len(test_psm_df) > 0:
        try:
            import scipy.stats
            from sklearn.metrics import median_absolute_error,r2_score
            test_psm_df['ccs_pred'] = ccs_model.predict(test_psm_df.copy())['ccs_pred']
            # Also need mobility_pred for plotting
            test_psm_df['mobility_pred'] = ccs_model.ccs_to_mobility_pred(test_psm_df)
            
            y_obs_ccs = test_psm_df['ccs'].values
            y_pred_ccs = test_psm_df['ccs_pred'].values
            pre_r_ccs = scipy.stats.pearsonr(y_obs_ccs, y_pred_ccs)[0]
            pre_mae_ccs = median_absolute_error(y_obs_ccs, y_pred_ccs)
            pre_r2_ccs = r2_score(y_obs_ccs, y_pred_ccs)
            
            y_obs_mobility = test_psm_df['mobility'].values
            y_pred_mobility = test_psm_df['mobility_pred'].values
            pre_r_mobility = scipy.stats.pearsonr(y_obs_mobility, y_pred_mobility)[0]
            pre_mae_mobility = median_absolute_error(y_obs_mobility, y_pred_mobility)
            pre_r2_mobility = r2_score(y_obs_mobility, y_pred_mobility)
            
            pretrained_metrics = {
                'r2_ccs': pre_r2_ccs,
                'mae_ccs': pre_mae_ccs,
                'r2_mobility': pre_r2_mobility,
                'mae_mobility': pre_mae_mobility
            }
            update_model_evaluation_metrics(out_dir, "im", pretrained_metrics=pretrained_metrics)
            logging.info(f"Testing pretrained CCS model (CCS space): R2 (CCS)={pre_r2_ccs:.4f}, MAE (CCS)={pre_mae_ccs:.4f}, R2 (mobility)={pre_r2_mobility:.4f}, MAE (mobility)={pre_mae_mobility:.4f}")
            
            # Save pretrained results and generate plot
            test_psm_df.to_csv(os.path.join(out_dir, "mobility_test_pretrained.tsv"), sep="\t", index=False)
            plot_mobility(out_dir, test_file="mobility_test_pretrained.tsv", 
                          output_plot="mobility_performance_pretrained.png", title="Mobility Prediction Performance (Pretrained)")
        except Exception as e:
            logging.warning(f"Could not perform pre-training CCS evaluation: {e}")
    
    ccs_model._prepare_training(tr_df, lr_to_train_rt_ccs)
    lr_scheduler = ccs_model._get_lr_schedule_with_warmup(warmup_epoch_to_train_rt_ccs, epoch_to_train_rt_ccs)
    
    best_test_loss = [float('inf')]
    epochs_without_improvement = [0]
    stopped_early = [False]
    epoch_records = []

    def run_training_loop():
        for ep in range(epoch_to_train_rt_ccs):
            epoch_started = time.monotonic()
            batch_cost = ccs_model._train_one_epoch(tr_df, ep, batch_size_to_train_rt_ccs, False)
            lr_scheduler.step()
            train_loss = np.mean(batch_cost)
            
            test_loss_str = ""
            test_loss = 0
            test_losses = []
            validation_predictions = []
            validation_targets = []
            if test_psm_df is not None and len(test_psm_df) > 0:
                ccs_model.model.eval()
                with torch.no_grad():
                    for nAA, df_group in test_psm_df.groupby('nAA', sort=False):
                        for i in range(0, len(df_group), batch_size_to_train_rt_ccs):
                            batch_df = df_group.iloc[i:i + batch_size_to_train_rt_ccs, :]
                            targets = ccs_model._get_targets_from_batch_df(batch_df)
                            features = ccs_model._get_features_from_batch_df(batch_df)
                            predicts = ccs_model.model(*features)
                            loss = torch.nn.L1Loss()(predicts, targets)
                            test_losses.append(loss.item())
                            validation_predictions.append(predicts.detach().cpu().numpy())
                            validation_targets.append(targets.detach().cpu().numpy())
                ccs_model.model.train()
                if test_losses:
                    test_loss = np.mean(test_losses)
                    test_loss_str = f", test_loss={test_loss}"
            
            ccs_model.history['epoch'].append(ep + 1)
            ccs_model.history['lr'].append(lr_scheduler.get_last_lr()[0])
            ccs_model.history['train_loss'].append(train_loss)
            ccs_model.history['test_loss'].append(test_loss if test_losses else 0)

            validation_metrics = regression_epoch_metrics(
                np.concatenate(validation_targets) if validation_targets else [],
                np.concatenate(validation_predictions) if validation_predictions else [],
            )
            epoch_records.append({
                "epoch": ep + 1,
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "train_loss": train_loss,
                "validation_loss": test_loss if test_losses else float("nan"),
                "epoch_seconds": time.monotonic() - epoch_started,
                **validation_metrics,
            })
            log_wandb_epoch("ccs", epoch_records[-1])
            if save_validation_checkpoints and test_losses:
                save_validation_checkpoint(ccs_model, out_dir, "ccs", epoch_records[-1])

            if test_losses:
                if test_loss < best_test_loss[0]:
                    best_test_loss[0] = test_loss
                    epochs_without_improvement[0] = 0
                    if use_best_model:
                        ccs_model.save(out_dir + "/ccs_model.pt")
                else:
                    epochs_without_improvement[0] += 1
            
            logging.info(f"[Training] Epoch={ep + 1}, lr={lr_scheduler.get_last_lr()[0]:.1e}, train_loss={train_loss}{test_loss_str}")
            
            if early_stop and epochs_without_improvement[0] >= patience:
                stopped_early[0] = True
                break
        torch.cuda.empty_cache()

    if device == 'cpu':
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=threads, user_api="blas"):
            with threadpool_limits(limits=threads, user_api="openmp"):
                torch.set_num_threads(threads)
                run_training_loop()
    else:
        run_training_loop()
    if use_best_model and os.path.exists(out_dir + "/ccs_model.pt"):
        ccs_model.load(out_dir + "/ccs_model.pt")
    
    ccs_model.save(out_dir + "/ccs_model.pt")

    # Post-training evaluation
    if len(test_psm_df) > 0:
        try:
            import scipy.stats
            from sklearn.metrics import median_absolute_error,r2_score
            test_psm_df['ccs_pred'] = ccs_model.predict(test_psm_df.copy())['ccs_pred']
            # Also need mobility_pred for plotting
            test_psm_df['mobility_pred'] = ccs_model.ccs_to_mobility_pred(test_psm_df)
            
            y_obs_ccs = test_psm_df['ccs'].values
            y_pred_ccs = test_psm_df['ccs_pred'].values
            post_r_ccs = scipy.stats.pearsonr(y_obs_ccs, y_pred_ccs)[0]
            post_mae_ccs = median_absolute_error(y_obs_ccs, y_pred_ccs)
            post_r2_ccs = r2_score(y_obs_ccs, y_pred_ccs)
            
            y_obs_mobility = test_psm_df['mobility'].values
            y_pred_mobility = test_psm_df['mobility_pred'].values
            post_r_mobility = scipy.stats.pearsonr(y_obs_mobility, y_pred_mobility)[0]
            post_mae_mobility = median_absolute_error(y_obs_mobility, y_pred_mobility)
            post_r2_mobility = r2_score(y_obs_mobility, y_pred_mobility)
            
            eval_label = "Final Evaluation (Best Model)" if use_best_model else "Final Evaluation (Last Model)"
            logging.info(f"{eval_label}: R2 (CCS)={post_r2_ccs:.4f}, MAE (CCS)={post_mae_ccs:.4f}, R2 (mobility)={post_r2_mobility:.4f}, MAE (mobility)={post_mae_mobility:.4f}")
            
            finetuned_metrics = {
                'r2_ccs': post_r2_ccs,
                'mae_ccs': post_mae_ccs,
                'r2_mobility': post_r2_mobility,
                'mae_mobility': post_mae_mobility
            }
            update_model_evaluation_metrics(
                out_dir,
                "im",
                pretrained_metrics=pretrained_metrics,
                finetuned_metrics=finetuned_metrics
            )

            if pretrained_metrics is not None:
                logging.info("\n=== Improvement Summary (CCS) ===")
                logging.info(f"{'Metric':<20} {'Pretrained':>12} {'Fine-tuned':>12} {'Improvement':>12}")
                logging.info("-" * 60)
                for metric in ['r2_ccs', 'mae_ccs', 'r2_mobility', 'mae_mobility']:
                    pre = pretrained_metrics[metric]
                    if metric == 'r2_ccs':
                        post = post_r2_ccs
                    elif metric == 'mae_ccs':
                        post = post_mae_ccs
                    elif metric == 'r2_mobility':
                        post = post_r2_mobility
                    elif metric == 'mae_mobility':
                        post = post_mae_mobility
                    delta = post - pre
                    # For R2, higher is better; for MAE, lower is better
                    if metric.startswith("r2"):
                        improved = "+" if delta >= 0 else ""
                    elif metric.startswith("mae"):
                        improved = "+" if delta <= 0 else ""
                    else:
                        print(f"Unknown metric: {metric}")
                        continue
                    logging.info(f"{metric:<20} {pre:>12.4f} {post:>12.4f} {improved}{delta:>11.4f}")
            
            test_psm_df.to_csv(os.path.join(out_dir, "mobility_test.tsv"), sep="\t", index=False)
        except Exception as e:
            logging.warning(f"Could not perform post-training CCS evaluation: {e}")

    ccs_model.save_training_history(os.path.join(out_dir, "ccs_history.tsv"))
    ccs_model.plot_training_history(os.path.join(out_dir, "ccs_history.png"), mark_best=use_best_model)
    save_training_diagnostics(out_dir, "ccs", epoch_records, {
        "validation_split": "sequence-disjoint hold-out; used for per-epoch model selection",
        "training_psms": len(tr_df), "validation_psms": len(test_psm_df),
        "batch_size": batch_size_to_train_rt_ccs, "epochs_requested": epoch_to_train_rt_ccs,
        "warmup_epochs": warmup_epoch_to_train_rt_ccs, "initial_learning_rate": lr_to_train_rt_ccs,
    })
    return model_mgr

def plot_rt(in_dir, out_dir, test_file="rt_test.tsv", 
            output_plot="rt_performance_test.png", 
            title="RT Prediction Performance",
            denormalize_rt=True):
    import os
    import json
    import pandas as pd
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import scipy.stats
        from sklearn.metrics import median_absolute_error, r2_score
    except ImportError:
        print("Required libraries for plotting (matplotlib, sklearn, scipy) not found.")
        return

    # Backwards compatibility for callers using positional args
    if test_file == "rt_test.tsv" and not os.path.exists(os.path.join(out_dir, test_file)):
         # If default not found, maybe it was passed in differently? 
         # But usually we'd expect it in out_dir.
         pass

    full_test_path = os.path.join(out_dir, test_file)
    if not os.path.exists(full_test_path):
        print(f"Test file {full_test_path} not found. Skipping plotting.")
        return

    # meta.json should be in in_dir
    meta_file = os.path.join(in_dir, "meta.json") if in_dir else None
    if meta_file and os.path.exists(meta_file):
        print(f"Found meta.json: {meta_file}")
    elif meta_file:
        print(f"meta.json not found in {in_dir}")
    
    rt_min = 0.0
    rt_max = 1.0
    if meta_file and os.path.exists(meta_file):
        try:
            # Use utf-8-sig to handle BOM which might cause JSON parsing 
            with open(meta_file, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                # Robust parsing: handle unescaped backslashes commonly found in Windows paths
                import re
                content = re.sub(r'(?<!\\)\\(?![\\"])', lambda m: r'\\', content)
                try:
                    meta_data = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback for very broken paths
                    with open(meta_file, 'r', encoding='utf-8-sig') as f2:
                         content = f2.read()
                    content = content.replace('\\', '\\\\').replace('\\\\"', '\\"')
                    meta_data = json.loads(content)
                
                if meta_data:
                    all_rt_min = []
                    all_rt_max = []
                    for key in meta_data:
                        if isinstance(meta_data[key], dict):
                            if 'rt_min' in meta_data[key]:
                                all_rt_min.append(meta_data[key]['rt_min'])
                            if 'rt_max' in meta_data[key]:
                                all_rt_max.append(meta_data[key]['rt_max'])
                    if all_rt_min: rt_min = min(all_rt_min)
                    if all_rt_max: rt_max = max(all_rt_max)
                    print(f"Using global RT normalization factors: rt_min={rt_min}, rt_max={rt_max}")
        except Exception as e:
            print(f"Error reading meta.json: {e}")

    try:
        df = pd.read_csv(full_test_path, sep="\t")
        if 'rt_norm' not in df.columns or 'rt_pred' not in df.columns:
            print(f"Columns 'rt_norm' or 'rt_pred' not found in {full_test_path}")
            return
            
        # Denormalize to minutes
        y_obs = df['rt_norm'] * (rt_max - rt_min) + rt_min
        if denormalize_rt:
            y_pred = df['rt_pred'] * (rt_max - rt_min) + rt_min
        else:
            y_pred = df['rt_pred']

        cor = scipy.stats.pearsonr(y_obs, y_pred)[0]
        mae = median_absolute_error(y_obs, y_pred)
        r2 = r2_score(y_obs, y_pred)
        range_95 = np.percentile(np.abs(y_obs - y_pred), 95)

        plt.rcParams['figure.figsize'] = [5, 5]
        plt.scatter(y_obs, y_pred, s=4, c="blue", alpha=0.5)
        
        max_rt = max(rt_max, y_obs.max(), y_pred.max())*1.02

        if denormalize_rt:
            plt.plot([0, max_rt], [0, max_rt], color='red', linestyle='--', linewidth=1)
        
        if denormalize_rt:
            stats_text = f"PCC = {cor:.4f}\nMAE = {mae:.2f} min\n$R^2$ = {r2:.4f}\n95% Range = {range_95:.2f} min\nN = {len(df)}"
        else:
            stats_text = f"PCC = {cor:.4f}\nN = {len(df)}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel('Observed RT (Minute)')
        if denormalize_rt:
            plt.ylabel('Predicted RT (Minute)')
        else:
            plt.ylabel('Predicted RT')
        plt.xlim(0, max_rt)
        if denormalize_rt:
            plt.ylim(0, max_rt)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.title(title)
        
        output_plot_path = os.path.join(out_dir, output_plot)
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        print(f"RT performance plot saved to {output_plot_path}")
    except Exception as e:
        print(f"Error generating RT plot: {e}")


def plot_rt_final_heavy_comparison(out_dir, pretrained_metrics, finetuned_metrics):
    """Plot final normalized MAE for global and heavy-dimethyl RT subsets."""
    if not pretrained_metrics or not finetuned_metrics:
        logging.warning("Skipping final heavy-dimethyl plot: pretrained or fine-tuned metrics are unavailable.")
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logging.warning("Skipping final heavy-dimethyl plot: matplotlib is unavailable.")
        return

    subsets = (
        ("All validation", "mae_normalized"),
        ("Any heavy\ndimethyl", "heavy_dimethyl_mae"),
        ("N-term heavy\ndimethyl", "heavy_nterm_dimethyl_mae"),
    )
    available = [
        (label, key, pretrained_metrics.get(key), finetuned_metrics.get(key))
        for label, key in subsets
        if pretrained_metrics.get(key) is not None and finetuned_metrics.get(key) is not None
    ]
    if not available:
        logging.warning("Skipping final heavy-dimethyl plot: no comparable subset metrics available.")
        return

    labels, _, pretrained, finetuned = zip(*available)
    positions = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.bar(positions - width / 2, pretrained, width, label="Pretrained", color="#8b95a1")
    axis.bar(positions + width / 2, finetuned, width, label="Fine-tuned", color="#1f77b4")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Normalized RT MAE")
    axis.set_title("Final RT MAE by Heavy-Dimethyl Subset")
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    path = os.path.join(out_dir, "rt_final_heavy_dimethyl_comparison.png")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    logging.info("Final heavy-dimethyl RT comparison plot saved to %s", path)


def plot_mobility(out_dir, test_file="mobility_test.tsv", output_plot="mobility_performance_test.png", title="Mobility Prediction Performance"):
    import os
    import json
    import pandas as pd
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import scipy.stats
        from sklearn.metrics import median_absolute_error, r2_score
    except ImportError:
        print("Required libraries for plotting (matplotlib, sklearn, scipy) not found.")
        return

    full_test_path = os.path.join(out_dir, test_file)
    if not os.path.exists(full_test_path):
        print(f"Test file {full_test_path} not found. Skipping plotting.")
        return

    try:
        df = pd.read_csv(full_test_path, sep="\t")
        if 'mobility' not in df.columns or 'mobility_pred' not in df.columns:
            print(f"Columns 'mobility' or 'mobility_pred' not found in {full_test_path}")
            return
            
        y_obs = df['mobility']
        y_pred = df['mobility_pred']

        cor = scipy.stats.pearsonr(y_obs, y_pred)[0]
        mae = median_absolute_error(y_obs, y_pred)
        r2 = r2_score(y_obs, y_pred)
        range_95 = np.percentile(np.abs(y_obs - y_pred), 95)

        plt.rcParams['figure.figsize'] = [5, 5]
        plt.scatter(y_obs, y_pred, s=4, c="blue", alpha=0.5)
        
        max_mobility = max(y_obs.max(), y_pred.max())*1.02
        min_mobility = min(y_obs.min(), y_pred.min())*0.98
        plt.plot([min_mobility, max_mobility], [min_mobility, max_mobility], color='red', linestyle='--', linewidth=1)
        
        stats_text = f"PCC = {cor:.4f}\nMAE = {mae:.4f}\n$R^2$ = {r2:.4f}\n95% Range = {range_95:.4f}\nN = {len(df)}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.xlabel('Observed Mobility')
        plt.ylabel('Predicted Mobility')
        plt.xlim(min_mobility, max_mobility)
        plt.ylim(min_mobility, max_mobility)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.title(title)
        
        output_plot_path = os.path.join(out_dir, output_plot)
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        print(f"Mobility performance plot saved to {output_plot_path}")
    except Exception as e:
        print(f"Error generating mobility plot: {e}")


def generate_html_report(out_dir, in_dir=None, tf_type="all", args_dict=None, command_line=None):
    """Generate a self-contained HTML report with all training plots and metrics."""
    import base64
    import pandas as pd
    from datetime import datetime

    def img_to_base64(path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def diagnostics_plots_html(model_name):
        """Embed all per-epoch diagnostic plots, one image per logged metric."""
        directory = os.path.join(out_dir, "training_diagnostics", "plots", model_name)
        if not os.path.isdir(directory):
            return ""
        plots = []
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".png"):
                continue
            encoded = img_to_base64(os.path.join(directory, filename))
            if encoded:
                caption = os.path.splitext(filename)[0].replace("_", " ").title()
                plots.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
        if not plots:
            return ""
        return "<h3>Per-epoch diagnostics</h3><div class=\"diagnostic-plots\">" + "".join(plots) + "</div>"

    def stratified_ms2_html():
        """Embed the highest-value subgroup plots while retaining all plots on disk."""
        root = os.path.join(out_dir, "training_diagnostics", "stratified_ms2", "plots")
        selections = [
            ("trypticity", "SA"), ("n_terminal_modification", "SA"),
            ("dimethyl", "SA"), ("charge", "SA"),
            ("length_bin", "SA"), ("collision_energy_bin", "SA"),
        ]
        plots = []
        for dimension, metric in selections:
            filename = os.path.join(root, dimension, f"{metric}.png")
            encoded = img_to_base64(filename)
            if encoded:
                caption = f"{metric}: {dimension.replace('_', ' ')}"
                plots.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
        if not plots:
            return ""
        return ("<p>Complete PSM-level data and all metric plots are in "
                "<code>training_diagnostics/stratified_ms2/</code>.</p>"
                "<div class=\"diagnostic-plots\">" + "".join(plots) + "</div>")

    def read_metrics_csv(path, sep=","):
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            return None

    def metrics_summary_html(df, metrics_cols=None):
        """Generate an HTML table summarizing median/mean for key columns."""
        if df is None or df.empty:
            return ""
        if metrics_cols is None:
            metrics_cols = [c for c in df.columns if c in ('PCC', 'COS', 'SA', 'SPC')]
        if not metrics_cols:
            return ""
        rows = []
        for col in metrics_cols:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            rows.append(f"<tr><td>{col}</td><td>{vals.median():.4f}</td><td>{vals.mean():.4f}</td>"
                        f"<td>{vals.min():.4f}</td><td>{vals.max():.4f}</td></tr>")
        if not rows:
            return ""
        return (f'<table class="metrics"><tr><th>Metric</th><th>Median</th><th>Mean</th>'
                f'<th>Min</th><th>Max</th></tr>{"".join(rows)}</table>')

    def rt_metrics_html(path):
        """Generate HTML for RT test metrics from a TSV file."""
        df = read_metrics_csv(path, sep="\t")
        if df is None or df.empty:
            return ""
        html = ""
        try:
            import numpy as np
            y_obs = df['rt_norm'].values
            y_pred = df['rt_pred'].values
            from scipy.stats import pearsonr
            from sklearn.metrics import median_absolute_error, r2_score
            pcc = pearsonr(y_obs, y_pred)[0]
            mae = median_absolute_error(y_obs, y_pred)
            r2 = r2_score(y_obs, y_pred)
            html = (f'<table class="metrics"><tr><th>Metric</th><th>Value</th></tr>'
                    f'<tr><td>PCC</td><td>{pcc:.4f}</td></tr>'
                    f'<tr><td>R²</td><td>{r2:.4f}</td></tr>'
                    f'<tr><td>MAE (normalized)</td><td>{mae:.4f}</td></tr>'
                    f'<tr><td>N</td><td>{len(df)}</td></tr></table>')
        except Exception:
            pass
        return html

    def mobility_metrics_html(path):
        """Generate HTML for mobility test metrics from a TSV file."""
        df = read_metrics_csv(path, sep="\t")
        if df is None or df.empty:
            return ""
        html = ""
        try:
            import numpy as np
            from scipy.stats import pearsonr
            from sklearn.metrics import median_absolute_error, r2_score
            rows = []
            for obs_col, pred_col, label in [('mobility', 'mobility_pred', 'Mobility'),
                                              ('ccs', 'ccs_pred', 'CCS')]:
                if obs_col in df.columns and pred_col in df.columns:
                    y_obs = df[obs_col].dropna().values
                    y_pred = df[pred_col].dropna().values
                    n = min(len(y_obs), len(y_pred))
                    if n > 0:
                        y_obs, y_pred = y_obs[:n], y_pred[:n]
                        pcc = pearsonr(y_obs, y_pred)[0]
                        mae = median_absolute_error(y_obs, y_pred)
                        r2 = r2_score(y_obs, y_pred)
                        rows.append(f'<tr><td>{label} PCC</td><td>{pcc:.4f}</td></tr>')
                        rows.append(f'<tr><td>{label} R²</td><td>{r2:.4f}</td></tr>')
                        rows.append(f'<tr><td>{label} MAE</td><td>{mae:.4f}</td></tr>')
            if rows:
                rows.append(f'<tr><td>N</td><td>{len(df)}</td></tr>')
                html = f'<table class="metrics"><tr><th>Metric</th><th>Value</th></tr>{"".join(rows)}</table>'
        except Exception:
            pass
        return html

    def history_table_html(path, sep="\t"):
        """Generate an HTML summary of training history."""
        df = read_metrics_csv(path, sep=sep)
        if df is None or df.empty:
            return ""
        last = df.iloc[-1]
        best_idx = df['test_loss'].idxmin() if 'test_loss' in df.columns and df['test_loss'].sum() > 0 else None
        rows = [f'<tr><td>Total epochs</td><td>{len(df)}</td></tr>',
                f'<tr><td>Final train loss</td><td>{last.get("train_loss", "N/A"):.6f}</td></tr>']
        if 'test_loss' in df.columns and df['test_loss'].sum() > 0:
            rows.append(f'<tr><td>Final test loss</td><td>{last["test_loss"]:.6f}</td></tr>')
            if best_idx is not None:
                rows.append(f'<tr><td>Best test loss</td><td>{df.loc[best_idx, "test_loss"]:.6f} (epoch {df.loc[best_idx, "epoch"]})</td></tr>')
        return f'<table class="metrics"><tr><th>Training</th><th>Value</th></tr>{"".join(rows)}</table>'

    # Build sections
    sections = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Configuration summary ---
    config_html = f'<p><strong>Date:</strong> {timestamp}</p>'
    if command_line:
        config_html += f'<p><strong>Command:</strong></p><pre style="background:#eee;padding:10px;border-radius:4px;overflow-x:auto;">{command_line}</pre>'
    if args_dict:
        config_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(args_dict.items()))
        config_html += f'<table class="metrics"><tr><th>Parameter</th><th>Value</th></tr>{config_rows}</table>'
    sections.append(("Configuration", config_html))

    # --- RT section ---
    if tf_type in ("all", "test", "rt"):
        rt_html = ""
        # Pretrained
        rt_pre_metrics = rt_metrics_html(os.path.join(out_dir, "rt_test_pretrained.tsv"))
        if rt_pre_metrics:
            rt_html += "<h3>Pretrained Model</h3>" + rt_pre_metrics
        rt_pre_img = img_to_base64(os.path.join(out_dir, "rt_performance_pretrained.png"))
        if rt_pre_img:
            rt_html += f'<img src="data:image/png;base64,{rt_pre_img}" alt="RT Pretrained">'
        # Fine-tuned
        rt_post_metrics = rt_metrics_html(os.path.join(out_dir, "rt_test.tsv"))
        if rt_post_metrics:
            rt_html += "<h3>Fine-tuned Model</h3>" + rt_post_metrics
        rt_post_img = img_to_base64(os.path.join(out_dir, "rt_performance_test.png"))
        if rt_post_img:
            rt_html += f'<img src="data:image/png;base64,{rt_post_img}" alt="RT Fine-tuned">'
        rt_heavy_final_img = img_to_base64(os.path.join(out_dir, "rt_final_heavy_dimethyl_comparison.png"))
        if rt_heavy_final_img:
            rt_html += ("<h3>Final Heavy-Dimethyl Comparison</h3>"
                        f'<img src="data:image/png;base64,{rt_heavy_final_img}" '
                        'alt="RT Heavy-Dimethyl Comparison">')
        # Training history
        rt_hist = history_table_html(os.path.join(out_dir, "rt_history.tsv"))
        if rt_hist:
            rt_html += "<h3>Training History</h3>" + rt_hist
        rt_hist_img = img_to_base64(os.path.join(out_dir, "rt_history.png"))
        if rt_hist_img:
            rt_html += f'<img src="data:image/png;base64,{rt_hist_img}" alt="RT History">'
        rt_html += diagnostics_plots_html("rt")
        if rt_html:
            sections.append(("RT Model", rt_html))

    # --- CCS/Mobility section ---
    if tf_type in ("all", "test", "ccs"):
        ccs_html = ""
        mob_pre_metrics = mobility_metrics_html(os.path.join(out_dir, "mobility_test_pretrained.tsv"))
        if mob_pre_metrics:
            ccs_html += "<h3>Pretrained Model</h3>" + mob_pre_metrics
        mob_pre_img = img_to_base64(os.path.join(out_dir, "mobility_performance_pretrained.png"))
        if mob_pre_img:
            ccs_html += f'<img src="data:image/png;base64,{mob_pre_img}" alt="Mobility Pretrained">'
        mob_post_metrics = mobility_metrics_html(os.path.join(out_dir, "mobility_test.tsv"))
        if mob_post_metrics:
            ccs_html += "<h3>Fine-tuned Model</h3>" + mob_post_metrics
        mob_post_img = img_to_base64(os.path.join(out_dir, "mobility_performance_test.png"))
        if mob_post_img:
            ccs_html += f'<img src="data:image/png;base64,{mob_post_img}" alt="Mobility Fine-tuned">'
        ccs_hist = history_table_html(os.path.join(out_dir, "ccs_history.tsv"))
        if ccs_hist:
            ccs_html += "<h3>Training History</h3>" + ccs_hist
        ccs_hist_img = img_to_base64(os.path.join(out_dir, "ccs_history.png"))
        if ccs_hist_img:
            ccs_html += f'<img src="data:image/png;base64,{ccs_hist_img}" alt="CCS History">'
        ccs_html += diagnostics_plots_html("ccs")
        if ccs_html:
            sections.append(("CCS / Mobility Model", ccs_html))

    # --- MS2 section ---
    if tf_type in ("all", "test", "ms2"):
        ms2_html = ""
        # Pretrained
        ms2_pre = read_metrics_csv(os.path.join(out_dir, "test_res_pretrained.csv"))
        if ms2_pre is not None and not ms2_pre.empty:
            ms2_html += "<h3>Pretrained Model</h3>" + metrics_summary_html(ms2_pre)
        # Fine-tuned
        ms2_post = read_metrics_csv(os.path.join(out_dir, "test_res_fine_tuned.csv"))
        if ms2_post is not None and not ms2_post.empty:
            ms2_html += "<h3>Fine-tuned Model</h3>" + metrics_summary_html(ms2_post)
        # Training history
        ms2_hist = history_table_html(os.path.join(out_dir, "ms2_history.tsv"))
        if ms2_hist:
            ms2_html += "<h3>Training History</h3>" + ms2_hist
        ms2_hist_img = img_to_base64(os.path.join(out_dir, "ms2_history.png"))
        if ms2_hist_img:
            ms2_html += f'<img src="data:image/png;base64,{ms2_hist_img}" alt="MS2 History">'
        ms2_html += diagnostics_plots_html("ms2")
        stratified_html = stratified_ms2_html()
        if stratified_html:
            ms2_html += "<h3>Stratified validation</h3>" + stratified_html
        if ms2_html:
            sections.append(("MS2 Model", ms2_html))

    # --- Assemble HTML ---
    sections_html = ""
    for title, content in sections:
        sections_html += f'<div class="section"><h2>{title}</h2>{content}</div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Fine-tuning Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 960px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 6px; margin-top: 30px; }}
    h3 {{ color: #7f8c8d; margin-top: 18px; }}
    .section {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .metrics {{ border-collapse: collapse; margin: 10px 0; }}
    .metrics th, .metrics td {{ border: 1px solid #ddd; padding: 8px 14px; text-align: left; }}
    .metrics th {{ background: #3498db; color: white; }}
    .metrics tr:nth-child(even) {{ background: #f9f9f9; }}
    .metrics tr:hover {{ background: #eaf2f8; }}
    img {{ max-width: 480px; height: auto; margin: 10px 4px; border: 1px solid #ddd;
           border-radius: 4px; }}
    .diagnostic-plots {{ display: flex; flex-wrap: wrap; gap: 12px; }}
    .diagnostic-plots figure {{ margin: 0; max-width: 480px; }}
    .diagnostic-plots figcaption {{ color: #555; font-size: 0.9em; text-align: center; }}
    .footer {{ text-align: center; color: #999; font-size: 0.85em; margin-top: 30px; }}
</style>
</head>
<body>
<h1>Model Fine-tuning Report</h1>
{sections_html}
<div class="footer">Generated on {timestamp}</div>
</body>
</html>"""

    report_path = os.path.join(out_dir, "training_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    logging.info(f"HTML report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--out_dir", type=str, default="./")
    parser.add_argument("--out_prefix", type=str, default="")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--instrument", type=str, default="Astral")
    parser.add_argument("--tf_type", type=str, default="all")
    parser.add_argument("--nce", type=float, default=27.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--mode", type=str, default="general")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument('--log_transform', action='store_true', help='log transform intensity data')
    parser.add_argument('--no_masking', action='store_true', help='disable masking for training')
    parser.add_argument('--user_mod', default=None, help='User defined modification')
    parser.add_argument('--use_grid_nce_search', action='store_true', help='Use grid search for NCE and instrument (default: False)')
    parser.add_argument('--use_best_model', action='store_true', help='Save model with lowest test loss (default: False, saves last epoch)')
    parser.add_argument('--auto_tune', action='store_true', help='Dynamic scaling of hyperparameters for 1k-100k range (default: False)')
    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping when validation loss stops improving (default: False)')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait for improvement before stopping (default: 10)')
    parser.add_argument('--save_validation_checkpoints', action='store_true',
                        help='Save RT, CCS, and MS2 model snapshots after every validation evaluation')
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads to use")
    parser.add_argument("--torch_compile", action="store_true", help="Compile model with torch.compile for speed (CPU/GPU)")
    parser.add_argument("--profile", action='store_true', help="Profile the script execution")
    parser.add_argument("--ms2_model", type=str, default=None, help="Pretrained MS2 model path")
    parser.add_argument("--rt_model", type=str, default=None, help="Pretrained RT model path")
    parser.add_argument(
        "--rt_validation_file", type=str, default=None,
        help="External RT validation TSV; disjointness is controlled separately",
    )
    parser.add_argument(
        "--rt_validation_disjoint_by", choices=("sequence", "peptidoform"), default="sequence",
        help="Disjointness unit enforced for --rt_validation_file (default: sequence)",
    )
    parser.add_argument("--ccs_model", type=str, default=None, help="Pretrained CCS model path")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch (random initialization, skip all pretrained model loading)")
    parser.add_argument("--ms2_epoch", type=int, default=20, help="Number of epochs for MS2 model training (default: 20)")
    parser.add_argument("--rt_epoch", type=int, default=40, help="Number of epochs for RT/CCS model training (default: 40)")
    parser.add_argument("--rt_batch_size", type=int, default=1024, help="Batch size for RT/CCS model training (default: 1024)")
    parser.add_argument("--rt_lr", type=float, default=0.0001, help="Learning rate for RT/CCS model training (default: 0.0001)")
    parser.add_argument("--rt_warmup_epoch", type=int, default=10, help="Warm-up epochs for RT/CCS model training (default: 10)")
    parser.add_argument("--rt_heavy_nterm_oversample", type=float, default=1.0,
                        help="RT oversampling factor for Heavy-Dimethyl N-terminal rows")
    parser.add_argument("--rt_heavy_lysine_oversample", type=float, default=1.0,
                        help="RT oversampling factor for Heavy-Dimethyl lysine rows")
    parser.add_argument("--rt_heavy_both_oversample", type=float, default=1.0,
                        help="RT oversampling factor for doubly Heavy-Dimethyl rows")
    parser.add_argument("--rt_heavy_nterm_weight", type=float, default=1.0,
                        help="Per-peptide RT loss weight for Heavy-Dimethyl N-terminal rows")
    parser.add_argument("--rt_heavy_lysine_weight", type=float, default=1.0,
                        help="Per-peptide RT loss weight for Heavy-Dimethyl lysine rows")
    parser.add_argument("--rt_heavy_both_weight", type=float, default=1.0,
                        help="Per-peptide RT loss weight for doubly Heavy-Dimethyl rows")
    parser.add_argument("--rt_loss", choices=("l1", "huber", "mse"), default="l1",
                        help="Optimizer loss for RT training; validation selection remains MAE")
    parser.add_argument("--rt_huber_delta", type=float, default=0.03,
                        help="Huber transition delta when --rt_loss huber is selected")
    parser.add_argument("--rt_trainable", choices=("head", "rnn_head", "cnn_head", "all", "discriminative"),
                        default="all", help="RT transfer-learning freeze policy")
    parser.add_argument("--rt_backbone_lr", type=float, default=None,
                        help="Backbone LR for discriminative or partial RT fine-tuning")
    parser.add_argument("--rt_weight_decay", type=float, default=0.0,
                        help="Adam weight decay for RT fine-tuning")
    parser.add_argument("--no_adjust_rt_batch_size", action="store_true",
                        help="Keep --rt_batch_size fixed instead of targeting a minimum step count")
    parser.add_argument(
        "--rt_organism_sampling",
        choices=(
            "natural", "round_robin_batches", "ratio_1_1_1", "ratio_2_1_1",
            "ratio_4_1_1", "ratio_8_1_1", "ratio_1_2_1", "ratio_1_1_2",
            "ratio_2_2_1", "ratio_2_1_2", "ratio_4_2_1", "ratio_4_1_2",
            "inverse_frequency", "temperature_0_25", "temperature_0_50",
            "temperature_0_75", "curriculum_natural_to_uniform",
            "curriculum_uniform_to_natural", "curriculum_ce4_to_uniform",
            "curriculum_uniform_to_ce4",
        ),
        default=None, help="Per-epoch source-organism sampling policy",
    )
    parser.add_argument(
        "--rt_steps_per_epoch", type=int, default=None,
        help=("Hard optimizer-step budget. With no organism policy, draws a "
              "deterministic with-replacement epoch sample so Heavy-Dimethyl "
              "oversampling changes exposure rather than update count."),
    )
    parser.add_argument(
        "--sweep_minimal_artifacts", action="store_true",
        help=("For large RT sweeps, retain models and numeric metrics but skip per-run PNG diagnostics, "
              "HTML reports and redundant prediction tables. Does not change training or validation metrics."),
    )
    parser.add_argument("--ms2_batch_size", type=int, default=512, help="Batch size for MS2 model training (default: 512)")
    parser.add_argument("--ms2_lr", type=float, default=0.0001, help="Learning rate for MS2 model training (default: 0.0001)")
    parser.add_argument("--ms2_warmup_epoch", type=int, default=10, help="Number of warmup epochs for MS2 model training (default: 10)")
    # the max number of samples to use for testing ms2
    parser.add_argument("--ms2_max_test_samples", type=int, default=1000, help="Number of samples to use for testing MS2 model (default: 1000)")
    parser.add_argument("--wandb-config", type=str, default=None, help="JSON configuration enabling W&B tracking")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (default: output directory name)")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group for campaign or sweep")
    parser.add_argument("--wandb-run-id", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.rt_steps_per_epoch is not None and args.rt_steps_per_epoch < 1:
        parser.error("--rt_steps_per_epoch must be positive")

    # Early stopping requires best model checkpointing
    if args.early_stop and not args.use_best_model:
        args.use_best_model = True
        logging.info("Enabling --use_best_model since --early_stop is active")

    import psutil
    n_physical = psutil.cpu_count(logical=False) or 4
    
    if args.threads is None:
        args.threads = n_physical
        
    configure_env_for_device(args.device, args.threads)

    # Add models.py to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    print(f"Using local models.py from: {script_dir}")

    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    reset_model_evaluation_metrics(out_dir)
    start_wandb(
        args.wandb_config,
        args.wandb_run_name or os.path.basename(os.path.abspath(out_dir)),
        args.wandb_group,
        {key: value for key, value in vars(args).items()
         if key not in {"wandb_config", "wandb_run_name", "wandb_group", "wandb_run_id"}},
        args.wandb_run_id,
    )

    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg') # Set backend to non-interactive

    
    # Critical CPU tuning (matching ai.py)
    device = args.device.strip().lower()
    if device == "cpu":
        torch.set_num_threads(args.threads)
        # Re-enabling interop threads if physical core count is high
        if args.threads > 8:
            try:
                torch.set_num_interop_threads(2)
            except RuntimeError:
                pass

    if args.verbose >= 2:
        print("torch intra", torch.get_num_threads())
        print("torch interop", torch.get_num_interop_threads())
    
    # Set seed - MUST be after torch import, BEFORE check_models_from_docker (matching ai.py)
    set_seed(int(args.seed))
    
    if args.verbose >= 2:
        print_thread_state(tag="Thread/BLAS summary (startup)")
    
    from threadpoolctl import threadpool_limits, threadpool_info
    
    if args.user_mod is not None:
        from models import add_user_defined_modifications
        user_mod = args.user_mod
        user_mod = re.sub(r'^\"',"",user_mod)
        user_mod = re.sub(r'\"$',"",user_mod)
        umods = user_mod.split(";")
        mod_dict = {}
        # format: 597.1216,K,469.0266,C(19)H(17)Cl(2)N(3)O(5)S(1)
        # format: modification name,amino acid,monoisotopic mass,composition
        for umod in umods:
            um = umod.split(",")
            mod_name = um[0]+"@"+um[1]
            mod_dict[mod_name] = {}
            mod_dict[mod_name]["composition"] = um[3]
            mod_dict[mod_name]["modloss_composition"] = ""
        print("User defined modifications (Applied):")
        print(mod_dict)
        add_user_defined_modifications(mod_dict)



    tf_type = args.tf_type
    
    def run_main():
        if tf_type == "all" or tf_type == "test":
            if tf_type == "test":
                print("Test mode ...")
            train_rt(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                device=args.device, threads=args.threads, use_best_model=args.use_best_model, 
                                auto_tune=args.auto_tune, early_stop=args.early_stop, patience=args.patience, 
                                verbose=args.verbose, torch_compile=args.torch_compile,
                                pretrained_model=args.rt_model, from_scratch=args.from_scratch,
                                epoch_to_train_rt_ccs=args.rt_epoch,
                                warmup_epoch_to_train_rt_ccs=args.rt_warmup_epoch,
                                batch_size_to_train_rt_ccs=args.rt_batch_size,
                                lr_to_train_rt_ccs=args.rt_lr,
                                heavy_nterm_oversample=args.rt_heavy_nterm_oversample,
                                heavy_lysine_oversample=args.rt_heavy_lysine_oversample,
                                heavy_both_oversample=args.rt_heavy_both_oversample,
                                heavy_nterm_weight=args.rt_heavy_nterm_weight,
                                heavy_lysine_weight=args.rt_heavy_lysine_weight,
                                heavy_both_weight=args.rt_heavy_both_weight,
                                rt_loss=args.rt_loss, huber_delta=args.rt_huber_delta,
                                trainable_strategy=args.rt_trainable,
                                backbone_learning_rate=args.rt_backbone_lr,
                                weight_decay=args.rt_weight_decay,
                                organism_sampling=args.rt_organism_sampling,
                                steps_per_epoch=args.rt_steps_per_epoch,
                                sampling_seed=args.seed,
                                adjust_batch_size_for_steps=not args.no_adjust_rt_batch_size,
                                save_validation_checkpoints=args.save_validation_checkpoints,
                                validation_file=args.rt_validation_file,
                                validation_disjoint_by=args.rt_validation_disjoint_by,
                                save_run_artifacts=not args.sweep_minimal_artifacts)
            if not args.sweep_minimal_artifacts:
                plot_rt(in_dir, out_dir)
            # CCS training if data exists and has sufficient samples
            ccs_data_path = os.path.join(in_dir, "ccs_train_data.tsv")
            if os.path.exists(ccs_data_path):
                import pandas as pd
                ccs_data = pd.read_csv(ccs_data_path, sep="\t")
                if ccs_data.shape[0] >= 100:
                    train_ccs(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                            device=args.device, threads=args.threads, use_best_model=args.use_best_model,
                                            auto_tune=args.auto_tune, early_stop=args.early_stop, patience=args.patience, 
                                            verbose=args.verbose, torch_compile=args.torch_compile,
                                            pretrained_model=args.ccs_model, from_scratch=args.from_scratch,
                                            epoch_to_train_rt_ccs=args.rt_epoch,
                                            warmup_epoch_to_train_rt_ccs=args.rt_warmup_epoch,
                                            batch_size_to_train_rt_ccs=args.rt_batch_size,
                                            lr_to_train_rt_ccs=args.rt_lr,
                                            save_validation_checkpoints=args.save_validation_checkpoints)
                    plot_mobility(out_dir)
            train_ms2(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                log_transform=args.log_transform,
                                device=args.device, instrument=args.instrument, nce=args.nce,
                                use_valid=(not args.no_masking),
                                threads=args.threads, use_best_model=args.use_best_model, 
                                auto_tune=args.auto_tune, early_stop=args.early_stop,
                                patience=args.patience, verbose=args.verbose,
                                torch_compile=args.torch_compile,
                                pretrained_model=args.ms2_model, from_scratch=args.from_scratch,
                                epoch_to_train_ms2=args.ms2_epoch,
                                batch_size_to_train_ms2=args.ms2_batch_size,
                                lr_to_train_ms2=args.ms2_lr,
                                ms2_max_test_samples=args.ms2_max_test_samples,
                                warmup_epoch_to_train_ms2=args.ms2_warmup_epoch,
                                save_validation_checkpoints=args.save_validation_checkpoints)
        elif tf_type == "rt":
            train_rt(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                device=args.device, threads=args.threads, use_best_model=args.use_best_model, 
                                auto_tune=args.auto_tune, early_stop=args.early_stop, patience=args.patience, 
                                verbose=args.verbose, torch_compile=args.torch_compile,
                                pretrained_model=args.rt_model, from_scratch=args.from_scratch,
                                epoch_to_train_rt_ccs=args.rt_epoch,
                                warmup_epoch_to_train_rt_ccs=args.rt_warmup_epoch,
                                batch_size_to_train_rt_ccs=args.rt_batch_size,
                                lr_to_train_rt_ccs=args.rt_lr,
                                heavy_nterm_oversample=args.rt_heavy_nterm_oversample,
                                heavy_lysine_oversample=args.rt_heavy_lysine_oversample,
                                heavy_both_oversample=args.rt_heavy_both_oversample,
                                heavy_nterm_weight=args.rt_heavy_nterm_weight,
                                heavy_lysine_weight=args.rt_heavy_lysine_weight,
                                heavy_both_weight=args.rt_heavy_both_weight,
                                rt_loss=args.rt_loss, huber_delta=args.rt_huber_delta,
                                trainable_strategy=args.rt_trainable,
                                backbone_learning_rate=args.rt_backbone_lr,
                                weight_decay=args.rt_weight_decay,
                                organism_sampling=args.rt_organism_sampling,
                                steps_per_epoch=args.rt_steps_per_epoch,
                                sampling_seed=args.seed,
                                adjust_batch_size_for_steps=not args.no_adjust_rt_batch_size,
                                save_validation_checkpoints=args.save_validation_checkpoints,
                                validation_file=args.rt_validation_file,
                                validation_disjoint_by=args.rt_validation_disjoint_by,
                                save_run_artifacts=not args.sweep_minimal_artifacts)
            if not args.sweep_minimal_artifacts:
                plot_rt(in_dir, out_dir)
        elif tf_type == "ms2":
            train_ms2(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                log_transform=args.log_transform,
                                device=args.device, instrument=args.instrument, nce=args.nce,
                                use_valid=(not args.no_masking),
                                threads=args.threads, use_best_model=args.use_best_model, 
                                auto_tune=args.auto_tune, early_stop=args.early_stop,
                                patience=args.patience, verbose=args.verbose,
                                torch_compile=args.torch_compile,
                                pretrained_model=args.ms2_model, from_scratch=args.from_scratch,
                                epoch_to_train_ms2=args.ms2_epoch,
                                batch_size_to_train_ms2=args.ms2_batch_size,
                                lr_to_train_ms2=args.ms2_lr,
                                ms2_max_test_samples=args.ms2_max_test_samples,
                                warmup_epoch_to_train_ms2=args.ms2_warmup_epoch,
                                save_validation_checkpoints=args.save_validation_checkpoints)
        elif tf_type == "ccs":
            train_ccs(in_dir=in_dir, out_dir=out_dir, mode_type=args.mode,
                                device=args.device, threads=args.threads, use_best_model=args.use_best_model,
                                auto_tune=args.auto_tune, early_stop=args.early_stop, patience=args.patience, 
                                verbose=args.verbose, torch_compile=args.torch_compile,
                                pretrained_model=args.ccs_model, from_scratch=args.from_scratch,
                                epoch_to_train_rt_ccs=args.rt_epoch,
                                warmup_epoch_to_train_rt_ccs=args.rt_warmup_epoch,
                                batch_size_to_train_rt_ccs=args.rt_batch_size,
                                lr_to_train_rt_ccs=args.rt_lr,
                                save_validation_checkpoints=args.save_validation_checkpoints)
            plot_mobility(out_dir)

        # A self-contained report embeds all PNGs as base64 and is therefore
        # intentionally disabled for large RT sweeps.  It can be regenerated
        # later for validation-selected finalists.
        if not args.sweep_minimal_artifacts:
            args_dict = {k: str(v) for k, v in vars(args).items()}
            command_line = sys.executable + ' ' + ' '.join(sys.argv)
            generate_html_report(out_dir, in_dir=in_dir, tf_type=tf_type, args_dict=args_dict, command_line=command_line)

    if args.profile:
        import cProfile
        import pstats
        import io
        pr = cProfile.Profile()
        pr.enable()
        run_main()
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(30) # Print top 30 functions
        print(s.getvalue())
        # Also save to file
        prof_file = os.path.join(args.out_dir, "training_profile.prof")
        pr.dump_stats(prof_file)
        print(f"Detailed profile saved to {prof_file}")
    else:
        run_main()

    # This must happen before the explicit os._exit below, otherwise the W&B
    # client cannot flush the last metrics and diagnostic artifact.
    finish_wandb(out_dir)

    # Cleanup at exit (matching ai.py)
    try:
        from pyarrow import fs
        fs.finalize_s3()
    except:
        pass

    # Explicit GPU memory cleanup before forced exit
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all GPU ops to finish
        torch.cuda.empty_cache()  # Release cached memory back to driver

    os._exit(0)
