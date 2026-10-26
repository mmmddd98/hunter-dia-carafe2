# External Carafe2 RT experiments

This is the entry point for the **external Carafe2 RT fine-tuning** component
linked from the [project overview](https://github.com/mmmddd98/hunter-dia-model-adaptation).
The trainer is [`src/main/resources/py/v2/ai.py`](../src/main/resources/py/v2/ai.py),
with `--tf_type rt`. The local
[recipe launcher](../scripts/run_external_rt_recipe.py) translates the experiment
JSONs below into that trainer's native arguments. It uses the active Python
interpreter and requires the environment in the [main README](../README.md).

## Experiment configurations

| Experiment | Local configuration | Design |
| --- | --- | --- |
| Broad RT target/layer/LR screen | [broad-screen.json](../configs/external-rt/broad-screen.json) | 4 targets × 5 layer policies × 5 LR multipliers × 3 seeds = 300 runs |
| Focused LR refinement | [lr-refinement.json](../configs/external-rt/lr-refinement.json) | 2 targets × 2 policies × 6 LR multipliers × 3 seeds = 72 runs |
| Heavy-label weighting reported in the thesis | [heavy-finalists.json](../configs/external-rt/heavy-finalists.json) | Joint-anchor RT at LR 0.00045 and Chronologer HI at LR 0.0009; matched W0 (1/1/1) and W7 (1/1/6) controls; 3 seeds = 12 runs |
| Source-specific versus pooled models | [source-specific.json](../configs/external-rt/source-specific.json) | 12 target/source cells × 42 strategies × 3 seeds = 1,512 runs; fixed 120 optimiser steps per epoch |

Each file records the preparation source, its Git revision and SHA-256, common
native trainer arguments, seeds and named recipes. The snapshot describes the
preparation code at that revision; grid sizes describe the design, not a claim
that every run completed. The heavy-finalist recipes additionally use the
validation-selection and checkpoint metadata retained with the thesis analysis.
They describe the model-level weighting comparison, not the general external
RT model used for the final spectral-library searches.

All four designs use 40 epochs, 10 warm-up epochs, batch size 512 without
automatic adjustment, zero weight decay, and seeds 2024, 2025 and 2026.
`l1` is the trainer's spelling for MAE. Common heavy-label weights and sampling
factors are one unless a recipe overrides them. Weight/sampling triples are
ordered **N-terminal only, lysine only, both sites**. The source-specific JSON
contains each of the 42 exact strategy definitions, including Huber deltas,
learning rates, backbone rates, weights and oversampling factors.

The first three designs process the complete training set each epoch. Historical
oversampling could therefore increase optimiser updates. The source-specific
design uses replacement sampling within a fixed 120-step budget. The 294-step
budget described for later DIA-NN experiments is not a Carafe2 setting here.

## Prepare the data

Targets and globally fixed peptidoform splits come from
[ma/data_collect](https://github.com/mmmddd98/hunter-dia-workflows/tree/main/data_collect).
The relevant source files, relative to that repository's `data_collect/`, are:

| Stage | Source entry points |
| --- | --- |
| Global split and train-fitted RT targets | `prepare_finalrt_holdout_split.py`, `prepare_finalrt_trainfit_dataset.py` |
| Broad screen | `prepare_carafe2_trainfit_sweep.py` |
| LR refinement | `prepare_rt_lr_refinement_sweep.py` with `--family carafe2` |
| Full heavy-label search and target-scaled Huber thresholds | `prepare_heavy_dimethyl_rt_sweep.py` with `--family carafe2`, `heavy_dimethyl_sweep_common.py` |
| Source-specific target/source cells | `prepare_carafe2_source_specific_sweep.py` |
| Validation/final-test prediction and RT-minute conversion | `evaluate_carafe2_trainfit.py`, `evaluate_carafe2_source_specific_finalists.py`, `evaluate_external_carafe2_rt_models.py` |

For the original prepared dataset, targets contain
`train/diann_rt_input/rt_train_data.tsv`,
`validation/diann_rt_input/rt_validation_data.tsv` and `rt_alignment.json`.
Place the selected training TSV in an input directory as `rt_train_data.tsv`.
The Carafe2 trainer consumes that same table format. For source-specific runs,
use the matching `train/sources/<source_id>.tsv` and
`validation/sources/<source_id>.tsv` instead; the JSON records source IDs and
valid target/source combinations. Human joint-anchor is omitted from that
comparison because it duplicates the human source-native coordinate.

`anchor_irt` means joint-anchor RT, `linear` robust linear alignment,
`chronologer_hi` the Chronologer HI coordinate, `global_60min` the unaligned
common-duration control, and `source_native` the per-source duration control.
Alignment is fitted on training observations only and applied unchanged to
validation/test data. Do not substitute a differently normalised TSV.

## Run a documented recipe

From this repository root, with prepared joint-anchor training data under
`inputs/anchor_irt/carafe2_input/`:

```bash
python scripts/run_external_rt_recipe.py \
  --config configs/external-rt/heavy-finalists.json \
  --recipe anchor_irt/W7 --target anchor_irt --seed 2024 \
  --in-dir inputs/anchor_irt/carafe2_input \
  --validation-file inputs/anchor_irt/rt_validation_data.tsv \
  --out-dir outputs/anchor_irt_W7_seed2024 \
  --device cpu --dry-run
```

The dry run prints the exact native trainer command and writes nothing. Remove
`--dry-run` to train; choose `--device gpu` in the CUDA environment. The target
and scope labels validate recipe selection; the caller must supply TSVs in
that target coordinate and source scope. The launcher does not transform data.
Repeat W0/W7 for seeds 2024, 2025 and 2026, using separate output directories.
For the HI comparison choose `chronologer_hi/W0` or `chronologer_hi/W7` and
matching HI tables. `--rt-model` accepts an explicit compatible pretrained
AlphaPeptDeep RT checkpoint; the default is the trainer's generic pretrained
model loader.

For other designs select a key from the JSON's `recipes`, for example
`all/base` in the broad screen, `all/fifteen_x` in LR refinement, or
`all_mae_baseline` in the source-specific screen. Source-specific runs also
require `--scope`, for example `--target chronologer_hi --scope human_hunter`.
The launcher starts one run; the original campaign generators and GPU queue
runner in `data_collect/` manage complete campaigns.

## Outputs and evaluation

Each run saves `rt_model.pt`, `rt_history.tsv` and numerical training/evaluation
outputs. `--use_best_model` selects by validation MAE. The source-specific
recipes request minimal sweep artifacts, suppressing redundant per-run plots.
The trainer checks exact peptidoform non-overlap against the validation file.
Keep the final test partition outside training and model selection. Evaluation
and conversion to observed minutes use the prepared alignment metadata through
the evaluation tools above; target-space errors across differently aligned
coordinates must not be pooled. The recipe launcher trains models only.
