# Fine-tuning and sweep tools

These optional tools tune the fork's bundled v2 models on data that Carafe has already prepared. They do not generate peptide forms or spectral libraries. The normal Carafe GUI and command-line workflows do not need Weights & Biases.

## Fine-tune with the bundled v2 trainer

Carafe calls `src/main/resources/py/v2/ai.py` during a normal library-generation run. The Java flags `-tf all`, `-tf rt`, `-tf ms2`, and `-tf ccs` select the model types. `-fine_tune_only` stops after training; `-save_validation_checkpoints` stores a checkpoint at each validation epoch. Training diagnostics and model evaluation metrics are written under the run output directory.

The v2 trainer also has optional controls for RT loss, per-class heavy-dimethyl weights and oversampling, sequence- or peptidoform-disjoint validation, organism sampling, training duration, and model layers. Use the Java command's `-h` for Carafe flags and run the bundled Python trainer with `--help` for the complete v2 options.

The held-out split used during training is validation data because its metrics are reported after every epoch. It is not an independent final test set for comparisons between configurations.

## External RT training used for the thesis comparisons

See the [experiment index](external-rt-experiments.md) for named, local JSON
recipes and the command launcher, including the exact reported heavy-weighted
comparisons.

The external Carafe2 RT branch starts from the generic pretrained AlphaPeptDeep
RT model and consumes the same prepared peptidoform partitions and target
coordinates as the external DIA-NN branch. Preparation, train-only alignment,
campaign definitions and held-out evaluation are maintained in the separate
[ma/data_collect component](https://github.com/mmmddd98/hunter-dia-workflows/tree/main/data_collect),
linked by the [project overview](https://github.com/mmmddd98/hunter-dia-model-adaptation).

For one prepared run, put `rt_train_data.tsv` under `inputs/carafe2_input/` and
provide a separate validation TSV with the same `sequence`, `mods`, `mod_sites`
and `rt_norm` columns:

```bash
python src/main/resources/py/v2/ai.py \
  --in_dir inputs/carafe2_input \
  --out_dir outputs/rt_run \
  --tf_type rt --device cpu --seed 2024 \
  --rt_validation_file inputs/rt_validation_data.tsv \
  --rt_validation_disjoint_by peptidoform \
  --rt_epoch 40 --rt_warmup_epoch 10 \
  --rt_batch_size 512 --no_adjust_rt_batch_size \
  --rt_lr 0.0001 --rt_loss l1 --rt_trainable all \
  --use_best_model
```

Use `--device gpu` with the documented CUDA environment. Repeat with seeds
2025 and 2026 for the three-seed protocol. `--no_adjust_rt_batch_size` preserves
the requested 512-row batch size. `--use_best_model` selects the validation-best
checkpoint; without it the trainer saves the last epoch. The output includes
`rt_model.pt` and `rt_history.tsv`. Supply `--rt_model` to start from an explicit
compatible AlphaPeptDeep checkpoint; otherwise the trainer uses its generic
pretrained model loader.

The trainer checks exact peptidoform non-overlap for the supplied validation
file. Keep final test data separate. Target mappings must be fitted only on
training observations before preparing the TSVs. Validation and final-test
metrics in RT target coordinates cannot be pooled across different target
scales; the external evaluation tools provide conversion to observed minutes.

The baseline above processes the training set each epoch. The source-specific Carafe2 campaign uses
`--rt_steps_per_epoch 120` for a fixed optimiser budget and sampling with
replacement. The 294-step setting belongs to later DIA-NN campaigns. Heavy-label loss weights and oversampling are separate
controls. Exact campaign combinations and target definitions live in the
preparation repository; the generic sweep JSONs in this fork are additional
research recipes rather than a frozen record of every thesis run.

## Weights & Biases

W&B is opt-in. The sweep helpers need an online W&B account because W&B creates and schedules the sweep. To use your own project, copy the example config, edit its project and optional entity, and log in in the same Python environment:

    cp configs/wandb.example.json configs/wandb.json
    # Edit project and, if applicable, entity in configs/wandb.json
    python -m pip install wandb==0.28.1
    wandb login

`configs/wandb.json` is ignored by Git so account-specific settings stay local. The sweep scripts use it by default; some launchers also accept a config path as an argument.

## Sweep launchers

Sweep inputs must be prepared Carafe fine-tuning output. The RT sweep requires `rt_train_data.tsv`; MS2 or combined sweeps also require `psm_pdv.txt` and `fragment_intensity_df.tsv`. Results are written to the output path, which defaults to the ignored local `finetune_output/` directory when a launcher does not require an explicit output path.

| Launcher | Search definition | Purpose |
| --- | --- | --- |
| `scripts/run_carafe2_wandb_sweep.sh` | `configs/carafe2_finetune_sweep.json` | Random search over the combined fine-tuning settings |
| `scripts/run_carafe2_rt_wandb_sweep.sh` | `configs/carafe2_rt_finetune_sweep.json` | RT-only random search |
| `scripts/run_carafe2_rt_oversampling_sweep.sh` | `configs/carafe2_rt_oversampling_sweep.json` | RT loss and heavy-dimethyl oversampling search across seeds |
| `scripts/run_carafe2_rt_loss_weighting_sweep.sh` | `configs/carafe2_rt_loss_weighting_sweep.json` | RT per-class loss weighting search across seeds |
| `scripts/start_controlled_weights_12h.sh` | `configs/carafe2_rt_controlled_weights_sweep.json` and `configs/carafe2_rt_controlled_weights_map.json` | Fixed comparison of four heavy-dimethyl weight settings |
| `run_carafe2_wandb_sweep.py` with `configs/carafe2_rt_full_multiorganism_strategy_sweep.json` | Bayesian search over RT losses, trainable layers, learning rates, and heavy-dimethyl weights; requires `--validation-dir` with the combined and three organism hold-out tables plus `rt_alignment.json` |

The shell launchers print usage when called without the required input paths. For example:

    ./scripts/run_carafe2_rt_wandb_sweep.sh 6 /path/to/prepared-carafe-output

For the fixed weight comparison, pass the prepared Carafe output and one visible GPU ID, for example `./scripts/start_controlled_weights_12h.sh /path/to/prepared-carafe-output 0`. For the multi-organism strategy sweep, call `run_carafe2_wandb_sweep.py` directly with `--tf-type rt`, `--validation-dir`, `--sweep-config configs/carafe2_rt_full_multiorganism_strategy_sweep.json`, and `--wandb-config configs/wandb.json`.

The first argument to a time-bounded launcher is a wall-time budget in hours; a running trial is allowed to finish before the launcher stops starting new trials. RT comparisons use the configured validation metric. For an unbiased model comparison, use a separate fixed test set that was not used for tuning or model selection.

## MA organism-scaling helper

`scripts/start_carafe2_organism_scaling.sh` and `evaluate_organism_tests.py` are research helpers for the MA multi-organism study. They need the separate MA data-collection tools and prepared hold-out tables; those files are not part of this repository. Set `CARAFE_DATA_COLLECT_DIR` to the directory containing those tools and `CARAFE2_PYTHON` to the Python executable for this fork. For comparison plots, also set `DIANN_STUDY_ROOT` to the prepared DIA-NN study output and `DIANN_FINETUNE_ROOT` to the DIA-NN fine-tuning repository. This workflow is not required for ordinary Carafe use.
