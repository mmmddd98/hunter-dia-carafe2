# HUNTER DIA Carafe2

This repository is a research fork of [Noble-Lab/Carafe](https://github.com/Noble-Lab/Carafe), based on upstream Carafe2 v2.2.0. Carafe2 trains experiment-specific retention-time (RT), fragment-intensity, and optional ion-mobility models from DIA identifications and raw MS data, then uses them to generate a spectral library.

The fork adds semi-specific FASTA digestion, improved DIA-NN modification handling, and changes to training, prediction, progress reporting, and library assembly. It keeps upstream behavior by default: semi-specific digestion is opt-in with `-semi`.

The release downloads on the upstream Carafe page are upstream builds; they do not contain this fork's changes. Build this repository to use them.

## Start here

| Task | Entry point |
| --- | --- |
| Generate a semi-specific spectral library | [Build](#build-and-test), then [run Carafe2](#generate-a-spectral-library) |
| Train only the external RT model | [Prepared-data quick start](#train-an-external-rt-model) |
| Reproduce an external RT experiment configuration | [Experiment index](external-rt-experiments.md) |
| Configure losses, layers, validation or W&B | [Fine-tuning guide](fine-tuning.md) |
| Understand the source layout and common errors | [Developer guide](development.md) |

The [thesis overview](https://github.com/mmmddd98/hunter-dia-model-adaptation)
links this repository under **HUNTER-compatible Carafe2 fork** and
**External Carafe2 RT fine-tuning**. The separate
[DIA-NN trainer](https://github.com/mmmddd98/hunter-dia-diann-rt-finetuning) produces DIA-NN
TorchScript models; this fork trains AlphaPeptDeep/Carafe2 models. Their
checkpoints cannot be substituted for each other.

## Changes from upstream

The comparison point is the upstream v2.2.0 tag. Changes in this fork are grouped below by purpose.

- **Semi-specific digestion:** `-semi` adds peptides with at least one enzymatic terminus while retaining fully specific peptides. The selected enzyme, missed-cleavage limit, and peptide-length limits still apply.
- **DIA-NN modifications:** the parser supports multi-digit UniMod identifiers, multiple modifications, and DIA-NN N-terminal notation. Terminal modifications are carried into the model at AlphaBase site 0 and exported at the correct peptide terminus.
- **Prediction and recovery:** v2 MS2+RT prediction can keep one worker per GPU alive, stage model files in a local content-checked cache, and resume library assembly from validated completed prediction chunks. GPU selection respects visible physical GPU IDs. Prediction and assembly report chunk-level progress and ETA; Parquet assembly uses bounded memory and ordered output.
- **Input compatibility and diagnostics:** DIA-NN `Ms2.Scan` is accepted alongside `MS2.Scan`. Python warnings are labelled as warnings in the Java log. Fine-tuning can save validation checkpoints and structured per-epoch diagnostics.
- **RT fine-tuning controls:** the bundled v2 trainer adds validation split controls, organism sampling strategies, heavy-dimethyl oversampling and loss weights, selectable RT losses, and optional Weights & Biases tracking and sweeps. These research tools are optional.
- **Build and regression coverage:** Maven dependencies and source/test separation were corrected for this fork. Tests cover semi-specific digestion and DIA-NN terminal and multi-digit modification parsing.

The persistent prediction-worker path applies to v2 MS2+RT inference without CCS. Other model modes keep the per-chunk worker path.

## Requirements

| Part | Requirement |
| --- | --- |
| Build and command-line runtime | 64-bit JDK 21; Maven 3.9 or newer to build |
| Fine-tuning and prediction | Python 3.11 and the packages in `requirements-carafe2.txt`, plus PyTorch |
| GPU inference/training | NVIDIA GPU, a compatible NVIDIA driver, and the tested CUDA 12.1 PyTorch build in `requirements-torch-cu121.txt` |
| CPU mode | The same Python packages with a CPU-compatible PyTorch build; pass `-device cpu` |
| Bruker `.d` raw data | The platform-specific `timsquery_cli` executable; it is not included in this source repository |

The pinned package versions in these files describe the tested Carafe2 fork environment, rather than the original project's unspecified environment. PyTorch is separate so CPU users and systems with another CUDA version can select the matching PyTorch build without replacing it when installing the common requirements.

Weights & Biases is optional for normal Carafe runs. It is needed only for W&B tracking and the sweep helpers; see [Fine-tuning and sweeps](fine-tuning.md).

## Build and test

From the repository root, with JDK 21 and Maven installed:

    mvn clean test
    mvn package

Maven writes the runnable distribution, including its dependency directory, to `target/carafe-2.2.0/`. Keep the JAR together with its `lib/` directory when moving the build.

The source includes the GUI as well as the command-line interface. Launch the GUI from a local build with:

    java -jar target/carafe-2.2.0/carafe-2.2.0.jar

See [the GUI guide](gui.md) for workflow instructions and screenshots. Use a locally built fork JAR to access this repository's changes.

## Install the fork's Python environment

On Linux with an NVIDIA GPU and CUDA 12.1-compatible driver:

    python3.11 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements-torch-cu121.txt -r requirements-carafe2.txt

On Windows, activate `.venv\Scripts\activate` instead. For CPU-only execution, install PyTorch 2.5.1 from the official CPU wheel index, then install only `requirements-carafe2.txt`:

    python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements-carafe2.txt

Select the matching PyTorch wheel for other CUDA versions as well. Bruker `.d` input needs `timsquery_cli` under `bin/timsquery/<platform>/` (`windows`, `linux`, or `macos`), relative to the built JAR directory (the current working directory is also checked). This binary is not distributed here.

## Generate a spectral library

Provide a DIA-NN report, the corresponding raw MS run (`.mzML` or Bruker `.d`), a protein FASTA, and an output directory. Select the Python executable from the Carafe2 environment explicitly:

    java -jar target/carafe-2.2.0/carafe-2.2.0.jar \
      -i <diann-report.parquet> \
      -ms <raw-run.mzML-or-d-folder> \
      -db <proteins.fasta> \
      -o <output-directory> \
      -python <path-to-carafe-python> \
      -device gpu \
      -tf all \
      -semi

Remove `-semi` to use the original fully specific digest. Use `-device cpu` for CPU inference. For supported options, run the built JAR with `-h`.

For DIA-NN 2.2 reports, export quantification fields so the report includes the MS2 scan column used by Carafe. This fork accepts either `Ms2.Scan` or `MS2.Scan`.

To train models and stop before library prediction, use `-fine_tune_only`. To save a model checkpoint after each validation epoch, add `-save_validation_checkpoints`. The assembly-only recovery option `-assemble_existing_predictions` requires a complete, validated set of prediction chunks in the existing output directory; it does not regenerate peptides, retrain models, or run predictions.

## Train an external RT model

After installing the Python environment, place the prepared training table at
`inputs/carafe2_input/rt_train_data.tsv` and the validation table at
`inputs/rt_validation_data.tsv`. Both tables use tab-separated columns:

| Column | Meaning |
| --- | --- |
| `sequence` | Unmodified amino-acid sequence |
| `mods` | Semicolon-separated AlphaBase modification names; empty for unmodified peptides |
| `mod_sites` | Matching residue positions, starting at 1; N-terminus is 0 |
| `rt_norm` | RT target in the coordinate used by both partitions |

```bash
python src/main/resources/py/v2/ai.py \
  --in_dir inputs/carafe2_input \
  --out_dir outputs/rt_example \
  --tf_type rt --device cpu --seed 2024 \
  --rt_validation_file inputs/rt_validation_data.tsv \
  --rt_validation_disjoint_by peptidoform \
  --rt_epoch 40 --rt_warmup_epoch 10 \
  --rt_batch_size 512 --no_adjust_rt_batch_size \
  --rt_lr 0.0001 --rt_loss l1 --rt_trainable all \
  --use_best_model
```

The output includes `rt_model.pt` and `rt_history.tsv`. Use a fresh output
directory for each configuration and seed. The trainer starts from its generic
pretrained RT model unless `--rt_model` supplies a compatible checkpoint.
Model loading may require downloading the external pretrained weights.
Use `--device gpu` with a configured CUDA environment.

Training and validation must share the same RT coordinate and contain no
identical peptidoforms. Fit alignment on training observations only; keep the
final test set separate. The [experiment guide](external-rt-experiments.md)
explains target preparation and the recorded thesis configurations.

## Fine-tuning and sweeps

For the external RT experiments linked from the thesis, start with the
[experiment index and portable configurations](external-rt-experiments.md).
It includes the exact trainer, a recipe launcher, layer/LR grids, the reported
heavy-weighted controls and source-specific strategies.

The bundled v2 Python trainer supports RT-only, MS2-only, CCS-only, and combined training. It writes model evaluation metrics and, unless minimal sweep artifacts are requested, a diagnostics report under the chosen output directory. The external RT entry point with prepared train/validation splits, thesis training settings, advanced controls and optional W&B launchers is documented in [docs/fine-tuning.md](fine-tuning.md).

## Citation

Please cite the relevant Carafe paper for your instrument/workflow:

- Wen, B., Hsu, C., Shteynberg, D. et al. [Carafe enables high quality *in silico* spectral library generation for data-independent acquisition proteomics](https://doi.org/10.1038/s41467-025-64928-4). *Nature Communications* 16, 9815 (2025).
- Wen, B., Paez, J., Hsu, C. et al. [Carafe2 enables high quality *in silico* spectral library generation for timsTOF data-independent acquisition proteomics](https://doi.org/10.64898/2026.03.27.714846). *bioRxiv* (2026).
