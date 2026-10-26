# HUNTER DIA Carafe2

HUNTER-compatible fork of [Carafe2 v2.2.0](https://github.com/Noble-Lab/Carafe).
It adds semi-specific digestion, support for N-terminal heavy dimethylation
(UniMod:510) and acetylation (UniMod:1), and changes for predicting and assembling
large spectral libraries.

The [thesis](https://github.com/mmmddd98/hunter-dia-model-adaptation) used this
fork for integrated RT and fragmentation-model adaptation and for controlled
external RT fine-tuning. Final searches of held-out files were performed with
DIA-NN through [hunter-dia-workflows](https://github.com/mmmddd98/hunter-dia-workflows).

## Build and run

Use JDK 21, Maven 3.9 or newer, and the Python environment described in the
[detailed setup guide](docs/usage.md#requirements).

```bash
git clone https://github.com/mmmddd98/hunter-dia-carafe2.git
cd hunter-dia-carafe2
mvn clean test package
```

The runnable distribution is `target/carafe-2.2.0/`; keep the JAR and its `lib/`
directory together. Upstream release binaries do not contain this fork's changes.

```bash
java -jar target/carafe-2.2.0/carafe-2.2.0.jar \
  -i inputs/report.parquet -ms inputs/run.mzML \
  -db inputs/proteins.fasta -o outputs/library \
  -python .venv/bin/python -device gpu -tf all -semi
```

Supply your own report, matching measurements and FASTA. `-semi` enables
semi-specific digestion; omit it for fully specific digestion. See the
[library-generation guide](docs/usage.md#generate-a-spectral-library) for setup
and input requirements.

## External RT fine-tuning

The RT trainer is `src/main/resources/py/v2/ai.py` with `--tf_type rt`.
Use prepared, peptidoform-disjoint training and validation partitions with the
same RT target coordinate; keep the final test partition separate.

- [Training example](docs/usage.md#train-an-external-rt-model)
- [Thesis experiment configurations and data preparation](docs/external-rt-experiments.md)
- [Losses, layers, sampling and optional W&B sweeps](docs/fine-tuning.md)
- [Development and regression tests](docs/development.md)
- [GUI guide](docs/gui.md)

The separate [DIA-NN RT trainer](https://github.com/mmmddd98/hunter-dia-diann-rt-finetuning)
exports DIA-NN TorchScript models. Its checkpoints are not interchangeable
with Carafe2/AlphaPeptDeep checkpoints.

## Attribution

Based on [Noble-Lab/Carafe](https://github.com/Noble-Lab/Carafe), under
[Apache-2.0](LICENSE). See the [Carafe citations](docs/usage.md#citation).
