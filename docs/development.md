# Development and reproducibility

## Source map

| Location | Responsibility |
| --- | --- |
| `src/main/java/` | CLI/GUI, digestion, modification parsing, library generation and assembly |
| `src/main/resources/py/v2/ai.py` | Bundled v2 training entry point, including external RT training |
| `src/main/resources/py/v2/models.py` | Model training and prediction utilities |
| `scripts/run_external_rt_recipe.py` | Translate a named experiment recipe to native trainer arguments |
| `configs/external-rt/` | Portable, versioned experiment designs with source provenance |
| `src/test/java/` | Digestion, modification parsing and other Java regression tests |
| `tests/python/` | Recipe-launcher checks that do not require PyTorch or measurement data |

Run commands from the repository root. Java development needs JDK 21 and
Maven 3.9 or newer. The Python training environment is documented in the
[README](../README.md); the launcher tests use only the Python standard library.

```bash
mvn clean test
python -m unittest discover -s tests/python -v
mvn package
```

Training code, preparation code and final search results are separate stages.
The experiment JSONs preserve preparation commit IDs and hashes. Their grid
sizes describe experiment designs, not completed-run counts. This repository
contains no benchmark measurement files or frozen archive of trained models.

## Local configuration

Use relative input/output locations or supply your own paths at runtime.
Copy `configs/wandb.example.json` to the ignored `configs/wandb.json` only if
using W&B. Runtime tables, checkpoints, credentials and local environments
belong outside versioned source. `CARAFE_DATA_COLLECT_DIR`, `CARAFE2_PYTHON`
and the optional comparison-study variables are documented in
[the fine-tuning guide](fine-tuning.md).

Generated logs and manifests may contain the local paths supplied by the
caller. Review them separately before sharing a run.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing Python package | Activate the environment used with `-python` or the direct trainer |
| CUDA is unavailable | Use the matching PyTorch build or select CPU explicitly |
| Missing `rt_train_data.tsv` | Supply the directory containing the prepared table, not its parent |
| Validation overlap error | Regenerate disjoint partitions before fitting target mappings |
| Recipe/target mismatch | Select the target and source scope listed in the recipe JSON |
| Output directory already contains files | Choose a fresh directory to keep independent runs separate |
| Missing MS2 scan column | For DIA-NN 2.2, export quantification fields as described in the README |

Retain upstream copyright, license and citation information when distributing
this fork. The Apache-2.0 license is in [LICENSE](../LICENSE).
