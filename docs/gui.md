# Graphical interface

The Maven build contains the Carafe GUI. Start the locally built fork from the repository root:

    java -jar target/carafe-2.2.0/carafe-2.2.0.jar

The GUI is part of this fork's build. The installers and release binaries linked by the upstream Carafe project are upstream builds and do not include the fork changes. In the GUI's Python Executable field, select the interpreter for the environment installed from this repository's requirements files. The `-semi` digestion switch is currently available through the command-line interface.

The GUI offers three workflows:

1. Search training DIA data with DIA-NN, fine-tune Carafe models, and generate a library.
2. Start from an existing DIA-NN report, fine-tune on its identifications and raw MS data, and generate a library.
3. Run the training search, library generation, and project-data DIA-NN search as one workflow.

Workflows 1 and 3 require DIA-NN installed separately. Thermo raw files require MSConvert from ProteoWizard. Bruker `.d` files require the platform-specific `timsquery_cli` executable described in the [setup guide](usage.md#install-the-forks-python-environment). Use mzML input when the relevant vendor tools are unavailable.

## Select the Python environment

Use the Python executable from the pinned Carafe2 fork environment rather than installing an unpinned environment from an upstream release. On Linux this is usually `.venv/bin/python`; on Windows it is `.venv\Scripts\python.exe`.

After selecting the environment, its path should appear in the Python Executable field.

## Workflow examples

### Search DIA data and generate a library

Select the training MS file(s), the protein database or DIA-NN library used for the training search, and the protein FASTA for the output library. Set the shared digestion, modification, and peptide-length options in Carafe's library-generation panel. The related DIA-NN search uses those settings. Other DIA-NN options can be entered in the workflow panel.

The console shows run progress and diagnostics.

### Start from an existing DIA-NN report

Select the report, the matching raw MS file(s), and the FASTA for the output library. The report must come from the selected training run. DIA-NN 2.2 reports need quantification export enabled to include the MS2 scan column. This fork accepts either `Ms2.Scan` or `MS2.Scan`.

### Run the full DIA workflow

Select the training data and database, output-library database, and project MS file(s). Carafe runs DIA-NN on the training data, trains the models and builds a library, then runs DIA-NN on the project data with that library. An optional library-free search can be enabled for comparison.

The optional library-free result can be compared with the Carafe-library result.

## Optional timsTOF viewer

Carafe's library TSV can also be viewed with the third-party [TimsViewer](https://github.com/TalusBio/timsbuktoolkit/releases). TimsViewer is a separate application and is not required to run Carafe.
