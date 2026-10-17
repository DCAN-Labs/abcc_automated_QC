# ABCC Automated QC Pipeline

This repository contains a command-line QC pipeline for ABCC imaging data. It inspects subject/session anatomical and functional outputs, computes QC metrics, and writes TSV summaries that can be reviewed either per subject or across a cohort.

Entry point:

- `run_qc.py`

## Workflow

The pipeline operates in the following order:

1. Discovers all subject folders under the dataset root.
2. Finds each subject's sessions and inspects their `anat/` and `func/` directories.
3. Extracts structural measures such as subcortical volumes and sulcal morphology.
4. Evaluates functional QC for each configured task and atlas, including brain coverage and pconn presence checks.
5. Writes per-subject/session QC results as TSV files.
6. Aggregates temporary structural metrics across the cohort to identify outliers.
7. Updates the final QC outputs with cohort-level summary information when applicable.

This workflow supports both subject-level review and cohort-level QC monitoring.

## What this tool checks

The pipeline can evaluate:

- Structural QC
  - subcortical volumetric metrics from FreeSurfer outputs
  - sulcal morphometry metrics
- Functional QC
  - brain coverage metrics for requested tasks and atlases
  - presence of pconn connectivity files
- Cohort QC
  - outlier detection across subject/session QC outputs

## Project structure

```text
abcc_automated_QC/
├── code/
│   ├── config.py
│   ├── cohort.py
│   ├── subject.py
│   ├── functional/
│   ├── structural/
│   └── utils/
├── config/
│   └── config.yaml
├── run_qc.py
├── requirements.txt
├── setup.py
└── README.md
```

## Requirements

Python 3 and the dependencies in `requirements.txt`.

Install dependencies with:

```bash
cd /projects/standard/midb_abcd/shared/ABCC/code/abcc_automated_QC
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer to install the package in editable mode:

```bash
pip install -e .
```

## Configuration

The pipeline reads settings from `config/config.yaml`.

Example:

```yaml
output_filename: desc-abcc-automated-qc

tasks:
  - rest
  - mid
  - sst
  - nback

atlases:
  - Gordon
  - MIDB
  - HCP

coverage_threshold: 0.90
brain_coverage_pass: 90
outlier_sigma: 3
n_jobs: 32
```

### Configuration fields

- `output_filename`: Base name used for generated QC TSV files
- `tasks`: BOLD tasks to evaluate (for example `rest`, `mid`, `sst`, `nback`)
- `atlases`: Atlases used for brain coverage checks
- `coverage_threshold`: Minimum coverage threshold for pass/fail checks
- `brain_coverage_pass`: Pass threshold for brain coverage metrics
- `outlier_sigma`: Number of standard deviations used for cohort-level outlier detection
- `n_jobs`: Parallelism value for job execution where relevant

## Data layout expected by the pipeline

The root directory supplied to `--root` should be structured like:

```text
<root>/
├── sub-01/
│   ├── ses-01/
│   │   ├── anat/
│   │   └── func/
│   └── ses-02/
│       ├── anat/
│       └── func/
├── sub-02/
│   └── ...
└── tmp/
```

The tool expects subject directories such as:

- `sub-<id>`
- within each, `ses-<id>`
- containing `anat/` and `func/` folders

It also expects FreeSurfer-derived anatomical outputs, including files such as:

- `<subject>_<session>.tar.gz` in `anat/`

and functional outputs like:

- task-specific files in `func/`
- pconn connectivity files when connectivity QC is included

## CLI usage

From the project root, the command is:

```bash
python run_qc.py --config config/config.yaml --root /path/to/dataset --mode full
```

### Flags

The script accepts the following arguments:

- `--config` (required): path to the YAML configuration file
- `--root` (required): dataset root containing `sub-*` and `ses-*` directories
- `--mode` (optional): one of `subject`, `cohort`, `connectivity`, or `full`
  - `subject`: run subject/session QC only
  - `cohort`: run cohort-level outlier analysis only
  - `connectivity`: run connectivity QC only
  - `full`: run the full pipeline (default)

To see the built-in help:

```bash
python run_qc.py --help
```

### Example commands

Run subject-level QC only:

```bash
python run_qc.py --config config/config.yaml --root /path/to/derivatives --mode subject
```

Run cohort-level QC only:

```bash
python run_qc.py --config config/config.yaml --root /path/to/derivatives --mode cohort
```

Run connectivity QC only:

```bash
python run_qc.py --config config/config.yaml --root /path/to/derivatives --mode connectivity
```

Run the complete pipeline:

```bash
python run_qc.py --config config/config.yaml --root /path/to/derivatives --mode full
```

## Output files

### Subject-level outputs

The tool writes TSV QC files for each subject/session.

Typical output paths:

```text
<root>/sub-<id>/ses-<id>/<subject>_<session>_desc-abcc-automated-qc.tsv
```

For pre-split structural metrics, it may also write additional files under `tmp/`:

```text
<root>/tmp/<subject>_<session>_<output_name>_subcortical.tsv
<root>/tmp/<subject>_<session>_<output_name>_sulc.tsv
```

The cohort step aggregates those temporary file outputs and updates the final per-subject QC files.

## Troubleshooting

Common issues:

- Missing or misnamed directories under `sub-*/ses-*`
- Missing anatomical or functional files required by the selected metrics
- Incomplete file paths when the `--root` value is not the dataset parent directory
- Configuration file missing required keys

Use the printed logger output to identify which subject/session or metric set failed.

## Notes

- `--root` should point to the dataset root, not a single subject folder.
- The `full` mode is the most complete option when you want all QC stages.
- The tool writes intermediate results into a `tmp` directory before cohort-level aggregation.

## Example full run

```bash
cd /projects/standard/midb_abcd/shared/ABCC/code/abcc_automated_QC
python run_qc.py \
  --config config/config.yaml \
  --root /path/to/derivatives \
  --mode full
```
