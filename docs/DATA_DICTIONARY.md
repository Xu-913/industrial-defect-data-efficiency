# Data dictionary

Every field in every result file, with units and provenance.

---

## 1. `data/bench/*.json` - main benchmark (2,700 files)

One JSON object per completed training run. This is the primary evidence for all
four-dataset results in the paper.

File name pattern: `{dataset}_{model}_seed{seed}_n{per_class}_res{img_size}.json`

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `type` | str | - | Always `"final"` for this directory |
| `run_id` | str | - | Unique run identifier; equals the file stem |
| `dataset` | str | - | One of `neu_cls`, `gc10det`, `magnetic_tile`, `casting` |
| `model` | str | - | torchvision model key, e.g. `resnet18`, `vit_b_16` |
| `seed` | int | - | Random seed; the main benchmark uses 0-4 |
| `per_class` | int | images | Training images **per class** (the swept variable) |
| `img_size` | int | px | Square network input side; 224 except `gc10det` at 256 |
| `epochs` | int | - | Training epochs; 50 uniformly in this generation |
| `bs` | int | images | Batch size |
| `lr` | float | - | AdamW learning rate, cosine schedule with 2-epoch warmup |
| `wd` | float | - | AdamW weight decay |
| `n_train` | int | images | Actual training images after the per-class cap |
| `n_classes` | int | - | Number of classes in the dataset |
| `classes` | list[str] | - | Class names in label-index order |
| `n_params` | int | - | Trainable parameter count of the instantiated model |
| `device` | str | - | GPU product name, e.g. `Tesla T4` |
| `amp_dtype` | str | - | `fp16`, `bf16`, or `none` - the AMP mode actually used |
| `torch` | str | - | PyTorch version, e.g. `2.5.1+cu121` |
| `python` | str | - | Python version |
| `ts` | str | - | Run timestamp, `YYYY-MM-DD HH:MM:SS` |
| `best_val_acc` | float | % | Best validation accuracy over the run |
| `best_epoch` | int | - | Epoch at which `best_val_acc` was reached |
| `test_acc` | float | % | Test accuracy of the best-validation checkpoint |
| `test_loss` | float | - | Test cross-entropy of that checkpoint |
| `per_class_acc` | dict[str,float] | % | Per-class test accuracy, keyed by class name |
| `confusion` | list[list[int]] | counts | Row = true class, column = predicted class |
| `total_sec` | float | s | Wall-clock training + evaluation time |

**Notes.** The reported test accuracy always comes from the checkpoint with the best
validation accuracy, never from the final epoch. `n_train` can be smaller than
`per_class * n_classes` when a class is exhausted before the cap.

---

## 2. `data/pilot/*.json` - pilot sweep (3,730 files)

One JSON object per completed pilot run, on NEU-CLS only. Same fields as Section 1
for the shared keys, plus the following. The pilot uses a **training-fraction**
scheme instead of a per-class count scheme, and covers 11 models including several
not in the main benchmark.

File name pattern: `{model}_seed{seed}_frac{train_frac}.json`

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `family` | str | - | Model family label, e.g. `modern_cnn` |
| `train_frac` | float | fraction | Fraction of the training split used |
| `pretrained` | bool | - | Whether ImageNet pretrained weights were loaded |
| `split_file` | str | - | Path to the frozen split used (original cluster path) |
| `cuda` | str | - | CUDA version string |
| `git` | str | - | Git revision at run time; `no-git` if unavailable |
| `n_val` | int | images | Validation set size |
| `n_test` | int | images | Test set size |

---

## 3. `data/pilot_epochs/*.jsonl` - per-epoch training logs (3,789 files)

Newline-delimited JSON. The **first line** is an `env` record (identical in shape to
the pilot config fields above); every **subsequent line** is one epoch.

File name pattern: `{model}_seed{seed}_frac{train_frac}.jsonl`

Per-epoch record:

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `type` | str | - | `"env"` for line 1, `"epoch"` for the rest |
| `epoch` | int | - | 1-based epoch index |
| `train_loss` | float | - | Mean training cross-entropy for the epoch |
| `train_acc` | float | % | Training accuracy for the epoch |
| `val_loss` | float | - | Validation cross-entropy |
| `val_acc` | float | % | Validation accuracy |
| `lr` | float | - | Learning rate at the end of the epoch |

These logs are the raw data behind the training-dynamics figure and the
generalization-gap numbers. They exist only for the pilot sweep, because the main
benchmark did not persist per-epoch logs.

---

## 4. `data/analysis/bench_all.csv` - aggregated main benchmark

One row per main-benchmark run (2,700 rows), 138 columns. This is a flattened,
directly plottable form of Section 1.

| Column group | Meaning |
|---|---|
| `run_id`, `dataset`, `model`, `seed`, `per_class`, `img_size`, `n_train`, `n_classes` | Run identity and configuration |
| `test_acc`, `best_val_acc`, `total_sec`, `device`, `amp_dtype` | Scalar outcomes |
| `acc_{i}_{classname}` | Per-class accuracy, indexed by label order |
| `cm_{i}_{j}` | Confusion-matrix entry: true class `i`, predicted class `j` |

Because the four datasets have different class counts, the `acc_*` and `cm_*` columns
are the union over all datasets; cells that do not apply to a dataset are empty.
Class-name suffixes differ per dataset (e.g. `acc_0_def_front` for `casting`,
`acc_0_crazing` for `neu_cls`), so filter on `dataset` before aggregating.

---

## 5. `data/analysis/results_all.csv` - aggregated pilot sweep

One row per pilot run, 67 columns: the pilot analogue of Section 4, with
`acc_{classname}` for the six NEU-CLS classes and a 6x6 `cm_{i}_{j}` block.

---

## 6. `data/analysis/bench_summary.json`

Nested summary keyed by dataset. Contains the per-dataset, per-model, per-level
aggregates (means and standard deviations across seeds) used to build the
data-efficiency curves. Top-level keys: `neu_cls`, `gc10det`, `magnetic_tile`,
`casting`.

---

## 7. `data/analysis/common_level_control.json`

The common-dynamic-range control that removes the right-censoring confound.

| Key | Meaning |
|---|---|
| `common_levels` | The levels shared by all four datasets, `[1, 2, 4, 8]` |
| `per_level` | Per-dataset, per-level statistics restricted to those levels |
| `summary` | The controlled heterogeneity ratios reported in the paper |

This file is the direct source of the paper's central claim: the uncontrolled
comparison gives a heterogeneity ratio of 1.58x, while the controlled comparison
gives `ctrl_rel` of 0.262 (`neu_cls`), 0.181 (`casting`), 1.138 (`gc10det`) and
0.935 (`magnetic_tile`), i.e. a 4-6x amplification rather than a decrease.

---

## 8. `data/analysis/figs_diagnostics.json`

Per-figure diagnostic values, keyed by figure stem, plus `_summary` and
`_provenance` blocks. Records the exact numbers that were drawn into each figure so
that a figure can be checked without re-running matplotlib.

---

## 9. `data/audit/` - dataset audit records

| File | Meaning |
|---|---|
| `audit_report.json` | Full audit: per-dataset counts, image sizes, channel counts, duplicate groups, corrupt files |
| `audit_report.csv` | Same, flattened to one row per dataset |
| `audit_summary.txt` | Human-readable summary (English) |
| `audit_summary.zh-CN.txt` | The original summary as emitted by the first build, in Chinese; same numbers, kept for provenance |

---

## 10. `data/splits/` and `data/benchmark_splits/`

Frozen stratified split files, JSON, one per dataset and seed.

| Directory | Coverage |
|---|---|
| `data/splits/` | NEU-CLS seeds 0-9 (pilot sweep) |
| `data/benchmark_splits/` | Four datasets x seeds 0-4 (main benchmark, 20 files) |

Each file has three keys - `train`, `val`, `test` - each a list of dataset-relative
image paths. The split is frozen once and reused at every annotation level, so that
differences along a data-efficiency curve reflect data quantity alone and not split
noise. This is a methodological requirement, not an implementation detail.

---

## 11. `data/manifest.json`

The image index used by the trainer. Top-level keys are the four dataset names; each
maps to:

| Field | Meaning |
|---|---|
| `root` | Absolute path to the image directory **on the original cluster** |
| `note` | Free-text provenance note |
| `classes` | Ordered class names |
| `counts` | Images per class after deduplication |
| `items` | List of `[relative_path, label_index]` pairs |

**Important.** The `root` values are site-specific cluster paths
(`/var/hpc-root/...`) and will not exist on your machine. After downloading the
datasets yourself (see `docs/DATASETS.md`), update each `root` to point at your local
copy. The `items` lists are the authoritative record of which images were used.

---

## 12. Definitions used throughout

| Symbol | Definition |
|---|---|
| `n@rel95` | Smallest images-per-class at which a class reaches 95% of its own maximum observed accuracy over the sweep |
| Level cap | Number of training images available for the rarest class; bounds the sweep |
| Imbalance ratio | Largest class size divided by smallest class size |
| `ctrl_rel` | Heterogeneity ratio recomputed on the common dynamic range `[1,2,4,8]` |
| Outflow error rate | Fraction of a class's true instances predicted as some other class |
| Inflow error rate | Fraction of predictions of a class that truly belong to another class |
| Generalization gap | Training accuracy minus test accuracy at the same epoch |
