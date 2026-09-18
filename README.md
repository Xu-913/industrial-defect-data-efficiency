# Class-Level Data Efficiency in Industrial Defect Classification

Data, code, and complete experimental results for the study:

> **Class-level data efficiency in industrial defect classification: a four-dataset
> audit, a dynamic-range confound, and annotation-budget allocation**

This repository is the data-availability artifact for the manuscript submitted to
*Engineering Applications of Artificial Intelligence* (Elsevier).

It contains **every experimental result** reported in the paper: 6,430 completed
training runs (2,700 main benchmark + 3,730 pilot), their per-class accuracies and
confusion matrices, the frozen data splits, the audit records, and the full analysis
and figure-generation pipeline. All numbers in the manuscript can be regenerated
from the files here without re-training anything.

---

## 1. What this study measures

Industrial surface-defect classifiers are usually reported as a single accuracy
number per dataset. In practice, a plant does not have a uniform annotation budget:
some defect classes are abundant and some are rare, and the question that decides
the cost of a deployment is *how many labelled images each class needs*.

This study defines and measures that quantity per class, across four public
industrial defect datasets and eleven torchvision classifiers, and reports three
results:

1. **A measurement protocol** for class-level data efficiency (global deduplication,
   stratified frozen splits, per-class sample-count sweeps, five seeds, per-class
   confusion matrices).

2. **A dynamic-range confound.** Comparing class-level heterogeneity *across*
   datasets is only meaningful if each dataset's sweep covers the same relative
   range. Sweeping every dataset up to its own rarest class's cap produces
   right-censoring, and the censoring alone can reverse the sign of the
   heterogeneity comparison. Controlling for the common dynamic range reverses a
   naive `4.00x -> 1.58x` "heterogeneity falls with imbalance" trend into a
   4-6x *increase*.

3. **Annotation-budget decision tables** derived from the measured per-class curves.

The dynamic-range confound (result 2) is the general contribution: it applies to any
cross-dataset comparison of per-class learning curves, not only to defect detection.

---

## 2. Repository layout

```
.
|-- code/                    Analysis and training pipeline (Python)
|-- data/
|   |-- bench/               2,700 main-benchmark result JSONs
|   |-- pilot/               3,730 pilot-sweep result JSONs
|   |-- pilot_epochs/        3,789 per-epoch training logs (JSONL)
|   |-- splits/              10 frozen NEU-CLS split files (pilot)
|   |-- benchmark_splits/    20 frozen split files (main, 4 datasets x 5 seeds)
|   |-- audit/               Dataset audit records
|   |-- analysis/            Aggregated tables and derived statistics
|   |-- manifest.json        Image index for all four datasets
|   `-- build_report.txt     Dataset construction log
|-- figures/                 10 manuscript figures (PDF vector + 600 dpi PNG)
|-- docs/
|   |-- DATA_DICTIONARY.md   Every field in every result file
|   |-- DATASETS.md          Dataset provenance, licences, how to obtain them
|   `-- REPRODUCIBILITY.md   How to regenerate each result and figure
|-- requirements.txt
|-- LICENSE                  MIT (code)
`-- LICENSE-DATA             CC BY 4.0 (our data and results)
```

---

## 3. Quick start

You do **not** need a GPU or the original datasets to inspect the results. Everything
in Section 4 is plain JSON/CSV.

```bash
git clone https://github.com/Xu-913/industrial-defect-data-efficiency.git
cd industrial-defect-data-efficiency

# Aggregate every main-benchmark run into one table
python code/analyze_bench.py --results data/bench --out /tmp/bench_all.csv
```

To reproduce the paper's numbers from the frozen results:

```bash
python code/analyze_bench.py     # Tables 5-15, Section 4 statistics
python code/analyze_common.py    # common-dynamic-range control (Table 11)
python code/p03_relative_budget.py
python code/figs_bench.py        # regenerate the 10 figures
```

To re-run training from scratch you additionally need the datasets and a GPU; see
`docs/REPRODUCIBILITY.md` and `docs/DATASETS.md`.

---

## 4. Headline numbers

All values below are computed from the files in `data/`.

| Quantity | Value |
|---|---|
| Completed runs (main benchmark) | 2,700 of 2,750 launched |
| Completed runs (pilot sweep) | 3,730 of 3,789 launched |
| Datasets / classes / models | 4 / 24 / 11 |
| Budget gain, `magnetic_tile` @120 images | +2.66 pp |
| Budget gain, `neu_cls` @600 images | +0.04 pp |
| Per-class spread @8 images/class, `gc10det` | 42.9 pp |
| Per-class spread @8 images/class, `neu_cls` | 6.3 pp |
| Heterogeneity ratio, uncontrolled vs controlled | 1.58x -> 4-6x increase |
| `rho` (outflow error rate vs `n@rel95`), NEU-CLS | +0.824 |
| Generalization gap, 1 image/class -> 180 | 29.9 pp -> 0.2 pp |

`n@rel95` is defined as the smallest number of images per class at which a class
reaches 95% of its own maximum observed accuracy. See `docs/DATA_DICTIONARY.md`.

---

## 5. Compute environment

All 2,700 main-benchmark runs were executed on a heterogeneous institutional
cluster (SLURM 23.02.4) at Hunan University of Information Technology, on
**NVIDIA Tesla T4 (SM 7.5, FP16)** nodes with PyTorch 2.5.1 / CUDA 12.1.
Cross-hardware evidence comes from a historical 198-pair scan on
**NVIDIA RTX A4000 (SM 8.6, native BF16)**; see `docs/REPRODUCIBILITY.md` for the
caveat that applies to those pairs.

Each result JSON records the exact `torch` and `python` versions, the device name,
and the AMP dtype actually used, so the provenance of every number is self-describing.

---

## 6. Licensing

- **Code** (`code/`): MIT, see `LICENSE`.
- **Our data and results** (`data/`, `figures/`): CC BY 4.0, see `LICENSE-DATA`.
- **The four source datasets are NOT redistributed here.** They remain under their
  original licences and must be downloaded from their original sources. In
  particular, the casting dataset is **CC BY-NC-ND 4.0**. See `docs/DATASETS.md`.

---

## 7. Citation

If you use these results, please cite the paper and the archived artifact. A
machine-readable record is in `CITATION.cff`.

The artifact is archived on Zenodo:

- **Version DOI (v1.0.0):** [10.5281/zenodo.22833116](https://doi.org/10.5281/zenodo.22833116)
- **Concept DOI (all versions):** [10.5281/zenodo.22833115](https://doi.org/10.5281/zenodo.22833115)

```bibtex
@article{xu2026classlevel,
  title   = {Class-level data efficiency in industrial defect classification:
             a measurement protocol, a difficulty map, and
             annotation-budget decision tables},
  author  = {Xu, Xiang and Nie, Mincan and Liu, Yiwen and Li, Xianxin and
             Sun, Yihui and Zhan, Zhu and Hu, Wei and Liu, Jun and
             Gong, Zhi and Su, Haibin},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  doi     = {10.5281/zenodo.22833116}
}
```

---

## 8. Contact

Xiang Xu (corresponding author) - xu913913@gmail.com
Hunan University of Information Technology, Changsha, China
