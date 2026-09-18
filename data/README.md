# `data/` - contents and how to read them

Everything in this directory is data, not code. See `../docs/DATA_DICTIONARY.md` for
the meaning of every field.

| Path | Files | Size | What it is |
|---|---|---|---|
| `bench/` | 2,700 | 2.4 MB | Main benchmark result JSONs - the primary evidence |
| `pilot/` | 3,730 | 4.7 MB | Pilot sweep result JSONs (NEU-CLS only) |
| `pilot_epochs/` | 3,789 | 38.4 MB | Per-epoch training logs, JSONL |
| `splits/` | 10 | 0.6 MB | Frozen NEU-CLS splits, seeds 0-9 (pilot) |
| `benchmark_splits/` | 20 | 2.5 MB | Frozen splits, 4 datasets x seeds 0-4 (main) |
| `audit/` | 4 | 45 KB | Dataset audit records (English summary plus the original) |
| `analysis/` | 5 | 2.4 MB | Aggregated tables and derived statistics |
| `manifest.json` | 1 | 0.8 MB | Image index for all four datasets |
| `build_report.txt` | 1 | 3 KB | Dataset construction log |

---

## Which file answers which question

| Question | Read |
|---|---|
| What accuracy did model M reach on dataset D at N images per class? | `bench/{D}_{M}_seed{k}_n{N}_res{S}.json`, field `test_acc` |
| What was the per-class accuracy in that run? | same file, field `per_class_acc` |
| What was the full confusion matrix? | same file, field `confusion` |
| What GPU and AMP mode produced that number? | same file, fields `device`, `amp_dtype` |
| How did training accuracy evolve epoch by epoch? | `pilot_epochs/{model}_seed{k}_frac{f}.jsonl` |
| Which exact images were in the training set? | `benchmark_splits/{D}_seed{k}.json`, key `train` |
| Which images exist in the dataset and what are their labels? | `manifest.json`, key `{D}` -> `items` |
| How many duplicates were removed? | `audit/audit_report.json` |
| The ready-made flat table for plotting | `analysis/bench_all.csv` |

---

## Reading the results in three lines

```python
import json, glob
runs = [json.load(open(f)) for f in glob.glob("data/bench/*.json")]
print(len(runs), "runs;", runs[0]["dataset"], runs[0]["test_acc"])
```

---

## Two cautions

1. **`manifest.json` contains absolute paths from the original cluster.** The `root`
   values (`/var/hpc-root/...`) will not exist on your machine. Update them after
   downloading the datasets yourself. See `../docs/DATASETS.md`.

2. **The pilot and main generations are not interchangeable.** They use different
   budget schemes (training fraction vs images per class), different split bases
   (1,800 vs 1,799 images), and coexisting epoch schedules. Agreement between them
   was measured at a mean absolute difference of 1.41 pp across 1,265 matched runs.
   Absolute accuracies from the pilot must never be cited alongside those of the
   main experiment. See `../docs/REPRODUCIBILITY.md`.
