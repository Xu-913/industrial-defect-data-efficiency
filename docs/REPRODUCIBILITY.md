# Reproducibility guide

Two levels of reproduction are supported.

- **Level A - from the frozen results.** No GPU, no datasets, no training. Regenerates
  every table and figure in the paper from the JSON files in `data/`. Takes minutes.
- **Level B - from scratch.** Re-trains the models. Needs the datasets and a GPU.
  Takes roughly 100 GPU-hours for the main benchmark.

---

## Level A: regenerate every result from the frozen JSON

```bash
pip install -r requirements.txt

# 1. Aggregate the main benchmark into one flat table (2,700 rows)
python code/analyze_bench.py --results data/bench --out /tmp/bench_all.csv

# 2. Common-dynamic-range control - the paper's central claim
python code/analyze_common.py --results data/bench \
    --out /tmp/common_level_control.json

# 3. Relative-budget comparison
python code/p03_relative_budget.py --results data/bench

# 4. Regenerate all ten figures
python code/figs_bench.py --results data/bench --out figures/
```

Compare your output against the shipped copies in `data/analysis/`. The shipped
`bench_all.csv` was produced by exactly step 1.

### Verifying a single number

Every scalar in the manuscript can be traced to a single run file. For example, the
`magnetic_tile` budget gain of +2.66 pp at 120 images per class:

```bash
python - <<'PY'
import json, glob, statistics as st
rows = []
for f in glob.glob("data/bench/magnetic_tile_*_n120_*.json"):
    d = json.load(open(f))
    rows.append((d["model"], d["test_acc"]))
# group by model, average over the five seeds, compare against n=100
PY
```

---

## Level B: re-train from scratch

### 1. Environment

The main benchmark ran on PyTorch 2.5.1 / CUDA 12.1 / Python 3.11.15, on NVIDIA
Tesla T4 (SM 7.5) hardware with FP16 automatic mixed precision.

```bash
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Prepare the data

Follow `docs/DATASETS.md` to download the four datasets, then rebuild the manifest
and splits:

```bash
python code/build_dataset.py --root /path/to/datasets --out /path/to/benchmark
```

This writes `manifest.json` and `splits/{dataset}_seed{k}.json`. Deduplication runs
before splitting and is not optional - see `docs/DATASETS.md` Section 7.

### 3. Point the code at your data

`train_bench.py` and `run_bench.py` contain absolute paths from the original cluster.
Set them to your own layout. In `train_bench.py`:

```python
BENCH = "/var/hpc-root/HPC_SHARED_DIR/benchmark"   # <- change to your benchmark dir
```

and in `run_bench.py`:

```python
WORK = "/var/hpc-root/xiangxu/work"   # <- change
CODE = "/var/hpc-root/xiangxu/code"   # <- change
```

Alternatively export them as environment variables if you prefer; the scripts read
the module-level constants only, so editing those two lines is sufficient.

### 4. Run the main benchmark

The full grid is 2,750 configurations (4 datasets x 11 models x per-dataset levels x
5 seeds). It is sharded across independent workers, and is **idempotent**: a result
JSON that already exists is skipped, so a killed job can be restarted unchanged.

```bash
# inspect the plan first
python code/run_bench.py --list

# run shard 0 of 24
python code/run_bench.py --job-index 0 --job-count 24
```

Per-dataset levels and input resolutions, exactly as used:

| Dataset | Input | Levels (images per class) |
|---|---|---|
| `neu_cls` | 224 | 1,2,3,4,6,8,10,12,15,20,25,30,40,50,65,80,100,125,150,180 |
| `gc10det` | 256 | 1,2,3,4,6,8,10,12,15,18 |
| `magnetic_tile` | 224 | 1,2,3,4,6,8,12,16,24,38 |
| `casting` | 224 | 1,2,4,8,16,32,64,128,256,512 |

Every level is bounded above by the rarest class's available images (the level cap).
This bounding is exactly what produces the right-censoring confound the paper
analyses, so do not extend the levels when reproducing.

### 5. Training protocol

Held fixed across all runs: 50 epochs, batch size 32, AdamW with learning rate
3e-4 and weight decay 5e-4, cosine schedule with 2 epochs of warmup, ImageNet
pretrained initialisation, greyscale images replicated to 3 channels, random
horizontal and vertical flips, ImageNet normalisation. The reported test accuracy is
always taken from the checkpoint with the **best validation accuracy**, never from
the final epoch.

AMP mode is selected automatically by compute capability: BF16 when the device
reports SM >= 8.0, otherwise FP16 with a gradient scaler. On T4 (SM 7.5) this means
FP16. Each result JSON records the mode actually used in its `amp_dtype` field.

---

## Cross-hardware caveat

All 2,700 main-benchmark runs were executed on **Tesla T4** hardware. There are
**zero A4000 runs** in `data/bench/`.

The cross-hardware evidence reported in the paper comes from a separate historical
scan of **198 paired configurations** on RTX A4000 (SM 8.6, native BF16). Those
pairs are **not** part of `data/bench/` and are not redistributed here; they were
produced under a different sweep scope and a different AMP mode. Their aggregate
statistics, as reported in the paper, are:

| Statistic | Value |
|---|---|
| Signed mean difference (A4000 - T4) | +0.238 pp |
| Mean absolute difference | 1.105 pp |
| Maximum absolute difference | 15.83 pp |
| Seed spread | 4.03 pp |

Because the seed spread (4.03 pp) exceeds the signed mean difference (0.238 pp), the
device effect is not separable from seed noise at this sample size. The paper states
this explicitly and does not claim a hardware ranking. Any reuse of these numbers
must carry the same caveat.

---

## Pilot sweep

The pilot sweep (`data/pilot/`, `data/pilot_epochs/`) is a separate generation on
NEU-CLS only, with 3,730 completed runs of 3,789 launched. It uses a
**training-fraction** scheme (`train_frac` in {0.05, 0.10, 0.25, 0.50, 1.00}) rather
than a per-class count scheme, and it is the **only** generation with per-epoch logs.

It also splits on the **1,800 pre-deduplication images** (`data/splits/`), whereas
the main benchmark splits on the 1,799 post-deduplication images
(`data/benchmark_splits/`). The two differ by one image in the training set (0.09%).
This is an internal heterogeneity of the pilot batch and is noted in the paper as
limitation L5.

Agreement between the two generations was quantified directly: across the 1,265 runs
that share an identical model, seed, level, resolution, learning rate and training
mode, test accuracies differ by a **mean absolute difference of 1.41 pp**, which is
smaller than the frozen benchmark's own mean within-cell seed standard deviation of
3.93 pp over 540 cells. This is what licenses using the pilot per-epoch logs to
discuss curve shape. Absolute accuracies from the pilot are never cited alongside
those of the main experiment.

---

## Known limitations carried by the artifact

| ID | Limitation |
|---|---|
| L2 | `gc10det` uses 256x256 input rather than 224x224, so it is not perfectly resolution-comparable with the other three |
| L5 | Pilot and main generations split on slightly different image sets (1,800 vs 1,799) and coexist 20- and 50-epoch schedules |
| - | The main benchmark did not persist per-epoch logs, so training-dynamics analysis rests on the pilot sweep alone |
| - | Model checkpoints (242 GB) are not published; they are regenerable from the code, splits and manifest |

---

## Environment provenance

Each result JSON is self-describing: it records `device`, `amp_dtype`, `torch` and
`python`. No result in this artifact depends on a version that is not recorded in the
file itself.
