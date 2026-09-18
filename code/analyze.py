"""Paper A result aggregation and plotting.

Collapses the thousands of individual runs in results/*.json into **one reusable wide table** plus an interim figure.

Outputs
-------
  analysis/results_all.csv     one row per run, flat columns (6 per-class accuracies + 36 confusion-matrix cells)
  analysis/fig_interim.png     interim figure (3 panels)
  stdout                       summary statistics

Discipline
----------
  * read-only on results/, never modify the raw results
  * bad files are skipped and counted, never silently swallowed
  * figures use English labels (no CJK font on the cluster; Chinese renders as boxes)
"""

import json
import os
import re
import sys
from collections import defaultdict

WORK = "/var/hpc-root/xiangxu/work"
RESULTS = os.path.join(WORK, "results")
OUT = os.path.join(WORK, "analysis")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]

os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- summary wide table
rows, bad = [], []
files = sorted(f for f in os.listdir(RESULTS) if f.endswith(".json"))
for fn in files:
    p = os.path.join(RESULTS, fn)
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("type") != "final":
            continue
        r = {
            "run_id": d["run_id"], "model": d["model"], "family": d.get("family"),
            "seed": d["seed"], "per_class": d.get("per_class"), "img_size": d.get("img_size"),
            "mode": d.get("mode"), "lr": d.get("lr"), "epochs": d.get("epochs"),
            "bs": d.get("bs"), "wd": d.get("wd"), "pretrained": d.get("pretrained"),
            "n_params": d.get("n_params"), "n_train": d.get("n_train"),
            "n_val": d.get("n_val"), "n_test": d.get("n_test"),
            "test_acc": d.get("test_acc"), "test_loss": d.get("test_loss"),
            "best_val_acc": d.get("best_val_acc"), "best_epoch": d.get("best_epoch"),
            "total_sec": d.get("total_sec"),
            "device": d.get("device"), "amp_dtype": d.get("amp_dtype"),
            "torch": d.get("torch"), "ts": d.get("ts"),
        }
        pca = d.get("per_class_acc") or {}
        for c in CLASSES:
            r["acc_" + c] = pca.get(c)
        cm = d.get("confusion")
        if cm:
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    r["cm_%d_%d" % (i, j)] = cm[i][j]
        rows.append(r)
    except Exception as e:
        bad.append((fn, str(e)[:80]))

print("scanned %d json files, parsed %d runs, %d bad files" % (len(files), len(rows), len(bad)))
for fn, e in bad[:5]:
    print("  BAD %s : %s" % (fn, e))

if not rows:
    sys.exit("no results to analyze")

cols = list(rows[0].keys())
for r in rows:
    for c in r:
        if c not in cols:
            cols.append(c)
csv_path = os.path.join(OUT, "results_all.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
print("wrote %s  (%d rows x %d cols)" % (csv_path, len(rows), len(cols)))

# ---------------------------------------------------------------- cohort definition
# Main cohort = W1/W2: res224 + finetune + lr3e-4 + epochs50
main = [r for r in rows if r["img_size"] == 224 and r["mode"] == "finetune"
        and abs((r["lr"] or 0) - 3e-4) < 1e-9 and r["epochs"] == 50]
print("\nmain cohort (res224/finetune/lr3e-4/ep50): %d runs" % len(main))
print("  covers %d models, per-class image levels %s, seeds %s"
      % (len(set(r["model"] for r in main)),
         sorted(set(r["per_class"] for r in main)),
         sorted(set(r["seed"] for r in main))))

# ---------------------------------------------------------------- per-class curves (core)
agg = defaultdict(list)          # (per_class, cls) -> [acc...]
for r in main:
    for c in CLASSES:
        v = r.get("acc_" + c)
        if v is not None:
            agg[(r["per_class"], c)].append(v)

print("\n=== INTERIM: per-class accuracy vs training images per class (main cohort, mean over models/seeds) ===")
levels = sorted(set(k[0] for k in agg))
print("  n/class | " + " | ".join("%-16s" % c[:16] for c in CLASSES))
for n in levels:
    cells = []
    for c in CLASSES:
        vals = agg.get((n, c), [])
        cells.append("%-16s" % ("%.1f (k=%d)" % (sum(vals) / len(vals), len(vals)) if vals else "-"))
    print("  %7d | %s" % (n, " | ".join(cells)))

# Annotations per class needed to reach 95% / 99% -- the core number for the paper.
# The old "first drop below 90%" metric is retired: at n=1 almost every class is below 90%, so it degenerates.
print("\n=== annotations per class needed to reach 95% / 99% (mean over models/seeds) ===")
print("  %-18s %-9s %-9s %s" % ("class", "n@95%", "n@99%", "gain(n=1 -> max)"))
for c in CLASSES:
    def need(thr):
        for n in levels:
            v = agg.get((n, c), [])
            if v and sum(v) / len(v) >= thr:
                return n
        return None
    n95, n99 = need(95.0), need(99.0)
    a1, aX = agg.get((min(levels), c), []), agg.get((max(levels), c), [])
    gain = (sum(aX) / len(aX) - sum(a1) / len(a1)) if a1 and aX else float("nan")
    print("  %-18s %-9s %-9s %+.1f pp"
          % (c, ("n=%d" % n95) if n95 else ">max", ("n=%d" % n99) if n99 else ">max", gain))

# ---------------------------------------------------------------- device reproducibility (T4 FP16 vs A4000 BF16)
print("\n=== device reproducibility: same config on T4(FP16) vs A4000(BF16) ===")
byid = {r["run_id"]: r for r in rows}
pairs = []
for r in rows:
    if r["run_id"].endswith("_a4"):
        t4 = byid.get(r["run_id"][:-3])
        if t4:
            pairs.append((r["run_id"][:-3], t4["test_acc"], r["test_acc"], t4.get("amp_dtype"), r.get("amp_dtype")))
if pairs:
    import statistics as st
    d = [abs(a - b) for _, a, b, _, _ in pairs]
    print("  %d paired runs  |delta test_acc| mean %.2f pp  median %.2f pp  max %.2f pp"
          % (len(pairs), sum(d) / len(d), st.median(d), max(d)))
    print("  T4 amp=%s   A4000 amp=%s" % (pairs[0][3], pairs[0][4]))
    sp = {}
    for r in main:
        sp.setdefault((r["model"], r["per_class"]), []).append(r["test_acc"])
    spd = [max(v) - min(v) for v in sp.values() if len(v) >= 2]
    if spd:
        print("  reference baseline: same-config cross-seed spread, mean %.2f pp" % (sum(spd) / len(spd)))
        print("  -> device difference %s seed difference; hardware %s affect the conclusions"
              % ("far smaller than" if sum(d) / len(d) < 0.3 * (sum(spd) / len(spd)) else "comparable to",
                 "does not" if sum(d) / len(d) < 0.3 * (sum(spd) / len(spd)) else "may"))
    print("  5 pairs with the largest difference:")
    for bid, a, b, _, _ in sorted(pairs, key=lambda x: -abs(x[1] - x[2]))[:5]:
        print("    %-48s T4=%6.2f  A4000=%6.2f  delta=%+.2f" % (bid, a, b, b - a))
else:
    print("  no paired _a4 results found")

# ---------------------------------------------------------------- plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # Figure 1: per-class curves
    ax = axes[0]
    for c in CLASSES:
        xs = [n for n in levels if agg.get((n, c))]
        ys = [sum(agg[(n, c)]) / len(agg[(n, c)]) for n in xs]
        ax.plot(xs, ys, marker="o", ms=3, label=c)
    ax.axhline(90, ls="--", c="gray", lw=1)
    ax.set_xlabel("Training images per class"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Per-class accuracy vs data budget\n(res224, finetune, lr3e-4, ep50; mean over models/seeds)")
    ax.set_xscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)

    # Figure 2: model x budget level (overall)
    ax = axes[1]
    for m in sorted(set(r["model"] for r in main)):
        d = defaultdict(list)
        for r in main:
            if r["model"] == m:
                d[r["per_class"]].append(r["test_acc"])
        xs = sorted(d); ys = [sum(d[n]) / len(d[n]) for n in xs]
        ax.plot(xs, ys, marker=".", ms=4, label=m, alpha=.85)
    ax.set_xlabel("Training images per class"); ax.set_ylabel("Overall test accuracy (%)")
    ax.set_title("Overall accuracy by model\n(dashed = 90%)")
    ax.axhline(90, ls="--", c="gray", lw=1)
    ax.set_xscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=6, ncol=2)

    # Figure 3: model x class heatmap at the lowest data budget
    import math
    ax = axes[2]
    low = min(levels)
    ms = sorted(set(r["model"] for r in main))
    M = []
    for m in ms:
        row = []
        for c in CLASSES:
            v = [r.get("acc_" + c) for r in main if r["model"] == m and r["per_class"] == low]
            v = [x for x in v if x is not None]
            row.append(sum(v) / len(v) if v else float("nan"))
        M.append(row)
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels([c[:9] for c in CLASSES], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=7)
    ax.set_title("Per-class accuracy at the lowest data budget\n(n=%d per class)" % low)
    for i in range(len(ms)):
        for j in range(len(CLASSES)):
            if M[i][j] == M[i][j]:
                ax.text(j, i, "%.0f" % M[i][j], ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=.046)

    plt.tight_layout()
    fig_path = os.path.join(OUT, "fig_interim.png")
    plt.savefig(fig_path, dpi=110)
    print("\nwrote figure %s" % fig_path)
except Exception as e:
    print("\nplotting failed: %r" % e)

print("\nDONE")
