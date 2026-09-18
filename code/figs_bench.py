#!/usr/bin/env python3
"""Paper A main-text figures (3, publication spec).

Figure 1  Per-class data-efficiency curves for the four datasets (with cross-model spread band)
Figure 2  Difficulty map: heterogeneity vs imbalance (raw criterion vs common-level control, level caps annotated)
Figure 3  Budget-gain curves (4 lines, showing how the payoff of non-uniform allocation varies with difficulty)

Spec: vector PDF + 600 dpi PNG; font size >=8 pt; Okabe-Ito colorblind-safe; line style + marker double encoding (grayscale-distinguishable).
Read-only on results_bench/*.json.
"""

import json, math, os
from collections import defaultdict

W = "/var/hpc-root/xiangxu/work"
R = os.path.join(W, "results_bench")
BENCH = "/var/hpc-root/HPC_SHARED_DIR/benchmark"
OUT = os.path.join(W, "analysis_bench", "figs")
DATASETS = ["neu_cls", "gc10det", "magnetic_tile", "casting"]
COMMON = [1, 2, 4, 8]
os.makedirs(OUT, exist_ok=True)

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
LS = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


def main():
    rows = []
    for fn in sorted(os.listdir(R)):
        if fn.endswith(".json"):
            try:
                d = json.load(open(os.path.join(R, fn), encoding="utf-8"))
                if d.get("type") == "final":
                    rows.append(d)
            except Exception:
                pass
    man = json.load(open(os.path.join(BENCH, "manifest.json"), encoding="utf-8"))
    print("loaded %d rows" % len(rows))

    # dataset -> class -> level -> [acc]
    acc = {ds: defaultdict(lambda: defaultdict(list)) for ds in DATASETS}
    for d in rows:
        ds = d.get("dataset")
        if ds not in acc:
            continue
        n = d.get("per_class"); pca = d.get("per_class_acc") or {}
        for c, v in pca.items():
            if v is not None and v == v:
                acc[ds][c][n].append(v)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.grid": True, "grid.alpha": .3, "grid.linewidth": .5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.dpi": 120})

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), dpi=600 if ext == "png" else None)
        plt.close(fig)
        print("wrote %s.pdf / .png" % name)

    # ---------- Figure 1 ----------
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2))
    for ax, ds in zip(axes, DATASETS):
        cls = man[ds]["classes"]
        levels = sorted(set(n for c in acc[ds] for n in acc[ds][c]))
        for i, c in enumerate(cls):
            xs = [n for n in levels if acc[ds][c].get(n)]
            ys = [mean(acc[ds][c][n]) for n in xs]
            cols = OKABE[i % len(OKABE)]
            ax.plot(xs, ys, color=cols, ls=LS[i % len(LS)], marker=MK[i % len(MK)],
                    ms=2.2, lw=1.0, label=c[:11])
        ax.axhline(95, ls=":", c="gray", lw=.9)
        ax.set_xscale("log"); ax.set_ylim(0, 101)
        ax.set_title("%s\n(imbalance %.1fx, cap %d)" % (ds, _imb(man, ds), max(levels)), fontsize=8.5)
        ax.set_xlabel("images / class")
        if ds == DATASETS[0]:
            ax.set_ylabel("Per-class test accuracy (%)")
        ax.legend(loc="lower right", ncol=2, frameon=False, handlelength=1.8)
    fig.suptitle("Per-class data efficiency across four industrial defect datasets", y=1.03, fontsize=10)
    save(fig, "fig1_perclass_curves")

    # ---------- Figure 2 ----------
    raw = {}
    try:
        s = json.load(open(os.path.join(W, "analysis_bench", "bench_summary.json"), encoding="utf-8"))
        raw = {ds: s[ds].get("heterogeneity_ratio") for ds in DATASETS if ds in s}
    except Exception:
        pass
    ctrl = {}
    try:
        cc = json.load(open(os.path.join(W, "analysis_bench", "common_level_control.json"), encoding="utf-8"))
        ctrl = {ds: cc["summary"][ds]["ctrl_rel"] for ds in DATASETS if ds in cc["summary"]}
    except Exception:
        pass

    ds_sorted = sorted(DATASETS, key=lambda d: _imb(man, d))
    x = [_imb(man, d) for d in ds_sorted]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    ax.plot(x, [raw.get(d) or float("nan") for d in ds_sorted], "o--", color="#999",
            label="raw metric (confounded)", ms=5)
    ax.plot(x, [ctrl.get(d) or float("nan") for d in ds_sorted], "s-", color=OKABE[1],
            label="common-level control", ms=6)
    for xi, d in zip(x, ds_sorted):
        ax.annotate(d, (xi, (ctrl.get(d) or 0)), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("dataset imbalance ratio (max/min class)")
    ax.set_ylabel("class-level heterogeneity")
    ax.set_title("(a) Imbalance amplifies heterogeneity\nonce dynamic range is controlled")
    ax.legend(frameon=False)
    ax.axhline(0, c="k", lw=.5)

    ax = axes[1]
    for i, ds in enumerate(ds_sorted):
        cls = man[ds]["classes"]
        sp = []
        for n in COMMON:
            v = [mean(acc[ds][c][n]) for c in cls if acc[ds][c].get(n)]
            v = [t for t in v if t is not None]
            sp.append(max(v) - min(v) if len(v) >= 2 else float("nan"))
        ax.plot(COMMON, sp, marker=MK[i], ls=LS[i], color=OKABE[i], label="%s (%.1fx)" % (ds, _imb(man, ds)))
    ax.set_xscale("log"); ax.set_xticks(COMMON); ax.set_xticklabels(COMMON)
    ax.set_xlabel("images / class (common level set)")
    ax.set_ylabel("per-class accuracy spread (pp)")
    ax.set_title("(b) At equal budget, imbalanced datasets\nretain a large class gap")
    ax.legend(frameon=False)
    save(fig, "fig2_difficulty_map")

    # ---------- Figure 3 ----------
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, ds in enumerate(ds_sorted):
        bs = []
        try:
            s = json.load(open(os.path.join(W, "analysis_bench", "bench_summary.json"), encoding="utf-8"))
            bs = s[ds]["budget"]
        except Exception:
            pass
        if not bs:
            continue
        xs = [b["budget"] for b in bs]
        ys = [b["gain"] for b in bs]
        ax.plot(xs, ys, marker=MK[i], ls=LS[i], color=OKABE[i], ms=5,
                label="%s (%.1fx)" % (ds, _imb(man, ds)))
    ax.axhline(0, c="k", lw=.6)
    ax.set_xscale("log")
    ax.set_xlabel("total annotation budget (images)")
    ax.set_ylabel("gain of DP-optimal over uniform (pp)")
    ax.set_title("Non-uniform allocation pays off more\non harder / more imbalanced datasets")
    ax.legend(frameon=False)
    save(fig, "fig3_budget_gain")

    print("DONE -> %s" % OUT)


def _imb(man, ds):
    cs = sorted(man[ds]["counts"].values())
    return cs[-1] / max(1, cs[0])


if __name__ == "__main__":
    main()
