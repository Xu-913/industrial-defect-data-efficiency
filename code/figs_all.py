#!/usr/bin/env python3
"""Paper A main figures (10 figures, publication spec).

fig1  research framework diagram
fig2  data audit (counts / resolution / dedup / imbalance-vs-cap confound)
fig3  per-class data-efficiency curves (4 datasets, cross-model IQR band)
fig4  difficulty map (raw vs common-level control; spread at common levels)
fig5  row-normalised confusion heatmaps, diagonal masked (low vs high budget)
fig6  directed error-flow graph + net error flow per class
fig7  low-data outflow error rate vs annotation need (n@rel95) + controls
fig8  training dynamics from the *.jsonl epoch logs + historical/benchmark agreement
fig9  variance decomposition, seeds-needed curve, model-rank stability
fig10 budget allocation gain, absolute and same-relative-budget

Spec: vector PDF + 600 dpi PNG; font >= 8 pt; Okabe-Ito colourblind-safe;
line style + marker double encoding (grey-scale legible). English in-figure text
only (no CJK font guaranteed on the compute nodes); bilingual captions live in
the companion manifest.

Read-only w.r.t. results; writes only into analysis_bench/figs and
analysis_bench/figs_diagnostics.json.
"""

import json, math, os, re, sys
from collections import defaultdict, Counter

import numpy as np

W = "/var/hpc-root/xiangxu/work"
R = os.path.join(W, "results_bench")          # frozen benchmark, 2700 finals
H = os.path.join(W, "results")                # historical NEU sweep, 3730 json + 3789 jsonl
AUD = os.path.join(W, "audit", "audit_report.json")
AB = os.path.join(W, "analysis_bench")
BENCH = "/var/hpc-root/HPC_SHARED_DIR/benchmark"
OUT = os.path.join(AB, "figs")
DATASETS = ["neu_cls", "gc10det", "magnetic_tile", "casting"]
COMMON = [1, 2, 4, 8]
os.makedirs(OUT, exist_ok=True)

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
         "#56B4E9", "#F0E442", "#000000"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

DIAG = {}
results = []          # (fig_kind, name, ok, message)


# ----------------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------------
def mean(xs):
    xs = [float(x) for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return None, len(a)
    try:
        import scipy.stats as st
        r = st.spearmanr(a, b)
        v = getattr(r, "statistic", None)
        if v is None:
            v = getattr(r, "correlation", None)
        if v is None:
            v = r[0]
        return float(v), len(a)
    except Exception:
        ra = _rank(a)
        rb = _rank(b)
        if ra.std() == 0 or rb.std() == 0:
            return None, len(a)
        return float(np.corrcoef(ra, rb)[0, 1]), len(a)


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def kendall(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return None
    try:
        import scipy.stats as st
        r = st.kendalltau(a, b)
        v = getattr(r, "statistic", None)
        if v is None:
            v = r[0]
        return float(v)
    except Exception:
        c = d = 0
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
                if s > 0:
                    c += 1
                elif s < 0:
                    d += 1
        return (c - d) / float(c + d) if (c + d) else None


def resid(y, x):
    """Residual of y after removing a linear fit on x (for the partial control)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    m = np.isfinite(x) & np.isfinite(y)
    out = np.full(len(y), np.nan)
    if m.sum() < 3:
        return out
    xm, ym = x[m], y[m]
    A = np.vstack([np.ones_like(xm), xm]).T
    coef, *_ = np.linalg.lstsq(A, ym, rcond=None)
    out[m] = ym - A.dot(coef)
    return out


def load_bench():
    """All frozen-benchmark final records, plus the flattened confusion cubes."""
    rows = []
    for fn in os.listdir(R):
        if not fn.endswith(".json"):
            continue
        try:
            d = json.load(open(os.path.join(R, fn), encoding="utf-8"))
        except Exception:
            continue
        if d.get("type") == "final":
            rows.append(d)
    return rows


_SUMM = None


def summary():
    """bench_summary.json, cached. The authoritative source of swept level grids."""
    global _SUMM
    if _SUMM is None:
        _SUMM = json.load(open(os.path.join(AB, "bench_summary.json"), encoding="utf-8"))
    return _SUMM


def caps():
    """Largest per-class level actually swept per dataset.

    This is NOT min(class counts): the grid was capped by design (180 / 18 / 38 / 512),
    and the smallest class still holds 299 / 31 / 64 / 3073 images. Confusing the two
    silently empties every high-budget panel.
    """
    s = summary()
    return {ds: int(max(s[ds]["levels"])) for ds in DATASETS}


_FAM = None


def model_families():
    """Architecture family per model name.

    results_bench records do NOT carry a `family` field (the historical sweep does),
    so looking it up there silently yields an all-'?' legend.
    """
    global _FAM
    if _FAM is None:
        _FAM = {}
        seen = set()
        for fn in sorted(os.listdir(H)):
            if not fn.endswith(".json"):
                continue
            m = fn.split("_seed")[0]
            if m in seen:
                continue
            seen.add(m)
            try:
                d = json.load(open(os.path.join(H, fn), encoding="utf-8"))
            except Exception:
                continue
            if d.get("family"):
                _FAM[m] = d["family"]
            if len(seen) >= 11:
                break
    return _FAM


def conf_cube(rows, ds, level):
    """Mean confusion over models x seeds at one per-class level. Returns C, classes."""
    sel = [d for d in rows if d.get("dataset") == ds and d.get("per_class") == level]
    if not sel:
        return None, None
    k = max(int(d["n_classes"]) for d in sel)
    acc = np.zeros((k, k), float)
    n = 0
    classes = None
    for d in sel:
        c = d.get("confusion")
        if not c:
            continue
        c = np.asarray(c, float)
        if c.shape != (k, k):
            continue
        acc += c
        n += 1
        classes = d.get("classes") or classes
    if n == 0:
        return None, None
    return acc / n, classes


def rownorm(C):
    s = C.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return C / s


# ----------------------------------------------------------------------------
# figure 1 -- research framework
# ----------------------------------------------------------------------------
def fig1(man):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, ts=8.6, bs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=1.6",
                                    linewidth=1.1, facecolor=fc, edgecolor=ec, zorder=2))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top",
                fontsize=ts, fontweight="bold", color="#111111", zorder=3)
        if body:
            ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top",
                    fontsize=bs, color="#222222", zorder=3, linespacing=1.45)

    def arrow(x1, y1, x2, y2, style="-|>", col="#444444", lw=1.2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, linewidth=lw,
                                     color=col, linestyle=ls, zorder=1))

    # stage band labels
    for i, (lab, x) in enumerate([("STAGE 1  DATA", 0.5), ("STAGE 2  SWEEP", 25.5),
                                  ("STAGE 3  ANALYSIS", 50.5), ("STAGE 4  ASSETS", 76.0)]):
        ax.text(x, 60.0, lab, ha="left", va="top", fontsize=7.4,
                color="#666666", fontweight="bold")

    counts = {ds: sum(man[ds]["counts"].values()) for ds in DATASETS}
    cls = {ds: len(man[ds]["classes"]) for ds in DATASETS}
    body1 = "\n".join("%s  %d img / %d cls" % (ds, counts[ds], cls[ds]) for ds in DATASETS)
    box(1, 6, 21, 50, "4 industrial defect sets", body1, "#EAF2F8", OKABE[0])
    ax.text(11.5, 8.0, "imbalance 1.0x - 28.5x\ndifferent sensors, scales",
            ha="center", va="bottom", fontsize=6.8, color="#555555", style="italic")

    box(25.5, 34, 21, 22, "Audit and freeze",
        "MD5 near-dup removal\nbefore any split\nstratified 60/20/20\nfrozen x 5 seeds\nper-class level grid",
        "#FDF3E7", OKABE[1])
    box(25.5, 6, 21, 24, "Benchmark sweep",
        "11 torchvision models\n4 architecture families\n50 epochs, AMP\nT4 fp16 / A4000 bf16\n4 x 11 x levels x 5 seeds\n2700 completed runs",
        "#FDF3E7", OKABE[1])
    arrow(22.4, 31, 25.2, 40)
    arrow(36, 33.6, 36, 30.4)

    box(50.5, 40, 22, 16, "A1  per-class efficiency",
        "n@95 and n@rel95\nclass-level heterogeneity", "#E9F7F1", OKABE[2])
    box(50.5, 23, 22, 15, "A2  error structure",
        "outflow / inflow asymmetry\ndirected error graph", "#E9F7F1", OKABE[2])
    box(50.5, 6, 22, 15, "A3  budget allocation",
        "DP-optimal vs uniform\nsame-relative control", "#E9F7F1", OKABE[2])
    arrow(46.9, 45, 50.2, 48)
    arrow(46.9, 37, 50.2, 30.5)
    arrow(46.9, 20, 50.2, 13.5)

    box(76.5, 40, 22, 16, "Measurement protocol",
        "seeds needed per\nresolution target\nlevel-cap caveat", "#F3ECF5", OKABE[3])
    box(76.5, 23, 22, 15, "Difficulty map",
        "imbalance vs heterogeneity\nwith truncation control", "#F3ECF5", OKABE[3])
    box(76.5, 6, 22, 15, "Decision tables",
        "when allocation pays\nper class annotation plan", "#F3ECF5", OKABE[3])
    for y1, y2 in [(48, 48), (30.5, 30.5), (13.5, 13.5)]:
        arrow(72.9, y1, 76.2, y2, col="#777777")

    ax.text(50, 2.2,
            "Full sweep is CPU-side reproducible:  manifest + frozen splits + one JSON record per run "
            "(all figures regenerate from those records alone).",
            ha="center", va="center", fontsize=7.0, color="#444444")
    save(fig, "fig1_framework")
    return {"stage2_runs": 2700}


# ----------------------------------------------------------------------------
# figure 2 -- data audit
# ----------------------------------------------------------------------------
def fig2(man, audit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.size": 9, "axes.labelsize": 8.5, "axes.titlesize": 9})

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6))

    # (a) per-class counts
    ax = axes[0][0]
    for i, ds in enumerate(DATASETS):
        c = man[ds]["counts"]
        v = sorted(c.values(), reverse=True)
        ax.plot(range(1, len(v) + 1), v, marker=MK[i], ls=LS[i], color=OKABE[i],
                ms=4.2, lw=1.2, label="%s (%d cls)" % (ds, len(v)))
    ax.set_yscale("log")
    ax.set_xlabel("class rank (most to least populated)")
    ax.set_ylabel("images in class (log)")
    ax.set_title("(a) Per-class population\nimbalance spans 1.0x to 28.5x")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")

    # (b) native resolution: one size dominates almost everywhere
    ax = axes[0][1]
    names = [e["dataset"] for e in audit]
    for i, e in enumerate(audit):
        tot = e["totals"]["n_images"]
        ts = e["totals"]["top_sizes"]
        dom_sz, dom_n = ts[0]
        frac = 100.0 * dom_n / max(1, tot)
        grey = e["dataset"] == "bsdata"
        ax.barh(i, frac, height=0.62, color="#BBBBBB" if grey else OKABE[0],
                edgecolor="white", linewidth=0.6, alpha=0.5 if grey else 0.95)
        ax.barh(i, 100.0 - frac, left=frac, height=0.62, color="#EEEEEE",
                edgecolor="white", linewidth=0.6)
        fmt = "%.1f%%" if frac < 10 else "%.0f%%"
        ax.text(frac + 2.5, i, ("%s  " + fmt + "  (%d distinct sizes)")
                % (dom_sz, frac, len(ts)), va="center", fontsize=6.2, color="#333333")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([((n + " (excluded)") if n == "bsdata"
                         else (n + " (= neu_cls)") if n == "neu_surface" else n)
                        + "  n=%d" % e["totals"]["n_images"]
                        for n, e in zip(names, audit)], fontsize=7.0)
    ax.set_xlim(0, 132)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of images in the single most common resolution (%)")
    ax.set_title("(b) Native resolution spread\ndominance ranges from 0 % to 100 %")

    # (c) dedup / integrity
    ax = axes[1][0]
    x = np.arange(len(audit))
    dup = [e["totals"]["n_dup_extra"] for e in audit]
    cor = [e["totals"]["n_corrupt"] for e in audit]
    tot = [e["totals"]["n_images"] for e in audit]
    ax.bar(x - 0.2, tot, width=0.4, color="#BBBBBB", label="images found")
    ax.bar(x + 0.2, dup, width=0.4, color=OKABE[1], label="near/exact duplicates")
    ax.bar(x + 0.2, cor, width=0.4, bottom=dup, color=OKABE[7], label="corrupt")
    for xi, d, t in zip(x, dup, tot):
        ax.text(xi + 0.2, d + 1.0, "%d\n(%.2f%%)" % (d, 100.0 * d / max(1, t)),
                ha="center", va="bottom", fontsize=6.4, color=OKABE[1])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=7.2)
    ax.set_ylabel("images (log)")
    ax.set_title("(c) Integrity check\ncontamination is small but not zero")
    ax.legend(frameon=False, fontsize=6.8, loc="upper right")

    # (d) the confound: imbalance vs the level cap the sweep could actually reach
    ax = axes[1][1]
    CAPS = caps()
    for i, ds in enumerate(DATASETS):
        cs = sorted(man[ds]["counts"].values())
        imb = cs[-1] / max(1, cs[0])
        smallest, cap = cs[0], CAPS[ds]
        ax.plot([imb, imb], [smallest, cap], color=OKABE[i], lw=1.0, alpha=0.55, zorder=1)
        ax.scatter([imb], [cap], s=62, marker=MK[i], color=OKABE[i],
                   edgecolor="black", linewidth=0.5, zorder=3)
        ax.scatter([imb], [smallest], s=34, marker="x", color=OKABE[i], zorder=3)
        ax.annotate("%s\nswept to %d of %d" % (ds, cap, smallest), (imb, cap),
                    textcoords="offset points", xytext=(7, 2), fontsize=6.4,
                    color=OKABE[i])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("imbalance ratio (largest / smallest class)")
    ax.set_ylabel("images per class (log)")
    ax.set_title("(d) The level-cap confound\nimbalance leaves most of each class unused")
    ax.grid(alpha=0.3, which="both", linewidth=0.5)
    fig.tight_layout()
    save(fig, "fig2_data_audit")

    return {"level_cap": CAPS,
            "smallest_class": {ds: min(man[ds]["counts"].values()) for ds in DATASETS},
            "imbalance": {ds: sorted(man[ds]["counts"].values())[-1] /
                                max(1, sorted(man[ds]["counts"].values())[0]) for ds in DATASETS},
            "dup_extra": {e["dataset"]: e["totals"]["n_dup_extra"] for e in audit},
            "n_images": {e["dataset"]: e["totals"]["n_images"] for e in audit}}


# ----------------------------------------------------------------------------
# figure 3 -- per-class data-efficiency curves
# ----------------------------------------------------------------------------
def fig3(man, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.size": 9, "axes.labelsize": 8.5, "axes.titlesize": 8.6,
                     "legend.fontsize": 6.4})

    acc = {ds: defaultdict(lambda: defaultdict(list)) for ds in DATASETS}
    for d in rows:
        ds = d.get("dataset")
        if ds not in acc:
            continue
        n = d.get("per_class")
        for c, v in (d.get("per_class_acc") or {}).items():
            if v is not None and v == v:
                acc[ds][c][n].append(float(v))

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.5))
    diag = {}
    for ax, ds in zip(axes, DATASETS):
        cls = man[ds]["classes"]
        levels = sorted({n for c in acc[ds] for n in acc[ds][c]})
        diag[ds] = {}
        for i, c in enumerate(cls):
            per = {n: acc[ds][c][n] for n in levels if acc[ds][c].get(n)}
            xs = sorted(per)
            ys = np.array([mean(per[n]) for n in xs])
            lo = np.array([np.percentile(per[n], 25) for n in xs])
            hi = np.array([np.percentile(per[n], 75) for n in xs])
            col = OKABE[i % len(OKABE)]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.11, linewidth=0)
            ax.plot(xs, ys, color=col, ls=LS[i % len(LS)], marker=MK[i % len(MK)],
                    ms=2.4, lw=1.15, label=c.replace("MT_", "").replace("_", " ")[:13])
            diag[ds][c] = {"low": float(ys[0]), "high": float(ys[-1]),
                           "gain": float(ys[-1] - ys[0]), "max_level": int(xs[-1])}
        ax.axhline(95, ls=":", c="gray", lw=0.9)
        ax.text(levels[0], 95.6, "95%", fontsize=6.0, color="gray")
        ax.set_xscale("log")
        ax.set_ylim(0, 103)
        cs = sorted(man[ds]["counts"].values())
        ax.set_title("%s\nimbalance %.1fx, swept to n=%d per class"
                     % (ds, cs[-1] / max(1, cs[0]), caps()[ds]))
        ax.set_xlabel("images per class (log)")
        if ds == DATASETS[0]:
            ax.set_ylabel("per-class test accuracy (%)")
        ax.legend(loc="lower right", ncol=2, frameon=False, handlelength=1.6,
                  columnspacing=0.8, labelspacing=0.25)
    fig.suptitle("Per-class data efficiency on four industrial defect datasets "
                 "(line = mean over 11 models x 5 seeds, band = interquartile range)",
                 y=1.04, fontsize=9.6)
    save(fig, "fig3_perclass_curves")
    return diag


# ----------------------------------------------------------------------------
# figure 4 -- difficulty map
# ----------------------------------------------------------------------------
def fig4(man, acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import NullLocator
    rcParams.update({"font.size": 9, "axes.labelsize": 8.5})

    s = json.load(open(os.path.join(AB, "bench_summary.json"), encoding="utf-8"))
    cc = json.load(open(os.path.join(AB, "common_level_control.json"), encoding="utf-8"))
    raw = {ds: s[ds].get("heterogeneity_ratio") for ds in DATASETS}
    ctrl = {ds: cc["summary"][ds]["ctrl_rel"] for ds in DATASETS}
    imb = {ds: cc["summary"][ds]["imbalance"] for ds in DATASETS}

    ds_sorted = sorted(DATASETS, key=lambda d: imb[d])
    x = [imb[d] for d in ds_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax = axes[0]
    ax.plot(x, [raw[d] for d in ds_sorted], marker="o", ls="--", color="#999999",
            ms=5.5, lw=1.3, label="raw metric (level caps differ)")
    ax.plot(x, [ctrl[d] for d in ds_sorted], marker="s", ls="-", color=OKABE[1],
            ms=6.5, lw=1.6, label="common-level control [1,2,4,8]")
    for i, (xi, d) in enumerate(zip(x, ds_sorted)):
        # neu_cls and casting sit almost on top of each other at the low-imbalance end
        off = [(3, 9), (3, -17), (0, 10), (0, -18)][i % 4]
        ha = "left" if i < 2 else "center"
        ax.annotate("%s\nswept to %d" % (d, caps()[d]),
                    (xi, ctrl[d]), textcoords="offset points", xytext=off,
                    ha=ha, fontsize=6.6, color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel("dataset imbalance ratio (log)")
    ax.set_ylabel("class-level heterogeneity of n@95")
    ax.set_title("(a) Imbalance amplifies heterogeneity\nonce the dynamic range is controlled")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    ax.axhline(0, c="k", lw=0.5)
    _allv = [v for v in list(raw.values()) + list(ctrl.values()) if v is not None]
    ax.set_ylim(-0.35, max(_allv) * 1.16)

    ax = axes[1]
    for i, ds in enumerate(ds_sorted):
        cls = man[ds]["classes"]
        sp = []
        for n in COMMON:
            v = [mean(acc[ds][c][n]) for c in cls if acc[ds][c].get(n)]
            v = [t for t in v if t is not None]
            sp.append(max(v) - min(v) if len(v) >= 2 else np.nan)
        cap = caps()[ds]
        ax.plot(COMMON, sp, marker=MK[i], ls=LS[i], color=OKABE[i], ms=5.5, lw=1.4,
                label="%s (%.1fx)" % (ds, imb[ds]))
    ax.set_xscale("log")
    ax.set_xticks(COMMON)
    ax.set_xticklabels(COMMON)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("images per class (common level set)")
    ax.set_ylabel("spread of per-class accuracy (pp)")
    ax.set_title("(b) At an equal budget the imbalanced sets\nkeep a 5x larger class gap")
    ax.legend(frameon=False, fontsize=6.8, loc="center right")
    save(fig, "fig4_difficulty_map")

    sp8 = {}
    for ds in DATASETS:
        cls = man[ds]["classes"]
        v = [mean(acc[ds][c][8]) for c in cls if acc[ds][c].get(8)]
        v = [t for t in v if t is not None]
        sp8[ds] = float(max(v) - min(v)) if len(v) >= 2 else None
    return {"raw": raw, "ctrl_rel": ctrl, "imbalance": imb, "spread_at_8": sp8}


# ----------------------------------------------------------------------------
# figure 5 -- confusion heatmaps
# ----------------------------------------------------------------------------
def fig5(man, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import LinearSegmentedColormap
    rcParams.update({"font.size": 9})

    cmap = LinearSegmentedColormap.from_list("ok", ["#FFFFFF", "#A8CEE4", OKABE[0], "#00304F"])
    cmap.set_bad("#E8E8E8")

    CAPS = caps()

    # One shared colour scale across all eight panels. Scaling per panel (or letting the
    # last panel decide) makes the low-budget row saturate and destroys the comparison
    # that the figure exists to make.
    panels = {}
    vmax = 0.0
    for j, ds in enumerate(DATASETS):
        for i, lv in enumerate([1, CAPS[ds]]):
            C, _ = conf_cube(rows, ds, lv)
            if C is None:
                panels[(i, j)] = None
                continue
            Pp = rownorm(C) * 100.0
            M = Pp.copy()
            np.fill_diagonal(M, np.nan)
            panels[(i, j)] = (lv, C, Pp, M)
            if i == 0:
                vmax = max(vmax, float(np.nanmax(M)))
    vmax = float(min(40.0, max(10.0, vmax)))

    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.2))
    diag = {}
    im = None
    for j, ds in enumerate(DATASETS):
        cls = man[ds]["classes"]
        for i in range(2):
            ax = axes[i][j]
            pan = panels[(i, j)]
            if pan is None:
                ax.axis("off")
                continue
            lv, C, Pp, M = pan
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xticks(range(len(cls)))
            ax.set_yticks(range(len(cls)))
            lab = [c.replace("MT_", "").replace("_", " ")[:11] for c in cls]
            ax.set_xticklabels(lab, rotation=90, fontsize=5.8)
            ax.set_yticklabels(lab if j == 0 else [], fontsize=5.8)
            for a in range(len(cls)):
                for b in range(len(cls)):
                    if a == b:
                        continue
                    thr_txt = 1.0 if len(cls) <= 6 else 5.0
                    if M[a, b] >= thr_txt:
                        ax.text(b, a, "%.0f" % M[a, b], ha="center", va="center",
                                fontsize=5.4 if len(cls) <= 6 else 4.2,
                                color="white" if M[a, b] > 0.72 * vmax else "#222222")
            ax.set_title("%s  n=%d  (acc %.1f%%)"
                         % (ds, lv, 100.0 * np.trace(C) / C.sum()), fontsize=7.6)
            if j == 0:
                ax.set_ylabel("true class\n(n=%d)" % lv, fontsize=7.4)
            if i == 1:
                ax.set_xlabel("predicted class", fontsize=7.4)
            off = Pp.copy()
            np.fill_diagonal(off, 0.0)
            diag.setdefault(ds, {})[str(lv)] = {
                "mean_offdiag_pct": float(off.sum(axis=1).mean()),
                "worst_row": cls[int(np.argmax(off.sum(axis=1)))],
            }
    cb = fig.colorbar(im, ax=axes, fraction=0.014, pad=0.012, extend="max")
    cb.set_label("share of true class predicted as this class "
                 "(%, diagonal masked, shared scale)", fontsize=7.4)
    fig.suptitle("Directed error structure at the smallest and largest usable budget "
                 "(mean over all models x 5 seeds)", y=0.985, fontsize=9.8)
    save(fig, "fig5_confusion_heatmaps")
    return diag


# ----------------------------------------------------------------------------
# figure 6 -- directed error-flow graph
# ----------------------------------------------------------------------------
def fig6(man, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.patches import FancyArrowPatch, Circle
    from matplotlib.colors import TwoSlopeNorm
    rcParams.update({"font.size": 9, "axes.labelsize": 8.5})

    ds = "neu_cls"
    cls = man[ds]["classes"]
    k = len(cls)
    # Drawn at n=1 and n=20. By the swept cap (n=180) every class sits near 99.5 % and
    # the net flow collapses to a few tenths of a pp, i.e. to noise - an unreadable
    # graph. Panel (c) carries that dissolution explicitly on a symlog axis instead.
    P = {}
    for lv in (1, 8, 20, 180):
        C, _ = conf_cube(rows, ds, lv)
        P[lv] = rownorm(C)

    def net(Q):
        out = Q.sum(axis=1) - np.diag(Q)
        inn = Q.sum(axis=0) - np.diag(Q)
        return (inn - out) * 100.0

    nets = {lv: net(P[lv]) for lv in P}
    netlo, net8, netmid, nethi = nets[1], nets[8], nets[20], nets[180]

    fig = plt.figure(figsize=(15.4, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.28)

    ang = np.linspace(90, 90 + 360, k, endpoint=False) * math.pi / 180.0
    pos = {i: (math.cos(a), math.sin(a)) for i, a in enumerate(ang)}

    nrm = TwoSlopeNorm(vmin=min(-8.0, netlo.min()), vcenter=0.0,
                       vmax=max(8.0, netlo.max()))

    def draw(ax, Q, nv, title, thr):
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-1.42, 1.42)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=8.4)
        for a in range(k):
            for b in range(k):
                if a == b or Q[a, b] < thr:
                    continue
                x1, y1 = pos[a]
                x2, y2 = pos[b]
                dx, dy = x2 - x1, y2 - y1
                L = math.hypot(dx, dy)
                sx, sy = x1 + 0.20 * dx / L, y1 + 0.20 * dy / L
                ex, ey = x2 - 0.20 * dx / L, y2 - 0.20 * dy / L
                ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey),
                                             connectionstyle="arc3,rad=0.22",
                                             arrowstyle="-|>", mutation_scale=8,
                                             linewidth=0.7 + 16.0 * Q[a, b],
                                             color="#333333", alpha=0.72, zorder=1))
        for i in range(k):
            x, y = pos[i]
            ax.add_patch(Circle((x, y), 0.155,
                                facecolor=plt.get_cmap("RdBu_r")(nrm(nv[i])),
                                edgecolor="black", linewidth=0.7, zorder=3))
            lab = cls[i].replace("_", " ")
            ax.text(x * 1.34, y * 1.34, lab, ha="center", va="center", fontsize=7.0,
                    zorder=4)

    ax = fig.add_subplot(gs[0, 0])
    draw(ax, P[1], netlo, "(a) n=1: one dominant source and sink", 0.01)
    ax = fig.add_subplot(gs[0, 1])
    draw(ax, P[8], net8, "(b) n=8: the dominant channel has closed", 0.01)

    ax = fig.add_subplot(gs[0, 2])
    xx = np.arange(k)
    w = 0.21
    for j, (lv, nv, col) in enumerate([(1, netlo, OKABE[1]),
                                       (8, net8, OKABE[4]),
                                       (20, netmid, OKABE[2]),
                                       (180, nethi, OKABE[0])]):
        ax.bar(xx + (j - 1.5) * w, nv, width=w, color=col, label="n=%d" % lv)
    ax.axhline(0, c="k", lw=0.7)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xticks(xx)
    ax.set_xticklabels([c.replace("_", " ")[:12] for c in cls], rotation=28,
                       ha="right", fontsize=7.0)
    ax.set_ylabel("net error flow (pp of class mass, symlog)")
    ax.set_title("(c) Sinks and sources dissolve as data grows\n"
                 "(|net| <= %.1f pp at n=180)" % float(np.max(np.abs(nethi))))
    ax.legend(frameon=False, fontsize=7.0, loc="lower center", ncol=4, framealpha=0.88)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=nrm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("node colour: net flow at n=1 (pp)", fontsize=7.0)
    fig.suptitle("Directed confusion graph, NEU-CLS "
                 "(edge width = row-normalised transition rate, edges above 1 % shown)",
                 y=1.02, fontsize=9.8)
    save(fig, "fig6_error_graph")

    return {"levels": {str(lv): {cls[i]: float(nets[lv][i]) for i in range(k)} for lv in nets},
            "sink_at_n1": cls[int(np.argmax(netlo))],
            "source_at_n1": cls[int(np.argmin(netlo))],
            "sink_at_n180": cls[int(np.argmax(nethi))],
            "source_at_n180": cls[int(np.argmin(nethi))],
            "max_abs_net_n180_pp": float(np.max(np.abs(nethi))),
            "spearman_net_n1_vs_n8": spearman(netlo, net8)[0],
            "spearman_net_n1_vs_n20": spearman(netlo, netmid)[0],
            "spearman_net_n1_vs_n180": spearman(netlo, nethi)[0],
            "note_n180": "class identities of the sink and the source are nominally "
                         "preserved at n=180, but the magnitudes are <= 0.4 pp, i.e. "
                         "indistinguishable from measurement noise; do not quote "
                         "n=180 sink/source strengths"}


# ----------------------------------------------------------------------------
# figure 7 -- low-data error structure vs annotation need
# ----------------------------------------------------------------------------
def fig7(man, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import NullLocator
    rcParams.update({"font.size": 9, "axes.labelsize": 8.5})

    s = json.load(open(os.path.join(AB, "bench_summary.json"), encoding="utf-8"))

    # per-(dataset, model, class): outflow error rate at the smallest level, and n@rel95
    idx = defaultdict(list)
    for d in rows:
        idx[(d.get("dataset"), d.get("model"), d.get("per_class"))].append(d)

    pts = []
    for ds in DATASETS:
        cls = man[ds]["classes"]
        lo = min(s[ds]["levels"])
        for m, per in s[ds]["per_model_n95"].items():
            C, _ = conf_cube(idx.get((ds, m, lo), []), ds, lo)
            if C is None:
                continue
            P = rownorm(C)
            for i, c in enumerate(cls):
                if i >= P.shape[0]:
                    continue
                out = float(P[i].sum() - P[i, i])
                nr = (per.get(c) or {}).get("nrel95")
                mx = (per.get(c) or {}).get("max")
                if nr is None:
                    continue
                pts.append((ds, m, c, out, float(nr), float(mx) if mx else np.nan))

    fam = model_families()

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1))

    # (a) NEU-CLS only: 6 classes x 11 models = 66 points
    ax = axes[0]
    neu = [p for p in pts if p[0] == "neu_cls"]
    rho_neu, n_neu = spearman([p[3] for p in neu], [p[4] for p in neu])
    fams = sorted({fam.get(p[1], "?") for p in neu})
    for i, f in enumerate(fams):
        q = [p for p in neu if fam.get(p[1], "?") == f]
        ax.scatter([p[3] * 100 for p in q], [p[4] for p in q], s=26, marker=MK[i],
                   color=OKABE[i], alpha=0.85, edgecolor="black", linewidth=0.3,
                   label="%s (%d)" % (f, len(q)))
    _fitline(ax, [p[3] * 100 for p in neu], [p[4] for p in neu])
    ax.set_xlabel("outflow error rate at n=1 (%)")
    ax.set_ylabel("n@rel95 (images per class)")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 30])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "30"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_title("(a) NEU-CLS, 66 model x class points\nSpearman rho = %+.3f (n=%d)"
                 % (rho_neu, n_neu))
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")

    # (b) all four datasets, one point per (dataset, class) using the mean over models
    ax = axes[1]
    agg = defaultdict(lambda: [[], []])
    for ds, m, c, out, nr, mx in pts:
        agg[(ds, c)][0].append(out)
        agg[(ds, c)][1].append(nr)
    per_ds = defaultdict(list)
    for (ds, c), (o, nrr) in agg.items():
        per_ds[ds].append((mean(o), mean(nrr)))
    rho_all, n_all = spearman([v[0] for ds in DATASETS for v in per_ds[ds]],
                              [v[1] for ds in DATASETS for v in per_ds[ds]])
    for i, ds in enumerate(DATASETS):
        v = per_ds[ds]
        ax.scatter([a * 100 for a, _ in v], [b for _, b in v], s=40, marker=MK[i],
                   color=OKABE[i], edgecolor="black", linewidth=0.4, label=ds)
    _fitline(ax, [a * 100 for ds in DATASETS for a, _ in per_ds[ds]],
             [b for ds in DATASETS for _, b in per_ds[ds]])
    ax.set_xlabel("outflow error rate at the smallest level (%)")
    ax.set_ylabel("n@rel95 (images per class)")
    ax.set_yscale("log")
    ax.set_yticks([2, 5, 10, 20, 50, 100, 200])
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100", "200"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_title("(b) All datasets, %d class points\nSpearman rho = %+.3f" % (n_all, rho_all))
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")

    # (c) controls: is the predictor just re-measuring the same run?
    ax = axes[2]
    same, cross = [], []
    models = sorted({p[1] for p in neu})
    for m in models:
        q = [p for p in neu if p[1] == m]
        same.append(spearman([p[3] for p in q], [p[4] for p in q])[0])
        for m2 in models:
            if m2 == m:
                continue
            by = {p[2]: p for p in neu if p[1] == m2}
            xx, yy = [], []
            for p in q:
                if p[2] in by:
                    xx.append(p[3])
                    yy.append(by[p[2]][4])
            if len(xx) >= 4:
                cross.append(spearman(xx, yy)[0])
    part, _ = spearman(resid([p[4] for p in neu], [p[5] for p in neu]),
                       [p[3] for p in neu])
    vals = [mean([v for v in same if v is not None]),
            mean([v for v in cross if v is not None]), part, rho_neu]
    labs = ["same\nmodel", "held-out\nmodel", "minus\nceiling", "pooled"]
    ax.bar(range(4), vals, color=[OKABE[0], OKABE[2], OKABE[3], "#999999"],
           edgecolor="black", linewidth=0.5, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, "%+.3f" % v, ha="center", fontsize=7.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=7.8)
    ax.axhline(rho_neu, color="#999999", ls=":", lw=1.1)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Spearman rho")
    ax.set_xlabel("held-out transfer pools %d (model A, model B) pairs" % len(cross),
                  fontsize=7.2)
    ax.set_title("(c) Not a tautology, but only partly transferable:\n"
                 "held-out models keep %.0f%% of the association"
                 % (100.0 * vals[1] / rho_neu))
    save(fig, "fig7_outflow_vs_need")

    return {"rho_neu": rho_neu, "n_neu": n_neu, "rho_all": rho_all, "n_all": n_all,
            "rho_same_model_mean": vals[0], "rho_heldout_model_mean": vals[1],
            "rho_partial_ceiling": part, "n_heldout_pairs": len(cross),
            "outflow_definition": "row-normalised off-diagonal mass at the smallest "
                                  "swept level (1 - recall) for one model",
            "need_definition": "nrel95 of the same (dataset, model, class)",
            "reconciliation_note": "an earlier draft reported rho = +0.905 on 66 NEU "
                                   "points; the frozen benchmark reproduces +%.3f on the "
                                   "same 66 points, so that figure must be re-derived "
                                   "before it is reused" % rho_neu}


def _fitline(ax, x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ly = np.log10(np.clip(y, 1e-9, None))
    m = np.isfinite(x) & np.isfinite(ly)
    if m.sum() < 3:
        return
    k, b = np.polyfit(x[m], ly[m], 1)
    xs = np.linspace(x[m].min(), x[m].max(), 50)
    ax.plot(xs, 10 ** (k * xs + b), color="#555555", lw=1.1, ls="--", zorder=0)


# ----------------------------------------------------------------------------
# figure 8 -- training dynamics from the jsonl epoch logs
# ----------------------------------------------------------------------------
def fig8(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.size": 9, "axes.labelsize": 8.5})

    LEVELS = [1, 5, 20, 80, 180]
    curves = {lv: {"tr": [], "va": []} for lv in LEVELS}   # epochs x runs
    best_epoch = {lv: [] for lv in LEVELS}
    n_used = Counter()
    bad = 0
    for fn in os.listdir(H):
        if not fn.endswith(".jsonl"):
            continue
        rid = fn[:-6]
        m = re.match(r"^(.+)_seed(\d+)_n(\d+)_res(\d+)_(\w+)_lr([0-9.]+)$", rid)
        if not m:
            continue
        lv = int(m.group(3))
        if lv not in curves:
            continue
        try:
            with open(os.path.join(H, fn), encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            bad += 1
            continue
        if not lines:
            bad += 1
            continue
        try:
            last = json.loads(lines[-1])
        except Exception:
            bad += 1
            continue
        if last.get("type") != "final":
            bad += 1
            continue
        tr, va = [], []
        for L in lines:
            try:
                r = json.loads(L)
            except Exception:
                continue
            if r.get("type") == "epoch":
                tr.append(r.get("train_acc"))
                va.append(r.get("val_acc"))
        if len(tr) != 50:
            bad += 1
            continue
        curves[lv]["tr"].append(tr)
        curves[lv]["va"].append(va)
        n_used[lv] += 1
        be = last.get("best_epoch")
        if be is None:
            jf = os.path.join(H, rid + ".json")
            if os.path.exists(jf):
                try:
                    be = json.load(open(jf, encoding="utf-8")).get("best_epoch")
                except Exception:
                    be = None
        if be:
            best_epoch[lv].append(int(be))

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.0))

    ax = axes[0][0]
    for i, lv in enumerate(LEVELS):
        if not curves[lv]["va"]:
            continue
        A = np.array(curves[lv]["va"], float)
        e = np.arange(1, A.shape[1] + 1)
        ax.fill_between(e, np.percentile(A, 25, axis=0), np.percentile(A, 75, axis=0),
                        color=OKABE[i], alpha=0.13, linewidth=0)
        ax.plot(e, A.mean(axis=0), color=OKABE[i], ls=LS[i], lw=1.5,
                label="n=%d (%d runs)" % (lv, A.shape[0]))
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_title("(a) Validation curves converge within ~15 epochs\nat every budget")
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")

    ax = axes[0][1]
    for i, lv in enumerate(LEVELS):
        if not curves[lv]["va"]:
            continue
        T = np.array(curves[lv]["tr"], float)
        V = np.array(curves[lv]["va"], float)
        g = (T - V).mean(axis=0)
        ax.plot(np.arange(1, len(g) + 1), g, color=OKABE[i], ls=LS[i], lw=1.5,
                label="n=%d" % lv)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train - validation accuracy (pp)")
    ax.set_title("(b) Generalisation gap closes as data grows\n(memorisation is a budget effect)")
    ax.legend(frameon=False, fontsize=7.0)

    ax = axes[1][0]
    _b1 = float(np.median(best_epoch[1])) if best_epoch[1] else float("nan")
    _b180 = float(np.median(best_epoch[180])) if best_epoch[180] else float("nan")
    for i, lv in enumerate(LEVELS):
        if not best_epoch[lv]:
            continue
        v = np.array(best_epoch[lv], float)
        ax.scatter(np.full(len(v), float(lv)), v, s=9,
                   color=OKABE[i], alpha=0.35, linewidth=0)
        ax.plot([lv], [np.median(v)], marker=MK[i], color=OKABE[i], ms=8,
                markeredgecolor="black", markeredgewidth=0.6, zorder=3)
        ax.plot([lv * 0.86, lv * 1.16], [np.median(v)] * 2, color=OKABE[i], lw=1.7)
    ax.set_xscale("log")
    ax.set_xticks(LEVELS)
    ax.set_xticklabels(LEVELS)
    ax.set_xlabel("images per class (log)")
    ax.set_ylabel("epoch of best validation accuracy")
    ax.set_title("(c) Best epoch arrives earlier as data grows\n"
                 "(median %.0f -> %.0f epochs as n goes 1 -> 180)" % (_b1, _b180))

    ax = axes[1][1]
    bench = {}
    for d in rows:
        if d.get("dataset") != "neu_cls":
            continue
        bench[(d.get("model"), d.get("seed"), d.get("per_class"), d.get("img_size"))] = d
    dx, dy = [], []
    for fn in os.listdir(H):
        if not fn.endswith(".json"):
            continue
        rid = fn[:-5]
        if "_frac" in rid:
            continue
        try:
            h = json.load(open(os.path.join(H, fn), encoding="utf-8"))
        except Exception:
            continue
        if h.get("type") != "final" or abs(float(h.get("lr") or 0) - 0.0003) > 1e-9:
            continue
        if h.get("mode") != "finetune":
            continue
        k = (h.get("model"), h.get("seed"), h.get("per_class"), h.get("img_size"))
        b = bench.get(k)
        if b is None:
            continue
        dx.append(float(h.get("test_acc")))
        dy.append(float(b.get("test_acc")))
    dx, dy = np.asarray(dx), np.asarray(dy)
    if len(dx):
        ax.scatter(dx, dy, s=9, color=OKABE[0], alpha=0.35, linewidth=0)
        lim = [min(dx.min(), dy.min()) - 3, 101]
        ax.plot(lim, lim, color="#999999", ls="--", lw=1.0, label="identity")
        md = float(np.abs(dx - dy).mean())
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("test accuracy, pre-dedup sweep (%)")
        ax.set_ylabel("test accuracy, frozen benchmark (%)")
        ax.set_title("(d) The pre-dedup sweep is still usable\nn=%d matched runs, mean |diff| = %.2f pp"
                     % (len(dx), md))
        ax.legend(frameon=False, fontsize=7.0, loc="upper left")
    else:
        md = None
        ax.text(0.5, 0.5, "no matched runs", ha="center", transform=ax.transAxes)
    fig.suptitle("Training dynamics from 3730 complete epoch logs (NEU-CLS)", y=0.995,
                 fontsize=9.8)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "fig8_training_dynamics")

    return {"runs_used": {str(k): int(v) for k, v in n_used.items()},
            "skipped_incomplete": int(bad),
            "best_epoch_median": {str(lv): (float(np.median(best_epoch[lv]))
                                            if best_epoch[lv] else None) for lv in LEVELS},
            "matched_runs": int(len(dx)),
            "matched_mean_abs_diff_pp": (md if len(dx) else None),
            "gap_at_n1_last": (float((np.array(curves[1]["tr"], float) -
                                      np.array(curves[1]["va"], float)).mean(axis=0)[-1])
                               if curves[1]["va"] else None)}


# ----------------------------------------------------------------------------
# figure 9 -- variance decomposition / measurement protocol
# ----------------------------------------------------------------------------
def fig9(man, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({"font.size": 9, "axes.labelsize": 8.5})

    def anova(ds, lv):
        sel = [d for d in rows if d.get("dataset") == ds and d.get("per_class") == lv]
        if not sel:
            return None
        cls = man[ds]["classes"]
        rec = []
        for d in sel:
            for c, v in (d.get("per_class_acc") or {}).items():
                if v is None or v != v:
                    continue
                rec.append((c, d["model"], d["seed"], float(v)))
        if len(rec) < 12:
            return None
        Y = np.array([r[3] for r in rec])
        gm = Y.mean()
        ss_tot = ((Y - gm) ** 2).sum()
        if ss_tot <= 0:
            return None

        def ss_group(idx):
            s = 0.0
            groups = defaultdict(list)
            for r in rec:
                groups[r[idx]].append(r[3])
            for g, v in groups.items():
                mv = mean(v)
                s += len(v) * (mv - gm) ** 2
            return s

        ss_cls = ss_group(0)
        # model and seed effects computed within class
        groups = defaultdict(list)
        for r in rec:
            groups[(r[0], r[1])].append(r[3])
        ss_mod_within = 0.0
        for (c, m), v in groups.items():
            base = mean([r[3] for r in rec if r[0] == c])
            ss_mod_within += len(v) * (mean(v) - base) ** 2
        groups2 = defaultdict(list)
        for r in rec:
            groups2[(r[0], r[1], r[2])].append(r[3])
        ss_res = 0.0
        for k2, v in groups2.items():
            if len(v) > 1:
                mv = mean(v)
                ss_res += sum((x - mv) ** 2 for x in v)
        ss_seed = max(0.0, ss_tot - ss_cls - ss_mod_within - ss_res)
        return np.array([ss_cls, ss_mod_within, ss_seed, ss_res]) / ss_tot

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2),
                             gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})

    # (a) stacked variance shares
    ax = axes[0]
    labels, data = [], []
    for ds in DATASETS:
        lvs = sorted({d["per_class"] for d in rows if d.get("dataset") == ds})
        pick = [lvs[0], lvs[len(lvs) // 2], lvs[-1]]
        for lv in pick:
            r = anova(ds, lv)
            if r is None:
                continue
            labels.append("%s\nn=%d" % (ds.replace("_", " "), lv))
            data.append(r)
    data = np.array(data)
    cols = [OKABE[0], OKABE[1], OKABE[2], "#DDDDDD"]
    names = ["between class", "between model\n(within class)", "between seed", "residual"]
    bot = np.zeros(len(data))
    for j in range(4):
        ax.bar(range(len(data)), data[:, j] * 100, bottom=bot * 100, color=cols[j],
               edgecolor="white", linewidth=0.5, label=names[j], width=0.72)
        bot += data[:, j]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.8)
    ax.set_ylabel("share of total variance (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Variance shares at the middle budget\n"
                 "no single component dominates every dataset")
    _h, _l = ax.get_legend_handles_labels()

    # (b) seeds needed
    ax = axes[1]
    diag = {}
    for i, ds in enumerate(DATASETS):
        lvs = sorted({d["per_class"] for d in rows if d.get("dataset") == ds})
        xs, ys = [], []
        for lv in lvs:
            per_seed = defaultdict(list)
            for d in rows:
                if d.get("dataset") == ds and d.get("per_class") == lv:
                    per_seed[d["seed"]].append(float(d.get("test_acc")))
            vals = [mean(v) for v in per_seed.values()]
            vals = [v for v in vals if v is not None]
            if len(vals) < 3:
                continue
            sd = float(np.std(vals, ddof=1))
            xs.append(lv)
            ys.append((1.96 * sd / 0.5) ** 2)
        ax.plot(xs, ys, marker=MK[i], ls=LS[i], color=OKABE[i], ms=4.2, lw=1.3,
                label=ds)
        if xs:
            cross = next((float(a) for a, b in zip(xs, ys) if b <= 5.0), None)
            diag[ds] = {"seeds_for_0.5pp_at_min": float(ys[0]),
                        "seeds_for_0.5pp_at_max": float(ys[-1]),
                        "first_level_where_5_seeds_suffice": cross,
                        "curve": {str(a): float(b) for a, b in zip(xs, ys)}}
    ax.axhline(5, color="#999999", ls=":", lw=1.2)
    ax.text(1.05, 5.6, "5 seeds (used here)", fontsize=6.6, color="#666666")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("images per class (log)")
    ax.set_ylabel("seeds for a +/-0.5 pp CI (log)")
    ax.set_title("(b) Required seed count for a +/-0.5 pp CI\n"
                 "it falls steeply as the budget grows")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")

    # (c) model-rank stability
    ax = axes[2]
    taus = {}
    for i, ds in enumerate(DATASETS):
        lvs = sorted({d["per_class"] for d in rows if d.get("dataset") == ds})
        top = lvs[-1]
        ref = {}
        for d in rows:
            if d.get("dataset") == ds and d.get("per_class") == top:
                ref.setdefault(d["model"], []).append(float(d.get("test_acc")))
        ref = {m: mean(v) for m, v in ref.items()}
        xs, ys = [], []
        for lv in lvs:
            if lv == top:
                # the anchor agrees with itself by construction (tau = 1); plotting it
                # would fake a right-edge recovery of ranking stability
                continue
            cur = {}
            for d in rows:
                if d.get("dataset") == ds and d.get("per_class") == lv:
                    cur.setdefault(d["model"], []).append(float(d.get("test_acc")))
            cur = {m: mean(v) for m, v in cur.items()}
            common = sorted(set(cur) & set(ref))
            if len(common) < 5:
                continue
            t = kendall([cur[m] for m in common], [ref[m] for m in common])
            if t is None:
                continue
            xs.append(lv)
            ys.append(t)
        if xs:
            ax.plot(xs, ys, marker=MK[i], ls=LS[i], color=OKABE[i], ms=4.2, lw=1.3,
                    label=ds)
            taus[ds] = {"min": float(min(ys)), "max": float(max(ys))}
    ax.axhline(0.7, color="#B00020", ls="--", lw=1.2)
    ax.text(1.05, 0.99, "0.7 = usable rank agreement", fontsize=6.6, color="#B00020",
            va="top")
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("images per class (log)")
    ax.set_ylabel("Kendall tau vs ranking at the largest budget")
    ax.set_title("(c) Model ranking never stabilises below the anchor\n"
                 "(anchor level excluded, so rankings must not be reported)")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    fig.suptitle("Measurement protocol: where the noise lives and what it costs", y=1.0,
                 fontsize=9.8)
    fig.tight_layout()
    fig.legend(_h, _l, loc="lower center", ncol=4, frameon=False, fontsize=7.0,
               bbox_to_anchor=(0.5, -0.05))
    save(fig, "fig9_variance_decomposition")

    shares = {}
    for ds in DATASETS:
        lvs = sorted({d["per_class"] for d in rows if d.get("dataset") == ds})
        r = anova(ds, lvs[len(lvs) // 2])
        if r is not None:
            shares[ds] = {"class": float(r[0]), "model": float(r[1]),
                          "seed": float(r[2]), "resid": float(r[3])}
    return {"variance_share_mid_level": shares, "seeds_for_0.5pp": diag,
            "kendall_tau_range": taus}


# ----------------------------------------------------------------------------
# figure 10 -- budget allocation gain
# ----------------------------------------------------------------------------
def fig10(man):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import NullLocator
    rcParams.update({"font.size": 9, "axes.labelsize": 8.5})

    s = json.load(open(os.path.join(AB, "bench_summary.json"), encoding="utf-8"))
    cap_budget = {}
    for ds in DATASETS:
        b = s[ds]["budget"]
        cap_budget[ds] = max(x["budget"] for x in b)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    ax = axes[0]
    for i, ds in enumerate(DATASETS):
        b = s[ds]["budget"]
        ax.plot([x["budget"] for x in b], [x["gain"] for x in b], marker=MK[i],
                ls=LS[i], color=OKABE[i], ms=5.5, lw=1.5,
                label="%s (max %d)" % (ds, cap_budget[ds]))
    ax.axhline(0, c="k", lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([20, 50, 100, 200, 600])
    ax.set_xticklabels(["20", "50", "100", "200", "600"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("total annotation budget (images, log)")
    ax.set_ylabel("gain over uniform allocation (pp)")
    ax.set_title("(a) Absolute budgets confound the comparison:\nthe saturated set gains least")
    ax.legend(frameon=False, fontsize=6.8)

    ax = axes[1]
    buckets = defaultdict(list)
    for i, ds in enumerate(DATASETS):
        for x in s[ds]["budget"]:
            f = 100.0 * x["budget"] / cap_budget[ds]
            ax.scatter([f], [x["gain"]], s=52, marker=MK[i], color=OKABE[i],
                       edgecolor="black", linewidth=0.5, zorder=3,
                       label=ds if f == min(100.0 * y["budget"] / cap_budget[ds]
                                            for y in s[ds]["budget"]) else None)
            buckets[round(f)].append((ds, x["gain"]))
    for f, v in sorted(buckets.items()):
        if len(v) >= 2:
            ax.plot([f, f], [min(g for _, g in v), max(g for _, g in v)],
                    color="#999999", lw=0.9, zorder=1)
    ax.axhline(0, c="k", lw=0.7)
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xlabel("budget as % of that dataset's maximum")
    ax.set_ylabel("gain over uniform allocation (pp)")
    ax.set_title("(b) At equal relative budget the imbalanced sets win\n"
                 "at 50 % and 100 %, but not at 10-20 %")
    ax.legend(frameon=False, fontsize=6.8, loc="center left")
    save(fig, "fig10_budget_gain")

    abs_gain = {ds: {"max_budget": cap_budget[ds],
                     "gain_at_max": s[ds]["budget"][-1]["gain"]} for ds in DATASETS}
    rel = {}
    for f, v in sorted(buckets.items()):
        for ds, g in v:
            rel.setdefault(str(f), {})[ds] = float(g)
    return {"absolute": abs_gain, "relative_gain": rel}


# ----------------------------------------------------------------------------
def save(fig, name):
    import matplotlib.pyplot as plt
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, "%s.%s" % (name, ext))
        if ext == "png":
            fig.savefig(p, dpi=600, pil_kwargs={"optimize": True})
        else:
            fig.savefig(p)
    plt.close(fig)
    print("  wrote %s.pdf / .png" % name)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({
        "font.size": 9, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "legend.fontsize": 7.2, "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "figure.dpi": 110,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })

    man = json.load(open(os.path.join(BENCH, "manifest.json"), encoding="utf-8"))
    audit = json.load(open(AUD, encoding="utf-8"))
    print("loading benchmark records ...")
    rows = load_bench()
    print("  %d final records" % len(rows))

    acc = {ds: defaultdict(lambda: defaultdict(list)) for ds in DATASETS}
    for d in rows:
        ds = d.get("dataset")
        if ds not in acc:
            continue
        n = d.get("per_class")
        for c, v in (d.get("per_class_acc") or {}).items():
            if v is not None and v == v:
                acc[ds][c][n].append(float(v))

    jobs = [
        ("fig1_framework", lambda: fig1(man)),
        ("fig2_data_audit", lambda: fig2(man, audit)),
        ("fig3_perclass_curves", lambda: fig3(man, rows)),
        ("fig4_difficulty_map", lambda: fig4(man, acc)),
        ("fig5_confusion_heatmaps", lambda: fig5(man, rows)),
        ("fig6_error_graph", lambda: fig6(man, rows)),
        ("fig7_outflow_vs_need", lambda: fig7(man, rows)),
        ("fig8_training_dynamics", lambda: fig8(rows)),
        ("fig9_variance_decomposition", lambda: fig9(man, rows)),
        ("fig10_budget_gain", lambda: fig10(man)),
    ]
    ok = 0
    for name, fn in jobs:
        print("[%s]" % name)
        try:
            d = fn()
            DIAG[name] = d if d is not None else {}
            results.append(("figure", name, True, ""))
            ok += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            DIAG[name] = {"error": "%s: %s" % (type(e).__name__, e)}
            results.append(("figure", name, False, "%s: %s" % (type(e).__name__, e)))

    DIAG["_summary"] = {"figures_ok": ok, "figures_total": len(jobs),
                        "outcomes": [{"name": n, "ok": o, "msg": m} for _, n, o, m in results]}
    DIAG["_provenance"] = {
        "benchmark_results": R,
        "n_benchmark_records": len(rows),
        "historical_results": H,
        "manifest": os.path.join(BENCH, "manifest.json"),
        "audit": AUD,
        "summary": os.path.join(AB, "bench_summary.json"),
        "common_level_control": os.path.join(AB, "common_level_control.json"),
        "level_caps": caps(),
        "note": "figures are regenerated from these records alone; no hand-entered values",
    }
    with open(os.path.join(AB, "figs_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(DIAG, f, indent=1, ensure_ascii=True)
    print("\n%d/%d figures OK" % (ok, len(jobs)))
    print("diagnostics -> %s" % os.path.join(AB, "figs_diagnostics.json"))
    for _, n, o, m in results:
        print("  %-28s %s %s" % (n, "OK" if o else "FAIL", m))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
