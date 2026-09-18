"""Paper A third-stage analysis: H3 retest (low-data regime) + H1 per-model + publication figures.

Fixing a design flaw of my own: H3 previously used the **full-data** confusion matrix to
measure error flow, but full-data accuracy is already 99.8%, so almost no errors flow -> the metric degenerates.
We now use the confusion matrix of the **low-data regime (where errors are abundant)**.

Publication figures: vector PDF + 600 dpi PNG, font size >=8 pt, Okabe-Ito colorblind-safe
palette, line style + marker double encoding (distinguishable in grayscale).
"""

import json, os, math
from collections import defaultdict

WORK = "/var/hpc-root/xiangxu/work"
RESULTS = os.path.join(WORK, "results")
OUT = os.path.join(WORK, "analysis")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
SHORT = ["crz", "inc", "pat", "pit", "rol", "scr"]
CN = {"crazing": "crazing", "inclusion": "inclusion", "patches": "patches",
      "pitted_surface": "pitted surf.", "rolled_in_scale": "rolled-in scale", "scratches": "scratches"}
os.makedirs(OUT, exist_ok=True)

rows = []
for fn in sorted(os.listdir(RESULTS)):
    if fn.endswith(".json"):
        try:
            with open(os.path.join(RESULTS, fn), encoding="utf-8") as f:
                d = json.load(f)
            if d.get("type") == "final":
                rows.append(d)
        except Exception:
            pass
print("loaded %d results\n" % len(rows))

MAIN = [d for d in rows if d.get("img_size") == 224 and d.get("mode") == "finetune"
        and abs((d.get("lr") or 0) - 3e-4) < 1e-9 and d.get("epochs") == 50
        and not d["run_id"].endswith("_a4")]
acc = lambda d, c: (d.get("per_class_acc") or {}).get(c)
levels = sorted(set(d["per_class"] for d in MAIN))
models = sorted(set(d["model"] for d in MAIN))
print("main cohort %d runs | models %d | levels %d..%d\n" % (len(MAIN), len(models), min(levels), max(levels)))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def n_at(m, c, thr):
    for n in levels:
        v = mean([acc(d, c) for d in MAIN if d["model"] == m and d["per_class"] == n])
        if v is not None and v >= thr:
            return n
    return None


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for p, i in enumerate(o): r[i] = p + 1
        return r
    rx, ry = rk(x), rk(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if dx * dy else float("nan")


# ==================================================== H1 per-model version
print("=" * 88)
print("H1 per-model version: annotations per class needed to reach 95% (n@95%)")
print("=" * 88)
print("  %-22s %s   | range" % ("model", "  ".join("%-6s" % s for s in SHORT)))
N95 = {}
for m in models:
    vs = [n_at(m, c, 95) for c in CLASSES]
    N95[m] = vs
    shown = ["%-6s" % (v if v is not None else ">180") for v in vs]
    good = [v for v in vs if v is not None]
    print("  %-22s %s   | %s" % (m, "  ".join(shown), (max(good) - min(good)) if good else "-"))

print("\n  Per-class distribution across models (n@95%)")
print("  %-18s %-8s %-8s %-8s %s" % ("class", "min", "median", "max", "spread_x"))
for i, c in enumerate(CLASSES):
    vs = sorted(v for v in (N95[m][i] for m in models) if v is not None)
    if vs:
        print("  %-18s %-8d %-8d %-8d %.1fx" % (c, vs[0], vs[len(vs) // 2], vs[-1], vs[-1] / max(1, vs[0])))

# ==================================================== H3 retest (low-data regime)
print("\n" + "=" * 88)
print("H3 retest: predicting data efficiency from the **low-data** confusion matrix (where errors are abundant)")
print("=" * 88)
BAND = [2, 5, 10]           # low-data levels: where errors concentrate
band_acc = {}
for m in models:
    for n in BAND:
        v = mean([d["test_acc"] for d in MAIN if d["model"] == m and d["per_class"] == n])
        band_acc[(m, n)] = v
print("  mean accuracy on the low-data band n=%s: %.1f%% .. %.1f%% (confirming that errors are indeed abundant here)"
      % (BAND, min(v for v in band_acc.values() if v is not None),
         max(v for v in band_acc.values() if v is not None)))

out_low, in_low = {}, {}
for m in models:
    for i, c in enumerate(CLASSES):
        os_, ins_ = [], []
        for n in BAND:
            for d in MAIN:
                if d["model"] != m or d["per_class"] != n:
                    continue
                cm = d.get("confusion")
                if not cm:
                    continue
                rowsum = sum(cm[i]) or 1
                totsum = sum(sum(r) for r in cm) or 1
                os_.append((rowsum - cm[i][i]) / rowsum)
                ins_.append(sum(cm[j][i] for j in range(6) if j != i) / totsum)
        if os_:
            out_low[(m, c)] = sum(os_) / len(os_)
            in_low[(m, c)] = sum(ins_) / len(ins_)

gain = {}
for m in models:
    for c in CLASSES:
        lo = mean([acc(d, c) for d in MAIN if d["model"] == m and d["per_class"] == min(levels)])
        hi = mean([acc(d, c) for d in MAIN if d["model"] == m and d["per_class"] == max(levels)])
        if lo is not None and hi is not None:
            gain[(m, c)] = hi - lo

keys = [k for k in out_low if k in gain and N95.get(k[0]) and N95[k[0]][CLASSES.index(k[1])] is not None]
if keys:
    x = [out_low[k] * 100 for k in keys]
    y = [gain[k] for k in keys]
    z = [N95[k[0]][CLASSES.index(k[1])] for k in keys]
    print("\n  Sample: %d (model, class) points" % len(keys))
    print("  [H3-a] low-data outflow error rate vs data gain      Spearman rho = %+.3f   (expected rho < 0)" % spearman(x, y))
    print("  [H3-b] low-data outflow error rate vs n@95%%         Spearman rho = %+.3f   (expected rho > 0)" % spearman(x, z))
    print("  [H3-c] low-data inflow error rate vs data gain      Spearman rho = %+.3f"
          % spearman([in_low[k] * 100 for k in keys], y))
    print("\n  Summary by class (averaged over 11 models)")
    print("    %-18s %-12s %-12s %-10s %s" % ("class", "low_outflow_%", "low_inflow_%", "gain_pp", "n@95%_median"))
    for c in CLASSES:
        o = mean([out_low[k] for k in out_low if k[1] == c])
        i2 = mean([in_low[k] for k in in_low if k[1] == c])
        g = mean([gain[k] for k in gain if k[1] == c])
        nn = sorted(v for v in (N95[m][CLASSES.index(c)] for m in models) if v is not None)
        print("    %-18s %-12s %-12s %-10s %s"
              % (c, "%.2f" % (o * 100) if o else "-", "%.2f" % (i2 * 100) if i2 else "-",
                 "%+.1f" % g if g else "-", nn[len(nn) // 2] if nn else "-"))
else:
    print("  not enough samples")

# ==================================================== publication figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # Okabe-Ito colorblind-safe palette
    OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    MK = ["o", "s", "^", "D", "v", "P"]
    LS = ["-", "--", "-.", ":", "-", "--"]
    rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })
    VEC = {}
    for m in models:
        for n in levels:
            v = [mean([acc(d, c) for d in MAIN if d["model"] == m and d["per_class"] == n]) for c in CLASSES]
            if all(x is not None for x in v):
                VEC[(m, n)] = v

    # ---- Figure 1 (main): per-class data efficiency with cross-model spread band
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for i, c in enumerate(CLASSES):
        xs, ys, los, his = [], [], [], []
        for n in levels:
            v = [VEC[(m, n)][i] for m in models if (m, n) in VEC]
            if v:
                xs.append(n); ys.append(sum(v) / len(v)); los.append(min(v)); his.append(max(v))
        ax[0].plot(xs, ys, color=OKABE[i], ls=LS[i], marker=MK[i], ms=2.6, lw=1.3, label=CN[c])
        ax[0].fill_between(xs, los, his, color=OKABE[i], alpha=0.13, lw=0)
    ax[0].set_xscale("log"); ax[0].axhline(95, ls=":", c="gray", lw=1)
    ax[0].set_xlabel("Training images per class"); ax[0].set_ylabel("Test accuracy (%)")
    ax[0].set_title("(a) Per-class data efficiency\n(mean line, shaded = min..max over 11 models)")
    ax[0].legend(loc="lower right", ncol=2, frameon=False, handlelength=2.2)
    ax[0].set_ylim(40, 101)

    # ---- Figure 1(b): per-model n@95% heatmap
    import numpy as np
    M = np.array([[N95[m][i] if N95[m][i] is not None else 181 for i in range(6)] for m in models], float)
    im = ax[1].imshow(M, cmap="viridis", aspect="auto")
    ax[1].set_xticks(range(6)); ax[1].set_xticklabels(SHORT)
    ax[1].set_yticks(range(len(models))); ax[1].set_yticklabels(models)
    for r in range(len(models)):
        for cc in range(6):
            ax[1].text(cc, r, "%d" % M[r, cc] if M[r, cc] <= 180 else ">", ha="center", va="center",
                       fontsize=6.5, color="w" if M[r, cc] < 110 else "k")
    ax[1].set_title("(b) Images per class needed to reach 95%")
    ax[1].grid(False); plt.colorbar(im, ax=ax[1], fraction=.046, label="images/class")
    for ext in ("pdf", "png"):
        plt.figure(fig.number).savefig(os.path.join(OUT, "fig1_data_efficiency.%s" % ext),
                                       dpi=600 if ext == "png" else None)
    plt.close(); print("\nwrote fig1_data_efficiency.pdf / .png")

    # ---- Figure 2: H3 retest scatter
    if keys:
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
        for i, c in enumerate(CLASSES):
            ks = [k for k in keys if k[1] == c]
            ax[0].scatter([out_low[k] * 100 for k in ks], [gain[k] for k in ks],
                          color=OKABE[i], marker=MK[i], s=26, alpha=.9, label=CN[c], lw=.4, ec="k")
        ax[0].set_xlabel("Confusion outflow at low data (%)")
        ax[0].set_ylabel("Accuracy gain, 1 -> 180 images")
        ax[0].set_title("(a) H3: confusability vs data gain\nSpearman rho = %+.2f (n=%d)" % (spearman(x, y), len(keys)))
        ax[0].legend(frameon=False, ncol=2, fontsize=7)
        ax[1].scatter([in_low[k] * 100 for k in keys], [gain[k] for k in keys], color="#555", s=22, alpha=.8)
        ax[1].set_xlabel("Confusion inflow at low data (%)")
        ax[1].set_ylabel("Accuracy gain, 1 -> 180 images")
        ax[1].set_title("(b) Error-sink effect\nSpearman rho = %+.2f" % spearman([in_low[k] * 100 for k in keys], y))
        for ext in ("pdf", "png"):
            plt.savefig(os.path.join(OUT, "fig2_h3_lowdata.%s" % ext), dpi=600 if ext == "png" else None)
        plt.close(); print("wrote fig2_h3_lowdata.pdf / .png")
except Exception as e:
    import traceback; traceback.print_exc()

print("\nDONE")
