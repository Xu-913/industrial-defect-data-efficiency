"""Paper A second-stage analysis: H2 / H3 / W4 / W5 plus publication-quality figures.

Read-only on results/*.json; no training result is modified.
"""

import json, os, math
from collections import defaultdict

WORK = "/var/hpc-root/xiangxu/work"
RESULTS = os.path.join(WORK, "results")
OUT = os.path.join(WORK, "analysis")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
SHORT = ["crz", "inc", "pat", "pit", "rol", "scr"]
os.makedirs(OUT, exist_ok=True)

rows = []
for fn in sorted(os.listdir(RESULTS)):
    if not fn.endswith(".json"):
        continue
    try:
        with open(os.path.join(RESULTS, fn), encoding="utf-8") as f:
            d = json.load(f)
        if d.get("type") != "final":
            continue
        rows.append(d)
    except Exception:
        pass
print("loaded %d results\n" % len(rows))


def cohort(**kw):
    def ok(d):
        for k, v in kw.items():
            if k == "a4":
                if d["run_id"].endswith("_a4") != v:
                    return False
            elif k == "nota4":
                if d["run_id"].endswith("_a4"):
                    return False
            elif k == "epochs":
                if d.get("epochs") != v:
                    return False
            elif k == "lr_in":
                if round(d.get("lr") or 0, 6) not in [round(x, 6) for x in v]:
                    return False
            elif d.get(k) != v:
                return False
        return True
    return [d for d in rows if ok(d)]


MAIN = cohort(img_size=224, mode="finetune", lr_in=[3e-4], epochs=50, nota4=True)
FROZ = cohort(img_size=224, mode="frozen")
LRS = cohort(img_size=224, mode="finetune", epochs=20, nota4=True)
print("main cohort %d | frozen %d | LR sensitivity %d\n" % (len(MAIN), len(FROZ), len(LRS)))

acc = lambda d, c: (d.get("per_class_acc") or {}).get(c)


def kendall(a, b):
    conc = disc = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0: conc += 1
            elif s < 0: disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else float("nan")


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


# Per-class vector for each (model, n), averaged over seeds
vec = defaultdict(lambda: defaultdict(list))
for d in MAIN:
    for c in CLASSES:
        v = acc(d, c)
        if v is not None:
            vec[(d["model"], d["per_class"])][c].append(v)
VEC = {k: [sum(v[c]) / len(v[c]) for c in CLASSES] for k, v in vec.items() if all(v.get(c) for c in CLASSES)}
levels = sorted(set(n for _, n in VEC))
models = sorted(set(m for m, _ in VEC))

# ============================================================ H2 rank stability
print("=" * 74)
print("H2: is the class-level difficulty ranking stable across models (Kendall tau, closer to 1 = more stable)")
print("=" * 74)
print("  %-6s %-8s %-9s %s" % ("n/cls", "pairs", "tau_mean", "tau_min"))
for n in levels:
    ms = [m for m in models if (m, n) in VEC]
    if len(ms) < 3:
        continue
    taus = [kendall(VEC[(ms[i], n)], VEC[(ms[j], n)]) for i in range(len(ms)) for j in range(i + 1, len(ms))]
    taus = [t for t in taus if t == t]
    if taus:
        print("  %-6d %-8d %-9.3f %.3f" % (n, len(taus), sum(taus) / len(taus), min(taus)))

print("\n  Class difficulty ranking (1 = hardest, sorted by n@95%, main cohort averaged over all models)")
aggc = defaultdict(list)
for d in MAIN:
    for c in CLASSES:
        v = acc(d, c)
        if v is not None:
            aggc[(d["per_class"], c)].append(v)
need95 = {}
for c in CLASSES:
    for n in levels:
        v = aggc.get((n, c), [])
        if v and sum(v) / len(v) >= 95:
            need95[c] = n; break
    need95.setdefault(c, None)
for c in sorted(CLASSES, key=lambda x: (need95[x] is None, need95[x] or 999)):
    print("    %-18s n@95%% = %s" % (c, need95[c]))

# ============================================================ H3 confusion structure
print("\n" + "=" * 74)
print("H3: can confusion structure predict data efficiency (at the model x class level, n=%d points)" % (len(models) * len(CLASSES)))
print("=" * 74)
# Use the full-data confusion matrix to compute the "outflow error rate" (share of this class misassigned to others)
outflow, inflow, gain, n95mc = {}, {}, {}, {}
for m in models:
    d180 = [d for d in MAIN if d["model"] == m and d["per_class"] == max(levels)]
    if not d180:
        continue
    cm = d180[0].get("confusion")
    if not cm:
        continue
    tot = sum(sum(r) for r in cm) or 1
    for i, c in enumerate(CLASSES):
        out = sum(cm[i][j] for j in range(len(CLASSES)) if j != i) / (sum(cm[i]) or 1)
        inf = sum(cm[j][i] for j in range(len(CLASSES)) if j != i) / tot
        outflow[(m, c)] = out
        inflow[(m, c)] = inf
    for c in CLASSES:
        lo = [acc(d, c) for d in MAIN if d["model"] == m and d["per_class"] == min(levels)]
        hi = [acc(d, c) for d in MAIN if d["model"] == m and d["per_class"] == max(levels)]
        if lo and hi:
            gain[(m, c)] = (sum(hi) / len(hi)) - (sum(lo) / len(lo))
        seq = []
        for n in levels:
            v = [acc(d, c) for d in MAIN if d["model"] == m and d["per_class"] == n]
            seq.append((n, sum(v) / len(v)) if v else (n, None))
        hit = None
        for n, a in seq:
            if a is not None and a >= 95:
                hit = n; break
        n95mc[(m, c)] = hit

keys = [k for k in outflow if k in gain and n95mc.get(k) is not None]
if keys:
    x = [outflow[k] for k in keys]
    y = [gain[k] for k in keys]
    print("  [H3-a] outflow error rate vs data gain   n=%d points  Spearman rho = %+.3f" % (len(keys), spearman(x, y)))
    print("         Prediction: the more confusable a class, the more limited its gain -> expect rho < 0")
    z = [n95mc[k] for k in keys]
    print("  [H3-b] outflow error rate vs n@95%%        Spearman rho = %+.3f" % spearman(x, z))
    print("         Prediction: the more confusable a class, the more annotations it needs -> expect rho > 0")
    y2 = [inflow[k] for k in keys]
    print("  [H3-c] inflow error rate (error sink) vs data gain  rho = %+.3f" % spearman(y2, y))

print("\n  Summary by class (averaged over 11 models)")
print("    %-18s %-12s %-12s %s" % ("class", "outflow_rate", "gain_pp", "n@95%_median"))
for c in CLASSES:
    o = [outflow[k] for k in outflow if k[1] == c]
    g = [gain[k] for k in gain if k[1] == c]
    nn = sorted(n95mc[k] for k in n95mc if k[1] == c and n95mc[k] is not None)
    print("    %-18s %-12s %-12s %s" % (
        c, "%.3f" % (sum(o) / len(o)) if o else "-",
        "%+.1f" % (sum(g) / len(g)) if g else "-",
        nn[len(nn) // 2] if nn else "-"))

# ============================================================ W4 frozen-backbone control
print("\n" + "=" * 74)
print("W4: frozen backbone (linear probe) vs fine-tune -- is the class-level difficulty ranking consistent")
print("=" * 74)
FZ = defaultdict(lambda: defaultdict(list))
for d in FROZ:
    for c in CLASSES:
        v = acc(d, c)
        if v is not None:
            FZ[(d["model"], d["per_class"])][c].append(v)
FV = {k: [sum(v[c]) / len(v[c]) for c in CLASSES] for k, v in FZ.items() if all(v.get(c) for c in CLASSES)}
common = sorted(set(FV) & set(VEC))
if common:
    taus = [kendall(VEC[k], FV[k]) for k in common]
    taus = [t for t in taus if t == t]
    overall = []
    for i in range(len(CLASSES)):
        a = [VEC[k][i] for k in common]; b = [FV[k][i] for k in common]
        overall.append(spearman(a, b))
    print("  comparable configs: %d" % len(common))
    print("  rank agreement Kendall tau mean = %.3f" % (sum(taus) / len(taus)))
    print("  per-class accuracy correlation Spearman: " + "  ".join("%s=%+.2f" % (SHORT[i], overall[i]) for i in range(6)))
    for c in CLASSES:
        f = [FV[k][CLASSES.index(c)] for k in common]
        t = [VEC[k][CLASSES.index(c)] for k in common]
        print("    %-18s frozen %.1f  fine-tune %.1f  delta=%+.1f pp" % (c, sum(f) / len(f), sum(t) / len(t), sum(f) / len(f) - sum(t) / len(t)))

# ============================================================ W5 learning-rate sensitivity
print("\n" + "=" * 74)
print("W5: learning-rate sensitivity -- is the conclusion just an artifact of hyperparameter tuning")
print("=" * 74)
LRV = defaultdict(dict)
for d in LRS:
    LRV[(d["model"], d["lr"], d["per_class"])] = d["test_acc"]
lrs = sorted(set(k[1] for k in LRV))
print("  %-22s %s" % ("model", "  ".join("lr=%-8.0e" % l for l in lrs)))
best = {}
for m in models:
    cells = []
    for l in lrs:
        v = [LRV[k] for k in LRV if k[0] == m and k[1] == l]
        cells.append(sum(v) / len(v) if v else float("nan"))
    if any(c == c for c in cells):
        best[m] = lrs[max(range(len(lrs)), key=lambda i: (cells[i] if cells[i] == cells[i] else -1))]
    print("  %-22s %s   best=%s" % (m, "  ".join("%-11.2f" % c if c == c else "%-11s" % "-" for c in cells),
                                    "%.0e" % best[m] if m in best else "-"))
print("\n  Best-LR distribution:", {("%.0e" % k): sum(1 for v in best.values() if v == k) for k in lrs})
print("  -> if the best LR differs across models, one uniform LR is unfair to some models and the sensitivity must be reported in the paper")

# ============================================================ plotting
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # ---- Figure A: H2 rank stability + H3 scatter
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    tt = []
    for n in levels:
        ms = [m for m in models if (m, n) in VEC]
        if len(ms) < 3: tt.append((n, float("nan"))); continue
        ts = [kendall(VEC[(ms[i], n)], VEC[(ms[j], n)]) for i in range(len(ms)) for j in range(i + 1, len(ms))]
        ts = [t for t in ts if t == t]
        tt.append((n, sum(ts) / len(ts) if ts else float("nan")))
    ax[0].plot([a for a, _ in tt], [b for _, b in tt], "o-", color="tab:blue")
    ax[0].axhline(0.7, ls="--", c="red", lw=1)
    ax[0].set_xscale("log"); ax[0].set_ylim(-0.2, 1.05)
    ax[0].set_xlabel("Training images per class"); ax[0].set_ylabel("Kendall tau (mean over model pairs)")
    ax[0].set_title("H2: rank stability of class difficulty\n(red dashed = 0.7 threshold)")
    ax[0].grid(alpha=.3)

    if keys:
        cmap = {c: i for i, c in enumerate(CLASSES)}
        for c in CLASSES:
            ks = [k for k in keys if k[1] == c]
            ax[1].scatter([outflow[k] * 100 for k in ks], [gain[k] for k in ks],
                          label=c, s=42, alpha=.85)
        ax[1].set_xlabel("Confusion outflow at full data (%)")
        ax[1].set_ylabel("Accuracy gain, n=1 -> 180 (pp)")
        ax[1].set_title("H3: confusability vs data efficiency\n(Spearman rho = %+.2f, %d points)" % (spearman([outflow[k] for k in keys], [gain[k] for k in keys]), len(keys)))
        ax[1].grid(alpha=.3); ax[1].legend(fontsize=6, ncol=2)

    ax[2].bar(range(6), [sum(aggc.get((max(levels), c), [0])) / max(1, len(aggc.get((max(levels), c), [0]))) for c in CLASSES],
              color="tab:green", alpha=.8)
    ax[2].set_xticks(range(6)); ax[2].set_xticklabels(SHORT); ax[2].set_ylim(90, 101)
    ax[2].set_title("Per-class accuracy at full data\n(ceiling effect check)"); ax[2].grid(alpha=.3, axis="y")
    plt.tight_layout(); p = os.path.join(OUT, "fig_h2h3.png"); plt.savefig(p, dpi=110); print("\nwrote %s" % p)

    # ---- Figure B: W4 frozen vs fine-tune + W5 learning rate
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    if common:
        for i, c in enumerate(CLASSES):
            ax[0].scatter([VEC[k][i] for k in common], [FV[k][i] for k in common], label=SHORT[i], s=26, alpha=.8)
        ax[0].plot([0, 100], [0, 100], "k--", lw=1); ax[0].set_xlim(20, 101); ax[0].set_ylim(20, 101)
        ax[0].set_xlabel("Fine-tune per-class acc (%)"); ax[0].set_ylabel("Frozen backbone per-class acc (%)")
        ax[0].set_title("W4: frozen (linear probe) vs fine-tune"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7, ncol=2)
    for m in models:
        ys = []
        for l in lrs:
            v = [LRV[k] for k in LRV if k[0] == m and k[1] == l]
            ys.append(sum(v) / len(v) if v else float("nan"))
        ax[1].plot(lrs, ys, "o-", label=m, alpha=.8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("Learning rate"); ax[1].set_ylabel("Mean test accuracy (%)")
    ax[1].set_title("W5: LR sensitivity"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=6, ncol=2)
    plt.tight_layout(); p = os.path.join(OUT, "fig_w4w5.png"); plt.savefig(p, dpi=110); print("wrote %s" % p)
except Exception as e:
    print("\nplotting failed: %r" % e)

print("\nDONE")
