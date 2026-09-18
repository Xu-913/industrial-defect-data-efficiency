"""Four deep-dive analyses A/B/C/D (all read-only over existing results)."""
import json, os, math
from collections import defaultdict

W = "/var/hpc-root/xiangxu/work"
R = os.path.join(W, "results")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
SH = ["crz", "inc", "pat", "pit", "rol", "scr"]

rows = []
for fn in sorted(os.listdir(R)):
    if fn.endswith(".json"):
        try:
            d = json.load(open(os.path.join(R, fn), encoding="utf-8"))
            if d.get("type") == "final":
                rows.append(d)
        except Exception:
            pass
MAIN = [d for d in rows if d.get("img_size") == 224 and d.get("mode") == "finetune"
        and abs((d.get("lr") or 0) - 3e-4) < 1e-9 and d.get("epochs") == 50
        and not d["run_id"].endswith("_a4")]
acc = lambda d, c: (d.get("per_class_acc") or {}).get(c)
levels = sorted(set(d["per_class"] for d in MAIN))
models = sorted(set(d["model"] for d in MAIN))
mean = lambda xs: sum(xs) / len(xs) if xs else None
print("loaded %d | main cohort %d\n" % (len(rows), len(MAIN)))


def sm(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for p, i in enumerate(o): r[i] = p + 1
        return r
    rx, ry = rk(x), rk(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    a = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    b = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n))) * math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return a / b if b else float("nan")


# Mean accuracy curve per class (over models and seeds)
curve = {}
for c in CLASSES:
    curve[c] = {}
    for n in levels:
        v = [acc(d, c) for d in MAIN if d["per_class"] == n]
        v = [x for x in v if x is not None]
        if v:
            curve[c][n] = sum(v) / len(v)
    # Monotonize (cummax): noise in the raw curve makes the greedy "marginal gain"
    # spike falsely and suck all budget into one class (measured pit=1042/1080). More data never hurts.
    mx = -1.0
    for n in sorted(curve[c]):
        mx = max(mx, curve[c][n])
        curve[c][n] = mx

# ============================== A. annotation budget allocation optimization
print("=" * 90)
print("A. Annotation budget allocation: uniform vs greedy-optimal (per-class mean curves, across 11 models)")
print("=" * 90)
lv = [n for n in levels if n <= 180]


def acc_at(c, n):
    ks = sorted(curve[c])
    if n <= 0: return 0.0
    if n in curve[c]: return curve[c][n]
    lo = max([k for k in ks if k <= n], default=ks[0])
    hi = min([k for k in ks if k >= n], default=ks[-1])
    if hi == lo: return curve[c][lo]
    return curve[c][lo] + (curve[c][hi] - curve[c][lo]) * (n - lo) / (hi - lo)


print("  %-10s %-12s %-12s %-9s %s" % ("budget", "uniform(B/6)", "DP-optimal", "gain_pp", "optimal_alloc"))
print("  " + "-" * 84)
# Dynamic programming instead of greedy. Greedy failed three times on noisy curves
# (pit=1042 / crz=1059 / stuck at 23): it is hypersensitive to jitter in the marginal gain. DP is robust.
CAP, NEG = 180, -1e9
for B in [60, 120, 300, 600, 1080]:
    uni = B // 6
    u = mean([acc_at(c, uni) for c in CLASSES])
    dp = [[NEG] * (B + 1) for _ in range(len(CLASSES) + 1)]
    ch = [[0] * (B + 1) for _ in range(len(CLASSES) + 1)]
    dp[0][0] = 0.0
    for j, c in enumerate(CLASSES, 1):
        for b in range(B + 1):
            best, bn = NEG, 0
            for n in range(0, min(CAP, b) + 1):
                pv = dp[j - 1][b - n]
                if pv <= NEG / 2:
                    continue
                v = pv + (acc_at(c, n) if n > 0 else 0.0)
                if v > best:
                    best, bn = v, n
            dp[j][b] = best
            ch[j][b] = bn
    alloc = {}
    b = B
    for j in range(len(CLASSES), 0, -1):
        n = ch[j][b]
        alloc[CLASSES[j - 1]] = n
        b -= n
    o = mean([acc_at(c, alloc[c]) for c in CLASSES])
    print("  %-10d %-12s %-12s %-9s %s" % (B, "%.2f%%" % u, "%.2f%%" % o, "%+.2f" % (o - u),
          " ".join("%s=%d" % (SH[i], alloc[c]) for i, c in enumerate(CLASSES))))

# ============================== B. training dynamics
print("\n" + "=" * 90)
print("B. Training dynamics: when overfitting starts / how best_epoch shifts with data volume")
print("=" * 90)
print("  %-7s %-10s %-12s %-12s %-10s" % ("n/cls", "n_runs", "best_epoch", "last_train_acc", "gen_gap"))
for n in levels:
    ds = [d for d in MAIN if d["per_class"] == n]
    if not ds: continue
    be = mean([d.get("best_epoch") for d in ds if d.get("best_epoch")])
    # last-epoch train_acc has to be read from the jsonl
    ta, gap = [], []
    for d in ds[:40]:
        p = os.path.join(R, d["run_id"] + ".jsonl")
        if not os.path.exists(p): continue
        try:
            ls = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
            ep = [x for x in ls if x.get("type") == "epoch"]
            if ep:
                ta.append(ep[-1]["train_acc"]); gap.append(ep[-1]["train_acc"] - ep[-1]["val_acc"])
        except Exception:
            pass
    print("  %-7d %-10d %-12s %-12s %-10s"
          % (n, len(ds), "%.1f" % be if be else "-",
             "%.1f" % mean(ta) if ta else "-", "%.1f" % mean(gap) if gap else "-"))

# ============================== C. confusion structure as a directed graph
print("\n" + "=" * 90)
print("C. Confusion structure: asymmetry / error sinks / stability of the structure with data volume")
print("=" * 90)
for tag, ns in (("low n<=5", [n for n in levels if n <= 5]), ("full n=180", [180])):
    off = [[0.0] * 6 for _ in range(6)]
    for d in MAIN:
        if d["per_class"] not in ns: continue
        cm = d.get("confusion")
        if not cm: continue
        tot = sum(sum(r) for r in cm) or 1
        for i in range(6):
            for j in range(6):
                if i != j: off[i][j] += cm[i][j] / tot
    asy = [abs(off[i][j] - off[j][i]) / (off[i][j] + off[j][i]) for i in range(6) for j in range(i + 1, 6)
           if off[i][j] + off[j][i] > 1e-9]
    print("  [%s] mean asymmetry index = %.3f   (0 = fully symmetric, 1 = fully one-way)" % (tag, mean(asy) if asy else 0))
    sink = [(sum(off[j][i] for j in range(6) if j != i) - sum(off[i][j] for j in range(6) if j != i), CLASSES[i])
            for i in range(6)]
    sink.sort(reverse=True)
    print("     error sink ranking (net inflow): " + ", ".join("%s%+.2f" % (SH[CLASSES.index(c)], v) for v, c in sink))
    print("     3 most asymmetric pairs: " + ", ".join(
        "%s->%s %.2f" % (SH[i], SH[j], abs(off[i][j] - off[j][i]) / (off[i][j] + off[j][i]))
        for i, j in sorted([(i, j) for i in range(6) for j in range(i + 1, 6)
                            if off[i][j] + off[j][i] > 1e-9],
                           key=lambda p: -abs(off[p[0]][p[1]] - off[p[1]][p[0]]) / (off[p[0]][p[1]] + off[p[1]][p[0]]))[:3]))

# ============================== D. variance decomposition
print("\n" + "=" * 90)
print("D. Variance decomposition: how much seed / model / data volume each contribute; how many seeds are enough")
print("=" * 90)
print("  %-7s %-12s %-12s %-12s %s" % ("n/cls", "seed_std", "model_range", "seed/model", "seeds_for_0.5pp"))
for n in levels:
    bym = defaultdict(list)
    for d in MAIN:
        if d["per_class"] == n:
            bym[d["model"]].append(d["test_acc"])
    within = [ (max(v)-min(v))/2 for v in bym.values() if len(v) >= 2 ]
    # stricter: population standard deviation within each model
    import statistics as st
    sd = [st.pstdev(v) for v in bym.values() if len(v) >= 2]
    mm = [mean(v) for v in bym.values() if v]
    if sd and mm:
        s = mean(sd); r = max(mm) - min(mm)
        k = math.ceil((1.96 * s / 0.5) ** 2) if s > 0 else 1
        print("  %-7d %-12.2f %-12.2f %-12.2f %d" % (n, s, r, s / r if r else 0, k))
print("\nDONE")
