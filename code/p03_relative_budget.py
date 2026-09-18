#!/usr/bin/env python3
"""P0-3: same-protocol (relative budget) re-check -- does the allocation gain still rise with difficulty?

Problem
-------
The +0.04 pp (neu_cls, budget 600) and +2.66 pp (magnetic_tile, budget 120) in Sec. 8.3 come from
**absolute budget levels of different datasets**, i.e. apples to oranges:
  * neu_cls's +0.04 appears near the ceiling (already saturated)
  * magnetic_tile's +2.66 appears on the rising part of the curve
So the cross-dataset claim "allocation gain rises with difficulty" lacks same-protocol evidence.

Approach
--------
For each dataset, use **the same fraction of its largest available level** as the budget, e.g.:
  B_frac in {1/12, 1/6, 1/3, 1/2} of (largest level x number of classes)
Recompute the "uniform vs DP-optimal" gain under identical **relative budgets** and print a comparison table.

Criterion
---------
If the gain still rises monotonically with difficulty at equal relative budgets -> upgrade the claim to a regularity statement;
if not -> downgrade it to "a correlation observed under the absolute budgets of the four datasets in this paper".
"""

import json, os
from collections import defaultdict

W = "/var/hpc-root/xiangxu/work"
R = os.path.join(W, "results_bench")
BENCH = "/var/hpc-root/HPC_SHARED_DIR/benchmark"
DATASETS = ["neu_cls", "gc10det", "magnetic_tile", "casting"]
FRACS = [0.5, 0.3333, 0.1667, 0.0833]      # fraction of "largest level x number of classes"
NEG = -1e9


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


def solve_dp(curve, cls, B, CAP):
    def acc_at(c, n):
        ks = sorted(curve[c])
        if not ks or n <= 0:
            return 0.0
        if n in curve[c]:
            return curve[c][n]
        lo = max([k for k in ks if k <= n], default=ks[0])
        hi = min([k for k in ks if k >= n], default=ks[-1])
        return curve[c][lo] if hi == lo else curve[c][lo] + (curve[c][hi] - curve[c][lo]) * (n - lo) / (hi - lo)

    uni = B // len(cls)
    u = mean([acc_at(c, uni) for c in cls])
    dp = [[NEG] * (B + 1) for _ in range(len(cls) + 1)]
    ch = [[0] * (B + 1) for _ in range(len(cls) + 1)]
    dp[0][0] = 0.0
    for j, c in enumerate(cls, 1):
        for b in range(B + 1):
            best, bn = NEG, 0
            for n in range(0, min(CAP, b) + 1):
                pv = dp[j - 1][b - n]
                if pv <= NEG / 2:
                    continue
                v = pv + (acc_at(c, n) if n > 0 else 0.0)
                if v > best:
                    best, bn = v, n
            dp[j][b] = best; ch[j][b] = bn
    alloc = {}; b = B
    for j in range(len(cls), 0, -1):
        n = ch[j][b]; alloc[cls[j - 1]] = n; b -= n
    o = mean([acc_at(c, alloc[c]) for c in cls])
    return u, o, alloc


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

    imb = {}
    for ds in DATASETS:
        cs = sorted(man[ds]["counts"].values())
        imb[ds] = cs[-1] / max(1, cs[0])
    order = sorted(DATASETS, key=lambda d: imb[d])

    print("=" * 100)
    print("P0-3 same-protocol re-check: set the budget to the same fraction of the largest level, recompute uniform vs DP gain")
    print("=" * 100)
    print("  dataset imbalance: " + ", ".join("%s=%.1fx" % (d, imb[d]) for d in order))

    results = defaultdict(dict)
    for ds in order:
        cls = man[ds]["classes"]
        acc = defaultdict(lambda: defaultdict(list))
        for d in rows:
            if d["dataset"] != ds:
                continue
            n = d.get("per_class"); pca = d.get("per_class_acc") or {}
            for c in cls:
                v = pca.get(c)
                if v is not None and v == v:
                    acc[c][n].append(v)
        levels = sorted(set(n for c in cls for n in acc[c]))
        CAP = max(levels)
        curve = {}
        for c in cls:
            curve[c] = {}
            for n in levels:
                v = mean(acc[c][n])
                if v is not None:
                    curve[c][n] = v
            mx = -1.0
            for n in sorted(curve[c]):
                mx = max(mx, curve[c][n]); curve[c][n] = mx
        Bfull = CAP * len(cls)
        for f in FRACS:
            B = max(len(cls), int(Bfull * f))
            u, o, _ = solve_dp(curve, cls, B, CAP)
            results[ds][f] = (B, u, o, (o - u) if (u is not None and o is not None) else None)

    for f in FRACS:
        print("\n--- relative budget = %.0f%% of the largest budget ---" % (f * 100))
        print("  %-16s %8s %10s %10s %9s" % ("dataset", "budget", "uniform", "DP", "gain"))
        for ds in order:
            B, u, o, g = results[ds][f]
            print("  %-16s %8d %9s%% %9s%% %+8s"
                  % (ds, B, ("%.2f" % u) if u else "-", ("%.2f" % o) if o else "-",
                     ("%.2f" % g) if g is not None else "-"))

    print("\n" + "=" * 100)
    print("Reading")
    print("=" * 100)
    for f in FRACS:
        gs = [(ds, results[ds][f][3]) for ds in order if results[ds][f][3] is not None]
        mono = all(gs[i][1] <= gs[i + 1][1] + 0.05 for i in range(len(gs) - 1))
        print("  %.0f%% budget: " % (f * 100) +
              "  ".join("%s=%+.2f" % (d, g) for d, g in gs) +
              ("   [monotonically rising]" if mono else "   [non-monotonic]"))
    print("\n  If the gain rises monotonically with imbalance at every relative budget -> Sec. 8.3 can be upgraded to a regularity claim;")
    print("  otherwise it must be downgraded to a correlation observed under the absolute budgets of the four datasets in this paper.")
    print("\nDONE")


if __name__ == "__main__":
    main()
