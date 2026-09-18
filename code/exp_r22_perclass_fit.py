#!/usr/bin/env python3
"""R2-2 supplementary experiment: can the per-class data-requirement prediction framework of Jain et al. (2023) be "applied per class"?

Background
----------
Reviewer R2-2 (blocking) noted that if our difference from Jain 2023 is merely that "they did not ask whether allocation
should be uniform", that is a difference at the **assumption level**; Jain's PPL + meta-learning framework **can in
principle be applied per class**. We must give **empirical evidence** for why per-class application is infeasible in practice.

Approach (existing 2700 results only, no new training)
--------
For each dataset and each class, take the (n, acc) curve (averaged over models and seeds).
Fit an error power law to the **lower half of the levels** and predict the **upper half**, then compare:

  A. Per-class independent fit (per-class)
  B. Pooled fit over all classes (pooled)
  C. Per-class independent fit using only the first 3 points (simulating "fewer probes")

Reported: prediction RMSE, fraction of fittable classes, parameter dispersion (CV of alpha).

Criterion
---------
If per-class RMSE is substantially larger than pooled, with a low fittable fraction and high parameter dispersion
-> "applying the Jain framework per class fails in practice" holds, and R2-2 is upgraded from assumption-level to empirical.
"""

import json, math, os
from collections import defaultdict

W = "/var/hpc-root/xiangxu/work"
R = os.path.join(W, "results_bench")
BENCH = "/var/hpc-root/HPC_SHARED_DIR/benchmark"
DATASETS = ["neu_cls", "gc10det", "magnetic_tile", "casting"]


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


def fit_powerlaw(pts):
    """Fit err(n) = k * n^(-alpha) with err = max(eps, 1 - acc/100).
    Returns (alpha, logk) or None. Least-squares fit of log(err) = logk - alpha*log(n)."""
    xs, ys = [], []
    for n, acc in pts:
        if n <= 0 or acc is None:
            continue
        err = max(1e-4, 1.0 - acc / 100.0)
        xs.append(math.log(n)); ys.append(math.log(err))
    if len(xs) < 3:
        return None
    m = len(xs)
    mx, my = sum(xs) / m, sum(ys) / m
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(m))
    den = sum((xs[i] - mx) ** 2 for i in range(m))
    if den <= 0:
        return None
    alpha = -num / den
    logk = my + alpha * mx
    return alpha, logk


def predict(par, n):
    alpha, logk = par
    return 100.0 * (1.0 - math.exp(logk) * (n ** (-alpha)))


def rmse(pairs):
    if not pairs:
        return None
    return math.sqrt(sum((p - t) ** 2 for p, t in pairs) / len(pairs))


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

    print("=" * 92)
    print("R2-2: feasibility test of applying the Jain framework per class (error power law err(n)=k*n^-alpha)")
    print("=" * 92)

    grand = {"per": [], "pool": [], "per3": []}
    for ds in DATASETS:
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
        curves = {}
        for c in cls:
            pts = sorted((n, mean(v)) for n, v in acc[c].items() if mean(v) is not None)
            if pts:
                curves[c] = pts
        levels = sorted(set(n for c in curves for n, _ in curves[c]))
        if len(levels) < 4:
            continue
        cut = levels[len(levels) // 2]

        # A. per-class independent
        per_pred, per_alpha, fit_ok = [], [], 0
        for c, pts in curves.items():
            fitpts = [(n, a) for n, a in pts if n <= cut]
            testpts = [(n, a) for n, a in pts if n > cut]
            par = fit_powerlaw(fitpts)
            if par is None or not testpts:
                continue
            fit_ok += 1
            per_alpha.append(par[0])
            for n, a in testpts:
                per_pred.append((predict(par, n), a))

        # B. pooled fit (only 2 parameters, using points from all classes)
        allfit = [(n, a) for c in curves for n, a in curves[c] if n <= cut]
        alltest = [(n, a) for c in curves for n, a in curves[c] if n > cut]
        pool_par = fit_powerlaw(allfit)
        pool_pred = [(predict(pool_par, n), a) for n, a in alltest] if pool_par else []

        # C. per-class but using only the first 3 points (fewer probes)
        per3_pred = []
        for c, pts in curves.items():
            par = fit_powerlaw(pts[:3])
            if par is None:
                continue
            for n, a in pts:
                if n > (pts[2][0] if len(pts) > 2 else 0):
                    per3_pred.append((predict(par, n), a))

        rp, rl, r3 = rmse(per_pred), rmse(pool_pred), rmse(per3_pred)
        ma = (sum(per_alpha) / len(per_alpha)) if per_alpha else float("nan")
        if len(per_alpha) > 1:
            sd = math.sqrt(sum((x - ma) ** 2 for x in per_alpha) / len(per_alpha))
            cv = sd / ma if ma else float("nan")
        else:
            cv = float("nan")
        grand["per"].append(rp); grand["pool"].append(rl); grand["per3"].append(r3)
        print("\n[%s] %d classes, %d levels, fit/test split n<=%d" % (ds, len(cls), len(levels), cut))
        print("  classes with a per-class fit: %d/%d" % (fit_ok, len(cls)))
        print("  extrapolation RMSE  per-class %-7s | pooled %-7s | per-class (3 pts) %-7s"
              % ("%.2f" % rp if rp else "-", "%.2f" % rl if rl else "-", "%.2f" % r3 if r3 else "-"))
        print("  per-class alpha mean %.3f  coefficient of variation %.3f" % (ma, cv))

    def m(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None
    print("\n" + "=" * 92)
    print("cross-dataset summary")
    print("=" * 92)
    print("  mean extrapolation RMSE: per-class %.3f | pooled %.3f | per-class (3 pts) %.3f"
          % (m(grand["per"]) or 0, m(grand["pool"]) or 0, m(grand["per3"]) or 0))
    a, b = m(grand["per"]), m(grand["pool"])
    if a and b:
        print("  per-class / pooled = %.2fx" % (a / b))
        print("\n  Reading: if the per-class extrapolation error is clearly larger than the pooled fit (e.g. >1.5x),")
        print("        then naive per-class application of the Jain framework fails in practice, and R2-2 is upgraded from assumption-level to empirical evidence.")
    print("\nDONE")


if __name__ == "__main__":
    main()
