#!/usr/bin/env python3
"""Cross-dataset main analysis: per-class data-efficiency table + difficulty map + budget decision table.

Read-only on results_bench/*.json; no training result is modified.
Output is written to --out (default /var/hpc-root/xiangxu/work/analysis_bench)
"""

import argparse, csv, json, math, os
from collections import defaultdict

W = "/var/hpc-root/xiangxu/work"
R = os.path.join(W, "results_bench")
BENCH = "/var/hpc-root/HPC_SHARED_DIR/benchmark"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(W, "analysis_bench"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for fn in sorted(os.listdir(R)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(R, fn), encoding="utf-8") as f:
                d = json.load(f)
            if d.get("type") == "final":
                rows.append(d)
        except Exception:
            pass
    print("loaded %d results\n" % len(rows))

    man = json.load(open(os.path.join(BENCH, "manifest.json"), encoding="utf-8"))
    datasets = [d for d in ["neu_cls", "gc10det", "magnetic_tile", "casting"] if d in man]

    # ---------- wide table ----------
    flat = []
    for d in rows:
        r = {"run_id": d["run_id"], "dataset": d["dataset"], "model": d["model"],
             "seed": d["seed"], "per_class": d.get("per_class"), "img_size": d.get("img_size"),
             "n_train": d.get("n_train"), "n_classes": d.get("n_classes"),
             "test_acc": d.get("test_acc"), "best_val_acc": d.get("best_val_acc"),
             "total_sec": d.get("total_sec"), "device": d.get("device"),
             "amp_dtype": d.get("amp_dtype")}
        pca = d.get("per_class_acc") or {}
        for i, (k, v) in enumerate(pca.items()):
            r["acc_%d_%s" % (i, k)] = v
        cm = d.get("confusion")
        if cm:
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    r["cm_%d_%d" % (i, j)] = cm[i][j]
        flat.append(r)
    cols = []
    for r in flat:
        for k in r:
            if k not in cols:
                cols.append(k)
    csv_p = os.path.join(a.out, "bench_all.csv")
    with open(csv_p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat)
    print("wrote %s (%d rows x %d cols)\n" % (csv_p, len(flat), len(cols)))

    # ---------- per-class data efficiency ----------
    summary = {}
    for ds in datasets:
        cls = man[ds]["classes"]
        cnt = man[ds]["counts"]
        dr = [d for d in rows if d["dataset"] == ds]
        levels = sorted(set(d.get("per_class") for d in dr if d.get("per_class")))
        models = sorted(set(d["model"] for d in dr))

        # (model, level) -> per-class mean over seeds
        cell = defaultdict(lambda: defaultdict(list))
        for d in dr:
            pca = d.get("per_class_acc") or {}
            for c in cls:
                if c in pca:
                    cell[(d["model"], d.get("per_class"))][c].append(pca[c])
        VEC = {k: {c: mean(v[c]) for c in cls} for k, v in cell.items()}

        def n_at(model, c, thr):
            for n in levels:
                v = VEC.get((model, n), {}).get(c)
                if v is not None and v >= thr:
                    return n
            return None

        # A fixed 95% threshold is out of reach on hard datasets (only 1 of gc10det's 10 classes makes it).
        # Add a **relative criterion**: n@rel95 = images needed to reach "this class's own best accuracy x 0.95".
        # Every dataset can then be scored and compared across datasets; the best accuracy itself is reported too.
        per_model = {}
        per_class = {}
        for m in models:
            row = {}
            for c in cls:
                seq = [(n, VEC.get((m, n), {}).get(c)) for n in levels]
                seq = [(n, v) for n, v in seq if v is not None]
                mx = max((v for _n, v in seq), default=None)
                rel = None
                if mx is not None:
                    for n, v in seq:
                        if v >= 0.95 * mx:
                            rel = n
                            break
                row[c] = {"n95": n_at(m, c, 95.0), "nrel95": rel, "max": mx}
            per_model[m] = row
        for c in cls:
            for key in ("n95", "nrel95", "max"):
                vs = sorted(v[key] for v in (per_model[m][c] for m in models) if v[key] is not None)
                per_class.setdefault(c, {})[key + "_min"] = vs[0] if vs else None
                per_class[c][key + "_median"] = vs[len(vs) // 2] if vs else None
                per_class[c][key + "_max"] = vs[-1] if vs else None
            per_class[c]["n_models"] = len([m for m in models if per_model[m][c]["nrel95"] is not None])
        # Heterogeneity uses the relative criterion (per-class **best** n@rel95 across models, reflecting intrinsic difficulty)
        med = [per_class[c]["nrel95_median"] for c in cls if per_class[c]["nrel95_median"]]
        hetero = (max(med) / min(med)) if med and min(med) else None
        cs = sorted(cnt.values())
        imbalance = cs[-1] / max(1, cs[0])

        # budget allocation (DP, the same algorithm as the NEU version)
        curve = {}
        for c in cls:
            curve[c] = {}
            for n in levels:
                v = mean([VEC.get((m, n), {}).get(c) for m in models])
                if v is not None:
                    curve[c][n] = v
            mx = -1.0
            for n in sorted(curve[c]):
                mx = max(mx, curve[c][n]); curve[c][n] = mx

        def acc_at(c, n):
            ks = sorted(curve[c])
            if not ks or n <= 0:
                return 0.0
            if n in curve[c]:
                return curve[c][n]
            lo = max([k for k in ks if k <= n], default=ks[0])
            hi = min([k for k in ks if k >= n], default=ks[-1])
            return curve[c][lo] if hi == lo else curve[c][lo] + (curve[c][hi] - curve[c][lo]) * (n - lo) / (hi - lo)

        CAP = max(levels)
        budgets = [x for x in (len(cls) * m for m in (10, 20, 50, 100)) if x <= CAP * len(cls)]
        budget_rows = []
        NEG = -1e9
        for B in budgets:
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
            budget_rows.append({"budget": B, "uniform_acc": u, "dp_acc": o,
                                "gain": (o - u) if (u is not None and o is not None) else None,
                                "alloc": alloc})

        summary[ds] = {"classes": cls, "counts": cnt, "levels": levels, "models": models,
                       "per_model_n95": per_model, "per_class_n95": per_class,
                       "heterogeneity_ratio": hetero, "imbalance_ratio": imbalance,
                       "n_results": len(dr), "budget": budget_rows}

    with open(os.path.join(a.out, "bench_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    # ---------- printing ----------
    print("=" * 100)
    print("A. Per-class data efficiency: annotations needed to reach 95% (n@95%, across models)")
    print("=" * 100)
    for ds in datasets:
        s = summary[ds]
        print("\n[%s] %d classes | imbalance %.1fx | levels %s..%s | %d models | %d results"
              % (ds, len(s["classes"]), s["imbalance_ratio"], min(s["levels"]), max(s["levels"]),
                 len(s["models"]), s["n_results"]))
        print("  %-22s %7s %9s | %7s %9s | %7s %5s" %
              ("class", "rel95", "range", "abs95", "range", "maxacc", "nmod"))
        _k = lambda x: (s["per_class_n95"][x]["nrel95_median"] is None,
                        s["per_class_n95"][x]["nrel95_median"] or 0)
        for c in sorted(s["classes"], key=_k):
            v = s["per_class_n95"][c]
            print("  %-22s %7s %9s | %7s %9s | %7s %5d"
                  % (c, v["nrel95_median"], "%s-%s" % (v["nrel95_min"], v["nrel95_max"]),
                     v["n95_median"], "%s-%s" % (v["n95_min"], v["n95_max"]),
                     ("%.1f" % v["max_median"]) if v["max_median"] is not None else "-",
                     v["n_models"]))
        print("  -> class-level heterogeneity (max/min of median n@95%%) = %.1fx" % (s["heterogeneity_ratio"] or 0))

    print("\n" + "=" * 100)
    print("B. Difficulty map: heterogeneity vs imbalance")
    print("=" * 100)
    print("  %-16s %10s %12s %10s %10s" % ("dataset", "imbalance", "heterogeneity", "n_classes", "n_images"))
    for ds in sorted(datasets, key=lambda d: summary[d]["imbalance_ratio"]):
        s = summary[ds]
        print("  %-16s %9.1fx %11.1fx %10d %10d"
              % (ds, s["imbalance_ratio"], s["heterogeneity_ratio"] or 0,
                 len(s["classes"]), sum(s["counts"].values())))

    print("\n" + "=" * 100)
    print("C. Budget allocation decision table (uniform vs DP-optimal)")
    print("=" * 100)
    for ds in datasets:
        print("\n[%s]" % ds)
        for b in summary[ds]["budget"]:
            alloc = " ".join("%s=%d" % (k[:6], v) for k, v in list(b["alloc"].items())[:6])
            print("  budget %-5d uniform %-7s DP %-7s gain %-7s | %s"
                  % (b["budget"],
                     "%.2f%%" % b["uniform_acc"] if b["uniform_acc"] else "-",
                     "%.2f%%" % b["dp_acc"] if b["dp_acc"] else "-",
                     "%+.2f" % b["gain"] if b["gain"] is not None else "-", alloc))

    print("\nDONE -> %s/bench_summary.json" % a.out)


if __name__ == "__main__":
    main()
