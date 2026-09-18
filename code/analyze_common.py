#!/usr/bin/env python3
"""Control experiment: removing the measurement confound caused by differing level caps.

Problem
-------
Level caps differ wildly across datasets (neu_cls 180 / casting 512 / gc10det 18 / magnetic_tile 38).
"Heterogeneity falls as imbalance rises" (4.0x -> 2.2x -> 1.6x) is very likely an artifact of a
**truncated dynamic range** rather than a real pattern.

Approach
--------
Recompute the class-level spread only on the **levels shared by all datasets**:
  * shared levels = intersection of the level sets of all datasets
  * at each shared level, compute the "per-class accuracy range (max-min)" and the "relative range (max-min)/mean"
  * average across shared levels -> an apples-to-apples heterogeneity metric

The raw per-class accuracy at each level is also printed for manual checking.
"""

import json, os
from collections import defaultdict

W = "/var/hpc-root/xiangxu/work"
R = os.path.join(W, "results_bench")
BENCH = "/var/hpc-root/HPC_SHARED_DIR/benchmark"
DATASETS = ["neu_cls", "gc10det", "magnetic_tile", "casting"]


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

    # per dataset: level -> class -> [acc]
    tab = {}
    for ds in DATASETS:
        cls = man[ds]["classes"]
        acc = defaultdict(lambda: defaultdict(list))
        for d in rows:
            if d["dataset"] != ds:
                continue
            n = d.get("per_class")
            pca = d.get("per_class_acc") or {}
            for c in cls:
                if c in pca and pca[c] == pca[c]:
                    acc[n][c].append(pca[c])
        tab[ds] = {n: {c: mean(v) for c, v in d2.items()} for n, d2 in acc.items()}

    sets = {ds: set(tab[ds]) for ds in DATASETS}
    common = sorted(set.intersection(*[sets[ds] for ds in DATASETS]))
    print("level-set size per dataset:", {ds: len(sets[ds]) for ds in DATASETS})
    print("common levels:", common)
    if not common:
        print("!! no common levels")
        return

    print("\n" + "=" * 96)
    print("A. Per-class accuracy at each common level (averaged over models and seeds)")
    print("=" * 96)
    for n in common:
        print("\n--- %d images per class ---" % n)
        for ds in DATASETS:
            cls = man[ds]["classes"]
            vals = [(c, tab[ds][n].get(c)) for c in cls]
            vals = [(c, v) for c, v in vals if v is not None]
            if not vals:
                continue
            lo = min(v for _c, v in vals); hi = max(v for _c, v in vals)
            disp = " ".join("%s=%.0f" % (c[:5], v) for c, v in sorted(vals, key=lambda x: -x[1]))
            print("  %-16s spread=%5.1f  %s" % (ds, hi - lo, disp))

    print("\n" + "=" * 96)
    print("B. Heterogeneity after control (common levels only) vs raw heterogeneity")
    print("=" * 96)
    raw = {}
    try:
        s = json.load(open(os.path.join(W, "analysis_bench", "bench_summary.json"), encoding="utf-8"))
        raw = {ds: s[ds].get("heterogeneity_ratio") for ds in DATASETS if ds in s}
    except Exception:
        pass

    print("  %-16s %9s %12s %12s %12s" % ("dataset", "imbalance", "raw_hetero", "ctrl_spread", "ctrl_rel"))
    out = {}
    for ds in sorted(DATASETS, key=lambda d: _imb(man, d)):
        cls = man[ds]["classes"]
        sp, rel = [], []
        for n in common:
            vals = [tab[ds][n].get(c) for c in cls]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                sp.append(max(vals) - min(vals))
                m = mean(vals)
                if m:
                    rel.append((max(vals) - min(vals)) / m)
        out[ds] = {"ctrl_spread_pp": mean(sp), "ctrl_rel": mean(rel),
                   "imbalance": _imb(man, ds), "raw_hetero": raw.get(ds)}
        print("  %-16s %8.1fx %10s %11s %12s"
              % (ds, _imb(man, ds),
                 ("%.1fx" % raw[ds]) if raw.get(ds) else "-",
                 ("%.1f pp" % out[ds]["ctrl_spread_pp"]) if out[ds]["ctrl_spread_pp"] is not None else "-",
                 ("%.3f" % out[ds]["ctrl_rel"]) if out[ds]["ctrl_rel"] is not None else "-"))

    print("\n  Reading: ctrl_rel = mean over common levels of (per-class range / per-class mean).")
    print("        If ctrl_rel still tracks imbalance monotonically the pattern holds; if it vanishes, it was a level-truncation artifact.")

    with open(os.path.join(W, "analysis_bench", "common_level_control.json"), "w", encoding="utf-8") as f:
        json.dump({"common_levels": common, "per_level": {ds: tab[ds] for ds in DATASETS},
                   "summary": out}, f, ensure_ascii=False, indent=1)
    print("\nDONE -> analysis_bench/common_level_control.json")


def _imb(man, ds):
    cs = sorted(man[ds]["counts"].values())
    return cs[-1] / max(1, cs[0])


if __name__ == "__main__":
    main()
