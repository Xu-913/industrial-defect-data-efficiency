"""O-round (overnight) experiment plan executor -- Paper A.

Design principles
-----------------
1. **Idempotent**: skip when the result json exists; a killed job restarts safely without burning GPU time twice.
2. **Sharding**: `--job-index i --job-count n` cuts the configs into n shards, one SLURM job each, so they run in parallel.
3. **No duplicated training logic**: train.py is invoked via subprocess, keeping a single source of truth.
4. **Failures are not fatal**: a crashed run is logged and execution continues; it does not take down the whole job.
5. **5 seeds per config** (user requirement); low-data levels have high variance, so multiple seeds are mandatory.

Usage (on the cluster)
    python -u run_plan.py --job-index 0 --job-count 24
    python -u run_plan.py --list        # only show the plan size, do not run
"""

import argparse
import json
import os
import subprocess
import sys
import time

WORK = "/var/hpc-root/xiangxu/work"
CODE = "/var/hpc-root/xiangxu/code"
DATA = os.path.join(WORK, "NEU-CLS")
RESULTS = os.path.join(WORK, "results")

MODELS = ["resnet18", "resnet50", "densenet121", "mobilenet_v3_small", "mobilenet_v3_large",
          "shufflenet_v2_x1_0", "efficientnet_b0", "regnet_x_400mf", "convnext_tiny",
          "vit_b_16", "swin_t"]

# Training images per class (finer than percentages: low levels do not drift to the same count via rounding)
IPCS_MAIN = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 125, 150, 180]
IPCS_FINE = list(range(1, 31))                       # 1..30, threshold localization
IPCS_RES = [1, 2, 4, 8, 15, 30, 60, 100, 180]        # for the resolution axis
IPCS_LR = [10, 30, 100, 180]                         # for the LR axis

SEEDS5 = [0, 1, 2, 3, 4]
SEEDS3 = [0, 1, 2]


def build_plan():
    """Return the deduplicated config list, sorted by wave priority (important ones first)."""
    cfg = {}

    def add(wave, model, per_class, seed, res, mode, epochs, lr, sfx=""):
        rid = "%s_seed%d_n%d_res%d_%s_lr%s%s" % (
            model, seed, per_class, res, mode, lr, ("_" + sfx) if sfx else "")
        if rid not in cfg:
            cfg[rid] = dict(run_id=rid, wave=wave, model=model, per_class=per_class,
                            seed=seed, res=res, mode=mode, epochs=epochs, lr=lr, sfx=sfx)

    # Epoch count: 50 for the main wave (fuller convergence, and it uses the overnight compute budget); 20 for sensitivity waves
    E_MAIN = 50
    E_SEC = 20

    # ---- W1 main matrix: 11 models x 20 levels x 5 seeds (the paper's backbone, top priority) ----
    for m in MODELS:
        for n in IPCS_MAIN:
            for s in SEEDS5:
                add("W1", m, n, s, 224, "finetune", E_MAIN, 3e-4)

    # ---- W2 threshold densification: fine sweep over 1..30 images per class (locating the collapse threshold) ----
    for m in MODELS:
        for n in IPCS_FINE:
            for s in SEEDS5:
                add("W2", m, n, s, 224, "finetune", E_MAIN, 3e-4)

    # ---- W3 high resolution: do small defects demand more data ----
    # WARNING: vit_b_16 hard-codes its positional encoding to 224x224, so feeding 320^2 raises
    #    AssertionError: Wrong image height! Expected 224 but got 320!
    #    (all 27 attempts failed in the O round). This is a genuine model limitation, so we do not
    #    force positional-encoding interpolation -- the paper records honestly that the architecture does not support that resolution.
    for m in MODELS:
        if m == "vit_b_16":
            continue
        for n in IPCS_RES:
            for s in SEEDS3:
                add("W3", m, n, s, 320, "finetune", E_SEC, 3e-4)

    # ---- W4 frozen backbone (linear probe): separating "feature quality" from "adaptation ability" ----
    for m in MODELS:
        for n in IPCS_MAIN:
            for s in SEEDS3:
                add("W4", m, n, s, 224, "frozen", E_MAIN, 1e-3)

    # ---- W5 LR sensitivity: defusing the reviewer attack that "a single LR is unfair to some models" ----
    for m in MODELS:
        for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
            for n in IPCS_LR:
                for s in SEEDS3:
                    add("W5", m, n, s, 224, "finetune", E_SEC, lr)

    # ---- W6 device reproducibility (A4000 / BF16): testing whether training hardware affects the conclusions ----
    # Motivation: T4 runs FP16 while A4000 runs BF16 (native from SM 8.6), so numerical behavior differs.
    # If seeds of one config span both cards, hardware is confounded with seed -- the effect must be quantified.
    # Approach: resample 11 models x 6 data levels x 3 seeds on the A4000 (run_id suffix _a4) and
    #           compare them one by one with the matching T4 configs. A **measurement robustness check**, not a deployment study (that is Paper C's).
    for m in MODELS:
        for n in [2, 5, 10, 20, 50, 180]:
            for s in SEEDS3:
                add("W6", m, n, s, 224, "finetune", E_MAIN, 3e-4, sfx="a4")

    order = {"W1": 0, "W2": 1, "W4": 2, "W3": 3, "W5": 4, "W6": 5}
    return sorted(cfg.values(), key=lambda c: (order[c["wave"]], c["model"], c["per_class"], c["seed"]))


# Relative cost weight per run (used to spread configs evenly across shards, so no shard is all large models)
COST = {"resnet18": 1.0, "resnet50": 1.6, "densenet121": 1.3, "mobilenet_v3_small": 0.7,
        "mobilenet_v3_large": 0.8, "shufflenet_v2_x1_0": 0.7, "efficientnet_b0": 0.9,
        "regnet_x_400mf": 0.8, "convnext_tiny": 1.5, "vit_b_16": 5.0, "swin_t": 1.8}


def cost_of(c):
    w = COST.get(c["model"], 1.0)
    w *= 0.55 if c["mode"] == "frozen" else 1.0        # frozen is forward-only
    w *= 2.1 if c["res"] == 320 else 1.0               # resolution^2 approximation
    w *= c["per_class"] / 60.0 + 0.15                  # data volume
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-index", type=int, default=0)
    ap.add_argument("--job-count", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max runs for this shard (0 = unlimited)")
    ap.add_argument("--budget-sec", type=int, default=0, help="time budget in seconds for this shard (0 = unlimited)")
    ap.add_argument("--only-wave", default="", help="run only the given waves, comma-separated, e.g. W6")
    a = ap.parse_args()

    plan = build_plan()
    if a.only_wave:
        want = set(x.strip() for x in a.only_wave.split(","))
        plan = [c for c in plan if c["wave"] in want]
    total = len(plan)

    if a.list:
        from collections import Counter
        print("total configs = %d" % total)
        for w, k in sorted(Counter(c["wave"] for c in plan).items()):
            print("  %s: %d" % (w, k))
        tot = sum(cost_of(c) for c in plan)
        print("total cost weight = %.0f  ->  estimated GPU hours ~ %.1f h (empirical conversion of 26 runs per weight unit)" % (tot, tot * 68 / 3600))
        return

    # balance shards by cost
    shards = [[] for _ in range(a.job_count)]
    load = [0.0] * a.job_count
    for c in plan:
        i = load.index(min(load))
        shards[i].append(c)
        load[i] += cost_of(c)

    mine = shards[a.job_index]
    print("shard %d/%d: %d configs, cost weight %.1f (total %.1f)"
          % (a.job_index, a.job_count, len(mine), load[a.job_index], sum(load)), flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    t_start = time.time()
    done = skipped = failed = 0

    for k, c in enumerate(mine, 1):
        out_json = os.path.join(RESULTS, c["run_id"] + ".json")

        if os.path.exists(out_json):
            skipped += 1
            continue
        if a.limit and done >= a.limit:
            print("[stop] reached --limit %d" % a.limit, flush=True)
            break
        if a.budget_sec and (time.time() - t_start) > a.budget_sec:
            print("[stop] reached time budget %ds" % a.budget_sec, flush=True)
            break

        cmd = [sys.executable, "-u", os.path.join(CODE, "train.py"),
               "--model", c["model"], "--seed", str(c["seed"]),
               "--data", DATA, "--split-file", os.path.join(WORK, "splits", "neu_cls_seed%d.json" % c["seed"]),
               "--out", RESULTS, "--img-size", str(c["res"]), "--epochs", str(c["epochs"]),
               "--bs", "32", "--lr", str(c["lr"]), "--warmup", "2",
               "--per-class", str(c["per_class"]), "--mode", c["mode"],
               "--tag", "n%d_res%d_%s_lr%s%s" % (c["per_class"], c["res"], c["mode"], c["lr"],
                                                 ("_" + c["sfx"]) if c.get("sfx") else "")]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
            ok = os.path.exists(out_json)
            dt = time.time() - t0
            if ok:
                done += 1
                tail = [l for l in p.stdout.splitlines() if l.startswith("FINAL ")]
                print("[%d/%d] %s  %.0fs  %s" % (k, len(mine), c["run_id"], dt,
                                                 tail[-1] if tail else "ok"), flush=True)
            else:
                failed += 1
                print("[%d/%d] %s  FAILED %.0fs  %s" % (k, len(mine), c["run_id"], dt,
                                                        (p.stderr or "")[-300:].replace("\n", " ")), flush=True)
        except Exception as e:
            failed += 1
            print("[%d/%d] %s  EXC %s" % (k, len(mine), c["run_id"], str(e)[:200]), flush=True)

    print("===== SHARD %d DONE: ok=%d skipped=%d failed=%d  elapsed=%.0fs ====="
          % (a.job_index, done, skipped, failed, time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
