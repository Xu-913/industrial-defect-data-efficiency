"""P1 main-matrix executor: 4 datasets x 11 models x per-class levels x 5 seeds.

Idempotent: an existing result json is skipped, so a killed job can simply be rerun.
Usage: python -u run_bench.py --job-index 0 --job-count 24
"""

import argparse, os, subprocess, sys, time

WORK = "/var/hpc-root/xiangxu/work"
CODE = "/var/hpc-root/xiangxu/code"
BENCH_RESULTS = os.path.join(WORK, "results_bench")
TRAINER = os.path.join(CODE, "train_bench.py")

MODELS = ["resnet18", "resnet50", "densenet121", "mobilenet_v3_small", "mobilenet_v3_large",
          "shufflenet_v2_x1_0", "efficientnet_b0", "regnet_x_400mf", "convnext_tiny",
          "vit_b_16", "swin_t"]

# Input resolution and per-class levels for each dataset (the level cap is set by the rarest class)
DATASETS = {
    "neu_cls":       dict(img=224, levels=[1,2,3,4,6,8,10,12,15,20,25,30,40,50,65,80,100,125,150,180]),
    "gc10det":       dict(img=256, levels=[1,2,3,4,6,8,10,12,15,18]),
    "magnetic_tile": dict(img=224, levels=[1,2,3,4,6,8,12,16,24,38]),
    "casting":       dict(img=224, levels=[1,2,4,8,16,32,64,128,256,512]),
}
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 50

COST = {"resnet18":1.0,"resnet50":1.6,"densenet121":1.3,"mobilenet_v3_small":0.7,
        "mobilenet_v3_large":0.8,"shufflenet_v2_x1_0":0.7,"efficientnet_b0":0.9,
        "regnet_x_400mf":0.8,"convnext_tiny":1.5,"vit_b_16":5.0,"swin_t":1.8}


def build_plan():
    plan = []
    for ds, cfg in DATASETS.items():
        for m in MODELS:
            for n in cfg["levels"]:
                for s in SEEDS:
                    plan.append(dict(dataset=ds, model=m, per_class=n, seed=s,
                                     img=cfg["img"], epochs=EPOCHS))
    return plan


def cost_of(c):
    w = COST.get(c["model"], 1.0)
    w *= (c["img"] / 224.0) ** 2
    return w * (c["per_class"] / 60.0 + 0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-index", type=int, default=0)
    ap.add_argument("--job-count", type=int, default=24)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only-dataset", default="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    plan = build_plan()
    if a.only_dataset:
        want = set(x.strip() for x in a.only_dataset.split(","))
        plan = [c for c in plan if c["dataset"] in want]
    if a.list:
        from collections import Counter
        print("total configs = %d" % len(plan))
        for k, v in sorted(Counter(c["dataset"] for c in plan).items()):
            print("  %-16s %d" % (k, v))
        tot = sum(cost_of(c) for c in plan)
        print("cost weight %.0f -> estimated GPU hours ~ %.0f h" % (tot, tot * 68 / 3600))
        return

    shards = [[] for _ in range(a.job_count)]
    load = [0.0] * a.job_count
    for c in plan:
        i = load.index(min(load)); shards[i].append(c); load[i] += cost_of(c)
    mine = shards[a.job_index]
    print("shard %d/%d: %d configs, weight %.0f" % (a.job_index, a.job_count, len(mine), load[a.job_index]), flush=True)

    os.makedirs(BENCH_RESULTS, exist_ok=True)
    t0 = time.time(); done = skip = fail = 0
    for k, c in enumerate(mine, 1):
        rid = "%s_%s_seed%d_n%d_res%d" % (c["dataset"], c["model"], c["seed"], c["per_class"], c["img"])
        if os.path.exists(os.path.join(BENCH_RESULTS, rid + ".json")):
            skip += 1; continue
        if a.limit and done >= a.limit:
            break
        cmd = [sys.executable, "-u", TRAINER, "--dataset", c["dataset"], "--model", c["model"],
               "--seed", str(c["seed"]), "--per-class", str(c["per_class"]),
               "--img-size", str(c["img"]), "--epochs", str(c["epochs"]),
               "--out", BENCH_RESULTS]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
            if os.path.exists(os.path.join(BENCH_RESULTS, rid + ".json")):
                done += 1
                tail = [l for l in p.stdout.splitlines() if l.startswith("FINAL ")]
                print("[%d/%d] %s  %s" % (k, len(mine), rid, tail[-1] if tail else "ok"), flush=True)
            else:
                fail += 1
                print("[%d/%d] %s FAILED %s" % (k, len(mine), rid, (p.stderr or "")[-200:].replace("\n"," ")), flush=True)
        except Exception as e:
            fail += 1
            print("[%d/%d] %s EXC %s" % (k, len(mine), rid, str(e)[:150]), flush=True)
    print("===== SHARD %d DONE ok=%d skip=%d fail=%d %.0fs =====" % (a.job_index, done, skip, fail, time.time()-t0), flush=True)


if __name__ == "__main__":
    main()
