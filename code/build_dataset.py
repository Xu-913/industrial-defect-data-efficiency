#!/usr/bin/env python3
"""Unified data pipeline (reusable): raw datasets -> standardized manifest + frozen splits.

Why it is needed
----------------
1. Deduplication must precede splitting. The audit found 64 duplicate groups in casting and 65 duplicate images in
   magnetic_tile; without dedup the same image can land in both train and test (data leakage), inflating accuracy.
2. Cross-dataset work needs one protocol: identical class naming, split ratios, and seeds, otherwise conclusions are not comparable.
3. Images are not copied; only a "manifest + split files" is produced, and images stay in place.

Usage
-----
    python build_dataset.py --out /path/to/benchmark
    python build_dataset.py --only gc10det
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Structural directory names: their presence in a path only marks a split/wrapper layer, not a class
STRUCTURAL = {"imgs", "images", "train", "test", "val", "valid", "validation",
              "annotations", "labels", "masks", "gt", "groundtruth"}
SRC = "/var/hpc-root/HPC_SHARED_DIR/datasets"

REGISTRY = {
    "neu_cls": {
        "root": "/var/hpc-root/xiangxu/work/NEU-CLS",
        "note": "derived from NEU-DET images; SAME 1800 images as neu_surface - do not use both.",
    },
    "gc10det": {
        "root": os.path.join(SRC, "gc10det"),
        "note": "10 classes, 2048x1000 grayscale, imbalance 21x; dirs named 1..10.",
    },
    "magnetic_tile": {
        "root": os.path.join(SRC, "magnetic_tile", "Surface-Defect-Detection-master", "Magnetic-Tile-Defect"),
        "note": "6 MT_* classes; images live under MT_x/Imgs/, so class = parent of Imgs.",
    },
    "casting": {
        "root": os.path.join(SRC, "casting", "casting_data", "casting_data"),
        "note": "binary ok_front/def_front; original train/test ignored, re-split uniformly.",
    },
}


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def label_of(dirpath, root):
    """Take the last non-structural directory name in the path as the class label.

    This is required: magnetic_tile's directories are MT_Free/Imgs/, so taking the last segment
    directly would collapse all 6 classes into one class named "Imgs" (observed in practice).
    """
    rel = os.path.relpath(dirpath, root)
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in (".", "")]
    for p in reversed(parts):
        if p.lower() not in STRUCTURAL:
            return p
    return os.path.basename(dirpath.rstrip("/\\"))


def collect(root):
    items, seen, drops = [], {}, []
    for dirpath, _dirnames, filenames in os.walk(root):
        imgs = [f for f in filenames if os.path.splitext(f)[1].lower() in IMG_EXT]
        if not imgs:
            continue
        lab = label_of(dirpath, root)
        if lab.lower() in STRUCTURAL:
            continue
        for f in sorted(imgs):
            p = os.path.join(dirpath, f)
            m = md5_of(p)
            if m is None:
                drops.append((p, "unreadable"))
                continue
            if m in seen:
                drops.append((p, "dup-of:" + os.path.relpath(seen[m], root)))
                continue
            seen[m] = p
            items.append((p, lab))
    return items, drops


def make_splits(items, seed, ratios):
    by = defaultdict(list)
    for i, (_p, y) in enumerate(items):
        by[y].append(i)
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for y in sorted(by):
        idx = sorted(by[y])
        rng.shuffle(idx)
        n = len(idx)
        a = int(round(n * ratios[0]))
        b = int(round(n * ratios[1]))
        out["train"] += idx[:a]
        out["val"] += idx[a:a + b]
        out["test"] += idx[a + b:]
    for k in out:
        out[k].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/var/hpc-root/HPC_SHARED_DIR/benchmark")
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ratios", default="0.6,0.2,0.2")
    a = ap.parse_args()

    only = set(x.strip() for x in a.only.split(",") if x.strip())
    names = [n for n in REGISTRY if not only or n in only]
    seeds = [int(x) for x in a.seeds.split(",")]
    ratios = tuple(float(x) for x in a.ratios.split(","))
    os.makedirs(os.path.join(a.out, "splits"), exist_ok=True)

    manifest, log = {}, []
    for n in names:
        cfg = REGISTRY[n]
        root = cfg["root"]
        log += ["=" * 74, "%s  <- %s" % (n, root), "  note: %s" % cfg["note"]]
        if not os.path.isdir(root):
            log.append("  SKIP: root not found")
            print("[%s] SKIP (root not found)" % n)
            continue
        items, drops = collect(root)
        # Drop "pseudo-classes" with too few samples: typically wrapper directories
        # (magnetic_tile's Magnetic-Tile-Defect has only 2 images) that would become a degenerate 7th class.
        MIN_CLASS = 20
        _cnt = Counter(y for _p, y in items)
        _tiny = {y for y, c in _cnt.items() if c < MIN_CLASS}
        if _tiny:
            items = [(p, y) for p, y in items if y not in _tiny]
            log.append("  dropped tiny labels (<%d imgs): %s" % (MIN_CLASS, sorted(_tiny)))
        labs = Counter(y for _p, y in items)
        log.append("  kept %d, dropped %d" % (len(items), len(drops)))
        for p, why in drops[:8]:
            log.append("    drop %s (%s)" % (os.path.basename(p), why))
        log.append("  %d classes:" % len(labs))
        for k, v in labs.most_common():
            log.append("    %-24s %d" % (k, v))
        if len(labs) > 1:
            cs = sorted(labs.values())
            log.append("  imbalance = %.1fx (max %d / min %d)" % (cs[-1] / max(1, cs[0]), cs[-1], cs[0]))

        classes = sorted(labs)
        cidx = {c: i for i, c in enumerate(classes)}
        manifest[n] = {
            "root": root, "note": cfg["note"], "classes": classes,
            "counts": {c: labs[c] for c in classes},
            "items": [[os.path.relpath(p, root), cidx[y]] for p, y in items],
        }
        for s in seeds:
            sp = make_splits(items, s, ratios)
            doc = {
                "_meta": {"dataset": n, "root": root, "seed": s,
                          "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
                          "classes": classes, "n": {k: len(v) for k, v in sp.items()},
                          "note": "frozen split, deduped globally by build_dataset.py"},
                "train": [manifest[n]["items"][i][0] for i in sp["train"]],
                "val": [manifest[n]["items"][i][0] for i in sp["val"]],
                "test": [manifest[n]["items"][i][0] for i in sp["test"]],
            }
            with open(os.path.join(a.out, "splits", "%s_seed%d.json" % (n, s)), "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
        log.append("  wrote %d splits" % len(seeds))
        print("[%s] %d imgs / %d classes / %d dropped" % (n, len(items), len(labs), len(drops)))

    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "build_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("wrote %s/manifest.json, splits/, build_report.txt" % a.out)


if __name__ == "__main__":
    main()
