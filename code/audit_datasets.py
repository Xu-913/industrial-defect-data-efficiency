#!/usr/bin/env python3
"""Generic dataset auditing tool (reusable).

Design goal: for **any** image dataset directory, automatically discover the class structure and produce an audit report for experiment design.

Usage
-----
    python audit_datasets.py                          # audit every dataset under the shared directory
    python audit_datasets.py --root /path/to/dir      # audit the given directory
    python audit_datasets.py --only gc10det           # audit just one of them

Outputs (written to --out, by default alongside --root)
-------
    audit_report.json    machine-readable: per-dataset per-class counts, sizes, duplicates, corruption
    audit_report.csv     one row per "class bucket", convenient for comparison
    audit_summary.txt    human-readable summary

Auto-discovery logic
--------------------
  1. dataset = a first-level subdirectory of --root
  2. "class bucket" = any directory that **directly contains image files**; its name is the relative path
  3. per-bucket statistics: image count, size distribution, MD5 deduplication, corrupt files (undecodable)
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
# Structural directory names: their presence in a path only marks a split/wrapper layer, not a class
SPLIT_WORDS = {"train", "test", "val", "valid", "validation", "images", "imgs",
               "annotations", "labels", "masks", "gt", "groundtruth"}


def find_buckets(ds_dir):
    """Return [(bucket_relpath, [absolute image paths, ...]), ...]. Only directories holding images are kept."""
    out = []
    for dirpath, dirnames, filenames in os.walk(ds_dir):
        imgs = [os.path.join(dirpath, f) for f in filenames
                if os.path.splitext(f)[1].lower() in IMG_EXT]
        if imgs:
            out.append((os.path.relpath(dirpath, ds_dir), sorted(imgs)))
    return sorted(out)


def image_size(path):
    """Read the size from the file header only, without decoding the whole image (much faster). Returns None on failure."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size, im.mode
    except Exception:
        return None


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def class_label(bucket_rel):
    """Derive the "class label" from a bucket's relative path: take the last segment; if it is a structural name, walk upward."""
    parts = [p for p in bucket_rel.replace("\\", "/").split("/") if p not in (".", "")]
    for p in reversed(parts):
        if p.lower() not in SPLIT_WORDS:
            return p
    return parts[-1] if parts else bucket_rel


def audit_dataset(name, ds_dir, do_hash=True, do_size=True):
    buckets = find_buckets(ds_dir)
    rec = {"dataset": name, "path": ds_dir, "buckets": [], "totals": {}}
    md5_seen = defaultdict(list)
    per_class = defaultdict(int)
    size_counter = Counter()
    mode_counter = Counter()
    n_img = n_bad = 0

    for rel, imgs in buckets:
        lab = class_label(rel)
        b = {"bucket": rel, "label": lab, "n": len(imgs), "sizes": {}, "modes": {},
             "n_bad": 0, "n_dup": 0}
        for p in imgs:
            n_img += 1
            if do_size:
                r = image_size(p)
                if r is None:
                    n_bad += 1
                    b["n_bad"] += 1
                    continue
                (w, h), mode = r
                size_counter[(w, h)] += 1
                mode_counter[mode] += 1
                b["sizes"]["%dx%d" % (w, h)] = b["sizes"].get("%dx%d" % (w, h), 0) + 1
                b["modes"][mode] = b["modes"].get(mode, 0) + 1
            if do_hash:
                m = md5_of(p)
                if m:
                    md5_seen[m].append(p)
        per_class[lab] += len(imgs)
        rec["buckets"].append(b)

    dups = {m: v for m, v in md5_seen.items() if len(v) > 1}
    n_dup_extra = sum(len(v) - 1 for v in dups.values())

    # Only meaningful for single-class buckets: aggregate labels up to the dataset level
    lab_count = Counter()
    for b in rec["buckets"]:
        lab_count[b["label"]] += b["n"]
    counts = sorted(lab_count.values(), reverse=True)

    rec["totals"] = {
        "n_buckets": len(buckets),
        "n_images": n_img,
        "n_corrupt": n_bad,
        "n_dup_groups": len(dups),
        "n_dup_extra": n_dup_extra,
        "n_labels": len(lab_count),
        "top_sizes": [["%dx%d" % k, v] for k, v in size_counter.most_common(6)],
        "modes": dict(mode_counter),
        "label_counts": dict(sorted(lab_count.items(), key=lambda x: -x[1])),
        "imbalance_ratio": (counts[0] / counts[-1]) if len(counts) > 1 and counts[-1] else None,
        "min_class": counts[-1] if counts else 0,
        "max_class": counts[0] if counts else 0,
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/var/hpc-root/HPC_SHARED_DIR/datasets")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--no-hash", action="store_true", help="skip deduplication (much faster on large datasets)")
    a = ap.parse_args()

    out = a.out or a.root
    os.makedirs(out, exist_ok=True)
    only = set(x.strip() for x in a.only.split(",") if x.strip())

    datasets = sorted(d for d in os.listdir(a.root)
                      if os.path.isdir(os.path.join(a.root, d)) and not d.startswith(".")
                      and (not only or d in only))
    print("datasets found: %s\n" % ", ".join(datasets))

    reports = []
    for d in datasets:
        print(">>> auditing %s ..." % d, flush=True)
        try:
            r = audit_dataset(d, os.path.join(a.root, d), do_hash=not a.no_hash)
        except Exception as e:
            print("    failed: %r" % e)
            continue
        t = r["totals"]
        print("    buckets=%d  images=%d  labels=%d  corrupt=%d  dup_groups=%d  imbalance=%s"
              % (t["n_buckets"], t["n_images"], t["n_labels"], t["n_corrupt"],
                 t["n_dup_groups"],
                 ("%.1fx" % t["imbalance_ratio"]) if t["imbalance_ratio"] else "-"))
        reports.append(r)

    jp = os.path.join(out, "audit_report.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=1)

    cp = os.path.join(out, "audit_report.csv")
    with open(cp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "bucket", "label", "n_images", "n_bad",
                    "top_size", "top_mode"])
        for r in reports:
            for b in r["buckets"]:
                ts = max(b["sizes"].items(), key=lambda x: x[1])[0] if b["sizes"] else ""
                tm = max(b["modes"].items(), key=lambda x: x[1])[0] if b["modes"] else ""
                w.writerow([r["dataset"], b["bucket"], b["label"], b["n"], b["n_bad"], ts, tm])

    sp = os.path.join(out, "audit_summary.txt")
    with open(sp, "w", encoding="utf-8") as f:
        for r in reports:
            t = r["totals"]
            f.write("=" * 70 + "\n%s  (%s)\n" % (r["dataset"], r["path"]))
            f.write("  total images %d   class buckets %d   labels %d\n" % (t["n_images"], t["n_buckets"], t["n_labels"]))
            f.write("  corrupt %d   duplicate groups %d (%d extra images)\n" % (t["n_corrupt"], t["n_dup_groups"], t["n_dup_extra"]))
            f.write("  sizes: %s\n" % ", ".join("%s x %d" % (k, v) for k, v in t["top_sizes"]))
            f.write("  channels: %s\n" % t["modes"])
            if t["imbalance_ratio"]:
                f.write("  imbalance: max %d / min %d = %.1f x\n"
                        % (t["max_class"], t["min_class"], t["imbalance_ratio"]))
            f.write("  per-label counts:\n")
            for k, v in t["label_counts"].items():
                f.write("    %-26s %d\n" % (k, v))
            f.write("\n")

    print("\nwrote:\n  %s\n  %s\n  %s" % (jp, cp, sp))


if __name__ == "__main__":
    main()
