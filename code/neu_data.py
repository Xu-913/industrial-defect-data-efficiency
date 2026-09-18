"""NEU-CLS data loading + frozen splits (the S0 infrastructure of Paper A).

Core discipline (matching PROJECT.md):
  1. **Splits must be frozen and persisted**. The split file on disk is the single source of truth;
     every model / seed / Paper C deployment experiment reads the same one, or numbers are not comparable.
  2. **Splits must be stratified**: NEU-CLS is strictly balanced at 6 classes x 300, and
     even a random split must preserve the per-class ratio.
  3. **Self-check at load time**: if counts / sizes deviate from the official spec, raise and stop, never continue silently.

Official specification (see the project dataset notes):
    1800 images, 6 classes x 300 each, 200x200 grayscale

Usage:
    python neu_data.py --data /var/hpc-root/xiangxu/data/NEU-CLS --make-splits
    python neu_data.py --data ... --check
"""

import argparse
import json
import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Fixed class order (alphabetical), uniform across the project, must not follow directory scan order
CLASSES = ["crazing", "inclusion", "patches",
           "pitted_surface", "rolled_in_scale", "scratches"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Directory naming for NEU-CLS differs across mirrors; unify the mapping
CLASS_ALIASES = {
    "crazing": "crazing", "cr": "crazing",
    "inclusion": "inclusion", "in": "inclusion",
    "patches": "patches", "patch": "patches", "pa": "patches",
    "pitted_surface": "pitted_surface", "pitted surface": "pitted_surface",
    "pitted": "pitted_surface", "ps": "pitted_surface",
    "rolled-in_scale": "rolled_in_scale", "rolled_in_scale": "rolled_in_scale",
    "rolled in scale": "rolled_in_scale", "rolledinscale": "rolled_in_scale",
    "rs": "rolled_in_scale",
    "scratches": "scratches", "scratch": "scratches", "sc": "scratches",
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
OFFICIAL_PER_CLASS = 300
OFFICIAL_SIZE = (200, 200)


# --------------------------------------------------------------------------
# Scanning and self-check
# --------------------------------------------------------------------------
def scan(root):
    """Scan the class directories under root and return {canonical_class: [relative paths, ...]}.

    Unrecognized class directories raise an explicit error rather than being ignored -- silently ignoring them would leave the class count wrong.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError("data root directory does not exist: %s" % root)

    buckets = defaultdict(list)
    unknown = []

    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if not os.path.isdir(d):
            continue
        key = CLASS_ALIASES.get(entry.strip().lower())
        if key is None:
            unknown.append(entry)
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMG_EXTS):
                buckets[key].append(os.path.join(entry, fn))

    if unknown:
        raise ValueError(
            "unrecognized class directories found: %s\n(known aliases are listed in CLASS_ALIASES; if the naming is genuinely new, add the mapping first)"
            % unknown)
    missing = [c for c in CLASSES if c not in buckets]
    if missing:
        raise ValueError("missing these class directories: %s\nfound instead: %s" % (missing, sorted(buckets)))
    return buckets


def check_dataset(root, strict=True):
    """Self-check against the official spec and return the statistics. With strict=True a mismatch raises."""
    buckets = scan(root)
    counts = {c: len(v) for c, v in buckets.items()}
    total = sum(counts.values())

    sizes = set()
    for c in CLASSES:
        for rel in buckets[c][:5]:                       # sample 5 images per class
            with Image.open(os.path.join(root, rel)) as im:
                sizes.add(im.size)

    stats = {"total": total, "counts": counts, "sampled_sizes": sorted(sizes)}
    problems = []
    if total != 1800:
        problems.append("total %d != official 1800" % total)
    for c in CLASSES:
        if counts[c] != OFFICIAL_PER_CLASS:
            problems.append("%s has %d images != official %d" % (c, counts[c], OFFICIAL_PER_CLASS))
    if sizes != {OFFICIAL_SIZE}:
        problems.append("sampled non-200x200 sizes: %s" % sorted(sizes))

    stats["problems"] = problems
    if problems and strict:
        raise ValueError("dataset does not match the official spec, stopping:\n  - " + "\n  - ".join(problems))
    return stats


# --------------------------------------------------------------------------
# Frozen splits
# --------------------------------------------------------------------------
def make_split(root, seed, ratios=(0.6, 0.2, 0.2)):
    """Stratified split, returning {'train':[...], 'val':[...], 'test':[...]} (relative paths).

    Uses a standalone Random instance with a fixed seed, decoupled from the global random state and reproducible.
    """
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"
    buckets = scan(root)
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}

    for c in CLASSES:
        items = list(buckets[c])
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        out["train"] += items[:n_tr]
        out["val"] += items[n_tr:n_tr + n_va]
        out["test"] += items[n_tr + n_va:]

    for k in out:
        out[k].sort()
    return out


def save_splits(root, out_dir, seeds=(0, 1, 2), ratios=(0.6, 0.2, 0.2)):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for s in seeds:
        sp = make_split(root, s, ratios)
        sp["_meta"] = {
            "dataset": "NEU-CLS", "root": root, "seed": s,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "classes": CLASSES,
            "n": {k: len(v) for k, v in sp.items() if isinstance(v, list)},
            "note": "Frozen split, the single source of truth for the whole project. Changing this file invalidates every reported number.",
        }
        p = os.path.join(out_dir, "neu_cls_seed%d.json" % s)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(sp, f, ensure_ascii=False, indent=1)
        written.append((p, sp["_meta"]["n"]))
    return written


def load_split(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def build_transforms(img_size=224, train=True):
    """Grayscale -> 3 channels -> resize -> normalize.

    NEU-CLS is 200x200 single-channel; ImageNet-pretrained models need 3 channels and 224 input.
    Normalization uses ImageNet statistics (because the backbone is ImageNet-pretrained).
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),          # defect orientation is irrelevant, so a vertical flip is equally valid
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class NEUCLS(Dataset):
    """Load by the list of relative paths stored in the split file."""

    def __init__(self, root, rel_paths, transform=None):
        self.root = root
        self.rel = list(rel_paths)
        self.tf = transform
        self.labels = [CLASS_TO_IDX[os.path.basename(os.path.dirname(p))] if
                       os.path.basename(os.path.dirname(p)) in CLASS_TO_IDX
                       else CLASS_TO_IDX[CLASS_ALIASES[os.path.basename(os.path.dirname(p)).strip().lower()]]
                       for p in self.rel]

    def __len__(self):
        return len(self.rel)

    def __getitem__(self, i):
        p = os.path.join(self.root, self.rel[i])
        with Image.open(p) as im:
            im = im.convert("L")            # force a single channel in case a mirror stores RGB
            x = self.tf(im) if self.tf else transforms.ToTensor()(im)
        return x, self.labels[i]


def build_loaders(root, split_file, img_size=224, batch_size=64,
                  num_workers=4, train_frac=1.0, seed=0, per_class=None):
    """Build the train/val/test DataLoaders.

    The training set can be reduced in two ways (per_class takes priority):
      * per_class=N  -- take exactly N images per class (**recommended**: low levels avoid rounding drift)
      * train_frac=p -- take a fraction (backward compatible)
    """
    sp = load_split(split_file)
    tr = sp["train"]

    if per_class is not None or train_frac < 1.0:
        by_cls = defaultdict(list)
        for p in tr:
            by_cls[CLASS_ALIASES[os.path.basename(os.path.dirname(p)).strip().lower()]].append(p)
        rng = random.Random(seed)
        keep = []
        for c in CLASSES:
            items = sorted(by_cls[c])
            rng.shuffle(items)
            if per_class is not None:
                k = min(int(per_class), len(items))
            else:
                k = max(1, int(round(len(items) * train_frac)))
            keep += items[:k]
        tr = sorted(keep)

    mk = lambda rel, is_tr: DataLoader(
        NEUCLS(root, rel, build_transforms(img_size, is_tr)),
        batch_size=batch_size, shuffle=is_tr, num_workers=num_workers,
        pin_memory=True, drop_last=False)
    return mk(tr, True), mk(sp["val"], False), mk(sp["test"], False)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/var/hpc-root/xiangxu/data/NEU-CLS")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--make-splits", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()

    if a.check or a.make_splits:
        st = check_dataset(a.data, strict=False)
        print("total = %d" % st["total"])
        for c in CLASSES:
            print("  %-16s %d" % (c, st["counts"][c]))
        print("sampled sizes = %s" % st["sampled_sizes"])
        if st["problems"]:
            print("!! does not match the official spec:")
            for p in st["problems"]:
                print("   - " + p)
        else:
            print("OK: fully matches the official spec (1800 / 300x6 / 200x200)")

    if a.make_splits:
        seeds = tuple(int(s) for s in a.seeds.split(","))
        for p, n in save_splits(a.data, a.out, seeds=seeds):
            print("wrote %s  %s" % (p, n))
