"""Dataset-agnostic training script (reads the shared benchmark manifest, works on any dataset).

Difference from the old train.py: it is no longer tied to NEU-CLS's directory conventions and
instead reads `/var/hpc-root/HPC_SHARED_DIR/benchmark/manifest.json` + `splits/<ds>_seed<k>.json`.

Usage
-----
    python -u train_bench.py --dataset gc10det --seed 0 --per-class 5 \
        --img-size 256 --epochs 50 --out /path/results
"""

import argparse, json, os, platform, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model, count_params, MODEL_ZOO   # noqa: E402

BENCH = "/var/hpc-root/HPC_SHARED_DIR/benchmark"
_MAN = None


def manifest():
    global _MAN
    if _MAN is None:
        with open(os.path.join(BENCH, "manifest.json"), encoding="utf-8") as f:
            _MAN = json.load(f)
    return _MAN


def split_doc(dataset, seed):
    p = os.path.join(BENCH, "splits", "%s_seed%d.json" % (dataset, seed))
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def make_loaders(dataset, seed, per_class, img_size, bs, workers, train_frac=1.0):
    man = manifest()[dataset]
    root, classes = man["root"], man["classes"]
    # relative path -> label
    lab = {rp: y for rp, y in man["items"]}
    sp = split_doc(dataset, seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]

    if per_class is not None:
        import random
        by = {}
        for rp in tr:
            by.setdefault(lab[rp], []).append(rp)
        rng = random.Random(seed)
        keep = []
        for y in sorted(by):
            it = sorted(by[y]); rng.shuffle(it)
            keep += it[:min(per_class, len(it))]
        tr = sorted(keep)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def tf(train):
        ops = [transforms.Grayscale(num_output_channels=3), transforms.Resize((img_size, img_size))]
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(mean, std)])

    class DS(Dataset):
        def __init__(self, rel, training):
            self.rel, self.t = rel, tf(training)
        def __len__(self):
            return len(self.rel)
        def __getitem__(self, i):
            rp = self.rel[i]
            with Image.open(os.path.join(root, rp)) as im:
                x = self.t(im.convert("L"))
            return x, lab[rp]

    mk = lambda rel, tr_: DataLoader(DS(rel, tr_), batch_size=bs, shuffle=tr_,
                                     num_workers=workers, pin_memory=True)
    return mk(tr, True), mk(va, False), mk(te, False), classes, len(tr)


def confusion(preds, labels, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm.tolist()


@torch.no_grad()
def evaluate(model, loader, device, use_bf16):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss = 0.0; preds = []; labels = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(x)
        loss += crit(out.float(), y).item()
        preds += out.argmax(1).cpu().tolist(); labels += y.cpu().tolist()
    n = len(labels); ncls = out.shape[1]
    acc = 100.0 * sum(int(a == b) for a, b in zip(preds, labels)) / max(n, 1)
    per = []
    for c in range(ncls):
        idx = [i for i, t in enumerate(labels) if t == c]
        per.append(100.0 * sum(int(preds[i] == c) for i in idx) / len(idx) if idx else float("nan"))
    return {"loss": loss / max(n, 1), "acc": acc, "n": n, "per_class_acc": per,
            "confusion": confusion(preds, labels, ncls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=list(MODEL_ZOO.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rid = "%s_%s_seed%d_n%s_res%d%s" % (a.dataset, a.model, a.seed,
                                        a.per_class if a.per_class else "full",
                                        a.img_size, ("_" + a.tag) if a.tag else "")
    jp = os.path.join(a.out, rid + ".json")
    if os.path.exists(jp):
        print("SKIP exists %s" % jp, flush=True); return

    import random
    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = torch.cuda.get_device_capability(0)[0] if device.type == "cuda" else 0
    use_bf16 = device.type == "cuda" and cap >= 8

    tr, va, te, classes, ntr = make_loaders(a.dataset, a.seed, a.per_class,
                                            a.img_size, a.bs, a.workers)
    model = build_model(a.model, num_classes=len(classes), pretrained=True).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = max(1, len(tr)) * a.epochs; warm = max(1, len(tr)) * a.warmup
    sched = optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it / warm) if it < warm else
        0.5 * (1 + np.cos(np.pi * (it - warm) / max(1, steps - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not use_bf16))

    env = {"run_id": rid, "dataset": a.dataset, "model": a.model, "seed": a.seed,
           "per_class": a.per_class, "img_size": a.img_size, "epochs": a.epochs,
           "bs": a.bs, "lr": a.lr, "wd": a.wd, "n_train": ntr, "n_classes": len(classes),
           "classes": classes, "n_params": count_params(model),
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "amp_dtype": "bf16" if use_bf16 else ("fp16" if device.type == "cuda" else "none"),
           "torch": torch.__version__, "python": platform.python_version(),
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps({k: env[k] for k in ("run_id", "n_train", "n_classes", "amp_dtype")}), flush=True)

    best, bep, bstate = -1.0, -1, None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = crit(model(x), y)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step()
        vm = evaluate(model, va, device, use_bf16)
        if vm["acc"] > best:
            best, bep = vm["acc"], ep
            bstate = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if bstate:
        model.load_state_dict(bstate)
    tm = evaluate(model, te, device, use_bf16)
    final = {"type": "final", **env, "best_val_acc": best, "best_epoch": bep,
             "test_acc": tm["acc"], "test_loss": tm["loss"],
             "per_class_acc": dict(zip(classes, tm["per_class_acc"])),
             "confusion": tm["confusion"], "total_sec": round(time.time() - t0, 1)}
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False)
    print("FINAL %s test_acc=%.2f best_val=%.2f@ep%d %.0fs" %
          (rid, tm["acc"], best, bep, final["total_sec"]), flush=True)


if __name__ == "__main__":
    main()
