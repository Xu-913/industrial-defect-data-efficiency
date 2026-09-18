"""Paper A single training run: one model x one seed.

Design discipline (matching PROJECT.md / the S3 gate of ai-paper-pipeline):
  * read the **frozen split file**, never split randomly on the fly
  * emit a **complete reproducible record**: config + seed + per-epoch metrics + confusion matrix + environment versions
  * T4 does not support BF16 -> automatically pick FP16+GradScaler; A4000 supports BF16 -> no scaler needed
  * all results go to JSONL/JSON, never scraped from stdout

Usage:
    python train.py --model resnet18 --seed 0 \
        --data /var/hpc-root/xiangxu/data/NEU-CLS \
        --split-file splits/neu_cls_seed0.json \
        --out results/
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model, count_params, MODEL_ZOO          # noqa: E402
from neu_data import CLASSES, build_loaders                      # noqa: E402


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def confusion_matrix(preds, labels, n=len(CLASSES)):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm.tolist()


@torch.no_grad()
def evaluate(model, loader, device, use_bf16):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss = 0.0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(x)
        loss += crit(out.float(), y).item()
        preds += out.argmax(1).cpu().tolist()
        labels += y.cpu().tolist()
    n = len(labels)
    acc = 100.0 * sum(int(a == b) for a, b in zip(preds, labels)) / max(n, 1)
    per_cls = []
    for c in range(len(CLASSES)):
        idx = [i for i, t in enumerate(labels) if t == c]
        per_cls.append(100.0 * sum(int(preds[i] == c) for i in idx) / len(idx) if idx else float("nan"))
    return {"loss": loss / max(n, 1), "acc": acc, "n": n,
            "per_class_acc": per_cls, "confusion": confusion_matrix(preds, labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_ZOO.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="/var/hpc-root/xiangxu/data/NEU-CLS")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--per-class", type=int, default=None,
                    help="take N training images per class (takes priority over --train-frac; low levels do not drift from rounding)")
    ap.add_argument("--mode", default="finetune", choices=["finetune", "frozen"],
                    help="frozen = freeze the backbone and train only the classification head (linear probe)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    run_id = "%s_seed%d%s" % (a.model, a.seed, ("_" + a.tag) if a.tag else "")
    jsonl_path = os.path.join(a.out, run_id + ".jsonl")
    json_path = os.path.join(a.out, run_id + ".json")

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # WARNING: do not use torch.cuda.is_bf16_supported(): since PyTorch 2.5 it defaults to including_emulation=True,
    # which returns True on a T4 (SM 7.5) -- but that is "software-emulated bf16", slow and numerically unlike native bf16.
    # Measured evidence: the 2026-09-16 smoke test printed amp_dtype=bf16 on a T4, which is a misjudgment.
    # Correct approach: check the compute capability explicitly; only SM >= 8.0 (Ampere and newer, e.g. RTX A4000) has native BF16.
    cap_major = torch.cuda.get_device_capability(0)[0] if device.type == "cuda" else 0
    use_bf16 = (device.type == "cuda") and (cap_major >= 8)

    tr, va, te = build_loaders(a.data, a.split_file, img_size=a.img_size,
                               batch_size=a.bs, num_workers=a.workers,
                               train_frac=a.train_frac, seed=a.seed,
                               per_class=a.per_class)
    model = build_model(a.model, num_classes=len(CLASSES),
                        pretrained=not a.no_pretrained)

    if a.mode == "frozen":
        # Linear probe: freeze the whole backbone and train only the classification head.
        # Purpose: separate "are the pretrained features good enough by themselves" from "fine-tuning adaptation ability" -- the key control for H2.
        head_prefix = ("fc.", "classifier.", "head.", "heads.")
        n_tr = n_fr = 0
        for name, p in model.named_parameters():
            if name.startswith(head_prefix):
                p.requires_grad = True
                n_tr += p.numel()
            else:
                p.requires_grad = False
                n_fr += p.numel()
        print("[mode] frozen: trainable=%.2fM frozen=%.2fM" % (n_tr / 1e6, n_fr / 1e6), flush=True)

    model = model.to(device)

    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = max(1, len(tr)) * a.epochs
    warm = max(1, len(tr)) * a.warmup

    def lr_at(it):
        if it < warm:
            return it / warm
        prog = (it - warm) / max(1, steps - warm)
        return 0.5 * (1 + np.cos(np.pi * prog))

    sched = optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not use_bf16))

    env = {
        "run_id": run_id, "model": a.model, "family": MODEL_ZOO[a.model][0],
        "seed": a.seed, "img_size": a.img_size, "epochs": a.epochs, "bs": a.bs,
        "lr": a.lr, "wd": a.wd, "train_frac": a.train_frac,
        "per_class": a.per_class, "mode": a.mode,
        "pretrained": not a.no_pretrained, "split_file": os.path.abspath(a.split_file),
        "n_params": count_params(model),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "amp_dtype": "bf16" if use_bf16 else ("fp16" if device.type == "cuda" else "none"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(), "git": git_hash(),
        "n_train": len(tr.dataset), "n_val": len(va.dataset), "n_test": len(te.dataset),
        "classes": CLASSES, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(env, ensure_ascii=False), flush=True)

    flog = open(jsonl_path, "w", encoding="utf-8")
    flog.write(json.dumps({"type": "env", **env}, ensure_ascii=False) + "\n")

    best_val, best_epoch, best_state = -1.0, -1, None
    t_start = time.time()

    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, seen, correct = 0.0, 0, 0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(x)
                loss = crit(out, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)
            correct += (out.argmax(1) == y).sum().item()

        vm = evaluate(model, va, device, use_bf16)
        rec = {"type": "epoch", "epoch": ep, "epochs": a.epochs,
               "train_loss": run_loss / max(seen, 1),
               "train_acc": 100.0 * correct / max(seen, 1),
               "val_loss": vm["loss"], "val_acc": vm["acc"],
               "lr": opt.param_groups[0]["lr"],
               "sec": round(time.time() - t0, 2)}
        print("ep %3d/%d  train_loss=%.4f train_acc=%.2f  val_loss=%.4f val_acc=%.2f  lr=%.2e  %.1fs"
              % (ep, a.epochs, rec["train_loss"], rec["train_acc"],
                 rec["val_loss"], rec["val_acc"], rec["lr"], rec["sec"]), flush=True)
        flog.write(json.dumps(rec) + "\n")
        flog.flush()

        if vm["acc"] > best_val:
            best_val, best_epoch = vm["acc"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    tm = evaluate(model, te, device, use_bf16)
    final = {"type": "final", "run_id": run_id, "best_val_acc": best_val,
             "best_epoch": best_epoch, "test_acc": tm["acc"], "test_loss": tm["loss"],
             "per_class_acc": dict(zip(CLASSES, tm["per_class_acc"])),
             "confusion": tm["confusion"],
             "total_sec": round(time.time() - t_start, 1), **env}
    flog.write(json.dumps(final, ensure_ascii=False) + "\n")
    flog.close()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=1)

    ckpt = os.path.join(a.out, run_id + ".pt")
    torch.save({"state_dict": model.state_dict(), "env": env}, ckpt)

    print("FINAL %s  test_acc=%.2f  best_val=%.2f@ep%d  %.0fs  ->  %s"
          % (a.model, tm["acc"], best_val, best_epoch, final["total_sec"], json_path), flush=True)
    print("CONFUSION " + json.dumps(tm["confusion"]), flush=True)


if __name__ == "__main__":
    main()
