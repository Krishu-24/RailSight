"""Fine-tune a segmentation model on RailSem19 and evaluate it.

    python scripts/train/train_seg.py --arch segformer-b2
    python scripts/train/train_seg.py --arch deeplab-r50
    python scripts/train/train_seg.py --arch segformer-b2 --eval-only --split test
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from railsight.datasets import RailSem19
from railsight.seg import ARCHS, IGNORE_INDEX, NUM_CLASSES, RS19_CLASSES, TRACK_IDS, build_model


def param_groups(net, arch, lr):
    """Pretrained backbone gets the base LR, the freshly initialised head 10x."""
    inner = net.model
    head = inner.decode_head if arch.startswith("segformer") else inner.classifier
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in inner.parameters() if id(p) not in head_ids]
    return [{"params": backbone, "lr": lr}, {"params": list(head.parameters()), "lr": lr * 10}]


@torch.inference_mode()
def evaluate(net, loader, device):
    net.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
    track = torch.zeros(2, 2, dtype=torch.int64, device=device)  # binary track vs not
    track_lut = torch.zeros(256, dtype=torch.int64, device=device)
    track_lut[TRACK_IDS] = 1
    t0, n = time.time(), 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            pred = net(x).argmax(1)
        valid = y != IGNORE_INDEX
        conf += torch.bincount(y[valid] * NUM_CLASSES + pred[valid],
                               minlength=NUM_CLASSES**2).view(NUM_CLASSES, NUM_CLASSES)
        yt, pt = track_lut[y[valid]], track_lut[pred[valid]]
        track += torch.bincount(yt * 2 + pt, minlength=4).view(2, 2)
        n += x.shape[0]
    fps = n / (time.time() - t0)

    def ious(c):
        c = c.double()
        return (c.diag() / (c.sum(0) + c.sum(1) - c.diag()).clamp(min=1)).cpu().numpy()

    iou = ious(conf)
    present = conf.sum(1).cpu().numpy() > 0
    net.train()
    return {
        "mIoU": float(iou[present].mean()),
        "pixel_acc": float(conf.diag().sum() / conf.sum()),
        "track_IoU": float(ious(track)[1]),
        "per_class_IoU": {RS19_CLASSES[i]: float(iou[i]) for i in range(NUM_CLASSES) if present[i]},
        "eval_fps": fps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHS, default="segformer-b2")
    ap.add_argument("--name", default=None)
    ap.add_argument("--data", default="data/processed/rs19")
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=640)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from runs/seg/<name>/last.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-val", type=int, default=None, help="evaluate on a subset (smoke tests)")
    ap.add_argument("--device", default="cuda", help="cpu for --eval-only while the GPU trains")
    args = ap.parse_args()

    name = args.name or args.arch
    out = Path("runs/seg") / name
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    torch.backends.cudnn.benchmark = True

    val_ds = RailSem19(args.data, args.split if args.eval_only else "val", train=False)
    if args.max_val:
        val_ds.ids = val_ds.ids[: args.max_val]
    # full-resolution eval one image at a time: larger batches overflow an 8 GB GPU while the
    # training state is still resident
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=args.workers)

    if args.eval_only:
        net = build_model(args.arch, weights=str(out / "best")).to(device)
        metrics = evaluate(net, val_dl, device)
        with open(out / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_class_IoU"}, indent=2))
        return

    train_ds = RailSem19(args.data, "train", crop=args.crop, train=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                          num_workers=args.workers, pin_memory=True, persistent_workers=True)

    net = build_model(args.arch).to(device).train()
    lr = args.lr or (6e-5 if args.arch.startswith("segformer") else 1e-4)
    opt = torch.optim.AdamW(param_groups(net, args.arch, lr), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / 500) * (1 - it / args.iters) ** 0.9
    )
    tb = SummaryWriter(out / "tb")
    with open(out / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best, it, history = -1.0, 0, []
    ckpt = out / "last.pt"
    if args.resume and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        best, it, history = state["best"], state["iter"], state["history"]
        print(f"resumed from iteration {it} (best val mIoU {best:.4f})")
    pbar = tqdm(total=args.iters, initial=it, desc=name)
    while it < args.iters:
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x)
            loss = F.cross_entropy(logits.float(), y, ignore_index=IGNORE_INDEX)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            pbar.update()
            if pbar.n - pbar.initial == 10:
                torch.cuda.synchronize()
                t_probe = time.time()
            elif pbar.n - pbar.initial == 30:
                torch.cuda.synchronize()
                sec_per_it = (time.time() - t_probe) / 20
                if sec_per_it > 3.0:
                    # On Windows the driver silently spills to system RAM when VRAM runs out,
                    # making training ~100x slower instead of raising an out-of-memory error.
                    raise SystemExit(f"{sec_per_it:.1f} s/it: GPU memory is spilling into shared "
                                     "memory. Lower --batch or --crop.")
            if it % 50 == 0:
                tb.add_scalar("train/loss", loss.item(), it)
                tb.add_scalar("train/lr", sched.get_last_lr()[0], it)
                pbar.set_postfix(loss=f"{loss.item():.3f}")
            if it % args.eval_every == 0 or it == args.iters:
                del x, y, logits, loss
                torch.cuda.empty_cache()
                m = evaluate(net, val_dl, device)
                torch.cuda.empty_cache()
                history.append({"iter": it, **{k: m[k] for k in ("mIoU", "track_IoU")}})
                tb.add_scalar("val/mIoU", m["mIoU"], it)
                tb.add_scalar("val/track_IoU", m["track_IoU"], it)
                tqdm.write(f"[{it}] val mIoU={m['mIoU']:.4f} track_IoU={m['track_IoU']:.4f}")
                if m["mIoU"] > best:
                    best = m["mIoU"]
                    net.save(out / "best")
                    with open(out / "metrics_val.json", "w", encoding="utf-8") as f:
                        json.dump(m, f, indent=2)
                torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "iter": it, "best": best,
                            "history": history}, ckpt)
            if it >= args.iters:
                break
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"done. best val mIoU={best:.4f}; weights in {out / 'best'}")


if __name__ == "__main__":
    main()
