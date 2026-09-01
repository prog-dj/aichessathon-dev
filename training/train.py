"""Supervised training of the policy + value net. Runs on the GPU box (Kaggle).

    python -m training.prepare eval --input ... --out data/shards/eval --limit 8000000
    python -m training.prepare pgn  --input ... --out data/shards/games --limit 6000000
    python -m training.train --shards data/shards/eval data/shards/games \
        --config medium --epochs 3 --batch 2048 --out weights

Each epoch writes ``weights/model.e{n}.pt`` and exports ``weights/model.onnx``.
If ``GH_TOKEN`` is set in the environment, every export is also force-pushed to
the ``trained-net`` branch, so a lost Kaggle session never loses the model.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, random_split

from training.data import ShardDataset
from training.export import export_onnx
from training.model import CONFIGS, ChessNet, parameter_count

_REPO = "github.com/prog-dj/aichessathon-dev.git"


def _push_weights(onnx_path: Path, note: str) -> None:
    token = os.environ.get("GH_TOKEN")
    if not token:
        return

    def run(*args: str, check: bool = True) -> None:
        subprocess.run(args, check=check, capture_output=True, text=True)

    try:
        run("git", "config", "user.email", "kaggle@bot")
        run("git", "config", "user.name", "kaggle")
        run("git", "add", "-f", str(onnx_path))
        run("git", "commit", "-m", f"trained net: {note}", check=False)
        run("git", "push", "-f", f"https://{token}@{_REPO}", "HEAD:trained-net")
        print(f"pushed {onnx_path.name} to trained-net")
    except subprocess.CalledProcessError as error:
        print(f"weights push failed: {error.stderr or error}")


def _soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _load_shards(
    paths: list[Path], value_scale: float | None = None
) -> ConcatDataset[tuple[torch.Tensor, int, torch.Tensor]]:
    shards: list[ShardDataset] = []
    for parent in paths:
        children = sorted(p for p in parent.glob("[0-9]" * 4) if p.is_dir())
        shards.extend(ShardDataset(c, value_scale) for c in (children or [parent]))
    if not shards:
        raise SystemExit(f"no shards under {paths}")
    return ConcatDataset(shards)


def _evaluate(model: ChessNet, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    policy_hits = value_hits = seen = 0
    with torch.no_grad():
        for planes, policy_target, wdl_target in loader:
            planes = planes.to(device)
            policy_logits, value_logits = model(planes)
            policy_hits += int(
                (policy_logits.argmax(1).cpu() == policy_target).sum().item()
            )
            value_hits += int(
                (value_logits.argmax(1).cpu() == wdl_target.argmax(1)).sum().item()
            )
            seen += planes.shape[0]
    return policy_hits / seen, value_hits / seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="medium")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument(
        "--value-scale",
        type=float,
        help="recompute WDL targets from the stored centipawns at this logistic scale",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("weights"))
    parser.add_argument(
        "--init-from", type=Path, help="warm-start weights from a .pt (LR schedule restarts)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _load_shards(args.shards, args.value_scale)
    val_len = max(1, int(len(dataset) * args.val_frac))
    train_set, val_set = random_split(
        dataset, [len(dataset) - val_len, val_len], generator=torch.Generator().manual_seed(0)
    )
    common = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)

    model = ChessNet(CONFIGS[args.config]).to(device)
    if args.init_from:
        blob = torch.load(args.init_from, map_location=device)
        model.load_state_dict(blob["state_dict"])
        print(f"warm-started from {args.init_from}")
    print(f"{args.config}: {parameter_count(model) / 1e6:.2f}M params, {len(dataset):,} samples")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    policy_loss_fn = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        for step, (planes, policy_target, wdl_target) in enumerate(train_loader):
            planes = planes.to(device, non_blocking=True)
            policy_target = policy_target.to(device, non_blocking=True)
            wdl_target = wdl_target.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                policy_logits, value_logits = model(planes)
                loss = policy_loss_fn(policy_logits, policy_target) + args.value_weight * (
                    _soft_cross_entropy(value_logits, wdl_target)
                )
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 200 == 0:
                rate = (step + 1) * args.batch / (time.time() - started)
                print(f"epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running / (step + 1):.3f} {rate:,.0f} pos/s")

        policy_acc, value_acc = _evaluate(model, val_loader, device)
        print(f"epoch {epoch}: val policy top1 {policy_acc:.3f}  value acc {value_acc:.3f}")
        torch.save(
            {"config": args.config, "state_dict": model.state_dict()},
            args.out / f"model.e{epoch}.pt",
        )
        try:
            export_onnx(model, args.out / "model.onnx")
            print(f"saved model.e{epoch}.pt and model.onnx")
            _push_weights(
                args.out / "model.onnx",
                f"{args.config} e{epoch} p{policy_acc:.3f} v{value_acc:.3f}",
            )
        except Exception as error:  # a bad export must not kill the run
            print(f"epoch {epoch}: ONNX export failed ({error}); checkpoint is still saved")


if __name__ == "__main__":
    main()
