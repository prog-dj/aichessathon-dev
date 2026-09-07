"""Train a tapered *linear* eval ("PSQT-net") on the nnue.prepare output.

  python -m nnue.train_psqt --data /kaggle/tmp/nd --epochs 10 --branch nnue-psqt

This is option 2 from the from-scratch rethink: a big linear model over the
king-bucketed HalfKA features (24576 -> [mg, eg]), phase-blended exactly like
the hand eval.  No hidden layer, no matmul -> at runtime the incremental
accumulator *is* the eval, cost ~= HCE.  Captures learned piece values
conditioned on king square (most of what full HalfKA buys) without the
1.5-2 ply speed hole the dense head costs.

Shares everything with train_t4: same data, same sigmoid win-prob loss, same
70/30 band/decisive resample, same in-band MAE/r headline metric.  Safe to run
as a second training loop in the same notebook right after train_t4 - it
reads the identical memmaps.

If GH_TOKEN is set, force-pushes weights/nnue_psqt.npz + nnue/psqt_log.json to
--branch after every epoch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnue.features import N_FEATURES

SCALE = 100.0                                  # cp = model_output * SCALE
WDL_DIV = 173.72 / SCALE                       # model units -> logistic win prob
# PeSTO material, own minus opp, model units - seeds the embedding so it starts
# as a sane material eval and only learns positional corrections from there.
_MG = [82, 337, 365, 477, 1025, 0]            # P N B R Q K
_EG = [94, 281, 297, 512, 936, 0]
_PHW = np.array([0, 1, 1, 2, 4, 0], np.float32)   # phase weight per piece type
_PH_MAX = 24.0

REPO = "github.com/prog-dj/aichessathon-dev.git"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "weights", "nnue_psqt.npz")
LOG = os.path.join(HERE, "psqt_log.json")


class PSQTNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(N_FEATURES + 1, 2, padding_idx=N_FEATURES)
        self._seed()

    def _seed(self) -> None:
        # halved: a piece is +v in its own perspective and -v in the other, and
        # forward is (white-persp - black-persp), so it lands as 2v in the output
        # - seed at v/2 so the material eval starts calibrated, not 2x hot.
        w = np.zeros((N_FEATURES + 1, 2), np.float32)
        for kb in range(32):
            base = kb * 768
            for pt in range(6):
                for sq in range(64):
                    w[base + (pt * 2 + 0) * 64 + sq] = (_MG[pt] / SCALE / 2, _EG[pt] / SCALE / 2)
                    w[base + (pt * 2 + 1) * 64 + sq] = (-_MG[pt] / SCALE / 2, -_EG[pt] / SCALE / 2)
        with torch.no_grad():
            self.emb.weight.copy_(torch.from_numpy(w))
            self.emb.weight[N_FEATURES].zero_()

    def forward(self, fw, fb, wtm, phase):
        d = self.emb(fw).sum(1) - self.emb(fb).sum(1)          # [B, 2] white POV
        wpov = (d[:, 0] * phase + d[:, 1] * (_PH_MAX - phase)) / _PH_MAX
        return torch.where(wtm, wpov, -wpov)                    # [B] stm POV


def _phase_from_feat_w(feat_w: np.ndarray, n: int, chunk: int = 2_000_000) -> np.ndarray:
    """Total game phase per row, decoded from the white-perspective feature idx.
    idx = kb*768 + (pt*2 + own)*64 + sq  ->  pt = (idx % 768) // 64 // 2."""
    out = np.zeros(n, np.float32)
    for i in range(0, n, chunk):
        block = np.asarray(feat_w[i:i + chunk]).astype(np.int64)
        valid = block < N_FEATURES
        pt = (block % 768) // 64 // 2
        w = np.where(valid, _PHW[np.clip(pt, 0, 5)], 0.0)
        out[i:i + block.shape[0]] = np.minimum(w.sum(1), _PH_MAX)
    return out


def export(model: PSQTNet) -> None:
    model.eval()
    w = model.state_dict()["emb.weight"][:N_FEATURES].cpu().numpy().astype(np.float32)
    q = float(32000.0 / max(float(np.abs(w).max()), 1e-6))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(
        OUT,
        w_mg=np.round(w[:, 0] * q).astype(np.int16),   # [24576]
        w_eg=np.round(w[:, 1] * q).astype(np.int16),
        q=np.float32(q),                               # int16 units per model unit
        scale=np.float32(SCALE),                       # cp = (blend / q) * scale
        ph_max=np.float32(_PH_MAX),
    )
    z = np.load(OUT)
    assert z["w_mg"].shape == (N_FEATURES,) and z["w_mg"].dtype == np.int16


def push(branch: str, note: str) -> None:
    token = os.environ.get("GH_TOKEN")
    if not token:
        return
    def run(*a, check=True):
        return subprocess.run(a, cwd=ROOT, check=check, capture_output=True, text=True)
    try:
        run("git", "config", "user.email", "t4@bot"); run("git", "config", "user.name", "t4")
        run("git", "add", "-f", "weights/nnue_psqt.npz", "nnue/psqt_log.json")
        run("git", "commit", "-m", f"nnue-psqt: {note}", check=False)
        run("git", "push", "-f", f"https://{token}@{REPO}", f"HEAD:{branch}")
        print(f"  pushed -> {branch}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"  push failed: {e.stderr or e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--branch", default="nnue-psqt")
    ap.add_argument("--max_lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    d = args.data
    try:
        n = int(json.load(open(os.path.join(d, "meta.json")))["n"])
    except Exception:
        n = None
    feat_w = np.load(os.path.join(d, "feat_w.npy"), mmap_mode="r")
    feat_b = np.load(os.path.join(d, "feat_b.npy"), mmap_mode="r")
    cp = np.asarray(np.load(os.path.join(d, "cp.npy"), mmap_mode="r")).astype(np.float32)
    wtm = np.asarray(np.load(os.path.join(d, "wtm.npy"), mmap_mode="r")).copy()
    if n is None:
        n = len(cp)
    feat_w, feat_b, cp, wtm = feat_w[:n], feat_b[:n], cp[:n], wtm[:n]
    cp_sign = np.where(wtm, 1.0, -1.0).astype(np.float32)
    target = (cp * cp_sign / SCALE).astype(np.float32)
    print(f"{n:,} positions   band(<=300cp) {np.mean(np.abs(cp) <= 300):.1%}")

    t = time.time()
    phase = _phase_from_feat_w(feat_w, n)
    print(f"phase decoded  mean {phase.mean():.1f}  ({time.time()-t:.0f}s)")

    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    n_val = max(1, n // 50)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    band = np.abs(target) <= 300.0 / SCALE
    tr_band, tr_dec = train_idx[band[train_idx]], train_idx[~band[train_idx]]

    y_all = torch.from_numpy(target)
    wtm_t = torch.from_numpy(wtm)
    ph_t = torch.from_numpy(phase)

    def tensors(rows):
        fw = torch.from_numpy(np.asarray(feat_w[rows]).astype(np.int64)).to(dev, non_blocking=True)
        fb = torch.from_numpy(np.asarray(feat_b[rows]).astype(np.int64)).to(dev, non_blocking=True)
        return (fw, fb, wtm_t[rows].to(dev), ph_t[rows].to(dev), y_all[rows].to(dev))

    model = PSQTNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
    steps = max(1, len(train_idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.max_lr, epochs=args.epochs, steps_per_epoch=steps)

    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        nb = int(len(train_idx) * 0.70)
        sb = rng.choice(tr_band, nb, replace=len(tr_band) < nb)
        sd = rng.choice(tr_dec, len(train_idx) - nb, replace=len(tr_dec) < len(train_idx) - nb)
        perm = rng.permutation(np.concatenate([sb, sd]))
        run_loss = k = 0
        t = time.time()
        for i in range(0, len(perm) - args.batch, args.batch):
            fw, fb, w, ph, y = tensors(perm[i:i + args.batch])
            opt.zero_grad(set_to_none=True)
            pred = model(fw, fb, w, ph)
            loss = F.mse_loss(torch.sigmoid(pred / WDL_DIV), torch.sigmoid(y / WDL_DIV))
            loss.backward(); opt.step(); sched.step()
            run_loss += loss.item(); k += 1

        model.eval()
        with torch.no_grad():
            vp = torch.empty(len(val_idx))
            for j in range(0, len(val_idx), args.batch):
                vr = val_idx[j:j + args.batch]
                fw, fb, w, ph, _ = tensors(vr)
                vp[j:j + len(vr)] = model(fw, fb, w, ph).cpu()
            vy = y_all[val_idx]
            mae = (vp - vy).abs().mean().item() * SCALE
            bm = vy.abs() * SCALE <= 300.0
            bp, by = vp[bm], vy[bm]
            bmae = (bp - by).abs().mean().item() * SCALE
            bc, yc = bp - bp.mean(), by - by.mean()
            r = (bc @ yc / (bc.norm() * yc.norm() + 1e-9)).item()
        rec = dict(epoch=ep, loss=round(run_loss / max(k, 1), 5),
                   val_mae_cp=round(mae, 1), band_mae_cp=round(bmae, 1),
                   band_r=round(r, 3), sec=round(time.time() - t))
        hist.append(rec)
        print(f"epoch {ep}: loss {rec['loss']}  MAE {mae:.1f}cp  "
              f"| in +/-300cp: MAE {bmae:.1f}cp r {r:.3f}  ({rec['sec']}s)", flush=True)
        export(model)
        with open(LOG, "w") as f:
            json.dump(dict(config=dict(model="psqt-linear", scale=SCALE), history=hist), f, indent=2)
        push(args.branch, f"e{ep} band_mae {bmae:.0f}cp r{r:.3f}")

    print(f"\ndone. best band MAE {min(h['band_mae_cp'] for h in hist):.1f}cp  "
          f"best r {max(h['band_r'] for h in hist):.3f}")


if __name__ == "__main__":
    main()
