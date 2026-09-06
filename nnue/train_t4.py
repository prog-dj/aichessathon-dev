"""Train the leaf NNUE on a GPU (T4/P100).  Reads nnue.prepare output.

  python -m nnue.train_t4 --data /kaggle/tmp/nd --epochs 8 --branch nnue-t4

Changes vs the old CPU trainer that plateaued at in-band MAE ~79cp / r 0.445
and still lost ~70 Elo:
  * L1 16 -> 32          (the "choke" — 512->16 can't hold an eval)
  * PSQT skip            eval = (psqt_stm - psqt_opp) + nn_head; the FT emits a
                         material scalar directly, so the tiny head only learns
                         positional corrections
  * runs on CUDA         (old run was 1600s/epoch on CPU)

Keeps the parts that were right: sigmoid win-prob loss, 70/30 band/decisive
resampling, in-band MAE+r as the headline metric.

If GH_TOKEN is set, force-pushes weights/nnue.npz + nnue/train_log.json to
--branch after every epoch (Kaggle sessions are not reliable).
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

FT = 256
L1W = 16                            # set from --l1 in main()
SCALE = 100.0                       # cp = model_output * SCALE
# model units (cp/SCALE), own-perspective minus opp-perspective, for the PSQT init
# PSQT init = true piece values. With 4x LR it drifted ~2x hot and the L1/L2
# head learned a big opposing correction (mean |nn_head| ~450cp) which kills
# lazy eval. Low LR mult keeps PSQT ~= material so the head only refines it.
_PIECE_VAL = [1.0, 3.2, 3.3, 5.0, 9.5, 0.0]   # P N B R Q K
WDL_DIV = 173.72 / SCALE            # model units -> logistic-Elo win prob
REPO = "github.com/prog-dj/aichessathon-dev.git"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "weights", "nnue.npz")
LOG = os.path.join(HERE, "train_log.json")


class NNUE(nn.Module):
    def __init__(self, l1w: int = L1W) -> None:
        super().__init__()
        self.ft = nn.Embedding(N_FEATURES + 1, FT, padding_idx=N_FEATURES)
        self.psqt = nn.Embedding(N_FEATURES + 1, 1, padding_idx=N_FEATURES)
        self.ft_bias = nn.Parameter(torch.zeros(FT))
        self.l1 = nn.Linear(FT * 2, l1w)
        self.l2 = nn.Linear(l1w, 1)
        nn.init.normal_(self.ft.weight, std=0.03)
        self._init_psqt()

    def _init_psqt(self) -> None:
        """Seed the PSQT skip with real piece values so it carries material from
        step 0 - otherwise the FT/L1 path absorbs material badly and the skip
        never engages (the failure mode of the L1=32 run: +9cp on a full queen).
        feat = (pt*2 + (0 own | 1 opp)) * 64 + sq_rel, per king bucket."""
        import numpy as _np
        w = _np.zeros(N_FEATURES + 1, _np.float32)
        for kb in range(32):
            base = kb * 768
            for pt, val in enumerate(_PIECE_VAL):
                for sq in range(64):
                    w[base + (pt * 2 + 0) * 64 + sq] = val    # own piece
                    w[base + (pt * 2 + 1) * 64 + sq] = -val   # their piece
        with torch.no_grad():
            self.psqt.weight.copy_(torch.from_numpy(w).unsqueeze(1))
            self.ft.weight[N_FEATURES].zero_()
            self.psqt.weight[N_FEATURES].zero_()

    def forward(self, fw: torch.Tensor, fb: torch.Tensor, wtm: torch.Tensor) -> torch.Tensor:
        aw = self.ft(fw).sum(1) + self.ft_bias          # [B, FT] white perspective
        ab = self.ft(fb).sum(1) + self.ft_bias
        pw = self.psqt(fw).sum(1).squeeze(-1)           # [B] white-perspective psqt
        pb = self.psqt(fb).sum(1).squeeze(-1)
        m = wtm.unsqueeze(1)
        a_stm = torch.where(m, aw, ab)
        a_opp = torch.where(m, ab, aw)
        p_stm = torch.where(wtm, pw, pb)
        p_opp = torch.where(wtm, pb, pw)
        h = torch.cat([F.relu(a_stm), F.relu(a_opp)], dim=1)
        h = F.relu(self.l1(h))
        return (p_stm - p_opp) + self.l2(h).squeeze(-1)  # model units, stm POV


def export(model: NNUE) -> None:
    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    l1w = sd["l2.weight"].shape[1]
    w_ft = sd["ft.weight"][:N_FEATURES].numpy().astype(np.float32)
    w_psqt = sd["psqt.weight"][:N_FEATURES].reshape(-1).numpy().astype(np.float32)
    ft_scale = float(32000.0 / max(float(np.abs(w_ft).max()), 1e-6))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(
        OUT,
        w_ft=np.round(w_ft * ft_scale).astype(np.int16),          # [24576, 256]
        ft_scale=np.float32(ft_scale),
        w_psqt=w_psqt,                                            # [24576] model units
        b_ft=sd["ft_bias"].numpy().astype(np.float32),            # [256]
        w_l1=sd["l1.weight"].numpy().T.astype(np.float32),        # [512, l1w]
        b_l1=sd["l1.bias"].numpy().astype(np.float32),            # [l1w]
        w_l2=sd["l2.weight"].numpy().reshape(-1).astype(np.float32),  # [l1w]
        b_l2=np.float32(sd["l2.bias"].item()),
        scale=np.float32(SCALE),
        l1w=np.int32(l1w),
    )
    z = np.load(OUT)
    assert z["w_ft"].shape == (N_FEATURES, FT) and z["w_ft"].dtype == np.int16
    assert z["w_l1"].shape == (2 * FT, l1w)
    assert z["w_psqt"].shape == (N_FEATURES,)


def push(branch: str, note: str) -> None:
    token = os.environ.get("GH_TOKEN")
    if not token:
        return
    def run(*a: str, check: bool = True):
        return subprocess.run(a, cwd=ROOT, check=check, capture_output=True, text=True)
    try:
        run("git", "config", "user.email", "t4@bot"); run("git", "config", "user.name", "t4")
        run("git", "add", "-f", "weights/nnue.npz", "nnue/train_log.json")
        run("git", "commit", "-m", f"nnue-t4: {note}", check=False)
        run("git", "push", "-f", f"https://{token}@{REPO}", f"HEAD:{branch}")
        print(f"  pushed -> {branch}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"  push failed: {e.stderr or e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--branch", default="nnue-t4")
    ap.add_argument("--max_lr", type=float, default=3e-3)
    ap.add_argument("--l1", type=int, default=16)
    ap.add_argument("--psqt_lr_mult", type=float, default=0.5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU'})")

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
    target = (cp * cp_sign / SCALE).astype(np.float32)    # stm-POV, model units
    print(f"{n:,} positions   band(<=300cp) {np.mean(np.abs(cp) <= 300):.1%}")

    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    n_val = max(1, n // 50)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    band_mask = np.abs(target) <= 300.0 / SCALE
    tr_band = train_idx[band_mask[train_idx]]
    tr_dec = train_idx[~band_mask[train_idx]]

    y_all = torch.from_numpy(target)
    wtm_t = torch.from_numpy(wtm)

    def batch_tensors(rows: np.ndarray):
        fw = torch.from_numpy(feat_w[rows].astype(np.int64)).to(dev, non_blocking=True)
        fb = torch.from_numpy(feat_b[rows].astype(np.int64)).to(dev, non_blocking=True)
        w = wtm_t[rows].to(dev, non_blocking=True)
        y = y_all[rows].to(dev, non_blocking=True)
        return fw, fb, w, y

    model = NNUE(l1w=args.l1).to(dev)
    psqt_params = list(model.psqt.parameters())
    rest = [p for n_, p in model.named_parameters() if not n_.startswith("psqt.")]
    opt = torch.optim.Adam([{"params": rest}, {"params": psqt_params}], lr=1e-3)
    steps = max(1, len(train_idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.max_lr, args.max_lr * args.psqt_lr_mult],
        epochs=args.epochs, steps_per_epoch=steps)
    print(f"L1={args.l1}  PSQT seeded with piece values, {args.psqt_lr_mult}x LR")

    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        n_band = int(len(train_idx) * 0.70)
        n_dec = len(train_idx) - n_band
        sb = rng.choice(tr_band, n_band, replace=len(tr_band) < n_band)
        sd_ = rng.choice(tr_dec, n_dec, replace=len(tr_dec) < n_dec)
        perm = rng.permutation(np.concatenate([sb, sd_]))
        run_loss = nb = 0
        t = time.time()
        for i in range(0, len(perm) - args.batch, args.batch):
            fw, fb, w, y = batch_tensors(perm[i:i + args.batch])
            opt.zero_grad(set_to_none=True)
            pred = model(fw, fb, w)
            loss = F.mse_loss(torch.sigmoid(pred / WDL_DIV), torch.sigmoid(y / WDL_DIV))
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item(); nb += 1

        model.eval()
        with torch.no_grad():
            vp = torch.empty(len(val_idx))
            for j in range(0, len(val_idx), args.batch):
                vr = val_idx[j:j + args.batch]
                fw, fb, w, _ = batch_tensors(vr)
                vp[j:j + len(vr)] = model(fw, fb, w).cpu()
            vy = y_all[val_idx]
            mae = (vp - vy).abs().mean().item() * SCALE
            bm = vy.abs() * SCALE <= 300.0
            bp, by = vp[bm], vy[bm]
            bmae = (bp - by).abs().mean().item() * SCALE
            bc, yc = bp - bp.mean(), by - by.mean()
            r = (bc @ yc / (bc.norm() * yc.norm() + 1e-9)).item()
        rec = dict(epoch=ep, loss=round(run_loss / max(nb, 1), 5),
                   val_mae_cp=round(mae, 1), band_mae_cp=round(bmae, 1),
                   band_r=round(r, 3), sec=round(time.time() - t))
        hist.append(rec)
        print(f"epoch {ep}: loss {rec['loss']}  MAE {mae:.1f}cp  "
              f"| in +/-300cp: MAE {bmae:.1f}cp r {r:.3f}  ({rec['sec']}s)", flush=True)
        export(model)
        with open(LOG, "w") as f:
            json.dump(dict(config=dict(FT=FT, L1W=L1W, scale=SCALE), history=hist), f, indent=2)
        push(args.branch, f"e{ep} band_mae {bmae:.0f}cp r{r:.3f}")

    print(f"\ndone. best band MAE {min(h['band_mae_cp'] for h in hist):.1f}cp")


if __name__ == "__main__":
    main()
