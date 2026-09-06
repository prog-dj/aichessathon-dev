"""Texel tune the eval weights (torch Adam) against Stockfish-eval targets.

  python -m texel.tune [texel_data.npz] [epochs] [--groups g1,g2,..] [--reg R]

Target (per position):  R = sigmoid(cp / 173.72)   — SF's win-prob judgment
                          (173.72 = 400 / ln 10, the logistic Elo model)
Loss:  mean( (R - sigmoid(K * eval_white_cp))^2 )  +  reg * anchor_penalty
K is fitted once (1-D) with the frozen starting weights.
anchor_penalty pulls each weight toward its current value scaled by its
magnitude, so weights only move where the data clearly demands it (stops
material / passed-pawn tables collapsing on correlated features).
PST frozen unless 'pst' in --groups.
"""
import os, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastchess as fc
from texel.model import W0, KD_CAP

HERE = os.path.dirname(os.path.abspath(__file__))
WPROB_DIV = 173.72
TUNABLE = ["mat_mg", "mat_eg", "mob", "kdw", "kdc", "bp_mg", "bp_eg",
           "iso_mg", "iso_eg", "dbl_mg", "dbl_eg", "pass_mg", "pass_eg",
           "rook_open", "rook_half", "tempo"]
# per-weight anchor scale (a "1 sigma" move); reg penalises (dw/scale)^2
ANCHOR = dict(
    mat_mg=np.array([12, 30, 30, 40, 80.]), mat_eg=np.array([12, 30, 30, 40, 80.]),
    mob=np.array([1.5, 1.5, 1.5, 1.5]), kdw=np.array([1.0, 1.0, 1.5, 2.0]),
    kdc=np.array([0.4]), bp_mg=np.array([20.]), bp_eg=np.array([20.]),
    iso_mg=np.array([8.]), iso_eg=np.array([8.]), dbl_mg=np.array([6.]), dbl_eg=np.array([8.]),
    pass_mg=np.array([8, 10, 12, 16, 22, 30.]), pass_eg=np.array([10, 12, 16, 22, 30, 40.]),
    rook_open=np.array([12.]), rook_half=np.array([8.]), tempo=np.array([10.]),
)


def eval_cp(F, W, pst_mg, pst_eg):
    mg = F["mat"] @ W["mat_mg"] + F["pst_mg"] @ pst_mg
    eg = F["mat"] @ W["mat_eg"] + F["pst_eg"] @ pst_eg
    mg = mg + F["mob"] @ W["mob"]
    ub = torch.clamp(F["kd"][:, 0:4] @ W["kdw"], max=KD_CAP)
    uw = torch.clamp(F["kd"][:, 4:8] @ W["kdw"], max=KD_CAP)
    mg = mg + (ub * ub - uw * uw) * W["kdc"]
    mg = mg + F["bp"] * W["bp_mg"] + F["iso"] * W["iso_mg"] + F["dbl"] * W["dbl_mg"]
    eg = eg + F["bp"] * W["bp_eg"] + F["iso"] * W["iso_eg"] + F["dbl"] * W["dbl_eg"]
    mg = mg + F["passed"] @ W["pass_mg"]
    eg = eg + F["passed"] @ W["pass_eg"]
    mg = mg + F["rook_open"] * W["rook_open"] + F["rook_half"] * W["rook_half"]
    ph = F["phase"]
    return (mg * ph + eg * (24.0 - ph)) / 24.0 + W["tempo"] * F["wtm"]


def main():
    args = sys.argv[1:]
    npz = args[0] if args and not args[0].startswith("-") else os.path.join(HERE, "texel_data.npz")
    epochs = 60
    for a in args:
        if a.isdigit():
            epochs = int(a)
    groups = None
    if "--groups" in args:
        groups = args[args.index("--groups") + 1].split(",")
    reg = 0.002
    if "--reg" in args:
        reg = float(args[args.index("--reg") + 1])

    torch.set_num_threads(8)
    d = np.load(npz)
    T = lambda k: torch.tensor(d[k], dtype=torch.float32)
    F = dict(mat=T("mat"), mob=T("mob"), kd=T("kd"), passed=T("passed"),
             pst_mg=T("pst_mg"), pst_eg=T("pst_eg"), phase=T("phase"), wtm=T("wtm"),
             bp=T("bp"), iso=T("iso"), dbl=T("dbl"),
             rook_open=T("rook_open"), rook_half=T("rook_half"))
    R = torch.sigmoid(T("cp") / WPROB_DIV)
    n = R.shape[0]
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n, generator=g)
    nv = n // 20
    val, tr = idx[:nv], idx[nv:]
    print(f"{n:,} positions ({nv:,} val)  reg={reg}")

    pst_mg = torch.tensor(fc._MG_PST.astype(np.float32).reshape(-1))
    pst_eg = torch.tensor(fc._EG_PST.astype(np.float32).reshape(-1))
    do_pst = groups is not None and "pst" in groups
    pst_mg.requires_grad_(do_pst)
    pst_eg.requires_grad_(do_pst)

    W = {k: torch.tensor(v.astype(np.float32)) for k, v in W0.items()}
    W0t = {k: torch.tensor(v.astype(np.float32)) for k, v in W0.items()}
    ANC = {k: torch.tensor(v.astype(np.float32)) for k, v in ANCHOR.items()}
    tune_keys = TUNABLE if groups is None else [x for x in groups if x in W]
    for k in W:
        W[k].requires_grad_(k in tune_keys)
    print("tuning:", tune_keys, "+pst" if do_pst else "")

    sub = lambda FF, s: {k: v[s] for k, v in FF.items()}
    Ftr, Fva = sub(F, tr), sub(F, val)
    Rtr, Rva = R[tr], R[val]

    with torch.no_grad():
        e0 = eval_cp(Fva, W, pst_mg, pst_eg)
    # fit K by golden-section on val
    def Ek(K):
        return torch.mean((torch.sigmoid(K * e0) - Rva) ** 2).item()
    lo, hi = 1e-4, 2e-2
    for _ in range(60):
        m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        if Ek(m1) < Ek(m2):
            hi = m2
        else:
            lo = m1
    K = 0.5 * (lo + hi)
    print(f"fitted K={K:.5f}  (eval scale vs SF: {1/K/WPROB_DIV:.2f}x)   start val-MSE {Ek(K):.5f}")

    params = [W[k] for k in tune_keys] + ([pst_mg, pst_eg] if do_pst else [])
    opt = torch.optim.Adam(params, lr=0.5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 262144
    best = 1e9
    for ep in range(1, epochs + 1):
        perm = torch.randperm(Rtr.shape[0])
        for i in range(0, perm.shape[0] - bs, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            pred = torch.sigmoid(K * eval_cp({k: v[b] for k, v in Ftr.items()}, W, pst_mg, pst_eg))
            loss = torch.mean((pred - Rtr[b]) ** 2)
            pen = sum(torch.mean(((W[k] - W0t[k]) / ANC[k]) ** 2) for k in tune_keys)
            (loss + reg * pen).backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            vm = torch.mean((torch.sigmoid(K * eval_cp(Fva, W, pst_mg, pst_eg)) - Rva) ** 2).item()
        best = min(best, vm)
        if ep % 6 == 0 or ep == 1:
            print(f"ep{ep:3d}  val-MSE {vm:.5f}")

    outw = {k: W[k].detach().numpy() for k in W0}
    if do_pst:
        outw["_pst_mg"] = pst_mg.detach().numpy()
        outw["_pst_eg"] = pst_eg.detach().numpy()
    np.savez(os.path.join(HERE, "tuned_weights.npz"), K=K, **outw)
    print(f"\nbest val-MSE {best:.5f}   saved tuned_weights.npz\n--- deltas ---")
    for k in tune_keys:
        print(f"  {k:10s} {np.round(W0[k],1)}  ->  {np.round(W[k].detach().numpy(),1)}")


if __name__ == "__main__":
    main()
