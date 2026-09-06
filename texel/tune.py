"""Texel tune the eval weights (torch Adam).

  python -m texel.tune [data.npz] [epochs] [--groups g1,g2,..] [--reg R] [--wdl]
                       [--bs N] [--pstreg P]   (--pstreg anchors PST when 'pst' in groups)

Target (per position), default:  R = sigmoid(cp / 173.72)   — SF's win-prob judgment
                          (173.72 = 400 / ln 10, the logistic Elo model)
With --wdl: R is the game result (1 / 0.5 / 0, white POV) straight from column 2
  — the real Texel target. A fit to SF evals measured +28% correlation and still
  played 160 Elo worse; matching an eval is not the same as predicting a win.
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
    _flag_vals = {args[i + 1] for i, a in enumerate(args[:-1])
                  if a in ("--groups", "--reg", "--bs", "--pstreg")}
    epochs = 60
    for a in args:
        if a.isdigit() and a not in _flag_vals:
            epochs = int(a)
    groups = None
    if "--groups" in args:
        groups = args[args.index("--groups") + 1].split(",")
    reg = 0.002
    if "--reg" in args:
        reg = float(args[args.index("--reg") + 1])
    pstreg = 0.0   # anchor PST toward its fastchess starting values (0 = free)
    if "--pstreg" in args:
        pstreg = float(args[args.index("--pstreg") + 1])
    wdl = "--wdl" in args   # column 2 is a game result (1/0.5/0), not an SF cp

    torch.set_num_threads(8)
    d = np.load(npz)
    T = lambda k: torch.tensor(d[k], dtype=torch.float32)
    F = dict(mat=T("mat"), mob=T("mob"), kd=T("kd"), passed=T("passed"),
             pst_mg=T("pst_mg"), pst_eg=T("pst_eg"), phase=T("phase"), wtm=T("wtm"),
             bp=T("bp"), iso=T("iso"), dbl=T("dbl"),
             rook_open=T("rook_open"), rook_half=T("rook_half"))
    R = T("cp") if wdl else torch.sigmoid(T("cp") / WPROB_DIV)
    n = R.shape[0]
    print(("game-result (WDL)" if wdl else "SF-eval") + " targets")
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

    pst_mg0 = torch.tensor(fc._MG_PST.astype(np.float32).reshape(-1))
    pst_eg0 = torch.tensor(fc._EG_PST.astype(np.float32).reshape(-1))
    params = [W[k] for k in tune_keys] + ([pst_mg, pst_eg] if do_pst else [])
    opt = torch.optim.Adam(params, lr=0.5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 262144
    if "--bs" in args:
        bs = int(args[args.index("--bs") + 1])
    bs = min(bs, Rtr.shape[0])
    best = 1e9
    for ep in range(1, epochs + 1):
        perm = torch.randperm(Rtr.shape[0])
        for i in range(0, perm.shape[0], bs):
            b = perm[i:i + bs]
            if b.shape[0] < 64:
                continue
            opt.zero_grad()
            pred = torch.sigmoid(K * eval_cp({k: v[b] for k, v in Ftr.items()}, W, pst_mg, pst_eg))
            loss = torch.mean((pred - Rtr[b]) ** 2)
            pen = sum(torch.mean(((W[k] - W0t[k]) / ANC[k]) ** 2) for k in tune_keys)
            reg_term = reg * pen
            if do_pst and pstreg > 0.0:
                reg_term = reg_term + pstreg * (
                    torch.mean(((pst_mg - pst_mg0) / 12.0) ** 2)
                    + torch.mean(((pst_eg - pst_eg0) / 12.0) ** 2))
            (loss + reg_term).backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            vm = torch.mean((torch.sigmoid(K * eval_cp(Fva, W, pst_mg, pst_eg)) - Rva) ** 2).item()
        best = min(best, vm)
        if ep % 6 == 0 or ep == 1:
            print(f"ep{ep:3d}  val-MSE {vm:.5f}")

    # Diagnostic: how well does each eval predict the val target?
    with torch.no_grad():
        cpv = T("cp")[val]
        e_new = eval_cp(Fva, W, pst_mg, pst_eg)
        W_start = {k: torch.tensor(v.astype(np.float32)) for k, v in W0.items()}
        e_old = eval_cp(Fva, W_start,
                        torch.tensor(fc._MG_PST.astype(np.float32).reshape(-1)),
                        torch.tensor(fc._EG_PST.astype(np.float32).reshape(-1)))

        def corr(a, b):
            a = a - a.mean(); b = b - b.mean()
            return float((a @ b) / (a.norm() * b.norm() + 1e-9))

        if wdl:
            # targets are 1/0.5/0 game results; report win-prob prediction quality
            p_old = torch.sigmoid(K * e_old)
            p_new = torch.sigmoid(K * e_new)
            mse_old = float(torch.mean((p_old - Rva) ** 2))
            mse_new = float(torch.mean((p_new - Rva) ** 2))
            dec = cpv != 0.5
            acc_old = float((((p_old > 0.5) == (Rva > 0.5))[dec]).float().mean())
            acc_new = float((((p_new > 0.5) == (Rva > 0.5))[dec]).float().mean())
            print(f"\n  val Brier (WDL)   : start {mse_old:.5f} -> tuned {mse_new:.5f}")
            print(f"  decisive-pos acc  : start {acc_old:.3f} -> tuned {acc_new:.3f} "
                  f"(n={int(dec.sum()):,})")
            print(f"  eval corr w/ result: start {corr(e_old, Rva):.3f} -> tuned {corr(e_new, Rva):.3f}")
        else:
            band = cpv.abs() <= 300
            print(f"\n  corr with SF  ALL      : start {corr(e_old, cpv):.3f} -> tuned {corr(e_new, cpv):.3f}")
            print(f"  corr with SF  +/-300cp : start {corr(e_old[band], cpv[band]):.3f} -> "
                  f"tuned {corr(e_new[band], cpv[band]):.3f}   <-- the one that matters "
                  f"(n={int(band.sum()):,})")

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
