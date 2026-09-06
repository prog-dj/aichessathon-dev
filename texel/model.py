"""The eval as  white_cp = f(features, weights)  — a differentiable (torch)
reconstruction of fastchess.evaluate_hce, plus the current weight values.

Weight vector is a flat dict of named tensors so tune.py can freeze/unfreeze
groups (PST is frozen by default — 768 params, tuned only in a later phase).
"""
from __future__ import annotations

import numpy as np

# ---- current weights (exact, from fastchess) ----------------------------
W0 = dict(
    mat_mg=np.array([82, 337, 365, 477, 1025], np.float64),
    mat_eg=np.array([94, 281, 297, 512, 936], np.float64),
    mob=np.array([2.2, 2.2, 2.2, 2.2], np.float64),          # per-piece N B R Q
    kdw=np.array([2.0, 2.0, 3.0, 5.0], np.float64),          # ring-hit weight N B R Q
    ksw=np.array([3.0, 6.0, 5.0], np.float64),               # shelter: dist-score, kfile-open, flank-open
    kdc=np.array([11.0 / 16.0], np.float64),                 # curve coeff (u^2 * kdc)
    bp_mg=np.array([25.0]), bp_eg=np.array([25.0]),
    iso_mg=np.array([-12.0]), iso_eg=np.array([-12.0]),
    dbl_mg=np.array([-5.0]), dbl_eg=np.array([-10.0]),
    pass_mg=np.array([6, 12, 17, 35, 63, 104], np.float64),
    pass_eg=np.array([12, 21, 35, 63, 109, 184], np.float64),
    rook_open=np.array([22.0]), rook_half=np.array([10.0]),
    tempo=np.array([14.0]),
)

# PST frozen separately (loaded from fastchess at import in tune.py)
KD_CAP = 40.0


def eval_white_cp_np(F: dict, W: dict, pst_mg: np.ndarray, pst_eg: np.ndarray) -> np.ndarray:
    """Vectorised numpy eval over a batch. F holds stacked arrays:
       F['mat'] (n,5) F['mob'] (n,4) F['kd'] (n,8) F['bp'/'iso'/'dbl'] (n,)
       F['passed'] (n,6) F['rook_open'/'rook_half'] (n,) F['phase'/'wtm'] (n,)
       F['pst_mg'/'pst_eg'] (n,384)
    """
    n = F["phase"].shape[0]
    mg = F["mat"] @ W["mat_mg"] + F["pst_mg"] @ pst_mg
    eg = F["mat"] @ W["mat_eg"] + F["pst_eg"] @ pst_eg

    mg = mg + F["mob"] @ W["mob"]

    ub = F["kd"][:, 0:4] @ W["kdw"] + F["ks"][:, 0:3] @ W["ksw"]
    uw = F["kd"][:, 4:8] @ W["kdw"] + F["ks"][:, 3:6] @ W["ksw"]
    ub = np.clip(ub, 0.0, KD_CAP)
    uw = np.clip(uw, 0.0, KD_CAP)
    mg = mg + (ub * ub - uw * uw) * W["kdc"][0]

    mg = mg + F["bp"] * W["bp_mg"][0] + F["iso"] * W["iso_mg"][0] + F["dbl"] * W["dbl_mg"][0]
    eg = eg + F["bp"] * W["bp_eg"][0] + F["iso"] * W["iso_eg"][0] + F["dbl"] * W["dbl_eg"][0]
    mg = mg + F["passed"] @ W["pass_mg"]
    eg = eg + F["passed"] @ W["pass_eg"]
    mg = mg + F["rook_open"] * W["rook_open"][0] + F["rook_half"] * W["rook_half"][0]

    ph = F["phase"]
    score = (mg * ph + eg * (24.0 - ph)) / 24.0
    return score + W["tempo"][0] * F["wtm"]
