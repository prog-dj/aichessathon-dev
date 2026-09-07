"""Cross-check: eval_white_cp_np(extract(fen), W0) must equal
fastchess.evaluate_hce(fen) (white POV) within a couple cp of integer rounding,
on a random sample. If this fails, tuned weights won't transfer — fix features
before tuning."""
import os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastchess as fc
from texel.features import extract
from texel.model import W0, eval_white_cp_np

PST_MG = fc._MG_PST.astype(np.float64).reshape(-1)   # (384,)
PST_EG = fc._EG_PST.astype(np.float64).reshape(-1)


def fc_eval_white(fen):
    bb, mb = fc.fen_to_arrays(fen)
    stm = fc.evaluate_hce(bb, mb)                     # side-to-move POV, +14 tempo baked
    return stm if fen.split()[1] == "w" else -stm


def stack(feats):
    keys = ["mat", "mob", "kd", "ks", "passed", "pst_mg", "pst_eg"]
    F = {k: np.stack([f[k] for f in feats]) for k in keys}
    for k in ["phase", "wtm", "bp", "iso", "dbl", "rook_open", "rook_half"]:
        F[k] = np.array([f[k] for f in feats], np.float64)
    return F


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\44783\AppData\Local\Temp\claude\c--Users-44783-AI-Chessathon-aichessathon-dev\d306b6d4-80d9-4b43-afd3-8bb06cc49ce3\scratchpad\nnue_positions.txt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    fens = []
    with open(src) as fh:
        for i, line in enumerate(fh):
            if i > 400000:
                break
            fens.append(line.split("\t")[0])
    random.seed(1)
    sample = random.sample(fens, n)

    feats = [extract(f) for f in sample]
    F = stack(feats)
    got = eval_white_cp_np(F, W0, PST_MG, PST_EG)
    want = np.array([fc_eval_white(f) for f in sample], np.float64)

    d = got - want
    bad = np.abs(d) > 3
    print(f"n={n}  mean|diff|={np.abs(d).mean():.2f}cp  max|diff|={np.abs(d).max():.0f}cp  "
          f">3cp: {bad.sum()} ({bad.mean()*100:.1f}%)")
    if bad.sum():
        order = np.argsort(-np.abs(d))[:12]
        for j in order:
            print(f"  diff {d[j]:+6.0f}   got {got[j]:+7.1f}  want {want[j]:+6.0f}   {sample[j]}")


if __name__ == "__main__":
    main()
