"""Extract Texel features for a chunk of rows -> a compact .npz.

  python -m texel.build_data <n_positions> [out.npz] [src.txt]

Input rows, tab-separated, auto-detected by column count:
  2 col:  fen <tab> cp_white               (Lichess SF-eval DB - distillation)
  3 col:  fen <tab> result <tab> cp_white  (self-play; result 1/0.5/0 white POV)

Always writes both `cp` (white-POV centipawns) and `result` (white-POV game
result, or -1.0 where unknown) so `tune.py` can pick the target with --wdl.
"""
import os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texel.features import extract

SRC_DEFAULT = (r"C:\Users\44783\AppData\Local\Temp\claude"
               r"\c--Users-44783-AI-Chessathon-aichessathon-dev"
               r"\d306b6d4-80d9-4b43-afd3-8bb06cc49ce3\scratchpad\nnue_positions.txt")

_KEYS_VEC = ["mat", "mob", "kd", "passed", "pst_mg", "pst_eg"]
_KEYS_SCALAR = ["phase", "wtm", "bp", "iso", "dbl", "rook_open", "rook_half"]


def _chunk(rows):
    out_vec = {k: [] for k in _KEYS_VEC}
    out_sc = {k: [] for k in _KEYS_SCALAR}
    cps, results = [], []
    for fen, cp, res in rows:
        try:
            f = extract(fen)
        except Exception:
            continue
        for k in _KEYS_VEC:
            out_vec[k].append(f[k])
        for k in _KEYS_SCALAR:
            out_sc[k].append(f[k])
        cps.append(cp)
        results.append(res)
    return (
        {k: np.asarray(v, np.float32) for k, v in out_vec.items()},
        {k: np.asarray(v, np.float32) for k, v in out_sc.items()},
        np.asarray(cps, np.float32),
        np.asarray(results, np.float32),
    )


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "texel_data.npz")
    src = sys.argv[3] if len(sys.argv) > 3 else SRC_DEFAULT

    rows = []
    with open(src) as fh:
        for line in fh:
            if len(rows) >= n:
                break
            p = line.rstrip("\n").split("\t")
            if len(p) == 2:
                rows.append((p[0], float(p[1]), -1.0))
            elif len(p) == 3:                       # fen, result, cp_white
                rows.append((p[0], float(p[2]), float(p[1])))
    print(f"{len(rows):,} rows, extracting features...", flush=True)

    t0 = time.time()
    CH = 20_000
    chunks = [rows[i:i + CH] for i in range(0, len(rows), CH)]
    parts = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for j, r in enumerate(ex.map(_chunk, chunks)):
            parts.append(r)
            if (j + 1) % 5 == 0:
                print(f"  {sum(len(p[2]) for p in parts):,} done "
                      f"({time.time()-t0:.0f}s)", flush=True)

    save = {}
    for k in _KEYS_VEC:
        save[k] = np.concatenate([p[0][k] for p in parts])
    for k in _KEYS_SCALAR:
        save[k] = np.concatenate([p[1][k] for p in parts])
    save["cp"] = np.concatenate([p[2] for p in parts])
    save["result"] = np.concatenate([p[3] for p in parts])
    np.savez_compressed(out, **save)
    have_r = int((save["result"] >= 0).sum())
    print(f"saved {out}  ({save['cp'].shape[0]:,} positions, {have_r:,} with result, "
          f"{os.path.getsize(out)/1e6:.0f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
