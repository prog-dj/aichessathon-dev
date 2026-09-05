"""NNUE accumulator maintenance must never drift from a full rebuild.

The search maintains acc_w/acc_b incrementally through make_move_acc /
unmake_move_acc (diff the touched squares; full rebuild on a king move because
the features are king-bucketed). If that ever desyncs from build_acc, the net
silently evaluates garbage and every A/B against it is meaningless. This test
DFS-walks real game trees and asserts byte-equality after every make and every
unmake.

NNUE is off by default (FASTCHESS_NNUE), so this runs in a subprocess with the
env-var the search reads at import.
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DRIVER = r"""
import os, sys, json
os.environ["FASTCHESS_NNUE"] = "1"
sys.path.insert(0, sys.argv[1])
import numpy as np
import chess
import fastchess as fc

if not fc.NNUE_OK:
    print(json.dumps({"skip": "no usable weights/nnue.npz"})); sys.exit(0)
if not fc.USE_NNUE:
    print(json.dumps({"fail": "FASTCHESS_NNUE=1 but USE_NNUE is False"})); sys.exit(0)

FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    "rn1q1rk1/pbppbppp/1p2pn2/8/2PP4/P4NP1/1P2PPBP/RNBQ1RK1 b - - 2 7",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                            # ep-heavy
    "4k2r/8/8/8/8/8/8/R3K3 w Qk - 0 1",                                     # castling
    "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",                                        # promotions
]

def full(bb, mb):
    aw = np.zeros(256, np.int32); ab = np.zeros(256, np.int32)
    fc.build_acc(bb, mb, aw, ab)
    return aw, ab

def walk(fen, depth):
    bb, mb = fc.fen_to_arrays(fen)
    gh = np.zeros((fc.MAX_PLY + 64, fc.HIST_W), np.uint64)
    acc_w, acc_b = full(bb, mb)
    n_checks = [0]

    def rec(ply, d):
        if d == 0:
            return None
        out = np.empty(256, np.int32)
        nm = fc.gen_moves(bb, mb, out, False)
        for i in range(nm):
            m = out[i]
            fc.make_move_acc(bb, mb, gh, ply, m, acc_w, acc_b)
            ew, eb = full(bb, mb)
            n_checks[0] += 1
            if not (np.array_equal(acc_w, ew) and np.array_equal(acc_b, eb)):
                return {"where": "make", "fen": fen, "uci": fc.move_to_uci(int(m)),
                        "dw": int(np.abs(acc_w - ew).max()),
                        "db": int(np.abs(acc_b - eb).max())}
            bad = rec(ply + 1, d - 1)
            if bad:
                return bad
            fc.unmake_move_acc(bb, mb, gh, ply, acc_w, acc_b)
            ew, eb = full(bb, mb)
            n_checks[0] += 1
            if not (np.array_equal(acc_w, ew) and np.array_equal(acc_b, eb)):
                return {"where": "unmake", "fen": fen, "uci": fc.move_to_uci(int(m)),
                        "dw": int(np.abs(acc_w - ew).max()),
                        "db": int(np.abs(acc_b - eb).max())}
        return None

    return rec(0, depth), n_checks[0]

bad_any, total = None, 0
for f in FENS:
    bad, n = walk(f, 3)
    total += n
    if bad:
        bad_any = bad
        break
print(json.dumps({"ok": bad_any is None, "checks": total, "bad": bad_any}))
"""


def _run():
    p = subprocess.run([sys.executable, "-c", _DRIVER, REPO],
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        pytest.fail(f"driver crashed:\nSTDOUT {p.stdout}\nSTDERR {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_accumulator_never_drifts():
    r = _run()
    if "skip" in r:
        pytest.skip(r["skip"])
    if "fail" in r:
        pytest.fail(r["fail"])
    assert r["ok"], f"accumulator drift: {r['bad']}"
    assert r["checks"] > 500, f"only {r['checks']} positions checked - tree too shallow"
