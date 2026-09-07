"""Build the NNUE training set: download the Lichess Stockfish-eval DB, filter,
extract HalfKA features, save as padded dense arrays ready for a GPU embedding.

  python -m nnue.prepare --limit 12000000 --out /kaggle/tmp/nd --workers 4

Output (in --out dir):
  feat_w.npy  int16 [N, 32]   white-perspective feature idx, PAD-padded
  feat_b.npy  int16 [N, 32]   black-perspective
  cnt.npy     uint8 [N]       pieces on board (= real entries per row)
  cp.npy      float32 [N]     Stockfish eval, white POV, centipawns, clamped
  wtm.npy     bool  [N]       white to move
  meta.json

A local  fen<TAB>cp  file can be passed with --src instead of downloading.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from nnue.features import MAX_PIECES, PAD, board_features

EVAL_DB_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
MIN_DEPTH = 18
CLAMP = 2000
CHUNK = 100_000


def _stream_rows(src: str | None, limit: int):
    """Yield (fen, cp_white, best_move_uci) from the eval DB or a local file.
    best_move_uci is the deepest eval's PV first move, "" if unknown - the
    quiet filter in _process drops positions whose best move is a capture."""
    if src and os.path.exists(src):
        with open(src) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    return
                parts = line.rstrip("\n").split("\t")
                yield parts[0], float(parts[1]), (parts[2] if len(parts) > 2 else "")
        return

    import zstandard

    raw = urllib.request.urlopen(EVAL_DB_URL)
    reader = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True)
    text = io.TextIOWrapper(reader, encoding="utf-8")
    kept = 0
    for line in text:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        evals = row.get("evals") or []
        if not evals:
            continue
        best = max(evals, key=lambda e: e.get("depth", 0))
        if best.get("depth", 0) < MIN_DEPTH:
            continue
        pvs = best.get("pvs") or []
        if not pvs:
            continue
        cp = pvs[0].get("cp")
        mate = pvs[0].get("mate")
        if cp is None and mate is None:
            continue
        cpw = float(cp) if cp is not None else (CLAMP if mate > 0 else -CLAMP)
        cpw = max(-CLAMP, min(CLAMP, cpw))
        fen = row.get("fen")
        if not fen:
            continue
        if len(fen.split()) == 4:
            fen += " 0 1"
        line_uci = (pvs[0].get("line") or "").split()
        best_move = line_uci[0] if line_uci else ""
        yield fen, cpw, best_move
        kept += 1
        if kept >= limit:
            return


def _process(rows: list[tuple[str, float, str]]):
    import chess

    fw_l, fb_l, cnt_l, cp_l, wtm_l = [], [], [], [], []
    for fen, cpw, best_move in rows:
        try:
            b = chess.Board(fen)
        except ValueError:
            continue
        # quiet filter (SF's rule): the net is a static eval called at leaves
        # where qsearch has resolved captures - training on in-check / tactical
        # positions makes it learn the search's job and pollutes the eval.
        if b.is_check():
            continue
        if best_move:
            try:
                mv = chess.Move.from_uci(best_move)
                if mv.promotion is not None or b.is_capture(mv):
                    continue
            except ValueError:
                pass
        iw, ib = board_features(b)
        k = len(iw)
        if k < 2 or k > MAX_PIECES:          # need both kings; skip illegal junk
            continue
        row_w = np.full(MAX_PIECES, PAD, np.int16); row_w[:k] = iw
        row_b = np.full(MAX_PIECES, PAD, np.int16); row_b[:k] = ib
        fw_l.append(row_w); fb_l.append(row_b)
        cnt_l.append(k); cp_l.append(cpw)
        wtm_l.append(b.turn == chess.WHITE)
    if not fw_l:
        z = np.zeros((0, MAX_PIECES), np.int16)
        return z, z, np.zeros(0, np.uint8), np.zeros(0, np.float32), np.zeros(0, np.bool_)
    return (np.stack(fw_l), np.stack(fb_l),
            np.array(cnt_l, np.uint8), np.array(cp_l, np.float32), np.array(wtm_l, np.bool_))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--src", default=None, help="local fen<TAB>cp file (skips download)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    from numpy.lib.format import open_memmap
    cap = int(args.limit * 1.02)
    pp = lambda nm: os.path.join(args.out, nm)
    m_fw = open_memmap(pp("feat_w.npy"), mode="w+", dtype=np.int16, shape=(cap, MAX_PIECES))
    m_fb = open_memmap(pp("feat_b.npy"), mode="w+", dtype=np.int16, shape=(cap, MAX_PIECES))
    m_cnt = open_memmap(pp("cnt.npy"), mode="w+", dtype=np.uint8, shape=(cap,))
    m_cp = open_memmap(pp("cp.npy"), mode="w+", dtype=np.float32, shape=(cap,))
    m_wtm = open_memmap(pp("wtm.npy"), mode="w+", dtype=np.bool_, shape=(cap,))
    done = 0

    def flush(fut):
        nonlocal done
        fw, fb, cnt, cp, wtm = fut.result()
        k = len(cnt)
        if k and done + k <= cap:
            s = slice(done, done + k)
            m_fw[s] = fw; m_fb[s] = fb; m_cnt[s] = cnt; m_cp[s] = cp; m_wtm[s] = wtm
            done += k
        print(f"  {done:,} kept  ({time.time()-t0:.0f}s)", flush=True)

    from collections import deque

    chunk: list[tuple[str, float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        pending: deque = deque()
        for row in _stream_rows(args.src, args.limit):
            chunk.append(row)
            if len(chunk) >= CHUNK:
                pending.append(ex.submit(_process, chunk))
                chunk = []
                if len(pending) >= args.workers * 3:
                    flush(pending.popleft())
        if chunk:
            pending.append(ex.submit(_process, chunk))
        while pending:
            flush(pending.popleft())

    for m in (m_fw, m_fb, m_cnt, m_cp, m_wtm):
        m.flush()
    cp = np.asarray(m_cp[:done])
    n = done
    meta = dict(n=int(n), cap=cap, min_depth=MIN_DEPTH, clamp=CLAMP,
               band_frac=float(np.mean(np.abs(cp) <= 300)),
               mean_abs_cp=float(np.mean(np.abs(cp))), seconds=round(time.time() - t0))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {n:,} positions to {args.out}  ({meta})")


if __name__ == "__main__":
    main()
