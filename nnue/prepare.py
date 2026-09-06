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
MIN_DEPTH = 13
CLAMP = 2000
CHUNK = 100_000


def _stream_rows(src: str | None, limit: int):
    """Yield (fen, cp_white) from the eval DB URL (default) or a local file."""
    if src and os.path.exists(src):
        with open(src) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    return
                fen, cp = line.rstrip("\n").split("\t")
                yield fen, float(cp)
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
        yield fen, cpw
        kept += 1
        if kept >= limit:
            return


def _process(rows: list[tuple[str, float]]):
    import chess

    n = len(rows)
    fw = np.full((n, MAX_PIECES), PAD, np.int16)
    fb = np.full((n, MAX_PIECES), PAD, np.int16)
    cnt = np.zeros(n, np.uint8)
    cp = np.zeros(n, np.float32)
    wtm = np.zeros(n, np.bool_)
    for i, (fen, cpw) in enumerate(rows):
        b = chess.Board(fen)
        iw, ib = board_features(b)
        k = len(iw)
        fw[i, :k] = iw
        fb[i, :k] = ib
        cnt[i] = k
        cp[i] = cpw
        wtm[i] = b.turn == chess.WHITE
    return fw, fb, cnt, cp, wtm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--src", default=None, help="local fen<TAB>cp file (skips download)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    fw_parts, fb_parts, cnt_parts, cp_parts, wtm_parts = [], [], [], [], []
    done = 0

    def flush(fut):
        nonlocal done
        fw, fb, cnt, cp, wtm = fut.result()
        fw_parts.append(fw); fb_parts.append(fb); cnt_parts.append(cnt)
        cp_parts.append(cp); wtm_parts.append(wtm)
        done += len(cnt)
        print(f"  {done:,} positions  ({time.time()-t0:.0f}s)", flush=True)

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

    fw = np.concatenate(fw_parts); fb = np.concatenate(fb_parts)
    cnt = np.concatenate(cnt_parts); cp = np.concatenate(cp_parts)
    wtm = np.concatenate(wtm_parts)
    n = len(cnt)
    np.save(os.path.join(args.out, "feat_w.npy"), fw)
    np.save(os.path.join(args.out, "feat_b.npy"), fb)
    np.save(os.path.join(args.out, "cnt.npy"), cnt)
    np.save(os.path.join(args.out, "cp.npy"), cp)
    np.save(os.path.join(args.out, "wtm.npy"), wtm)
    meta = dict(n=int(n), min_depth=MIN_DEPTH, clamp=CLAMP,
               band_frac=float(np.mean(np.abs(cp) <= 300)),
               mean_abs_cp=float(np.mean(np.abs(cp))), seconds=round(time.time() - t0))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {n:,} positions to {args.out}  ({meta})")


if __name__ == "__main__":
    main()
