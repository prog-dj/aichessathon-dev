"""Relabel self-play positions with Stockfish evals (offline distillation).

  python -m nnue.sflabel <in.txt> <out.txt> [nodes] [workers] [sf_path]

Input rows  : fen <tab> result <tab> our_cp_white
Output rows : fen <tab> result <tab> sf_cp_white     (same format, prepare.py reads it)

`result` (game outcome) is passed through untouched; only the cp column is
replaced with Stockfish's node-limited eval, White POV, clamped +/-2000.
"""
import os
import sys
import time
from multiprocessing import Process

import chess
import chess.engine

CLAMP = 2000
DEFAULT_NODES = 40000
DEFAULT_SF = os.environ.get("SF_PATH", "stockfish")


def worker(wid, rows, out_path, nodes, sf_path):
    eng = chess.engine.SimpleEngine.popen_uci(sf_path)
    eng.configure({"Threads": 1, "Hash": 64})
    fh = open(out_path, "w")
    t0 = time.time()
    lim = chess.engine.Limit(nodes=nodes)
    done = 0
    for fen, result in rows:
        try:
            board = chess.Board(fen)
            info = eng.analyse(board, lim)
            sc = info["score"].white().score(mate_score=30000)
            if sc is None:
                continue
            sc = max(-CLAMP, min(CLAMP, int(sc)))
        except Exception:
            continue
        fh.write(f"{fen}\t{result}\t{sc}\n")
        done += 1
        if done % 2000 == 0:
            fh.flush()
            print(f"  w{wid}: {done:,}/{len(rows):,}  {done / (time.time() - t0):.0f}/s", flush=True)
    fh.close()
    eng.quit()


def main():
    src = sys.argv[1]
    out = sys.argv[2]
    nodes = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_NODES
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, (os.cpu_count() or 2) - 1)
    sf_path = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_SF

    rows = []
    with open(src) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                rows.append((p[0], p[1]))
    print(f"{len(rows):,} positions  {nodes} nodes/pos  {nw} workers  sf={sf_path}", flush=True)

    per = (len(rows) + nw - 1) // nw
    parts = [f"{out}.w{i}" for i in range(nw)]
    ps = []
    for i in range(nw):
        chunk = rows[i * per:(i + 1) * per]
        if chunk:
            ps.append(Process(target=worker, args=(i, chunk, parts[i], nodes, sf_path)))
    t0 = time.time()
    for p in ps:
        p.start()
    for p in ps:
        p.join()

    n = 0
    with open(out, "w") as o:
        for p in parts:
            if os.path.exists(p):
                with open(p) as f:
                    for line in f:
                        o.write(line)
                        n += 1
                os.remove(p)
    print(f"\nwrote {out}: {n:,} labelled in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
