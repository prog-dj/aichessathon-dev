"""Generate Texel training data: self-play games labelled by their outcome.

  python -m texel.selfplay <games> [out.txt] [nodes] [workers]

Writes "<fen>\t<result>" where result is 1.0 / 0.5 / 0.0 from WHITE's point of
view - the classic Texel target. Tuning against game results rather than another
engine's evaluations is the whole point: a fit to Stockfish's numbers measured
+28% correlation and still played 160 Elo worse, because matching an eval is not
the same as predicting a win.

Diversity matters more than it looks: the search is deterministic at fixed nodes,
so the same opening replays the same game forever. Each game therefore gets a
random opening, 2-5 random plies on top, and a jittered node budget.

Sampled positions are "quiet" in the usual sense - not in check, the move played
from them was not a capture or promotion, and past the random opening plies.
"""
import os
import random
import sys
import time
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OPENINGS = os.path.join(
    r"C:\Users\44783\AppData\Local\Temp\claude"
    r"\c--Users-44783-AI-Chessathon-aichessathon-dev"
    r"\d306b6d4-80d9-4b43-afd3-8bb06cc49ce3\scratchpad", "openings.txt")

SAMPLE_EVERY = 6      # take a candidate position every N plies
SKIP_PLIES = 6        # ignore the first few plies after the random opening
MAX_FULLMOVES = 160


def worker(wid, n_games, out_path, nodes, seed):
    sys.path.insert(0, ROOT)
    import chess
    import fastchess as fc

    rng = random.Random(seed)
    opens = [ln.strip() for ln in open(OPENINGS) if ln.strip()]
    e = fc.Engine()
    fh = open(out_path, "w")
    kept = 0
    t0 = time.time()

    for g in range(n_games):
        e.tt[:, :] = 0
        board = chess.Board(rng.choice(opens))
        # random plies for diversity
        for _ in range(rng.randint(2, 5)):
            mv = list(board.legal_moves)
            if not mv or board.is_game_over():
                break
            board.push(rng.choice(mv))
        if board.is_game_over():
            continue
        root = board.fen()
        hist = []
        gn = int(nodes * (0.8 + 0.4 * rng.random()))
        samples = []
        ply = 0
        while not board.is_game_over(claim_draw=True) and board.fullmove_number < MAX_FULLMOVES:
            fen_before = board.fen()
            in_check = board.is_check()
            try:
                uci, sc, d, n = e.best_move(root, hist, 10 ** 9, 10 ** 9, 64, gn)
                mv = chess.Move.from_uci(uci)
            except Exception:
                break
            if mv not in board.legal_moves:
                break
            noisy = board.is_capture(mv) or mv.promotion is not None
            if ply >= SKIP_PLIES and (ply % SAMPLE_EVERY == 0) and not in_check and not noisy:
                samples.append(fen_before)
            board.push(mv)
            hist.append(uci)
            ply += 1

        r = board.result(claim_draw=True)
        if r == "1-0":
            res = "1.0"
        elif r == "0-1":
            res = "0.0"
        elif r == "1/2-1/2":
            res = "0.5"
        else:
            continue                      # hit the move cap unresolved: no label
        for f in samples:
            fh.write(f"{f}\t{res}\n")
            kept += 1
        if (g + 1) % 200 == 0:
            fh.flush()
            el = time.time() - t0
            print(f"  w{wid}: {g+1} games, {kept:,} positions, "
                  f"{(g+1)/el:.1f} games/s", flush=True)
    fh.close()


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "selfplay.txt")
    nodes = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    per = total // nw
    parts = [f"{out}.w{i}" for i in range(nw)]
    print(f"{total:,} games, {nodes} nodes/move, {nw} workers ({per:,} each)", flush=True)
    ps = [Process(target=worker, args=(i, per, parts[i], nodes, 1000 + i)) for i in range(nw)]
    t0 = time.time()
    for p in ps:
        p.start()
    for p in ps:
        p.join()

    n = 0
    with open(out, "w") as o:
        for p in parts:
            if not os.path.exists(p):
                continue
            with open(p) as f:
                for line in f:
                    o.write(line); n += 1
            os.remove(p)
    print(f"\nwrote {out}: {n:,} positions in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
