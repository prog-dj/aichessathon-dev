"""Generate on-distribution NNUE training data: self-play games with the current
engine, sampling quiet positions the search actually reaches.

  FASTCHESS_NNUE=1 python -m nnue.selfplay <games> [out.txt] [nodes] [workers]

Each kept row is  "<fen>\t<result>\t<cp_white>":
  result    1.0 / 0.5 / 0.0   from White's POV   (the target that transfers)
  cp_white  the engine's own search score of that position, White POV
            (a dense teacher signal; blended with `result` at train time)

Why self-play instead of the Lichess eval DB: the net is a leaf evaluator, so it
should be trained on the distribution of positions the *search* lands on, not on
positions humans chose to analyse. nnue-a (Lichess-DB, better offline r) played
flat vs the epoch-8 net - offline metric != strength unless the data matches.

Quiet filter, applied here where the info is free: past the random opening plies,
side to move not in check, and the move the engine played from the position was
neither a capture nor a promotion (so qsearch would have nothing to resolve).
"""
import os
import random
import sys
import time
from multiprocessing import Process

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_OPEN_CANDIDATES = [
    os.path.join(ROOT, "texel", "openings.txt"),
    os.path.join(ROOT, "openings.txt"),
]

SAMPLE_EVERY = 5
SKIP_PLIES = 8
MAX_FULLMOVES = 160
CLAMP_CP = 2000
# early adjudication - trims the dead-endgame tail (our engine shuffles for
# 40+ moves in won/drawn R+P endings). Conservative because our own eval is
# the judge and it's weakest exactly in the endgame.
RESIGN_CP = 1500          # |white-POV score| this large ...
RESIGN_PLIES = 8          # ... sustained this many plies -> adjudicate a result
CUTOFF_WIN_CP = 800       # at MAX_FULLMOVES, last score >= this -> win, else draw


def _openings():
    for p in _OPEN_CANDIDATES:
        if os.path.exists(p):
            return [ln.strip() for ln in open(p) if ln.strip()]
    raise SystemExit("no openings.txt found (looked in texel/ and repo root)")


def worker(wid: int, n_games: int, out_path: str, nodes: int, seed: int,
           use_nnue: str) -> None:
    os.environ["FASTCHESS_NNUE"] = use_nnue   # before import - numba freezes it
    sys.path.insert(0, ROOT)
    import chess
    import fastchess as fc

    if wid == 0:
        print(f"  engine: NNUE_OK={fc.NNUE_OK} USE_NNUE={getattr(fc, 'USE_NNUE', '?')}",
              flush=True)

    rng = random.Random(seed)
    opens = _openings()
    e = fc.Engine()
    fh = open(out_path, "w")
    kept = games_done = 0
    t0 = time.time()

    for g in range(n_games):
        e.tt[:, :] = 0
        board = chess.Board(rng.choice(opens))
        for _ in range(rng.randint(2, 6)):
            mv = list(board.legal_moves)
            if not mv or board.is_game_over():
                break
            board.push(rng.choice(mv))
        if board.is_game_over():
            continue
        root = board.fen()
        hist: list[str] = []
        gn = int(nodes * (0.8 + 0.4 * rng.random()))
        samples: list[tuple[str, int]] = []   # (fen, cp_white)
        ply = 0
        adj_res = None
        decisive = 0
        last_cpw = 0
        while not board.is_game_over(claim_draw=True) and board.fullmove_number < MAX_FULLMOVES:
            fen_before = board.fen()
            in_check = board.is_check()
            wtm = board.turn == chess.WHITE
            try:
                uci, sc, d, n = e.best_move(root, hist, 10 ** 9, 10 ** 9, 64, gn)
                mv = chess.Move.from_uci(uci)
            except Exception:
                break
            if mv not in board.legal_moves:
                break
            cpw = int(sc if wtm else -sc)
            last_cpw = cpw
            noisy = board.is_capture(mv) or mv.promotion is not None
            if (ply >= SKIP_PLIES and ply % SAMPLE_EVERY == 0
                    and not in_check and not noisy and abs(sc) < 15000):
                samples.append((fen_before, max(-CLAMP_CP, min(CLAMP_CP, cpw))))
            board.push(mv)
            hist.append(uci)
            ply += 1
            if abs(cpw) >= RESIGN_CP:
                decisive += 1
                if decisive >= RESIGN_PLIES:
                    adj_res = "1.0" if cpw > 0 else "0.0"
                    break
            else:
                decisive = 0

        if adj_res is not None:
            res = adj_res
        else:
            r = board.result(claim_draw=True)
            res = {"1-0": "1.0", "0-1": "0.0", "1/2-1/2": "0.5"}.get(r)
            if res is None:   # hit MAX_FULLMOVES with no natural result
                res = ("1.0" if last_cpw >= CUTOFF_WIN_CP
                       else "0.0" if last_cpw <= -CUTOFF_WIN_CP else "0.5")
        for fen, cp in samples:
            fh.write(f"{fen}\t{res}\t{cp}\n")
            kept += 1
        games_done += 1
        if games_done % 50 == 0:
            fh.flush()
            el = time.time() - t0
            print(f"  w{wid}: {games_done} games  {kept:,} pos  "
                  f"{games_done / el:.1f} g/s", flush=True)
    fh.close()


def main() -> None:
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "selfplay.txt")
    nodes = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, (os.cpu_count() or 2) - 1)

    per = total // nw
    parts = [f"{out}.w{i}" for i in range(nw)]
    use_nnue = os.environ.get("FASTCHESS_NNUE", "1")   # default on for this pipeline
    print(f"{total:,} games  {nodes} nodes/move  {nw} workers ({per:,} each)  "
          f"NNUE={use_nnue}", flush=True)
    ps = [Process(target=worker, args=(i, per, parts[i], nodes, 1000 + i, use_nnue))
          for i in range(nw)]
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
                    o.write(line)
                    n += 1
            os.remove(p)
    print(f"\nwrote {out}: {n:,} positions in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
