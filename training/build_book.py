"""Build weights/book.json.

Two sources:

    # from strong games (move played, weighted by frequency)
    python -m training.build_book pgn --input elite.pgn.zst --out weights/book.json \
        --min-elo 2400 --plies 16 --min-count 8

    # from the Lichess Stockfish-eval database (best move for opening positions)
    python -m training.build_book eval \
        --input https://database.lichess.org/lichess_db_eval.jsonl.zst \
        --out weights/book.json --limit 8000000

book.py reads the result: a map from Zobrist key (hex) to [[uci, weight], ...].
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot

from training.prepare import open_text

_MinCounts = dict[str, dict[str, int]]
_START_OCCUPIED = chess.Board().occupied


def _key(board: chess.Board) -> str:
    return f"{chess.polyglot.zobrist_hash(board):016x}"


def _is_opening(board: chess.Board, max_moved: int) -> bool:
    """No fullmove counter in eval-DB FENs, so gauge it by distance from the start:
    how many squares differ from the initial piece layout."""
    return chess.popcount(board.occupied ^ _START_OCCUPIED) <= 2 * max_moved


def from_pgn(pgn_path: str, plies: int, min_elo: int, limit: int) -> _MinCounts:
    counts: _MinCounts = defaultdict(lambda: defaultdict(int))
    stream = open_text(pgn_path)
    games = 0
    try:
        while games < limit:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            h = game.headers
            try:
                if min(int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))) < min_elo:
                    continue
            except ValueError:
                continue
            games += 1
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= plies:
                    break
                counts[_key(board)][move.uci()] += 1
                board.push(move)
            if games % 5000 == 0:
                print(f"{games} games, {len(counts)} positions")
    finally:
        stream.close()
    print(f"scanned {games} games")
    return counts


def from_eval(jsonl_path: str, max_moved: int, min_depth: int, limit: int) -> _MinCounts:
    counts: _MinCounts = defaultdict(lambda: defaultdict(int))
    stream = open_text(jsonl_path)
    seen = 0
    try:
        for line in stream:
            seen += 1
            if seen % 500_000 == 0:
                print(f"{seen:,} lines, {len(counts)} positions")
            if seen > limit:
                break
            row = json.loads(line)
            evals = row.get("evals") or []
            if not evals:
                continue
            best = max(evals, key=lambda e: e.get("depth", 0))
            if best.get("depth", 0) < min_depth:
                continue
            pvs = best.get("pvs") or []
            if not pvs or not pvs[0].get("line"):
                continue
            fen = row["fen"]
            board = chess.Board(fen if len(fen.split()) == 6 else fen + " 0 1")
            if not _is_opening(board, max_moved):
                continue
            move = pvs[0]["line"].split()[0]
            if chess.Move.from_uci(move) in board.legal_moves:
                counts[_key(board)][move] += max(1, best.get("depth", 20) - 15)
    finally:
        stream.close()
    print(f"scanned {seen:,} lines")
    return counts


def prune(counts: _MinCounts, min_count: int) -> dict[str, list[list[object]]]:
    book: dict[str, list[list[object]]] = {}
    for key, moves in counts.items():
        kept = [[uci, n] for uci, n in moves.items() if n >= min_count]
        if kept:
            book[key] = sorted(kept, key=lambda pair: pair[1], reverse=True)[:4]
    return book


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("pgn")
    p.add_argument("--input", required=True)
    p.add_argument("--out", type=Path, default=Path("weights/book.json"))
    p.add_argument("--plies", type=int, default=16)
    p.add_argument("--min-elo", type=int, default=2400)
    p.add_argument("--min-count", type=int, default=8)
    p.add_argument("--limit", type=int, default=1_000_000)

    e = sub.add_parser("eval")
    e.add_argument("--input", required=True)
    e.add_argument("--out", type=Path, default=Path("weights/book.json"))
    e.add_argument("--max-moved", type=int, default=7, help="pieces moved from home = ~plies/2")
    e.add_argument("--min-depth", type=int, default=24)
    e.add_argument("--min-count", type=int, default=2)
    e.add_argument("--limit", type=int, default=10_000_000)

    args = parser.parse_args()
    if args.mode == "pgn":
        counts = from_pgn(args.input, args.plies, args.min_elo, args.limit)
    else:
        counts = from_eval(args.input, args.max_moved, args.min_depth, args.limit)

    book = prune(counts, args.min_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(book, separators=(",", ":")))
    print(f"wrote {args.out} ({len(book)} positions, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
