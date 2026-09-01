"""Build weights/book.json from strong games.

    python -m training.build_book --input https://database.lichess.org/... \
        --out weights/book.json --min-elo 2400 --plies 16 --min-count 8

For every position seen up to --plies, tally the moves played in games where
both sides clear --min-elo. Keep moves played at least --min-count times; the
weight is the play count. book.py reads the result.
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


def build(pgn_path: str, plies: int, min_elo: int, limit: int) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    stream = open_text(pgn_path)
    games = 0
    try:
        while games < limit:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            headers = game.headers
            try:
                if min(int(headers.get("WhiteElo", 0)), int(headers.get("BlackElo", 0))) < min_elo:
                    continue
            except ValueError:
                continue
            games += 1
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= plies:
                    break
                key = f"{chess.polyglot.zobrist_hash(board):016x}"
                counts[key][move.uci()] += 1
                board.push(move)
            if games % 5000 == 0:
                print(f"{games} games, {len(counts)} positions")
    finally:
        stream.close()
    print(f"scanned {games} games")
    return {k: dict(v) for k, v in counts.items()}


def prune(counts: dict[str, dict[str, int]], min_count: int) -> dict[str, list[list[object]]]:
    book: dict[str, list[list[object]]] = {}
    for key, moves in counts.items():
        kept = [[uci, n] for uci, n in moves.items() if n >= min_count]
        if kept:
            book[key] = sorted(kept, key=lambda pair: pair[1], reverse=True)
    return book


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="PGN path or http(s) URL")
    parser.add_argument("--out", type=Path, default=Path("weights/book.json"))
    parser.add_argument("--plies", type=int, default=16)
    parser.add_argument("--min-elo", type=int, default=2400)
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--limit", type=int, default=1_000_000)
    args = parser.parse_args()

    counts = build(args.input, args.plies, args.min_elo, args.limit)
    book = prune(counts, args.min_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(book, separators=(",", ":")))
    print(f"wrote {args.out} ({len(book)} positions, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
