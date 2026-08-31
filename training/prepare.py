"""Turn raw public data into training shards.

    python -m training.prepare eval --input data/raw/lichess_db_eval.jsonl.zst \
        --out data/shards/eval --limit 8000000
    python -m training.prepare pgn  --input data/raw/elite.pgn.zst \
        --out data/shards/games --min-elo 2300 --limit 6000000

Stream A (eval): Lichess's Stockfish-annotated positions. Value target from the
deepest eval's score, policy target its best move. Scores in that file are White
POV; we flip to side-to-move POV.

Stream B (pgn): games where both players clear --min-elo. Value target is the
game result, policy target the move actually played, from ply --skip-plies on.
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import chess
import chess.pgn

from encoding import move_to_index
from training.labels import Wdl, cp_to_wdl, result_to_wdl, score_to_cp
from training.pack import pack_position

Sample = tuple["object", int, Wdl]  # (uint8 record, policy index, wdl)


def _board_from_eval_fen(fen: str) -> chess.Board:
    fields = fen.split()
    if len(fields) == 4:
        fen = fen + " 0 1"
    return chess.Board(fen)


def iter_eval_samples(lines: Iterable[str], min_depth: int = 12) -> Iterator[Sample]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        evals = row.get("evals") or []
        if not evals:
            continue
        best_eval = max(evals, key=lambda e: e.get("depth", 0))
        if best_eval.get("depth", 0) < min_depth:
            continue
        pvs = best_eval.get("pvs") or []
        if not pvs or "line" not in pvs[0] or not pvs[0]["line"]:
            continue
        try:
            board = _board_from_eval_fen(row["fen"])
            move = chess.Move.from_uci(pvs[0]["line"].split()[0])
        except (ValueError, KeyError):
            continue
        if move not in board.legal_moves:
            continue
        cp_white = score_to_cp(pvs[0].get("cp"), pvs[0].get("mate"))
        cp_stm = cp_white if board.turn == chess.WHITE else -cp_white
        yield pack_position(board), move_to_index(board, move), cp_to_wdl(cp_stm)


def iter_pgn_samples(
    pgn: io.TextIOBase, min_elo: int = 2200, skip_plies: int = 12, stride: int = 1
) -> Iterator[Sample]:
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            return
        headers = game.headers
        if headers.get("Result") not in {"1-0", "0-1", "1/2-1/2"}:
            continue
        try:
            white_elo = int(headers.get("WhiteElo", "0"))
            black_elo = int(headers.get("BlackElo", "0"))
        except ValueError:
            continue
        if white_elo < min_elo or black_elo < min_elo:
            continue
        result = headers["Result"]
        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= skip_plies and (ply - skip_plies) % stride == 0:
                wdl = result_to_wdl(result, board.turn == chess.WHITE)
                yield pack_position(board), move_to_index(board, move), wdl
            board.push(move)


def open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".zst":
        import zstandard

        reader = zstandard.ZstdDecompressor().stream_reader(path.open("rb"))
        return io.TextIOWrapper(reader, encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _run(samples: Iterator[Sample], out: Path, limit: int) -> None:
    from training.data import ShardWriter

    writer = ShardWriter(out)
    for record, policy_index, wdl in samples:
        writer.add(record, policy_index, wdl)  # type: ignore[arg-type]
        if writer.total % 100_000 == 0:
            print(f"{writer.total:,} samples")
        if writer.total >= limit:
            break
    writer.flush()
    print(f"done: {writer.total:,} samples in {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    eval_parser = sub.add_parser("eval", help="Lichess Stockfish eval database")
    eval_parser.add_argument("--input", type=Path, required=True)
    eval_parser.add_argument("--out", type=Path, required=True)
    eval_parser.add_argument("--limit", type=int, default=10_000_000)
    eval_parser.add_argument("--min-depth", type=int, default=12)

    pgn_parser = sub.add_parser("pgn", help="PGN archive filtered by rating")
    pgn_parser.add_argument("--input", type=Path, required=True)
    pgn_parser.add_argument("--out", type=Path, required=True)
    pgn_parser.add_argument("--limit", type=int, default=10_000_000)
    pgn_parser.add_argument("--min-elo", type=int, default=2200)
    pgn_parser.add_argument("--skip-plies", type=int, default=12)
    pgn_parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()
    stream = open_text(args.input)
    try:
        if args.mode == "eval":
            _run(iter_eval_samples(stream, args.min_depth), args.out, args.limit)
        else:
            _run(
                iter_pgn_samples(stream, args.min_elo, args.skip_plies, args.stride),
                args.out,
                args.limit,
            )
    finally:
        stream.close()


if __name__ == "__main__":
    main()
