"""Paired UCI-engine benchmark runner.

Example::

    uv run python -m harness.uci_benchmark \
        --new "./Engine_New --uci" \
        --old "/path/to/Engine_Old" \
        --openings openings.epd --pgn match.pgn

Opening files may contain one FEN per line, EPD records, or PGN games.  PGN
games contribute the position after ``--plies`` main-line plies.  Every
position is played twice with the colours swapped.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import math
import shlex
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn

DEFAULT_TIME_MS = 120_000
DEFAULT_PLIES = 4
DEFAULT_MAX_PLIES = 400
MATE_SCORE_CP = 100_000
PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


@dataclass(frozen=True)
class Opening:
    name: str
    fen: str


@dataclass(frozen=True)
class GameResult:
    opening: str
    new_is_white: bool
    result: str
    termination: str
    pgn: str


def load_openings(path: Path, plies: int = DEFAULT_PLIES) -> list[Opening]:
    """Load FEN, EPD, or PGN openings from ``path``.

    Plain-text files are detected as FEN/EPD line files unless their suffix is
    ``.pgn``.  Blank lines and lines beginning with ``#`` are ignored.
    """
    if path.suffix.lower() == ".pgn" or path.name.lower().endswith(".pgn.gz"):
        return _load_pgn(path, plies)
    records = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    records = [line for line in records if not line.startswith("#")]
    openings: list[Opening] = []
    for index, record in enumerate(records, 1):
        openings.append(Opening(f"{path.stem}-{index}", _record_to_fen(record)))
    if not openings:
        raise ValueError(f"opening file is empty: {path}")
    return openings


def _load_pgn(path: Path, plies: int) -> list[Opening]:
    openings = []
    opener = gzip.open if path.name.lower().endswith(".gz") else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        index = 0
        while game := chess.pgn.read_game(handle):
            board = game.board()
            for move_index, move in enumerate(game.mainline_moves()):
                if move_index >= plies:
                    break
                board.push(move)
            index += 1
            name = game.headers.get("Opening", f"{path.stem}-{index}")
            openings.append(Opening(name, board.fen()))
    if not openings:
        raise ValueError(f"PGN contains no games: {path}")
    return openings


def _record_to_fen(record: str) -> str:
    fields = record.split()
    fen: str
    try:
        fen = chess.Board(record).fen()
    except ValueError:
        looks_like_epd = len(fields) >= 5 and not fields[4].lstrip("-").isdigit()
        if looks_like_epd:
            try:
                board = chess.Board()
                board.set_epd(record)
                fen = board.fen()
            except ValueError as error:
                raise ValueError(f"not a FEN or EPD record: {record}") from error
        elif len(fields) >= 2 and fields[1] in {"w", "b"}:
            padded = [*fields[:6], "0", "1"]
            fen = " ".join(padded[:6])
            try:
                fen = chess.Board(fen).fen()
            except ValueError as error:
                raise ValueError(f"not a FEN or EPD record: {record}") from error
        else:
            try:
                board = chess.Board()
                board.set_epd(record)
                fen = board.fen()
            except ValueError as error:
                raise ValueError(f"not a FEN or EPD record: {record}") from error
    return fen


def run_benchmark(
    new_command: Sequence[str],
    old_command: Sequence[str],
    openings: Iterable[Opening],
    pgn_path: Path,
    time_ms: int = DEFAULT_TIME_MS,
    max_plies: int = DEFAULT_MAX_PLIES,
    draw_adjudication: bool = False,
    resign_cp: int | None = None,
) -> list[GameResult]:
    """Run a paired benchmark and append all games to ``pgn_path``."""
    opening_list = list(openings)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[GameResult] = []
    with pgn_path.open("w", encoding="utf-8") as pgn_handle:
        for round_index, opening in enumerate(opening_list, 1):
            for new_is_white in (True, False):
                game_result = _play_game(
                    new_command,
                    old_command,
                    opening,
                    new_is_white,
                    time_ms,
                    max_plies,
                    draw_adjudication,
                    resign_cp,
                )
                results.append(game_result)
                pgn_handle.write(game_result.pgn + "\n\n")
                pgn_handle.flush()
            _print_summary(results, round_index, opening.name)
    return results


def _play_game(
    new_command: Sequence[str],
    old_command: Sequence[str],
    opening: Opening,
    new_is_white: bool,
    time_ms: int,
    max_plies: int,
    draw_adjudication: bool,
    resign_cp: int | None,
) -> GameResult:
    board = chess.Board(opening.fen)
    engines = {
        chess.WHITE: chess.engine.SimpleEngine.popen_uci(
            list(new_command if new_is_white else old_command)
        ),
        chess.BLACK: chess.engine.SimpleEngine.popen_uci(
            list(old_command if new_is_white else new_command)
        ),
    }
    clocks = {chess.WHITE: float(time_ms), chess.BLACK: float(time_ms)}
    termination = "unknown"
    try:
        quiet_draw_plies = 0
        while len(board.move_stack) < max_plies:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                termination = outcome.termination.name.lower()
                return _game_result(opening, new_is_white, board, outcome.result(), termination)
            if draw_adjudication and _draw_adjudicated(board, quiet_draw_plies):
                return _game_result(opening, new_is_white, board, "1/2-1/2", "draw_adjudication")
            if resign_cp is not None and _resign_adjudicated(board, resign_cp):
                winner = chess.BLACK if _material_cp(board) > resign_cp else chess.WHITE
                result = "0-1" if winner == chess.BLACK else "1-0"
                return _game_result(
                    opening, new_is_white, board, result, "resignation_adjudication"
                )

            mover = board.turn
            engine = engines[mover]
            started = time.monotonic()
            try:
                play_result = engine.play(
                    board,
                    chess.engine.Limit(time=max(0.001, clocks[mover] / 1000.0)),
                )
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
                termination = "crash"
                return _game_result(
                    opening, new_is_white, board, _winner_result(not mover), termination
                )
            clocks[mover] -= (time.monotonic() - started) * 1000.0
            if clocks[mover] <= 0:
                return _game_result(opening, new_is_white, board, _winner_result(not mover), "flag")
            move = play_result.move
            if move is None or move not in board.legal_moves:
                return _game_result(
                    opening, new_is_white, board, _winner_result(not mover), "illegal"
                )
            board.push(move)
            quiet_draw_plies = quiet_draw_plies + 1 if abs(_material_cp(board)) <= 10 else 0
        termination = "ply_cap"
        return _game_result(opening, new_is_white, board, "1/2-1/2", termination)
    finally:
        for engine in engines.values():
            with contextlib.suppress(chess.engine.EngineError):
                engine.quit()


def _winner_result(winner: chess.Color) -> str:
    return "1-0" if winner == chess.WHITE else "0-1"


def _material_cp(board: chess.Board) -> int:
    return sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUES_CP.items()
    )


def _draw_adjudicated(board: chess.Board, quiet_draw_plies: int) -> bool:
    return board.fullmove_number > 40 and quiet_draw_plies >= 20


def _resign_adjudicated(board: chess.Board, threshold_cp: int) -> bool:
    return abs(_material_cp(board)) >= threshold_cp


def _game_result(
    opening: Opening, new_is_white: bool, board: chess.Board, result: str, termination: str
) -> GameResult:
    game = chess.pgn.Game.from_board(board)
    game.headers.update(
        {
            "Event": "UCI paired benchmark",
            "Opening": opening.name,
            "FEN": opening.fen,
            "SetUp": "1",
            "EngineNewColor": "White" if new_is_white else "Black",
            "Result": result,
            "Termination": termination,
        }
    )
    return GameResult(opening.name, new_is_white, result, termination, str(game))


def _print_summary(results: Sequence[GameResult], round_index: int, opening: str) -> None:
    wins = sum(_new_score(result) == 1.0 for result in results)
    draws = sum(_new_score(result) == 0.5 for result in results)
    losses = len(results) - wins - draws
    score = (wins + draws / 2) / len(results)
    elo = _elo_difference(score)
    margin = 1.96 * math.sqrt(max(score * (1 - score), 0.0) / len(results))
    low = max(0.0, score - margin)
    high = min(1.0, score + margin)
    los = _probability_above_half(score, len(results))
    print(
        f"round {round_index}: {opening}; {wins}W/{draws}D/{losses}L, "
        f"score {score:.1%}, win rate {wins / len(results):.1%}, "
        f"Elo {elo:+.0f} ({_elo_difference(low):+.0f}..{_elo_difference(high):+.0f}), "
        f"LOS {los:.1%}",
        flush=True,
    )


def _new_score(result: GameResult) -> float:
    new_won = (result.result == "1-0") == result.new_is_white
    if result.result == "1/2-1/2":
        return 0.5
    return 1.0 if new_won else 0.0


def _elo_difference(score: float) -> float:
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return 400.0 * math.log10(score / (1.0 - score))


def _probability_above_half(score: float, games: int) -> float:
    if games == 0 or score <= 0.0:
        return 0.0
    if score >= 1.0:
        return 1.0
    standard_error = math.sqrt(score * (1.0 - score) / games)
    if standard_error == 0.0:
        return 1.0 if score > 0.5 else 0.0
    return 0.5 * (1.0 + math.erf((score - 0.5) / (standard_error * math.sqrt(2.0))))


def _parse_command(value: str) -> list[str]:
    command = shlex.split(value)
    if not command:
        raise argparse.ArgumentTypeError("engine command cannot be empty")
    return command


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a paired UCI engine benchmark.")
    parser.add_argument("--new", required=True, type=_parse_command, help="Engine_New command line")
    parser.add_argument("--old", required=True, type=_parse_command, help="Engine_Old command line")
    parser.add_argument("--openings", required=True, type=Path, help="FEN, EPD, or PGN suite")
    parser.add_argument("--pgn", type=Path, default=Path("benchmark.pgn"))
    parser.add_argument("--plies", type=int, default=DEFAULT_PLIES, help="PGN setup plies")
    parser.add_argument("--time-ms", type=int, default=DEFAULT_TIME_MS)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument(
        "--draw-adjudication", action="store_true", help="Draw after 10 quiet moves past move 40"
    )
    parser.add_argument(
        "--resign-cp", type=int, help="Optional material-based resignation threshold in centipawns"
    )
    arguments = parser.parse_args(argv)
    if arguments.time_ms <= 0 or arguments.max_plies <= 0 or arguments.plies < 0:
        parser.error("time, max plies, and setup plies must be non-negative/positive")
    openings = load_openings(arguments.openings, arguments.plies)
    results = run_benchmark(
        arguments.new,
        arguments.old,
        openings,
        arguments.pgn,
        arguments.time_ms,
        arguments.max_plies,
        arguments.draw_adjudication,
        arguments.resign_cp,
    )
    score = sum(_new_score(result) for result in results) / len(results)
    print(f"completed {len(results)} games; Engine_New score {score:.1%}; PGN {arguments.pgn}")


if __name__ == "__main__":
    main()