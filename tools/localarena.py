"""Windows-friendly in-process arena for fast strength measurement.

The real ``harness/`` speaks the platform's stdin/stdout protocol and only runs
on Linux (it selects on pipes). This tool imports each ``agent.py`` in-process
instead, so it runs anywhere and skips the per-move subprocess round trip, which
matters when a single evaluation needs hundreds of games.

It mirrors ``harness/referee.py``: same natural-end detection, same 300-ply
material adjudication, same wall-clock accounting and flag rule. It does not
reproduce process isolation, the memory cap, or the 4 KB output cap. Use
``make arena`` / ``make gate`` on Linux or CI for a faithful protocol check
before spending upload quota.

    python -m tools.localarena --agent . --opponent baselines/greedy --games 100
    python -m tools.localarena --agent . --opponent ../old-version --games 400 --workers 8
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import chess

GetMove = Callable[[str, int], str]

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100
PLY_CAP = 300
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
FAILED_TERMINATIONS = frozenset({"crash", "illegal", "flag"})


@dataclass(frozen=True)
class GameResult:
    result: str  # "white" | "black" | "draw"
    termination: str
    plies: int


def load_agent(directory: Path) -> GetMove:
    """Import ``directory/agent.py`` under a unique module name and return get_move."""
    directory = directory.resolve()
    source = directory / "agent.py"
    if not source.is_file():
        raise RuntimeError(f"{source} does not exist")
    name = f"_agent_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    get_move = getattr(module, "get_move", None)
    if not callable(get_move):
        raise RuntimeError(f"{source} has no get_move function")
    return get_move  # type: ignore[no-any-return]


def _other(colour: chess.Color) -> str:
    return "black" if colour == chess.WHITE else "white"


def _adjudicate(board: chess.Board) -> str:
    balance = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUES.items()
    )
    if balance > 0:
        return "white"
    if balance < 0:
        return "black"
    return "draw"


def _parse_legal(board: chess.Board, uci: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(uci)
    except (chess.InvalidMoveError, ValueError):
        return None
    return move if move in board.legal_moves else None


def play_game(
    white: GetMove,
    black: GetMove,
    base_ms: int = FAST_BASE_MS,
    increment_ms: int = FAST_INCREMENT_MS,
    ply_cap: int = PLY_CAP,
) -> GameResult:
    board = chess.Board()
    agents = {chess.WHITE: white, chess.BLACK: black}
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            plies = len(board.move_stack)
            name = outcome.termination.name.lower()
            if outcome.winner is None:
                return GameResult("draw", name, plies)
            return GameResult("white" if outcome.winner else "black", name, plies)
        if len(board.move_stack) >= ply_cap:
            return GameResult(_adjudicate(board), "adjudication", len(board.move_stack))

        mover = board.turn
        started_at = time.monotonic()
        try:
            uci = agents[mover](board.fen(), int(clock[mover]))
        except Exception:  # an agent crash is a loss, exactly as on the platform
            return GameResult(_other(mover), "crash", len(board.move_stack))
        clock[mover] -= (time.monotonic() - started_at) * 1000.0
        if clock[mover] < 0:
            return GameResult(_other(mover), "flag", len(board.move_stack))

        move = _parse_legal(board, uci)
        if move is None:
            return GameResult(_other(mover), "illegal", len(board.move_stack))
        board.push(move)
        clock[mover] += increment_ms


@dataclass(frozen=True)
class _Job:
    agent_dir: str
    opponent_dir: str
    agent_is_white: bool
    base_ms: int
    increment_ms: int
    ply_cap: int


def _run_job(job: _Job) -> tuple[str, str, int]:
    """Play one game in a worker. Agents are loaded fresh, as a new process would."""
    agent = load_agent(Path(job.agent_dir))
    opponent = load_agent(Path(job.opponent_dir))
    white, black = (agent, opponent) if job.agent_is_white else (opponent, agent)
    result = play_game(white, black, job.base_ms, job.increment_ms, job.ply_cap)
    scored_for_agent = (
        "draw"
        if result.result == "draw"
        else ("win" if (result.result == "white") == job.agent_is_white else "loss")
    )
    return scored_for_agent, result.termination, result.plies


def _elo(score: float) -> float:
    score = min(max(score, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def _margin(wins: int, draws: int, losses: int) -> float:
    games = wins + draws + losses
    if games == 0:
        return 0.0
    score = (wins + 0.5 * draws) / games
    variance = (
        wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * (0.0 - score) ** 2
    ) / games
    stderr = math.sqrt(variance / games)
    low = _elo(max(score - 1.96 * stderr, 1e-9))
    high = _elo(min(score + 1.96 * stderr, 1 - 1e-9))
    return (high - low) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description="In-process arena with an Elo estimate.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/greedy"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--workers", type=int, default=1, help="0 picks a sensible default")
    arguments = parser.parse_args()

    workers = arguments.workers or max(1, min(8, os.cpu_count() or 2) - 1)
    jobs = [
        _Job(
            agent_dir=str(arguments.agent.resolve()),
            opponent_dir=str(arguments.opponent.resolve()),
            agent_is_white=(game % 2 == 0),
            base_ms=arguments.base_ms,
            increment_ms=arguments.increment_ms,
            ply_cap=arguments.ply_cap,
        )
        for game in range(arguments.games)
    ]

    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    results: list[tuple[str, str, int]]
    if workers == 1:
        results = [_run_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_job, jobs))

    total_plies = 0
    for scored, termination, plies in results:
        terminations[termination] = terminations.get(termination, 0) + 1
        total_plies += plies
        if scored == "win":
            wins += 1
        elif scored == "draw":
            draws += 1
        else:
            losses += 1

    games = wins + draws + losses
    score = (wins + 0.5 * draws) / games if games else 0.0
    print(f"\n{arguments.agent} vs {arguments.opponent} over {games} games")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print(f"Elo {_elo(score):+.0f} +/- {_margin(wins, draws, losses):.0f}")
    print(f"avg game length {total_plies / games:.0f} plies" if games else "no games")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terminations.items())))
    broken = {k: v for k, v in terminations.items() if k in FAILED_TERMINATIONS}
    agent_broke = any(
        scored == "loss" and termination in FAILED_TERMINATIONS
        for scored, termination, _ in results
    )
    if agent_broke:
        print("WARNING: your agent lost at least one game to crash / illegal / flag")
    elif broken:
        print(f"note: opponent failed some games ({broken})")


if __name__ == "__main__":
    main()
