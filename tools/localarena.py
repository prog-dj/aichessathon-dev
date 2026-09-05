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
import json
import math
import os
import subprocess
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
    white_min_clock_ms: float
    black_min_clock_ms: float


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
    # NOTE: only ONE build can be loaded per process. agent.py does a plain
    # `import fastchess`, which lands in sys.modules under its bare name, so a
    # second in-process load silently reuses the first build's engine - and
    # numba refuses to be re-imported if you try to purge and reload it
    # ("cannot load module more than once per process"). Comparing two builds
    # therefore requires one subprocess per build: see spawn_agent below.
    try:
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    get_move = getattr(module, "get_move", None)
    if not callable(get_move):
        raise RuntimeError(f"{source} has no get_move function")
    return get_move  # type: ignore[no-any-return]


_DRIVER = """
import sys, json, io
sys.path.insert(0, sys.argv[1])
_real = sys.stdout
sys.stdout = io.StringIO()          # agent.py prints diagnostics at import
import agent
sys.stdout = _real
sys.stdout.write(json.dumps({"ready": 1}) + chr(10)); sys.stdout.flush()
for line in sys.stdin:
    r = json.loads(line)
    sys.stdout = io.StringIO()
    try:
        uci = agent.get_move(r["fen"], r["time_left_ms"])
    except Exception as e:
        sys.stdout = _real
        sys.stdout.write(json.dumps({"error": str(e)}) + chr(10)); sys.stdout.flush()
        continue
    sys.stdout = _real
    sys.stdout.write(json.dumps({"move": uci}) + chr(10)); sys.stdout.flush()
"""


class _SubprocessAgent:
    """One build, in its own process. Required for correctness: two builds
    cannot coexist in a single process (see the note in load_agent), so any
    A/B where fastchess.py differs MUST use this, not load_agent."""

    def __init__(self, directory: Path, env: dict[str, str] | None = None) -> None:
        environment = dict(os.environ)
        if env:
            environment.update(env)
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _DRIVER, str(directory.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=environment,
        )
        line = self._proc.stdout.readline()  # type: ignore[union-attr]
        if not line or json.loads(line).get("ready") != 1:
            raise RuntimeError(f"{directory} failed to start")

    def __call__(self, fen: str, time_left_ms: int) -> str:
        self._proc.stdin.write(  # type: ignore[union-attr]
            json.dumps({"fen": fen, "time_left_ms": int(time_left_ms)}) + "\n"
        )
        self._proc.stdin.flush()  # type: ignore[union-attr]
        reply = json.loads(self._proc.stdout.readline())  # type: ignore[union-attr]
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return str(reply["move"])

    def close(self) -> None:
        self._proc.kill()


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
    min_clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}

    def _result(result: str, termination: str, plies: int) -> GameResult:
        return GameResult(result, termination, plies, min_clock[chess.WHITE], min_clock[chess.BLACK])

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            plies = len(board.move_stack)
            name = outcome.termination.name.lower()
            if outcome.winner is None:
                return _result("draw", name, plies)
            return _result("white" if outcome.winner else "black", name, plies)
        if len(board.move_stack) >= ply_cap:
            return _result(_adjudicate(board), "adjudication", len(board.move_stack))

        mover = board.turn
        min_clock[mover] = min(min_clock[mover], clock[mover])
        started_at = time.monotonic()
        try:
            uci = agents[mover](board.fen(), int(clock[mover]))
        except Exception:  # an agent crash is a loss, exactly as on the platform
            return _result(_other(mover), "crash", len(board.move_stack))
        clock[mover] -= (time.monotonic() - started_at) * 1000.0
        if clock[mover] < 0:
            return _result(_other(mover), "flag", len(board.move_stack))

        move = _parse_legal(board, uci)
        if move is None:
            return _result(_other(mover), "illegal", len(board.move_stack))
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
    # env overrides applied to the AGENT side only, so a search-parameter sweep
    # can test a candidate against the same build with stock settings. Tuple of
    # pairs rather than a dict to stay hashable/picklable for the process pool.
    agent_env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _JobResult:
    scored_for_agent: str  # "win" | "draw" | "loss"
    termination: str
    plies: int
    agent_min_clock_ms: float
    opponent_min_clock_ms: float
    agent_is_white: bool


def _run_job(job: _Job) -> _JobResult:
    """Play one game. Each build runs in its OWN subprocess - two builds cannot
    share a process (the second would silently reuse the first's fastchess),
    so this is what makes a fastchess.py-level A/B mean anything."""
    agent = _SubprocessAgent(Path(job.agent_dir), dict(job.agent_env))
    opponent = _SubprocessAgent(Path(job.opponent_dir))
    try:
        white, black = (agent, opponent) if job.agent_is_white else (opponent, agent)
        result = play_game(white, black, job.base_ms, job.increment_ms, job.ply_cap)
    finally:
        agent.close()
        opponent.close()
    scored_for_agent = (
        "draw"
        if result.result == "draw"
        else ("win" if (result.result == "white") == job.agent_is_white else "loss")
    )
    agent_min_clock, opponent_min_clock = (
        (result.white_min_clock_ms, result.black_min_clock_ms)
        if job.agent_is_white
        else (result.black_min_clock_ms, result.white_min_clock_ms)
    )
    return _JobResult(
        scored_for_agent, result.termination, result.plies,
        agent_min_clock, opponent_min_clock, job.agent_is_white,
    )


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
    parser.add_argument(
        "--per-game", action="store_true",
        help="print each game's result and both sides' min-clock, not just the aggregate",
    )
    parser.add_argument(
        "--agent-env", action="append", default=[], metavar="KEY=VALUE",
        help="env override applied to the agent side only (repeatable), e.g. "
             "--agent-env FASTCHESS_NULLMOVE_R_BASE=4",
    )
    arguments = parser.parse_args()

    agent_env: tuple[tuple[str, str], ...] = tuple(
        (k, v) for k, _, v in (pair.partition("=") for pair in arguments.agent_env)
    )

    workers = arguments.workers or max(1, min(8, os.cpu_count() or 2) - 1)
    jobs = [
        _Job(
            agent_dir=str(arguments.agent.resolve()),
            opponent_dir=str(arguments.opponent.resolve()),
            agent_is_white=(game % 2 == 0),
            base_ms=arguments.base_ms,
            increment_ms=arguments.increment_ms,
            ply_cap=arguments.ply_cap,
            agent_env=agent_env,
        )
        for game in range(arguments.games)
    ]

    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    results: list[_JobResult]
    if workers == 1:
        results = [_run_job(job) for job in jobs]
    else:
        # max_tasks_per_child=1: load_agent() gives every game a fresh,
        # uniquely-named module (so numba/TT state can't be shared between
        # games on purpose), but that also means a worker that plays several
        # games in a row never releases the previous game's Engine() - RSS
        # climbs without bound until allocations start failing and the
        # engine silently falls back to the much weaker Python fallback
        # mid-run, corrupting the result. Recycling the process after every
        # game costs a re-compile per game but keeps every game honest.
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
            results = list(pool.map(_run_job, jobs))

    if arguments.per_game:
        print("\nper-game:")
        for i, r in enumerate(results, 1):
            colour = "W" if r.agent_is_white else "B"
            print(
                f"  g{i:>3} agent={colour}  {r.scored_for_agent:<4s} {r.termination:<22s} "
                f"plies={r.plies:>4}  agent_min={r.agent_min_clock_ms/1000:>6.1f}s  "
                f"opp_min={r.opponent_min_clock_ms/1000:>6.1f}s"
            )

    total_plies = 0
    min_clocks: list[float] = []
    for r in results:
        terminations[r.termination] = terminations.get(r.termination, 0) + 1
        total_plies += r.plies
        min_clocks.append(r.agent_min_clock_ms)
        if r.scored_for_agent == "win":
            wins += 1
        elif r.scored_for_agent == "draw":
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
    if min_clocks:
        avg_min_clock = sum(min_clocks) / len(min_clocks)
        panic_frac = sum(1 for c in min_clocks if c < 4000) / len(min_clocks)
        danger_frac = sum(1 for c in min_clocks if c < 10_000) / len(min_clocks)
        print(
            f"agent min-clock: avg {avg_min_clock/1000:.1f}s, "
            f"{panic_frac:.0%} of games dipped under 4s (panic), "
            f"{danger_frac:.0%} under 10s (danger)"
        )
        # the actual question this instrumentation exists to answer: in games where
        # the *opponent* ran into real time trouble, did the agent (with whichever
        # time-management build it's running) come out ahead?
        opp_in_danger = [r for r in results if r.opponent_min_clock_ms < 10_000]
        if opp_in_danger:
            w = sum(1 for r in opp_in_danger if r.scored_for_agent == "win")
            d = sum(1 for r in opp_in_danger if r.scored_for_agent == "draw")
            losses_ = sum(1 for r in opp_in_danger if r.scored_for_agent == "loss")
            sc = (w + 0.5 * d) / len(opp_in_danger)
            print(
                f"in the {len(opp_in_danger)} game(s) where the OPPONENT dipped under 10s: "
                f"agent scored +{w} ={d} -{losses_} ({sc:.0%})"
            )
    broken = {k: v for k, v in terminations.items() if k in FAILED_TERMINATIONS}
    agent_broke = any(
        r.scored_for_agent == "loss" and r.termination in FAILED_TERMINATIONS for r in results
    )
    if agent_broke:
        print("WARNING: your agent lost at least one game to crash / illegal / flag")
    elif broken:
        print(f"note: opponent failed some games ({broken})")


if __name__ == "__main__":
    main()
