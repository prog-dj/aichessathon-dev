"""AlphaBeta search: legality, board safety, and basic tactics.

Runs with no evaluator (pure material + piece-square), so it is fast and
deterministic in CI.
"""

from __future__ import annotations

import time

import chess
import pytest

from alphabeta import AlphaBetaConfig, AlphaBetaSearch

_NO_NET = AlphaBetaConfig(use_net_policy=False, use_net_value=False)


def _run(fen: str, seconds: float = 1.0) -> chess.Move:
    return AlphaBetaSearch(None, _NO_NET).run(chess.Board(fen), time.monotonic() + seconds)


def test_returns_legal_move_and_does_not_touch_the_board() -> None:
    board = chess.Board("r2q1rk1/1b1nbppp/p2ppn2/1p6/3NP3/1BN1B3/PPP1QPPP/R4RK1 w - - 0 12")
    before = board.fen()
    move = AlphaBetaSearch(None, _NO_NET).run(board, time.monotonic() + 0.3)
    assert move in board.legal_moves
    assert board.fen() == before  # the search must leave the caller's board alone


def test_grabs_a_hanging_queen() -> None:
    assert _run("rnb1kbnr/pppp1ppp/8/7q/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3").uci() == "d1h5"


def test_finds_mate_in_one() -> None:
    assert _run("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1").uci() == "a1a8"


def test_recaptures_to_stay_level() -> None:
    # White just played Bxc6; Black must recapture or be down a piece.
    board = chess.Board("r1bqkbnr/pp1p1ppp/2B5/2p1p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4")
    move = AlphaBetaSearch(None, _NO_NET).run(board, time.monotonic() + 0.5)
    assert move in {chess.Move.from_uci("b7c6"), chess.Move.from_uci("d7c6")}


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "8/2k5/3p4/p2P1p2/P2P1P2/8/2K5/8 w - - 0 1",
        "7k/8/8/8/8/8/6q1/7K w - - 0 1",
    ],
)
def test_stays_legal_on_odd_positions(fen: str) -> None:
    board = chess.Board(fen)
    assert _run(fen, 0.2) in board.legal_moves
