"""PUCT search behaviour that does not depend on a trained net.

A stub evaluator returns a flat policy and zero value, so any good move the
search finds comes from the tree and the terminal handling, not the network.
"""

from __future__ import annotations

import time

import chess
import pytest

from search import PuctSearch


class FlatEvaluator:
    def evaluate(
        self, boards: list[chess.Board]
    ) -> list[tuple[dict[chess.Move, float], float]]:
        out = []
        for board in boards:
            moves = list(board.legal_moves)
            uniform = {move: 1.0 / len(moves) for move in moves} if moves else {}
            out.append((uniform, 0.0))
        return out


def _run(board: chess.Board, seconds: float = 2.0) -> chess.Move:
    return PuctSearch(FlatEvaluator()).run(board, time.monotonic() + seconds)  # type: ignore[arg-type]


def test_returns_a_legal_move() -> None:
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4")
    assert _run(board) in board.legal_moves


def test_single_legal_move_is_immediate() -> None:
    board = chess.Board("8/8/8/8/8/8/6q1/7K w - - 0 1")  # in check, only Kxg2
    moves = list(board.legal_moves)
    assert len(moves) == 1
    assert _run(board, 0.05) == moves[0]


def test_finds_mate_in_one() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    assert _run(board).uci() == "a1a8"


def test_avoids_stalemate_when_winning() -> None:
    # White is massively up; Qg6 would stalemate, many moves keep the win.
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    move = _run(board)
    board.push(move)
    assert not board.is_stalemate()


def test_forced_check_is_searched() -> None:
    # back-rank mate: Re8 is the only mate and it is a checking move
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    assert _run(board).uci() == "e1e8"


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "8/2k5/3p4/p2P1p2/P2P1P2/8/2K5/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
])
def test_never_crashes_and_stays_legal(fen: str) -> None:
    board = chess.Board(fen)
    assert _run(board, 0.3) in board.legal_moves
