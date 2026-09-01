"""Syzygy tablebase probing. Skipped unless the 3-4-man tables are present."""

from __future__ import annotations

from pathlib import Path

import chess
import pytest

from endgame import Tablebase

_TABLES = Path(__file__).resolve().parent.parent / "weights" / "syzygy"

pytestmark = pytest.mark.skipif(
    not (_TABLES / "KRvK.rtbw").exists(), reason="run training.get_tablebases first"
)


def _tb() -> Tablebase:
    return Tablebase(_TABLES)


def test_off_table_returns_none() -> None:
    assert _tb().best_moves(chess.Board()) is None


def test_krvk_every_approved_move_keeps_the_win() -> None:
    board = chess.Board("7k/8/6K1/8/8/8/8/1R6 w - - 0 1")  # KR vs K, White winning
    tb = _tb()
    approved = tb.best_moves(board)
    assert approved is not None and approved
    for move in approved:
        board.push(move)
        assert tb._tb.probe_wdl(board) <= 0  # after our move the opponent is not winning
        board.pop()


def test_kqvk_never_stalemates_or_drops_the_queen() -> None:
    board = chess.Board("4k3/8/4K3/8/8/8/8/7Q w - - 0 1")  # KQ vs K, White winning
    tb = _tb()
    approved = tb.best_moves(board)
    assert approved is not None and approved
    for move in approved:
        board.push(move)
        assert not board.is_stalemate()
        assert tb._tb.probe_wdl(board) <= 0
        board.pop()


def test_drawn_kqvkq_holds_the_draw() -> None:
    board = chess.Board("8/8/8/3qk3/8/8/3QK3/8 w - - 0 1")
    approved = _tb().best_moves(board)
    assert approved is not None and len(approved) >= 1
