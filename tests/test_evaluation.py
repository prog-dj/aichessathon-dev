"""The hand-crafted evaluation."""

from __future__ import annotations

import chess

from evaluation import cp_to_scalar, evaluate_cp, material_pst_cp


def test_startpos_is_roughly_balanced() -> None:
    assert abs(material_pst_cp(chess.Board())) <= 20
    assert abs(evaluate_cp(chess.Board()) - 12) <= 40  # tempo only


def test_side_to_move_perspective() -> None:
    # White is a whole rook up. Score is positive for White, negative for Black.
    white = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    black = chess.Board("4k3/8/8/8/8/8/8/R3K3 b - - 0 1")
    assert material_pst_cp(white) > 400
    assert material_pst_cp(black) < -400


def test_a_hung_queen_shows_up() -> None:
    balanced = chess.Board()
    down_a_queen = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    assert material_pst_cp(down_a_queen) < material_pst_cp(balanced) - 800


def test_bishop_pair_bonus() -> None:
    two_bishops = chess.Board("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1")
    one_bishop = chess.Board("2n1k3/8/8/8/8/8/8/2B1K3 w - - 0 1")
    two_extra = evaluate_cp(two_bishops) - material_pst_cp(two_bishops)
    one_extra = evaluate_cp(one_bishop) - material_pst_cp(one_bishop)
    assert two_extra > one_extra


def test_cp_to_scalar_monotone_and_bounded() -> None:
    assert cp_to_scalar(0.0) == 0.0
    assert 0.9 < cp_to_scalar(2000.0) <= 1.0
    assert -1.0 <= cp_to_scalar(-2000.0) < -0.9
    assert cp_to_scalar(300.0) > cp_to_scalar(100.0) > cp_to_scalar(-100.0)
