"""The hand-crafted evaluation."""

from __future__ import annotations

import chess
import pytest

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


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NP3/1BN1B3/PPP1QPPP/R4RK1 w - - 0 12",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    ],
)
def test_evaluate_is_colour_symmetric(fen: str) -> None:
    board = chess.Board(fen)
    assert evaluate_cp(board) == evaluate_cp(board.mirror())


def test_incremental_material_matches_recompute() -> None:
    import random

    from evaluation import material_pst_white, mpst_quiet_delta

    rng = random.Random(0)
    for _ in range(60):
        board = chess.Board()
        mw = material_pst_white(board)
        for _ in range(rng.randint(0, 40)):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            delta = mpst_quiet_delta(board, move)
            board.push(move)
            mw = mw + delta if delta is not None else material_pst_white(board)
            assert abs(mw - material_pst_white(board)) < 0.01


def test_passed_pawn_and_open_file_help() -> None:
    plain = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    passed = chess.Board("4k3/8/4P3/8/8/8/8/4K3 w - - 0 1")  # further advanced passer
    assert evaluate_cp(passed) > evaluate_cp(plain)


def test_cp_to_scalar_monotone_and_bounded() -> None:
    assert cp_to_scalar(0.0) == 0.0
    assert 0.9 < cp_to_scalar(2000.0) <= 1.0
    assert -1.0 <= cp_to_scalar(-2000.0) < -0.9
    assert cp_to_scalar(300.0) > cp_to_scalar(100.0) > cp_to_scalar(-100.0)
