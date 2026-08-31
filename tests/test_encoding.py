"""Round-trip and invariant checks for :mod:`encoding`."""

from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from encoding import (
    N_PLANES,
    POLICY_SIZE,
    encode_board,
    index_to_move,
    legal_policy_indices,
    move_to_index,
)


def _random_positions(count: int, seed: int = 0) -> list[chess.Board]:
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    for _ in range(count):
        board = chess.Board()
        for _ in range(rng.randint(0, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        boards.append(board)
    return boards


POSITIONS = _random_positions(400)


@pytest.mark.parametrize("board", POSITIONS, ids=lambda b: b.fen())
def test_every_legal_move_round_trips(board: chess.Board) -> None:
    for move in board.legal_moves:
        index = move_to_index(board, move)
        assert 0 <= index < POLICY_SIZE
        assert index_to_move(board, index) == move


@pytest.mark.parametrize("board", POSITIONS, ids=lambda b: b.fen())
def test_legal_indices_are_unique(board: chess.Board) -> None:
    indices = [index for _, index in legal_policy_indices(board)]
    assert len(indices) == len(set(indices))


def test_encode_shape_and_determinism() -> None:
    board = chess.Board()
    first = encode_board(board)
    assert first.shape == (N_PLANES, 8, 8)
    assert first.dtype == np.float32
    assert np.array_equal(first, encode_board(board))


def test_startpos_piece_planes() -> None:
    planes = encode_board(chess.Board())
    # White to move: own pawns on rank 2 (row 1), own king on e1.
    assert planes[0, 1].sum() == 8
    assert planes[5, 0, 4] == 1.0
    # Opponent pawns on rank 7 (row 6), opponent king on e8.
    assert planes[6, 6].sum() == 8
    assert planes[11, 7, 4] == 1.0


@pytest.mark.parametrize("board", POSITIONS[:50], ids=lambda b: b.fen())
def test_side_to_move_canonicalisation_is_symmetric(board: chess.Board) -> None:
    # board.mirror() swaps colours, flips vertically and swaps the side to move,
    # so a canonicalised encoding must be identical for a position and its mirror.
    assert np.array_equal(encode_board(board), encode_board(board.mirror()))


def test_underpromotion_indices() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    seen = set()
    for move in board.legal_moves:
        if move.promotion is not None:
            index = move_to_index(board, move)
            seen.add(move.promotion)
            assert index_to_move(board, index) == move
    assert seen == {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}


def test_black_promotion_round_trips() -> None:
    board = chess.Board("K6k/8/8/8/8/8/6p1/8 b - - 0 1")
    for move in board.legal_moves:
        index = move_to_index(board, move)
        assert index_to_move(board, index) == move
