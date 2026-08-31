"""Packed training records must rebuild exactly what ``encoding`` produces."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from encoding import encode_board
from tests.test_encoding import POSITIONS
from training.pack import RECORD_BYTES, pack_position, unpack_position


@pytest.mark.parametrize("board", POSITIONS, ids=lambda b: b.fen())
def test_pack_round_trips_to_encode_board(board: chess.Board) -> None:
    record = pack_position(board)
    assert record.shape == (RECORD_BYTES,)
    assert record.dtype == np.uint8
    assert np.array_equal(unpack_position(record), encode_board(board))


def test_en_passant_survives_packing() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("a7a6")
    board.push_uci("e4e5")
    board.push_uci("d7d5")  # ep target d6, white to move
    assert board.ep_square == chess.D6
    assert np.array_equal(unpack_position(pack_position(board)), encode_board(board))


def test_black_to_move_ep_survives_packing() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    board.push_uci("g1f3")
    board.push_uci("a7a6")
    board.push_uci("f1c4")
    board.push_uci("a6a5")
    board.push_uci("e1g1")
    board.push_uci("a5a4")
    board.push_uci("b2b4")  # ep target b3, black to move
    assert board.ep_square == chess.B3
    assert np.array_equal(unpack_position(pack_position(board)), encode_board(board))
