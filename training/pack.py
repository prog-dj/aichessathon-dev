"""Compact 36-byte position records for training shards.

Training-only, numpy + python-chess, no torch. :func:`unpack_position` must
rebuild exactly the planes that :func:`encoding.encode_board` produces, so a
shard can store 36 bytes per position instead of a 4864-byte float stack.

Layout (bytes):
    0-31  64 squares, one nibble each: 0 empty, 1-6 own P N B R Q K,
          7-12 opponent P N B R Q K  (already canonicalised to side to move)
    32    high nibble: castling  bit0 own K, bit1 own Q, bit2 opp K, bit3 opp Q
          low nibble:  en passant target file 0-7, or 15 for none
    33    halfmove clock, clamped to 255
    34    fullmove number, clamped to 255
    35    reserved, always 0
"""

from __future__ import annotations

import chess
import numpy as np
import numpy.typing as npt

from encoding import N_PLANES

RECORD_BYTES = 36
_EP_NONE = 15
_EP_CANON_RANK = 5  # after canonicalisation the ep target always sits on rank 6

Record = npt.NDArray[np.uint8]
Planes = npt.NDArray[np.float32]


def pack_position(board: chess.Board) -> Record:
    flip = board.turn == chess.BLACK
    stm = board.turn

    squares = np.zeros(64, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        cell = square ^ 56 if flip else square
        own = piece.color == stm
        squares[cell] = (piece.piece_type - 1) + (0 if own else 6) + 1

    record = np.zeros(RECORD_BYTES, dtype=np.uint8)
    record[:32] = (squares[0::2] << 4) | squares[1::2]

    castling = (
        (int(board.has_kingside_castling_rights(stm)) << 0)
        | (int(board.has_queenside_castling_rights(stm)) << 1)
        | (int(board.has_kingside_castling_rights(not stm)) << 2)
        | (int(board.has_queenside_castling_rights(not stm)) << 3)
    )
    ep_file = _EP_NONE
    if board.ep_square is not None:
        ep_cell = board.ep_square ^ 56 if flip else board.ep_square
        ep_file = ep_cell & 7
    record[32] = (castling << 4) | ep_file
    record[33] = min(board.halfmove_clock, 255)
    record[34] = min(board.fullmove_number, 255)
    return record


def unpack_position(record: Record) -> Planes:
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)

    squares = np.zeros(64, dtype=np.uint8)
    squares[0::2] = record[:32] >> 4
    squares[1::2] = record[:32] & 0x0F
    filled = np.nonzero(squares)[0]
    for cell in filled:
        planes[squares[cell] - 1, cell >> 3, cell & 7] = 1.0

    castling = record[32] >> 4
    planes[12] = float(castling & 1)
    planes[13] = float((castling >> 1) & 1)
    planes[14] = float((castling >> 2) & 1)
    planes[15] = float((castling >> 3) & 1)

    ep_file = record[32] & 0x0F
    if ep_file != _EP_NONE:
        planes[16, _EP_CANON_RANK, ep_file] = 1.0

    planes[17] = min(int(record[33]), 100) / 100.0
    planes[18] = min(int(record[34]), 100) / 100.0
    return planes
