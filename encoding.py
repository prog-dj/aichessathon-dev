"""Board and move encoding shared by training and the agent.

Both the network trainer and ``agent.py`` import this file, so it is the single
source of truth for how a position becomes a tensor and how a policy index maps
back to a move. Ship it inside the zip next to ``agent.py``.

Everything is canonicalised to the side to move: the board is mirrored vertically
when Black is to move and the piece colours are swapped, so the network always
sees "my pieces moving up the board". ``index_to_move`` and ``move_to_index``
undo and redo that mirror.

Input planes (19, each 8x8):
    0-5    own pawn, knight, bishop, rook, queen, king
    6-11   opponent pawn, knight, bishop, rook, queen, king
    12-13  own kingside / queenside castling rights (constant plane)
    14-15  opponent kingside / queenside castling rights
    16     en passant target square
    17     halfmove clock, min(n, 100) / 100 (fifty-move progress)
    18     fullmove number, min(n, 100) / 100

Policy: the AlphaZero 8x8x73 layout, ``from_square * 73 + move_plane``, 4672 total.
    0-55   queen-like moves: 8 directions x 7 distances
    56-63  knight moves
    64-72  underpromotions: 3 file directions x {knight, bishop, rook}
Queen promotions reuse the matching queen-like plane; decoding infers the queen
when a pawn lands on the last rank.
"""

from __future__ import annotations

import chess
import numpy as np
import numpy.typing as npt

N_PLANES = 19
POLICY_SIZE = 64 * 73

Planes = npt.NDArray[np.float32]

_PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

# (file delta, rank delta), rank increasing toward the promotion rank
_QUEEN_DIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
_KNIGHT_DIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
_UNDER_PIECES: tuple[int, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
_UNDER_FILE_DELTAS: tuple[int, ...] = (-1, 0, 1)


def _mirror(square: int) -> int:
    return square ^ 56


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def encode_board(board: chess.Board) -> Planes:
    """Return the (19, 8, 8) plane stack for ``board``, seen by the side to move."""
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    stm = board.turn
    opp = not stm
    flip = stm == chess.BLACK

    for type_index, piece_type in enumerate(_PIECE_TYPES):
        for colour_offset, colour in ((0, stm), (6, opp)):
            plane = colour_offset + type_index
            for square in board.pieces(piece_type, colour):
                cell = _mirror(square) if flip else square
                planes[plane, cell >> 3, cell & 7] = 1.0

    planes[12] = float(board.has_kingside_castling_rights(stm))
    planes[13] = float(board.has_queenside_castling_rights(stm))
    planes[14] = float(board.has_kingside_castling_rights(opp))
    planes[15] = float(board.has_queenside_castling_rights(opp))

    if board.ep_square is not None:
        cell = _mirror(board.ep_square) if flip else board.ep_square
        planes[16, cell >> 3, cell & 7] = 1.0

    planes[17] = min(board.halfmove_clock, 100) / 100.0
    planes[18] = min(board.fullmove_number, 100) / 100.0
    return planes


def _canonical_squares(board: chess.Board, move: chess.Move) -> tuple[int, int]:
    if board.turn == chess.BLACK:
        return _mirror(move.from_square), _mirror(move.to_square)
    return move.from_square, move.to_square


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    """Map a legal ``move`` in ``board`` to its policy index in ``[0, 4672)``."""
    from_square, to_square = _canonical_squares(board, move)
    from_file, from_rank = from_square & 7, from_square >> 3
    to_file, to_rank = to_square & 7, to_square >> 3
    file_delta, rank_delta = to_file - from_file, to_rank - from_rank

    if move.promotion is not None and move.promotion != chess.QUEEN:
        direction = _UNDER_FILE_DELTAS.index(file_delta)
        piece = _UNDER_PIECES.index(move.promotion)
        return from_square * 73 + 64 + 3 * direction + piece

    if (file_delta, rank_delta) in _KNIGHT_DIRS:
        return from_square * 73 + 56 + _KNIGHT_DIRS.index((file_delta, rank_delta))

    direction = _QUEEN_DIRS.index((_sign(file_delta), _sign(rank_delta)))
    distance = max(abs(file_delta), abs(rank_delta))
    return from_square * 73 + direction * 7 + (distance - 1)


def index_to_move(board: chess.Board, index: int) -> chess.Move | None:
    """Inverse of :func:`move_to_index`. ``None`` if the index is off the board.

    The result is not guaranteed legal; callers filter against ``legal_moves``.
    """
    from_square, plane = divmod(index, 73)
    from_file, from_rank = from_square & 7, from_square >> 3
    promotion: int | None = None

    if plane < 56:
        direction, distance = divmod(plane, 7)
        file_delta, rank_delta = _QUEEN_DIRS[direction]
        distance += 1
        to_file = from_file + file_delta * distance
        to_rank = from_rank + rank_delta * distance
    elif plane < 64:
        file_delta, rank_delta = _KNIGHT_DIRS[plane - 56]
        to_file = from_file + file_delta
        to_rank = from_rank + rank_delta
    else:
        direction, piece = divmod(plane - 64, 3)
        to_file = from_file + _UNDER_FILE_DELTAS[direction]
        to_rank = from_rank + 1
        promotion = _UNDER_PIECES[piece]

    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        return None
    to_square = to_rank * 8 + to_file

    if board.turn == chess.BLACK:
        real_from, real_to = _mirror(from_square), _mirror(to_square)
    else:
        real_from, real_to = from_square, to_square

    if promotion is None and to_rank == 7:
        piece_at = board.piece_at(real_from)
        if piece_at is not None and piece_at.piece_type == chess.PAWN:
            promotion = chess.QUEEN

    return chess.Move(real_from, real_to, promotion)


def legal_policy_indices(board: chess.Board) -> list[tuple[chess.Move, int]]:
    """Every legal move paired with its policy index, for masking network output."""
    return [(move, move_to_index(board, move)) for move in board.legal_moves]


def _build_mirror_planes() -> tuple[int, ...]:
    """Plane permutation under a horizontal (file) flip: (df, dr) -> (-df, dr)."""
    mapping = list(range(73))
    for direction, (file_delta, rank_delta) in enumerate(_QUEEN_DIRS):
        flipped = _QUEEN_DIRS.index((-file_delta, rank_delta))
        for distance in range(7):
            mapping[direction * 7 + distance] = flipped * 7 + distance
    for index, (file_delta, rank_delta) in enumerate(_KNIGHT_DIRS):
        mapping[56 + index] = 56 + _KNIGHT_DIRS.index((-file_delta, rank_delta))
    for direction in range(3):
        for piece in range(3):
            flipped_dir = 2 - direction  # file deltas (-1, 0, 1) reverse
            mapping[64 + 3 * direction + piece] = 64 + 3 * flipped_dir + piece
    return tuple(mapping)


_MIRROR_PLANES = _build_mirror_planes()


def mirror_index(index: int) -> int:
    """Policy index of the same move on the file-mirrored board."""
    from_square, plane = divmod(index, 73)
    return (from_square ^ 7) * 73 + _MIRROR_PLANES[plane]
