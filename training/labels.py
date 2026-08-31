"""Value-target maths. Training-only, numpy-free, so it is cheap to test.

The value head is 3-way win/draw/loss from the side-to-move point of view.
Stream A turns a Stockfish centipawn score into a soft target; stream B uses the
game result as a hard target.
"""

from __future__ import annotations

import math

MATE_CP = 12000.0

Wdl = tuple[float, float, float]


def score_to_cp(cp: float | None, mate: int | None) -> float:
    """One centipawn number from an eval that is either a score or a mate count."""
    if mate is not None:
        return math.copysign(MATE_CP - min(abs(int(mate)), 100) * 20.0, mate)
    return float(cp if cp is not None else 0.0)


def cp_to_wdl(cp: float, scale: float = 350.0, draw_margin: float = 90.0) -> Wdl:
    """Soft win/draw/loss from a side-to-move centipawn score.

    Two logistic shoulders ``draw_margin`` apart: outside the margin the score
    tends to a decisive result, inside it the mass sits on the draw.
    """
    cp = max(-MATE_CP, min(MATE_CP, cp))
    win = 1.0 / (1.0 + math.exp(-(cp - draw_margin) / scale))
    loss = 1.0 / (1.0 + math.exp(-(-cp - draw_margin) / scale))
    draw = max(0.0, 1.0 - win - loss)
    total = win + draw + loss
    return win / total, draw / total, loss / total


def result_to_wdl(result: str, side_to_move_is_white: bool) -> Wdl:
    """``result`` is a PGN tag: '1-0', '0-1' or '1/2-1/2'."""
    if result == "1/2-1/2":
        return 0.0, 1.0, 0.0
    white_won = result == "1-0"
    stm_won = white_won == side_to_move_is_white
    return (1.0, 0.0, 0.0) if stm_won else (0.0, 0.0, 1.0)
