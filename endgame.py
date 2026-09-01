"""Syzygy endgame tablebase probing. Shipped in the zip.

We ship 3-4-man WDL tables (~30 MB). When the position on the board has few
enough pieces, :meth:`Tablebase.best_moves` returns the moves that keep the best
achievable result (win over draw over loss), and the agent restricts its choice
to those - so it never throws away a won or drawn endgame. Progress toward mate
is left to the search, whose endgame piece-square tables push the king and pawns
the right way.
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.syzygy


class Tablebase:
    def __init__(self, directory: Path | str, max_pieces: int = 4) -> None:
        self._tb = chess.syzygy.open_tablebase(str(directory), load_dtz=False)
        self._max_pieces = max_pieces

    def best_moves(self, board: chess.Board) -> list[chess.Move] | None:
        """Legal moves that preserve the best tablebase result, or None if off-table."""
        if chess.popcount(board.occupied) > self._max_pieces:
            return None
        try:
            scored: list[tuple[chess.Move, int]] = []
            for move in board.legal_moves:
                board.push(move)
                wdl = -self._tb.probe_wdl(board)  # from our POV after the reply
                board.pop()
                scored.append((move, wdl))
        except (KeyError, chess.syzygy.MissingTableError, ValueError):
            return None
        if not scored:
            return None
        best = max(wdl for _, wdl in scored)
        return [move for move, wdl in scored if wdl == best]
