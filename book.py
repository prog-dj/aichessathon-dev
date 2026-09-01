"""Opening book lookup. Shipped in the zip.

The book is a JSON map from a position's Zobrist key (hex) to a list of
``[uci, weight]`` entries, built offline from strong games by
``training/build_book.py``. A move is chosen in proportion to its weight, so
play varies but stays inside sound theory.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess
import chess.polyglot


class Book:
    def __init__(self, path: Path | str) -> None:
        raw: dict[str, list[tuple[str, int]]] = json.loads(Path(path).read_text())
        self._entries: dict[int, list[tuple[chess.Move, int]]] = {
            int(key, 16): [(chess.Move.from_uci(uci), int(weight)) for uci, weight in moves]
            for key, moves in raw.items()
        }

    def __len__(self) -> int:
        return len(self._entries)

    def move(self, board: chess.Board) -> chess.Move | None:
        options = self._entries.get(chess.polyglot.zobrist_hash(board))
        if not options:
            return None
        legal = [(m, w) for m, w in options if m in board.legal_moves]
        if not legal:
            return None
        moves, weights = zip(*legal, strict=True)
        pick: chess.Move = random.choices(list(moves), weights=list(weights), k=1)[0]
        return pick
