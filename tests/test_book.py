"""Opening book lookup."""

from __future__ import annotations

import json

import chess
import chess.polyglot

from book import Book


def _write_book(tmp_path, positions):  # type: ignore[no-untyped-def]
    path = tmp_path / "book.json"
    payload = {
        f"{chess.polyglot.zobrist_hash(chess.Board(fen)):016x}": moves
        for fen, moves in positions.items()
    }
    path.write_text(json.dumps(payload))
    return path


def test_returns_a_weighted_book_move(tmp_path) -> None:  # type: ignore[no-untyped-def]
    book = Book(
        _write_book(tmp_path, {chess.STARTING_FEN: [["e2e4", 100], ["d2d4", 50]]})
    )
    assert len(book) == 1
    picks = {book.move(chess.Board()).uci() for _ in range(50)}  # type: ignore[union-attr]
    assert picks <= {"e2e4", "d2d4"}


def test_none_when_position_not_in_book(tmp_path) -> None:  # type: ignore[no-untyped-def]
    book = Book(_write_book(tmp_path, {chess.STARTING_FEN: [["e2e4", 1]]}))
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert book.move(board) is None


def test_skips_moves_that_are_not_legal(tmp_path) -> None:  # type: ignore[no-untyped-def]
    book = Book(_write_book(tmp_path, {chess.STARTING_FEN: [["e7e5", 100]]}))  # illegal for White
    assert book.move(chess.Board()) is None
