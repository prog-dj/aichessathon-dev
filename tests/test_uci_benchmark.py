from __future__ import annotations

from pathlib import Path

import chess
import chess.engine

from harness.uci_benchmark import Opening, _new_score, load_openings, run_benchmark


class FakeEngine:
    def __init__(self) -> None:
        self.moves = 0

    def play(self, board: chess.Board, limit: chess.engine.Limit) -> chess.engine.PlayResult:
        self.moves += 1
        return chess.engine.PlayResult(next(iter(board.legal_moves)), None)

    def quit(self) -> None:
        pass


def test_loads_fen_epd_and_pgn_openings(tmp_path: Path) -> None:
    fen = chess.STARTING_FEN
    fen_path = tmp_path / "openings.fen"
    fen_path.write_text(f"# comment\n{fen}\n")
    assert load_openings(fen_path)[0].fen == fen

    epd_path = tmp_path / "openings.epd"
    epd_path.write_text("8/8/8/8/8/8/4K3/4k3 w - - hmvc 0;\n")
    assert chess.Board(load_openings(epd_path)[0].fen).board_fen() == "8/8/8/8/8/8/4K3/4k3"

    pgn_path = tmp_path / "openings.pgn"
    pgn_path.write_text("[Opening \"Test\"]\n\n1. e4 e5 2. Nf3 Nc6 *\n")
    openings = load_openings(pgn_path, plies=4)
    assert openings[0].name == "Test"
    assert chess.Board(openings[0].fen).fullmove_number == 3


def test_runner_plays_each_opening_with_both_colors(tmp_path: Path, monkeypatch) -> None:
    engines: list[FakeEngine] = []

    def fake_popen(command: list[str]) -> FakeEngine:
        engine = FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", fake_popen)
    results = run_benchmark(
        ["new"],
        ["old"],
        [Opening("start", chess.STARTING_FEN)],
        tmp_path / "match.pgn",
        max_plies=2,
    )

    assert [result.new_is_white for result in results] == [True, False]
    assert len(engines) == 4
    assert all(result.result == "1/2-1/2" for result in results)
    assert sum(_new_score(result) for result in results) == 1.0
    assert len((tmp_path / "match.pgn").read_text().split("[Event ")) == 3