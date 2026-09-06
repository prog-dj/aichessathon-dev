from __future__ import annotations

import subprocess
import sys


def test_fast_engine_survives_tactical_position_after_opening_sequence() -> None:
    fen = "r1bqkbnr/ppp1pppp/2n5/8/2BP4/4P3/PP3PPP/RN1QKBNR w KQkq - 2 6"
    script = (
        "import chess, agent; "
        f"board=chess.Board({fen!r}); "
        "move=chess.Move.from_uci(agent.get_move(board.fen(), 120000)); "
        "assert move in board.legal_moves"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout