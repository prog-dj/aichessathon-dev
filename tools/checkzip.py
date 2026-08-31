"""Extract submission.zip into a clean directory and exercise it like the platform.

Catches the packaging bugs the local tree hides: a file the agent imports but the
zip does not include, a path that only works from the repo root, weights that
were not bundled. Run after `make zip`, before spending an upload.

    python -m tools.checkzip [--zip submission.zip]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

import chess

_START_FEN = chess.STARTING_FEN
_MIDGAME_FEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("submission.zip"))
    args = parser.parse_args()

    if not args.zip.is_file():
        raise SystemExit(f"{args.zip} does not exist; run `make zip` first")

    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        if "agent.py" not in names:
            raise SystemExit("agent.py is not at the zip root")
        with tempfile.TemporaryDirectory() as tmp:
            archive.extractall(tmp)
            _run_in(Path(tmp))

    total = args.zip.stat().st_size
    print(f"\nOK: {args.zip} ({total / 1e6:.1f} MB, {len(names)} entries) imports and moves")


def _run_in(directory: Path) -> None:
    script = textwrap.dedent(
        f"""
        import time, chess, agent
        for fen in ({_START_FEN!r}, {_MIDGAME_FEN!r}):
            board = chess.Board(fen)
            start = time.monotonic()
            uci = agent.get_move(fen, 120_000)
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves, f"illegal {{uci}} in {{fen}}"
            print(f"  {{fen[:30]}}... -> {{uci}}  ({{(time.monotonic() - start) * 1000:.0f}} ms)")
        """
    )
    print(f"extracted to {directory}, importing agent as the platform would:")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=directory, capture_output=True, text=True, check=False
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="")
        raise SystemExit("the packaged agent failed to import or made an illegal move")


if __name__ == "__main__":
    main()
