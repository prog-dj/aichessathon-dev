"""Raw-data parsers for the training shards."""

from __future__ import annotations

import io
import json

import chess

from encoding import index_to_move
from training.pack import unpack_position
from training.prepare import iter_eval_samples, iter_pgn_samples

_EVAL_ROWS = [
    {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "evals": [
            {"depth": 40, "pvs": [{"cp": 30, "line": "e2e4 e7e5"}]},
            {"depth": 20, "pvs": [{"cp": 18, "line": "d2d4"}]},
        ],
    },
    {
        # black to move, White is up a lot -> side-to-move target should be a loss
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -",
        "evals": [{"depth": 35, "pvs": [{"cp": 900, "line": "b8c6"}]}],
    },
]


def test_iter_eval_samples_picks_deepest_and_flips_pov() -> None:
    samples = list(iter_eval_samples([json.dumps(row) for row in _EVAL_ROWS]))
    assert len(samples) == 2

    record, policy_index, (win, _draw, loss) = samples[0]
    board = chess.Board(_EVAL_ROWS[0]["fen"] + " 0 1")
    assert index_to_move(board, policy_index) == chess.Move.from_uci("e2e4")  # deepest pv
    assert win > loss  # +30 cp for the side to move
    assert unpack_position(record).shape == (19, 8, 8)

    _, _, (win2, _, loss2) = samples[1]
    assert loss2 > win2  # +900 for White, but Black is to move


def test_iter_eval_samples_skips_shallow_and_illegal() -> None:
    shallow = {
        "fen": "8/8/8/8/8/8/8/K6k w - -",
        "evals": [{"depth": 3, "pvs": [{"cp": 0, "line": "a1a2"}]}],
    }
    illegal = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "evals": [{"depth": 30, "pvs": [{"cp": 0, "line": "e2e5"}]}],
    }
    assert list(iter_eval_samples([json.dumps(shallow), json.dumps(illegal)])) == []


_PGN = """[Event "x"]
[Result "1-0"]
[WhiteElo "2500"]
[BlackElo "2450"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0

[Event "y"]
[Result "0-1"]
[WhiteElo "1800"]
[BlackElo "1900"]

1. d4 d5 2. c4 e6 0-1
"""


def test_iter_pgn_samples_filters_by_elo_and_skips_opening() -> None:
    samples = list(iter_pgn_samples(io.StringIO(_PGN), min_elo=2200, skip_plies=4))
    # only the first game qualifies; it has 10 plies, sampled from ply 4 -> 6 samples
    assert len(samples) == 6
    for record, policy_index, wdl in samples:
        assert unpack_position(record).shape == (19, 8, 8)
        assert 0 <= policy_index < 64 * 73
        assert wdl in {(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)}  # decisive game, never a draw target


def test_iter_pgn_samples_result_pov_alternates() -> None:
    samples = list(iter_pgn_samples(io.StringIO(_PGN), min_elo=2200, skip_plies=0))
    # game 1 is 1-0: white-to-move plies are wins, black-to-move plies are losses
    assert samples[0][2] == (1.0, 0.0, 0.0)
    assert samples[1][2] == (0.0, 0.0, 1.0)
