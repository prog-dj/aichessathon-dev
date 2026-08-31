"""Value-target maths."""

from __future__ import annotations

import pytest

from training.labels import cp_to_wdl, result_to_wdl, score_to_cp


def test_score_to_cp_passthrough_and_mate() -> None:
    assert score_to_cp(50.0, None) == 50.0
    assert score_to_cp(None, None) == 0.0
    assert score_to_cp(None, 1) > 11_000
    assert score_to_cp(None, -3) < -11_000
    assert score_to_cp(None, 1) > score_to_cp(None, 5)  # faster mate scores higher


def test_cp_to_wdl_is_a_distribution() -> None:
    for cp in (-2000.0, -200.0, 0.0, 75.0, 200.0, 5000.0):
        win, draw, loss = cp_to_wdl(cp)
        assert win >= 0 and draw >= 0 and loss >= 0
        assert win + draw + loss == pytest.approx(1.0)


def test_cp_to_wdl_monotonic_and_symmetric() -> None:
    assert cp_to_wdl(400.0)[0] > cp_to_wdl(100.0)[0] > cp_to_wdl(-100.0)[0]
    win, draw, loss = cp_to_wdl(250.0)
    mirror_loss, mirror_draw, mirror_win = cp_to_wdl(-250.0)
    assert (win, draw, loss) == pytest.approx((mirror_win, mirror_draw, mirror_loss))


def test_cp_to_wdl_extremes() -> None:
    assert cp_to_wdl(8000.0)[0] > 0.99
    assert cp_to_wdl(-8000.0)[2] > 0.99


def test_result_to_wdl() -> None:
    assert result_to_wdl("1-0", side_to_move_is_white=True) == (1.0, 0.0, 0.0)
    assert result_to_wdl("1-0", side_to_move_is_white=False) == (0.0, 0.0, 1.0)
    assert result_to_wdl("0-1", side_to_move_is_white=True) == (0.0, 0.0, 1.0)
    assert result_to_wdl("1/2-1/2", side_to_move_is_white=True) == (0.0, 1.0, 0.0)
