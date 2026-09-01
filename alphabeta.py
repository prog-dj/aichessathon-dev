"""Iterative-deepening alpha-beta over a material + piece-square evaluation.

Shipped in the zip. The value net on one CPU core is too slow to call at every
node, so the deep search runs on the fast hand-crafted evaluation (~15k
nodes/s, depth 5-7 in the budget) and the network contributes where it is
affordable:

- the policy head orders the root moves, so alpha-beta settles quickly
- the value head is read once per legal root move (one batched call) and
  combined with that move's search score to pick the move actually played

    move = AlphaBetaSearch(evaluator).run(board, deadline_monotonic)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess

from evaluation import material_pst_cp
from inference import Evaluator

_MATE = 1_000_000
_INF = 2_000_000


@dataclass(frozen=True)
class AlphaBetaConfig:
    net_weight_cp: float = 200.0  # centipawns per unit of value-head output at the root
    qsearch_captures: int = 8
    max_depth: int = 40
    use_net_policy: bool = True
    use_net_value: bool = True


class _Timeout(Exception):
    pass


class AlphaBetaSearch:
    def __init__(self, evaluator: Evaluator | None, config: AlphaBetaConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config or AlphaBetaConfig()
        self._deadline = 0.0
        self._nodes = 0
        self._priors: dict[chess.Move, float] = {}
        self._root_scores: dict[chess.Move, float] = {}

    def run(self, board: chess.Board, deadline: float) -> chess.Move:
        self._deadline = deadline
        self._nodes = 0
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]

        self._priors = {}
        child_value: dict[chess.Move, float] = {}
        if self.evaluator is not None:
            self._priors, _ = self.evaluator.evaluate([board])[0]
            if self.config.use_net_value:
                child_value = self._root_child_values(board, legal)

        self._root_scores = {}
        best = max(legal, key=lambda m: self._priors.get(m, 0.0))
        for depth in range(1, self.config.max_depth + 1):
            try:
                # a timeout unwinds without popping; search a fresh copy each pass
                best = self._root(board.copy(), depth, best)
            except _Timeout:
                break

        if not self._root_scores:
            return best
        # combine the deep material score with the net's read of each move
        # (child_value is from the mover-after's view, so subtract it)
        weight = self.config.net_weight_cp
        return max(
            self._root_scores,
            key=lambda m: self._root_scores[m] - weight * child_value.get(m, 0.0),
        )

    def _root_child_values(
        self, board: chess.Board, legal: list[chess.Move]
    ) -> dict[chess.Move, float]:
        assert self.evaluator is not None
        children: list[chess.Board] = []
        for move in legal:
            board.push(move)
            children.append(board.copy(stack=False))
            board.pop()
        return {
            move: value
            for move, (_, value) in zip(legal, self.evaluator.evaluate(children), strict=True)
        }

    def _root(self, board: chess.Board, depth: int, prev_best: chess.Move) -> chess.Move:
        # full window at the root ply so every _root_scores entry is exact and can
        # be combined with the net value; deeper plies still prune normally
        best, best_score = prev_best, -_INF
        for move in self._ordered(board, prev_best):
            board.push(move)
            score = -self._negamax(board, depth - 1, -_INF, _INF)
            board.pop()
            self._root_scores[move] = score
            if score > best_score:
                best_score, best = score, move
        return best

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        self._nodes += 1
        if self._nodes % 1024 == 0 and time.monotonic() >= self._deadline:
            raise _Timeout
        if board.is_checkmate():
            return -_MATE + board.ply()
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3):
            return 0
        if depth <= 0:
            return self._quiescence(board, alpha, beta)

        best = -_INF
        for move in self._ordered(board, None):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def _quiescence(self, board: chess.Board, alpha: float, beta: float) -> float:
        stand_pat = float(material_pst_cp(board))
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        captures = sorted(
            (m for m in board.legal_moves if board.is_capture(m)),
            key=lambda m: _mvv_lva(board, m),
            reverse=True,
        )[: self.config.qsearch_captures]
        for move in captures:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _ordered(self, board: chess.Board, first: chess.Move | None) -> list[chess.Move]:
        moves = list(board.legal_moves)

        def key(move: chess.Move) -> tuple[bool, float, int]:
            prior = self._priors.get(move, 0.0) if self.config.use_net_policy else 0.0
            capture = _mvv_lva(board, move) if board.is_capture(move) else 0
            return (move == first, prior, capture)

        moves.sort(key=key, reverse=True)
        return moves


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_type_at(move.to_square) or 0
    attacker = board.piece_type_at(move.from_square) or 0
    return victim * 10 - attacker
