"""Iterative-deepening alpha-beta over a material + piece-square evaluation.

Shipped in the zip as an alternative to search.py. The value net on one CPU core
is too slow to call at every node, so the bulk of the search runs on the fast
hand-crafted evaluation and the network contributes where it is affordable:

- the policy head orders moves at every node, so alpha-beta cutoffs land early
- the value head is blended into the evaluation for positions near the root
  (a single batched call per move), where the move choice is actually decided

    move = AlphaBetaSearch(evaluator).run(board, deadline_monotonic)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess
import chess.polyglot

from evaluation import material_pst_cp
from inference import Evaluator

_MATE = 1_000_000
_INF = 2_000_000


@dataclass(frozen=True)
class AlphaBetaConfig:
    net_weight_cp: float = 300.0  # centipawns per unit of value-head output
    net_plies: int = 2  # blend the value head this many plies deep
    qsearch_captures: int = 8
    max_depth: int = 40
    use_net_policy: bool = True


class _Timeout(Exception):
    pass


class AlphaBetaSearch:
    def __init__(self, evaluator: Evaluator | None, config: AlphaBetaConfig | None = None) -> None:
        self.evaluator = evaluator
        self.config = config or AlphaBetaConfig()
        self._deadline = 0.0
        self._nodes = 0
        self._priors: dict[chess.Move, float] = {}
        self._net_value: dict[int, float] = {}

    def run(self, board: chess.Board, deadline: float) -> chess.Move:
        self._deadline = deadline
        self._nodes = 0
        self._net_value = {}
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]

        self._priors = {}
        if self.evaluator is not None:
            self._priors, _ = self.evaluator.evaluate([board])[0]
            self._prime_net_values(board)

        # a timeout unwinds through the recursion without popping, so give each
        # deepening pass a fresh copy and never touch the caller's board
        best = max(legal, key=lambda m: self._priors.get(m, 0.0))
        for depth in range(1, self.config.max_depth + 1):
            try:
                best = self._root(board.copy(), depth, best)
            except _Timeout:
                break
        return best

    def _prime_net_values(self, board: chess.Board) -> None:
        """Batch-evaluate the positions within net_plies of the root, once."""
        assert self.evaluator is not None
        frontier = [board.copy(stack=False)]
        boards = list(frontier)
        for _ in range(self.config.net_plies):
            nxt: list[chess.Board] = []
            for parent in frontier:
                for move in parent.legal_moves:
                    child = parent.copy(stack=False)
                    child.push(move)
                    nxt.append(child)
            boards.extend(nxt)
            frontier = nxt
            if len(boards) > 800:
                break
        for probe, (_, value) in zip(boards, self.evaluator.evaluate(boards), strict=True):
            self._net_value[chess.polyglot.zobrist_hash(probe)] = value

    def _root(self, board: chess.Board, depth: int, prev_best: chess.Move) -> chess.Move:
        alpha, best = -_INF, prev_best
        for move in self._ordered(board, prev_best):
            board.push(move)
            score = -self._negamax(board, depth - 1, -_INF, -alpha)
            board.pop()
            if score > alpha:
                alpha, best = score, move
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
        stand_pat = self._evaluate(board)
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

    def _evaluate(self, board: chess.Board) -> float:
        score = float(material_pst_cp(board))
        bonus = self._net_value.get(chess.polyglot.zobrist_hash(board))
        if bonus is not None:
            score += self.config.net_weight_cp * bonus
        return score

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
