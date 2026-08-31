"""Runtime wrapper around the ONNX net. Shipped in the zip.

The training code produces ``weights/model.onnx`` with two outputs:
    policy  (batch, 4672)  logits in encoding's from_square * 73 + plane order
    value   (batch, 3)     logits for win / draw / loss, side-to-move POV

``agent.py`` is the only caller. Everything here is plain numpy + onnxruntime so
a judge can read the whole move-selection path.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from encoding import encode_board, legal_policy_indices

Floats = npt.NDArray[np.float32]


def softmax(logits: Floats, axis: int = -1) -> Floats:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    result: Floats = (exp / exp.sum(axis=axis, keepdims=True)).astype(np.float32)
    return result


class Evaluator:
    """One onnxruntime session, pinned to a single core."""

    def __init__(self, model_path: Path | str, threads: int = 1) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name

    def run(self, planes: Floats) -> tuple[Floats, Floats]:
        """``planes`` is (batch, 19, 8, 8). Returns (policy logits, value logits)."""
        policy, value = self._session.run(None, {self._input: planes})
        return policy.astype(np.float32), value.astype(np.float32)

    def evaluate(self, boards: list[chess.Board]) -> list[tuple[dict[chess.Move, float], float]]:
        """Per board: a legal-move prior distribution and a scalar value in [-1, 1]."""
        planes = np.stack([encode_board(board) for board in boards]).astype(np.float32)
        policy_logits, value_logits = self.run(planes)
        value_probs = softmax(value_logits, axis=1)
        scalars = (value_probs[:, 0] - value_probs[:, 2]).tolist()

        out: list[tuple[dict[chess.Move, float], float]] = []
        for board, logits, scalar in zip(boards, policy_logits, scalars, strict=True):
            pairs = legal_policy_indices(board)
            masked = np.array([logits[index] for _, index in pairs], dtype=np.float32)
            priors = softmax(masked) if len(masked) else masked
            distribution = {
                move: float(prior)
                for (move, _), prior in zip(pairs, priors, strict=True)
            }
            out.append((distribution, scalar))
        return out
