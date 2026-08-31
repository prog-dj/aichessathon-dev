"""The shard dataset and writer. Training-only (needs torch).

A shard directory holds three parallel arrays, one row per position:
    records.npy  uint8   (N, 36)   packed position, see training.pack
    policy.npy   int32   (N,)      target move as an encoding policy index
    targets.npy  float32 (N, 4)    soft win / draw / loss (side-to-move POV),
                                   then the raw centipawn score or NaN

Storing the centipawn score means the win/draw/loss softness is a train-time
knob (``value_scale``) instead of something baked into the shard.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from training.labels import Wdl, cp_to_wdl
from training.pack import unpack_position


class ShardDataset(Dataset[tuple[torch.Tensor, int, torch.Tensor]]):
    def __init__(self, shard_dir: Path | str, value_scale: float | None = None) -> None:
        directory = Path(shard_dir)
        self.records: npt.NDArray[np.uint8] = np.load(directory / "records.npy", mmap_mode="r")
        self.policy: npt.NDArray[np.int32] = np.load(directory / "policy.npy", mmap_mode="r")
        self.targets: npt.NDArray[np.float32] = np.load(directory / "targets.npy", mmap_mode="r")
        self.value_scale = value_scale
        if not len(self.records) == len(self.policy) == len(self.targets):
            raise ValueError(f"shard {directory} arrays disagree in length")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        planes = unpack_position(np.asarray(self.records[index]))
        row = np.asarray(self.targets[index])
        cp = float(row[3]) if row.shape[0] > 3 else math.nan
        if self.value_scale is not None and not math.isnan(cp):
            wdl = np.array(cp_to_wdl(cp, scale=self.value_scale), dtype=np.float32)
        else:
            wdl = np.array(row[:3], dtype=np.float32)  # copy: mmap is read-only
        return torch.from_numpy(planes), int(self.policy[index]), torch.from_numpy(wdl)


class ShardWriter:
    """Accumulate samples and flush fixed-size shards to ``root/0000`` etc."""

    def __init__(self, root: Path | str, shard_size: int = 500_000) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self._records: list[npt.NDArray[np.uint8]] = []
        self._policy: list[int] = []
        self._targets: list[tuple[float, float, float, float]] = []
        self._next = len(list(self.root.glob("[0-9]" * 4)))
        self.total = 0

    def add(
        self, record: npt.NDArray[np.uint8], policy_index: int, wdl: Wdl, cp: float = math.nan
    ) -> None:
        self._records.append(record)
        self._policy.append(policy_index)
        self._targets.append((*wdl, cp))
        self.total += 1
        if len(self._records) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._records:
            return
        directory = self.root / f"{self._next:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "records.npy", np.stack(self._records))
        np.save(directory / "policy.npy", np.asarray(self._policy, dtype=np.int32))
        np.save(directory / "targets.npy", np.asarray(self._targets, dtype=np.float32))
        print(f"wrote {directory} ({len(self._records)} samples)")
        self._records.clear()
        self._policy.clear()
        self._targets.clear()
        self._next += 1
