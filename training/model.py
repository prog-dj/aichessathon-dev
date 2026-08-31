"""The network: a small Leela-style ResNet with policy and value heads.

Not shipped. The trainer runs this on a GPU and exports an ONNX file to
``weights/``; ``agent.py`` loads only the ONNX with onnxruntime.

Input is the (19, 8, 8) stack from :mod:`encoding`. The policy head emits one
logit per ``encoding`` policy index in ``from_square * 73 + plane`` order, so
``policy.reshape(-1)`` lines up with :func:`encoding.move_to_index` directly.
The value head is 3-way win/draw/loss; the search uses ``P(win) - P(loss)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from encoding import N_PLANES, POLICY_SIZE


@dataclass(frozen=True)
class NetConfig:
    blocks: int
    channels: int

    @property
    def name(self) -> str:
        return f"{self.blocks}x{self.channels}"


CONFIGS: dict[str, NetConfig] = {
    "tiny": NetConfig(blocks=4, channels=64),
    "small": NetConfig(blocks=8, channels=112),
    "medium": NetConfig(blocks=12, channels=160),
    "large": NetConfig(blocks=16, channels=192),
}


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class ChessNet(nn.Module):
    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        c = config.channels

        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock(c) for _ in range(config.blocks)))

        self.policy_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 73, 1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(c, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.tower(self.stem(x))
        policy = self.policy_conv(features)  # (B, 73, 8, 8)
        # (B, 73, rank, file) -> (B, rank, file, 73) -> (B, from_square * 73 + plane)
        policy = policy.permute(0, 2, 3, 1).reshape(x.shape[0], POLICY_SIZE)
        value_logits = self.value_head(features)  # (B, 3) = win, draw, loss
        return policy, value_logits


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for key, config in CONFIGS.items():
        net = ChessNet(config)
        dummy = torch.zeros(2, N_PLANES, 8, 8)
        policy, value = net(dummy)
        params = parameter_count(net)
        print(
            f"{key:7s} {config.name:8s} "
            f"params {params / 1e6:5.2f}M  policy {tuple(policy.shape)}  value {tuple(value.shape)}"
        )
