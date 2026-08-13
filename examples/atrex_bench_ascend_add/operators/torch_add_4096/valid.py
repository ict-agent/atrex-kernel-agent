from __future__ import annotations

import torch


def reference_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.add(x, y)

