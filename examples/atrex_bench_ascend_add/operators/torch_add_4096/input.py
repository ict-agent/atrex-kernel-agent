from __future__ import annotations

import torch
import torch_npu  # noqa: F401


def make_inputs(
    *, length: int = 4096, dtype: str = "float16", device: str = "npu:0", seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    tensor_dtype = getattr(torch, dtype)
    return (
        torch.randn((length,), dtype=tensor_dtype, device=device),
        torch.randn((length,), dtype=tensor_dtype, device=device),
    )

