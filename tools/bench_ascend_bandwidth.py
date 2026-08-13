#!/usr/bin/env python3
# Copyright 2026 Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure a practical Ascend device-to-device copy-bandwidth ceiling.

The reported traffic counts one read and one write for every ``copy_``.  It is
therefore a practical aggregate HBM-traffic ceiling for the selected size, not
a claim about the physical HBM interface rate.  Run this in the exact
torch-npu/CANN environment used by the evaluator.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any


def _tb_s(bytes_per_copy: int, copies: int, elapsed_seconds: float) -> float:
    if bytes_per_copy <= 0 or copies <= 0 or elapsed_seconds <= 0:
        raise ValueError("bytes, copies, and elapsed time must be positive")
    # A device-to-device copy reads the source and writes the destination.
    return 2.0 * bytes_per_copy * copies / elapsed_seconds / 1e12


def run_benchmark(
    *, device_index: int, size_mib: int, warmup: int, copies: int, trials: int
) -> dict[str, Any]:
    try:
        import torch
        import torch_npu  # noqa: F401  # registers the torch.npu backend
    except ImportError as exc:
        raise RuntimeError(
            "torch and torch-npu are required; activate the target CANN environment"
        ) from exc

    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise RuntimeError("torch.npu is unavailable in the selected environment")

    npu.set_device(device_index)
    device = f"npu:{device_index}"
    bytes_per_copy = size_mib * 1024 * 1024
    element_count = bytes_per_copy // 2
    src = torch.empty(element_count, dtype=torch.float16, device=device)
    dst = torch.empty_like(src)

    for _ in range(warmup):
        dst.copy_(src)
    npu.synchronize()

    samples: list[float] = []
    elapsed_samples: list[float] = []
    for _ in range(trials):
        npu.synchronize()
        started = time.perf_counter_ns()
        for _ in range(copies):
            dst.copy_(src)
        npu.synchronize()
        elapsed = (time.perf_counter_ns() - started) / 1e9
        elapsed_samples.append(elapsed)
        samples.append(_tb_s(bytes_per_copy, copies, elapsed))

    free_bytes = total_bytes = None
    mem_get_info = getattr(npu, "mem_get_info", None)
    if callable(mem_get_info):
        try:
            free_bytes, total_bytes = (int(value) for value in mem_get_info(device_index))
        except (RuntimeError, TypeError, ValueError):
            pass

    return {
        "schema_version": 1,
        "device": {
            "index": device_index,
            "name": npu.get_device_name(device_index),
            "torch": getattr(torch, "__version__", None),
            "torch_npu": getattr(torch_npu, "__version__", None),
            "free_bytes_after_allocation": free_bytes,
            "total_bytes": total_bytes,
        },
        "method": {
            "operation": "torch.Tensor.copy_",
            "dtype": "float16",
            "bytes_per_copy": bytes_per_copy,
            "traffic_bytes_per_copy": 2 * bytes_per_copy,
            "traffic_convention": "one source read plus one destination write",
            "warmup_copies": warmup,
            "timed_copies_per_trial": copies,
            "trials": trials,
            "timer": "host perf_counter_ns around synchronized NPU work",
        },
        "elapsed_seconds": elapsed_samples,
        "aggregate_tb_s": {
            "samples": samples,
            "median": statistics.median(samples),
            "maximum": max(samples),
            "minimum": min(samples),
        },
        "semantics": (
            "practical same-size copy ceiling; not a physical HBM-interface rating"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--size-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--copies", type=int, default=50)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args(argv)
    if min(args.size_mib, args.warmup, args.copies, args.trials) <= 0:
        parser.error("size, warmup, copies, and trials must all be positive")

    try:
        result = run_benchmark(
            device_index=args.device_index,
            size_mib=args.size_mib,
            warmup=args.warmup,
            copies=args.copies,
            trials=args.trials,
        )
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
