#!/usr/bin/env python3
"""Small NPU evaluator implementing the Atrex-Bench run_eval result contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--num-correctness-cases", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--candidate-timeout-s", type=float, default=20.0)
    parser.add_argument("--perf-timeout-s", type=float, default=120.0)
    return parser


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _entrypoint(module: ModuleType) -> Callable[..., Any]:
    model = getattr(module, "Model", None)
    if model is not None:
        instance = model()
        if hasattr(instance, "eval"):
            instance.eval()
        return instance
    run = getattr(module, "run", None)
    if callable(run):
        return run
    raise RuntimeError("candidate must export Model or run")


def _outputs(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def _result_path(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    return output / "eval_result.json"


def _write_failure(path: Path, error: BaseException) -> None:
    payload = {
        "eval_id": str(uuid.uuid4()),
        "error": f"{type(error).__name__}: {error}",
        "passed": {
            "compile": {"status": "failed", "reason": str(error)},
            "correctness": {},
        },
        "correctness": {"shapes": {}},
        "performance": {"shapes": {}},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    output_path = _result_path(Path(args.output))
    reference_dir = Path(args.reference_dir).resolve()
    sys.path.insert(0, str(reference_dir))

    try:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
        input_module = _load_module(reference_dir / "input.py", "atrex_input")
        reference_module = _load_module(reference_dir / "reference.py", "atrex_reference")
        candidate_module = _load_module(Path(args.input).resolve(), "atrex_candidate")
        reference = _entrypoint(reference_module)
        candidate = _entrypoint(candidate_module)
        shapes = json.loads((reference_dir / "shapes.json").read_text(encoding="utf-8"))

        correctness_status: dict[str, dict[str, str]] = {}
        correctness_shapes: dict[str, dict[str, Any]] = {}
        performance_shapes: dict[str, dict[str, Any]] = {}

        with torch.inference_mode():
            for shape_id, shape in shapes.items():
                kwargs = dict(shape.get("input_kwargs") or {})
                cases: list[dict[str, Any]] = []
                shape_passed = True
                reason = ""
                for case_index in range(args.num_correctness_cases):
                    inputs = input_module.make_inputs(seed=9001 + case_index, **kwargs)
                    expected = _outputs(reference(*inputs))
                    actual = _outputs(candidate(*inputs))
                    torch.npu.synchronize()
                    if len(expected) != len(actual):
                        raise RuntimeError(
                            f"output count mismatch: expected {len(expected)}, got {len(actual)}"
                        )
                    output_metrics: list[dict[str, float]] = []
                    for expected_tensor, actual_tensor in zip(expected, actual):
                        diff = (actual_tensor.float() - expected_tensor.float()).abs()
                        max_abs = float(diff.max().item()) if diff.numel() else 0.0
                        denom = expected_tensor.float().abs().clamp_min(1e-6)
                        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
                        output_metrics.append(
                            {
                                "max_elementwise_abs_diff": max_abs,
                                "max_elementwise_rel_diff": max_rel,
                            }
                        )
                        if not torch.allclose(
                            actual_tensor, expected_tensor, atol=args.atol, rtol=args.rtol
                        ):
                            shape_passed = False
                            reason = f"allclose failed: abs={max_abs}, rel={max_rel}"
                    cases.append({"outputs": output_metrics})
                correctness_status[str(shape_id)] = {
                    "status": "passed" if shape_passed else "failed",
                    **({"reason": reason} if reason else {}),
                }
                correctness_shapes[str(shape_id)] = {"cases": cases}

                perf_inputs = input_module.make_inputs(seed=12001, **kwargs)
                for _ in range(args.warmup_iters):
                    candidate(*perf_inputs)
                torch.npu.synchronize()
                samples = []
                for _ in range(args.bench_iters):
                    start_ns = time.perf_counter_ns()
                    candidate(*perf_inputs)
                    torch.npu.synchronize()
                    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                    samples.append({"end_to_end_time_ms": elapsed_ms})
                performance_shapes[str(shape_id)] = {"samples": samples}

        payload = {
            "eval_id": str(uuid.uuid4()),
            "passed": {
                "compile": {"status": "passed"},
                "correctness": correctness_status,
            },
            "correctness": {"shapes": correctness_shapes},
            "performance": {"shapes": performance_shapes},
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0 if all(v["status"] == "passed" for v in correctness_status.values()) else 1
    except BaseException as error:
        traceback.print_exc()
        _write_failure(output_path, error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

