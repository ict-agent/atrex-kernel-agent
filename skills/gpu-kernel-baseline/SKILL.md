---
name: gpu-kernel-baseline
description: Learn the target framework from gpu-wiki and implement a baseline GPU kernel. Use this skill to understand compute semantics, determine the target platform and framework, search reference implementations, and produce a correct V0 baseline with performance records for later profile-driven optimization.
---

# GPU Kernel Baseline

## When to Use

Use this skill when the user provides PyTorch logic or a kernel demo and asks to:

- Write a GPU kernel for the target platform.
- Build a baseline from scratch.
- Prepare `kernel.py`, `reference.py`, `test_kernel.py`, and `baseline_report.md` for later profile-driven optimization.

## Workflow

This stage first understands the PyTorch semantics, then learns the target framework APIs (CuteDSL,
FlyDSL, or AscendC) through `<gpu-wiki>/README.md`, implements the candidate and `test_kernel.py`,
validates correctness, records performance, writes `baseline_report.md`, and writes `memory/v0.json`.

The orchestrator exposes the knowledge base at `./gpu-wiki/` inside each campaign workspace,
referenced below as `<gpu-wiki>/`.

## Phase 1: Understand PyTorch Semantics

1. Read the user-provided PyTorch logic and `kernel_demo`.
2. Extract and record:
   - Compute pattern, such as `GEMM`, `Decode Attention`, `Reduction`, or `Elementwise`.
   - Input/output shape, stride, dtype, layout, and device.
   - Data dependencies, broadcasting, masks, boundary handling, and write-back semantics.
   - Accuracy requirements, tolerance, accumulation dtype, and special-value handling.
3. Determine target platform and framework:
   - H100/H20/H200 -> Hopper -> `CuteDSL`
   - MI300X/MI308X -> CDNA3 -> `FlyDSL`
   - MI355X -> CDNA4 -> `FlyDSL`
   - Ascend 910B1 -> `ascend910b1` -> `AscendC`
4. If the PyTorch logic is ambiguous, first create a minimal runnable reference, then continue.

## Phase 2: Learn Framework APIs from gpu-wiki

1. **Mandatory prerequisite**: read `<gpu-wiki>/README.md` and follow its indexed learning path.
2. Prioritize API docs, reference kernels, hardware constraints, and pitfalls directly related to the target platform, framework, and compute pattern.
3. Prefer implementations with the same framework and compute pattern.
4. Enter relevant documents through each directory-level `README.md`; do not blindly grep the full wiki first.
5. Query L1 with the real architecture, vendor, and DSL. For Ascend 910B1, use the Ascend scope
   explicitly rather than a CUDA/ROCm substitute:

   ```bash
   python3 gpu-wiki/scripts/query.py --arch ascend910b1 --vendor ascend \
     --area docs --dsl ascendc --operator <operator> \
     --section ref-docs --section pitfalls
   python3 gpu-wiki/scripts/query.py <operator-or-mechanism> --arch ascend910b1 \
     --vendor ascend --dsl ascendc --area reference-kernels --kind kernel
   ```

6. If L1 does not yield an actionable AscendC pattern, search L2 under `reference-projects/`. Start
   with the CANN 8.5 repository matching the operator class (`ops-math`, `ops-nn`,
   `ops-transformer`, or `ops-cv`), then use `vllm-ascend` and `cann-recipes-infer` for runtime and
   inference-integration patterns. Repositories carrying the Ascend Open Source Software License
   Agreement (OSLA) are **reference-only**: learn API usage and organization, but do not copy code
   verbatim, load their implementations, or declare them as candidate dependencies.
7. Record learned wiki paths, reference-project paths, API constraints, hardware constraints, and
   pitfalls in `plans/v0_plan.md` for implementation and reporting.

## Phase 3: Implement Baseline Kernel and Correctness Tests

1. Implement a correct baseline based on PyTorch semantics and the learned framework APIs. For CuteDSL
   and FlyDSL the executable candidate remains in `kernel.py`. For AscendC, keep the evaluator-facing
   Python entry point in `kernel.py` and place only self-authored kernel/tiling/host glue in `.cpp`, `.h`,
   or `.asc` files listed by relative path in `solution.json.sources`. Before editing an AscendC
   PyTorch binding, host tiling implementation, or Cube/Matmul launch path, invoke the repository-local
   `ascendc-custom-pytorch-op` skill and follow its Atrex compatibility contract.
2. Update `solution.json` languages and dependencies for the selected framework. An AscendC candidate's
   `sources` list must include `kernel.py` and every source/header needed to compile and launch it; do not
   rely on undeclared side files, prebuilt custom operators, or imported reference implementations.
   Compile those declared self-authored sources only through the sandbox with the preinstalled CANN
   toolchain. Do not install dependencies or build any reference-project/third-party source.
   When using the CANN CMake/Bisheng fast-launch route, also declare `CMakeLists.txt`. Candidate Python
   may invoke only direct checked literal-argv CMake commands such as
   `subprocess.run(["cmake", "-S", ...], check=True)`; shell commands, `shell=True`, command strings,
   setup-script sourcing, and non-CMake subprocess executables are forbidden. The gateway already has
   the CANN environment. Never import `torch_npu` in a host probe; use `importlib.util.find_spec` for
   static location discovery or perform the probe through `tools/sandbox.py`.
3. Write `test_kernel.py` using PyTorch logic directly as the correctness reference.
4. Cover representative inputs, including normal shapes, boundary shapes, and relevant dtype or stride cases.
5. Example correctness check:

```python
ref = pytorch_reference(inputs)
out = kernel_v1(inputs)
rel_err = (out.float() - ref).norm() / ref.norm()
assert rel_err < 0.01
```

6. The default BF16 threshold is `rel_err < 0.01`; lower precision formats may use task-specific relaxed thresholds.
7. Add per-case timeout guard in `test_kernel.py` to prevent hanging:

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Test case exceeded timeout limit")

signal.signal(signal.SIGALRM, timeout_handler)

TIMEOUT_SEC = int(os.environ.get("TEST_TIMEOUT_SEC", "30"))

for case in test_cases:
    signal.alarm(TIMEOUT_SEC)
    try:
        run_test(case)
    except TimeoutError:
        record_failure(case, "TIMEOUT_FAIL")
    finally:
        signal.alarm(0)
```
8. If API, compilation, accuracy, performance, or hardware issues appear, return to `<gpu-wiki>/` through README indexes, read the relevant docs/reference kernels/pitfalls, and then fix the implementation.
9. Record the baseline configuration, including tile size, core/block organization, and major data-movement patterns.

## Phase 4: Performance, Correctness, and Quality Gate

1. Run `test_kernel.py` with a per-case timeout to prevent hanging on compilation errors or infinite loops:

```bash
timeout 60 python tools/sandbox.py --kind run --no-sync -- python test_kernel.py
```

   - Each individual test case must complete within **30 seconds** (configurable via `TEST_TIMEOUT_SEC` env var).
   - If a case exceeds the timeout, mark it as `TIMEOUT_FAIL`, kill the process, and record the failure in `baseline_report.md`.
   - Common timeout causes: infinite loops in index calculation, deadlocks in synchronization, or excessive compilation time. Return to gpu-wiki to diagnose.

2. Verify all correctness cases pass and record max `rel_err` plus PASS/FAIL.
3. Measure baseline performance and record:

```text
latency(us) | TFLOPS | bandwidth(GB/s) | TFLOPS peak utilization(%) | bandwidth peak utilization(%)
```

4. Use `compute_utilization.py` to calculate TFLOPS and bandwidth utilization:

```bash
python tools/compute_utilization.py   --gpu <gpu> --dtype <dtype>   --flops-expr '<expr>' --bytes-expr '<expr>'   --time-ms <ms> --grid-blocks <blocks>
```

For `ascend910b1`, also pass `--execution-kind aic|aiv|mix`; AIV requires
`--operation-kind add|fma`. The provider rejects unsupported scalar MIX/FMA
peaks. Use `--allow-modeled-peak` only when its recorded assumptions match the
kernel. Select `--bandwidth-kind hbm_cycle_model` for the CANN msProf Roofline
model or `measured_copy_256m` for the recorded practical copy ceiling.

5. Every theoretical peak, bandwidth, and utilization calculation must cite the gpu-wiki spec sources registered in Step 0.
   On Ascend, do not claim hardware-counter or bottleneck evidence unless the sandbox has a working
   `msprof` integration and produced inspectable output for this candidate. If it is not connected,
   state that profiling is unavailable and report only measured evaluator timing; never fabricate
   `msprof` fields or reinterpret NCU/rocprof counters as Ascend metrics.
6. Write `baseline_report.md` with:
   - Baseline kernel path
   - Correctness test path
   - PyTorch reference logic description
   - Learned and searched gpu-wiki paths
   - Baseline configuration summary
   - Correctness results: case list, max `rel_err`, PASS/FAIL (include any TIMEOUT_FAIL cases)
   - Baseline performance: latency(us), TFLOPS, bandwidth(GB/s), and peak utilization percentages
7. Write baseline iteration data to `memory/v0.json` using `tools/memory_manager.py`:

   ```bash
   # Create the iteration file
   python tools/memory_manager.py create --workspace kernel_opt_<name> --version v0

   # Fill in performance and metadata
   python tools/memory_manager.py update --workspace kernel_opt_<name> --version v0 \
       --set 'performance.latency_us=<value>' \
       --set 'performance.tflops=<value>' \
       --set 'performance.bandwidth_gbps=<value>' \
       --set 'performance.tflops_peak_utilization_pct=<value>' \
       --set 'performance.bandwidth_peak_utilization_pct=<value>' \
       --set 'optimization.action_category=baseline' \
       --set 'optimization.action_description=<summary>' \
       --set 'correctness.rel_err=<value>' \
       --set 'correctness.status=PASS' \
       --set 'quality_gate.result=PASS'
   ```

   For array fields (`pitfalls_and_fixes`, `references`), update the JSON file directly or use `read` + manual edit + write-back. Fill in:
   - `pitfalls_and_fixes`: any errors encountered during implementation
   - `references`: gpu-wiki paths and docs referenced during learning

8. After the quality gate passes, commit:

```bash
git add kernel.py solution.json test_kernel.py baseline_report.md memory/v0.json README.md
# AscendC only: explicitly stage every additional solution.json.sources[].path here.
git commit -m "V0: baseline kernel"
```
Before the commit, an AscendC campaign must also run `git add --` with the actual additional source
paths and verify that every `solution.json.sources[].path` appears in the staged file list.

## memory/ Requirements

Each iteration produces a `memory/v<N>.json` file following the schema defined in `reference/v_iteration.schema.json`. The JSON structure captures performance data, optimization actions, profile evidence, correctness results, ISA metric progress, search logs, pitfalls and fixes, and references.

Key rules:
- The `masked` field defaults to `false`. When set to `true`, the file is skipped during reads.
- ISA optimization target thresholds are stored in `README.md` and must be derived from `<gpu-wiki>/` best practices, hardware specs, and Step 0 Roofline conclusions. Do not fabricate thresholds from experience.

## Deliverables

- Runnable and correct candidate using CuteDSL, FlyDSL, or AscendC (`kernel.py` plus any declared AscendC sources)
- PyTorch `reference.py`
- `test_kernel.py`
- `baseline_report.md`
- Created `memory/v0.json`
- Git commit

## Appendix: Prohibited Actions

- Do not use unspecified programming frameworks or import external projects.
- Do not reduce an AscendC campaign to CUDA/ROCm assumptions; use `vendor=ascend`,
  `arch=ascend910b1`, and `framework=AscendC` throughout discovery and implementation.
