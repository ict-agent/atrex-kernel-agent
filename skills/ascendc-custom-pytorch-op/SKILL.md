---
name: ascendc-custom-pytorch-op
description: Build or repair a self-authored AscendC operator that is compiled and launched from PyTorch inside an Atrex campaign. Use for PyBind/torch.ops registration, host tiling, CMake/BiSheng integration, NPU stream launch, or Cube/Matmul custom-op failures. Do not use merely to call an existing ACLNN or torch_npu operator.
---

# AscendC Custom PyTorch Operator

Produce one self-contained candidate whose Python entry point, PyTorch binding, host tiling, device
kernel, build recipe, and launch parameters agree end to end. Upstream CANN material is reference-only;
the candidate must contain its own declared implementation.

## Read first

Read [references/atrex-contract.md](references/atrex-contract.md) before editing candidate files.

Load additional material only when it matches the task:

- For PyTorch schema/dispatch, stream acquisition, and direct kernel launch, read
  `reference-projects/ops-transformer/experimental/README.md`. For a Cube-containing worked project,
  also read the complete files under
  `reference-projects/ops-transformer/experimental/posembedding/rope_matrix/` that implement
  `torch_interface.cpp`, `op_host/rope_matrix_tiling.h`, `inc/rope_matrix_extern.h`, and the device
  kernel.
- For a new full CANN operator project, read
  `cann-skills/skills/tutorial/cann-op-scaffold/SKILL.md`. Its vendor-package build/deploy commands
  are not valid in an Atrex candidate; apply the compatibility rules in the reference above.
- For custom tiling or multi-core partitioning, read
  `cann-skills/skills/tutorial/cann-tiling/SKILL.md`.
- For data movement, queues, alignment, or AIV kernels, read
  `cann-skills/skills/tutorial/cann-kernel/SKILL.md`.
- For a named CANN operator, read `cann-skills/skills/cann-op-router/SKILL.md`, then prefix every
  routed path with `cann-skills/`. Auto-generated operator cards describe installed ACLNN APIs; in
  production mode they are semantic references, not permission to delegate candidate compute.
- For Cube/Matmul, read [references/cube-matmul.md](references/cube-matmul.md) before choosing static
  versus host-generated tiling.

## Establish one binding contract

Choose one evaluator-facing binding and keep it consistent:

- A self-contained PyBind extension imported from the candidate build directory; or
- A narrow custom `torch.ops` namespace with an explicit schema, `PrivateUse1` implementation, and
  Meta implementation when required by the evaluator or graph capture.

In either form, the binding must:

1. Validate shape, dtype, device, and contiguity assumptions before launch.
2. Allocate the output with the required PyTorch shape and dtype.
3. Obtain the current NPU stream from the installed torch_npu runtime.
4. Generate or materialize all host tiling and launch metadata from the same shape constants used by
   the device kernel.
5. Launch only the declared self-authored kernel, propagate launch errors, and use synchronization
   only when the evaluator contract requires it.

Do not mix a PyBind-only API with an incomplete `TORCH_LIBRARY` registration. Do not load an installed
custom operator or copy a reference-project implementation.

## Keep cross-file invariants explicit

Before compiling, write down and check the values shared across files:

- logical shape and tensor layout;
- transpose flags and leading dimensions;
- tile/group width and total work-item count;
- `blockDim`, tiling `usedCoreNum`, and device block-index mapping;
- `singleCoreM/N/K`, base tile sizes, and output row stride;
- bias, workspace, tiling-buffer, and output pointer order in the kernel ABI;
- build module name, build directory, Python import name, and custom namespace/schema.

Change coupled host and device constants in the same edit. A build-directory-only change is not a
mathematical implementation change, although a version-specific directory is useful to prevent stale
extension reuse.

## Implement and validate

1. Inspect a release-matched working reference with the same execution kind and record the exact
   source paths used.
2. Implement the smallest end-to-end path, including host tiling and launch plumbing, before tuning.
3. Run static checks for source declarations and prohibited compute dependencies.
4. Build and execute only through `tools/sandbox.py`; use a clean, version-specific build directory.
5. On an address fault, first reduce to one block/one tile and separately validate tiling transport,
   A/B/C offsets, transpose, bias, output stride, and scheduler ownership. Do not treat a compiler-accepted
   pointer cast as evidence that a tiling object is semantically valid.
6. Require base and multi-seed correctness before performance conclusions.
7. If the user or campaign explicitly requires Cube, verify the source/compiler/profiler evidence for
   the Cube path and reject a scalar fallback even when it is numerically correct and faster.

Finish with a manifest audit: `kernel.py`, `CMakeLists.txt`, and every self-authored binding, host-tiling,
launch, header, and kernel source must appear in `solution.json.sources`.
