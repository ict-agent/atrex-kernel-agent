# AscendC Programming Model on Ascend 910B

AscendC kernels run SPMD device code declared with `__global__ __aicore__`.
Global-memory pointers use the `__gm__` address-space qualifier. Each launched
block obtains its logical identity with `GetBlockIdx()` and the launch width
with `GetBlockNum()`.

## Split AIC/AIV execution

The A2 architecture separates Cube (AIC) and Vector (AIV) cores. Each has a
Scalar control unit. A logical MIX group contains one AIC and a target-specific
number of AIVs; Ascend 910B1 uses 1:2. AIC and AIV do not share local memory and
communicate through global memory with explicit synchronization.

Choose the kernel type from the actual work:

- `AIV_ONLY`: vector/elementwise work; launch against the AIV count.
- `AIC_ONLY`: matrix/Cube work; launch against the AIC count.
- `MIX`: coupled Cube and Vector stages; launch logical groups and define the
  ratio instead of treating the AIV count as the group count.

## Data paths and pipelines

The normal Vector path is `GM -> UB -> Vector -> UB -> GM`. The normal Cube path
is `GM -> L1 -> L0A/L0B -> Cube -> L0C -> FixPipe -> GM or L1`.

Memory-transfer engines and compute can execute asynchronously. `TPipe` and
`TQue` express ownership and queue dependencies; double buffering overlaps a
tile's copy with the prior tile's compute. Lower-level `SetFlag`/`WaitFlag` and
`PipeBarrier` are legal when their producer/consumer scopes and event IDs are
proven. Reusing a buffer or event early is a correctness bug, not merely a
performance issue.

For AIC/AIV cross-core synchronization, flag IDs and outstanding counts are
limited. Keep every set/wait paired across all boundary and early-exit paths.

## Cube tiling

The configured base M/N/K for common half or BF16 Cube work is 16/16/16. INT8
uses a deeper K grouping. This is not a complete API-legality table: valid A/B,
C/accumulator, bias, quantization, and FixPipe combinations depend on the CANN
version and the selected high-level Matmul interface.

Host-side tiling must solve the documented L0A/L0B/L0C/L1/BT capacity equations
and emit a tiling key that selects matching device code. Always study
`op_host/` and `op_kernel/` together in a release-matched reference operator.

## Build and launch choices

CANN 8.5 supports two useful integration styles:

1. Build an Ascend CMake target with `find_package(ASC)` and `.asc` sources (or
   source language `ASC`), then launch through ACL.
2. For generated kernels, use ACL RTC to create and compile a program for the
   deployed 220x architecture, obtain the binary, load it, and launch it with a
   runtime configuration.

The standard host lifecycle initializes ACL, selects the device, creates a
stream, allocates/copies global memory, launches the kernel, synchronizes, and
releases resources. A Python wrapper is plumbing around this lifecycle; it is
not itself the AscendC implementation.

Compiler architecture flags, runtime SoC strings, torch-npu wheels, CANN ops
packages, driver, and firmware must be version-compatible. Do not substitute
the internal `dav-c220-*` platform label for the compiler's documented 220x
architecture flag.

## Primary references

- [Ascend C hardware abstraction](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html)
- [A2 architecture details](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html)
- [SPMD programming](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00028.html)
- [TPipe, TQue, and double buffering](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00033.html)
- [Ascend CMake build](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00039.html)
- [ACL RTC compilation](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00040.html)
- [ACL kernel launch](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00043.html)
- [AscendC API catalog](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)

## Related documents

- [Ascend 910B1 hardware facts](../../ascend910b1/hardware-specs/hardware_specs_ascend910b1.md)
- [Generation checklist](../../kernel-opt/ascendc/generation-checklist.md)

