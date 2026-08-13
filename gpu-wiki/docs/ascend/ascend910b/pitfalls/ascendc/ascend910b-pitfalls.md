# AscendC Pitfalls on Ascend 910B

## Family alias is not a hardware contract

`910B`, Atlas A2, and a short `Ascend910B` token do not identify the enabled
AIC/AIV topology. Use the long SoC name such as `Ascend910B1`. In particular,
never hard-code 24 AIC and 48 AIV for every B variant.

## AIV count is not MIX blockDim

On Ascend 910B1, AIV-only work can normally use 48 vector blocks, while a 1:2
MIX kernel has 24 logical AIC/AIV groups. Launching 48 MIX groups is not a free
way to expose more parallelism.

## Configured UB is not fully allocatable UB

The 192 KiB configured UB includes capacity that may be reserved or consumed by
queues, double buffers, work tensors, and API workspaces. Query the platform and
show the complete per-tile byte equation. A kernel that only fits when every
buffer is counted once will fail after enabling double buffering.

## A copied tail is not automatically correct

A `DataCopy` continuous length that violates the datablock requirement may be
rounded down. Padding global memory reads can also cross a valid allocation.
Use a documented padded-copy API or an explicit legal tail and test lengths on
both sides of every 32-byte boundary.

## Pipeline primitives carry correctness state

Queues, events, and barriers are not performance annotations. Reusing a local
tensor before `FreeTensor`, mismatching `EnQue`/`DeQue`, or skipping a flag wait
on an empty tile can race. Cross-core flag IDs and outstanding counts are
limited; test multi-block execution, not only a one-block debug case.

## Kernel-only examples omit required ABI work

A production custom operator also needs compatible host tiling, shape/type
inference, registration, build/package metadata, and Python/runtime launch glue.
Search `op_host/` and `op_kernel/` as a pair and declare every source file in the
candidate manifest.

## CANN examples are versioned API evidence

An implementation from a newer ops-nn or vLLM Ascend release may compile
against APIs absent from CANN 8.5. Conversely, the migrated `cann-ops` master is
not a release-matched ABI definition. Pin reference repositories and toolchain
versions before generation.

## No profiler artifact means no profiler conclusion

Public specifications do not reveal enough latency, throughput, cache, or board
bandwidth detail to choose a winning tile statically. Run msProf/simulator and
same-device measurements. Do not translate NCU metrics mechanically to AIC/AIV.

## References

- [DataCopy constraints](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0103.html)
- [Kernel type configuration](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0218.html)
- [Matmul API constraints](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0614.html)
- [TCubeTiling capacity equations](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0673.html)

