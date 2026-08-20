# AscendC Generation Checklist for Ascend 910B

Use this checklist before accepting an agent-generated AscendC candidate.

## 1. Freeze the target capability

- Record the long SoC name, not only `910B`.
- Record CANN, 910B ops package, driver, firmware, compiler, torch, and torch-npu versions.
- Query AIC/AIV counts and usable local memory; keep an explicit architecture
  override available when the Python runtime cannot import torch-npu.
- Reject a candidate built for another B variant or CANN ABI.

## 2. Establish the operator contract

- Enumerate every shape, dtype, layout, stride, device, tolerance, and aliasing case.
- Keep a reference result and multiple seeded boundary cases.
- Decide whether the implementation is AIV-only, AIC-only, or MIX before
  selecting `blockDim` and tiling.
- Pair host tiling/registration and device kernel sources in one declared
  candidate; a standalone device `.cpp` is rarely a runnable custom operator.

## 3. Search release-matched examples

Search the release-matched source for the operator class first:

1. `reference-projects/ops-math`, `ops-nn`, `ops-transformer`, or `ops-cv` on
   the deployed CANN maintenance line: production `op_host/` + `op_kernel/`
   patterns and paired examples.
2. `reference-projects/vllm-ascend` at the matching release: LLM integration,
   Python registration, packaging, and workload-specific custom operators.
3. `reference-projects/cann-recipes-infer`: end-to-end LLM and multimodal
   inference optimization patterns; validate its APIs against the deployed
   CANN release before use.

The four CANN operator repositories have hardware-limited CANN Open Software
licenses. Learn patterns and APIs; do not copy blocks verbatim without a
separate license/provenance review.

## 4. Prove memory and synchronization legality

- Budget all queues, work tensors, padding, and double buffers below queried
  usable UB/L0/L1 capacity.
- Make every GM/local transfer length and address legal for its destination.
- Cover non-32-byte tails explicitly.
- Validate each intrinsic's dtype, repeat, mask, stride, overlap, and architecture
  constraints against the installed CANN API documentation.
- Pair all queue and event transitions on tail/empty/early-exit paths.

## 5. Verify on the target

1. Compile with the target CANN environment and fail on warnings that imply an
   unsupported architecture or API.
2. Run correctness across all required shapes and multiple seeds.
3. Re-run the incumbent and candidate in the same device allocation using ABBA
   ordering; report distributions, not one timing.
4. Use msProf operator profiling to inspect pipelines, hotspots, memory traffic,
   and `ResourceConflictRatio`. Do not synthesize NVIDIA NCU or AMD rocprof
   conclusions when no Ascend profile artifact exists.

## References

- [Ascend 910B1 hardware facts](../../ascend910b1/hardware-specs/hardware_specs_ascend910b1.md)
- [AscendC programming model](../../ref-docs/ascendc/programming-model.md)
- [msProf operator profiling](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0082.html)
- [msKPP performance analysis](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0006.html)
