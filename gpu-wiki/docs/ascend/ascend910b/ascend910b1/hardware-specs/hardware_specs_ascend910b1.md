# Ascend 910B1 Hardware Specifications for AscendC

**Last updated:** 2026-08-11  
**Validated deployment:** CANN 8.5.0, long SoC name `Ascend910B1`

This page records the values needed to generate and tune an AscendC operator.
Treat the long runtime SoC name and the installed CANN platform data as the
authority. `Ascend 910B` is a family name, not a safe substitute for `910B1`.

## Identity and topology

| Property | Ascend 910B1 value | Generation consequence |
|---|---:|---|
| Architecture | A2 / NPU Arch 2201 | Compile 220x device code for the installed toolkit |
| Core organization | split AIC (Cube) and AIV (Vector) | Select AIC-only, AIV-only, or MIX explicitly |
| AIC / Cube cores | 24 | Maximum normal AIC-only block count |
| AIV / Vector cores | 48 | Maximum normal AIV-only block count |
| MIX grouping | 1 AIC : 2 AIV | MIX `blockDim` is 24 logical groups, not 48 |
| AI CPU cores | 6 | Host/runtime resource; not an AscendC data-parallel block count |
| Nominal core clock in platform data | 1850 MHz | Input to the versioned cycle-model peak table below |
| Config-reported device memory | 64 GiB | Confirm free memory at run time |
| Shared L2 | 192 MiB | Shared cache; effective behavior must be profiled |

The deployed CANN 8.5.0 platform file reports these values. A public mirror of
the CANN 8.1.RC1 platform data independently shows the same B1 topology and
capacities, but it is secondary evidence only. B2, B3, and B4 enable different
core counts or clocks, so a generator must not collapse their aliases into B1.

## Versioned peak table

The following values are the **dense CANN 8.5 cycle-model ceilings for the
validated `Ascend910B1` target**, not an unqualified 910B-family marketing
table. A multiply-accumulate counts as two operations. The common formula is:

```text
device peak = active cores × nominal cycles/s × work/cycle
```

The installed platform contract supplies 24 AICs, 48 AIVs, 1850 MHz, and the
configured per-dtype Cube M/K/N shapes. CANN 8.5 independently documents one
FP16 Cube `16 × 16 × 16` MAC and one 128-element FP16 Vector add per core per
cycle. A configured tile shape is not by itself proof that every dtype issues
one MMAD per cycle, so conditional rows say so explicitly.

| Execution path | Dense cycle-model peak | Derivation | Confidence / use |
|---|---:|---|---|
| AIC FP16 Cube | **363.7248 TFLOP/s** | `24 × 1.85 GHz × 16 × 16 × 16 × 2` | High; default FP16 matrix roofline |
| AIC BF16 Cube | **363.7248 TFLOP/s if FP16 issue rate applies** | BF16 is supported with the default 16 × 16 × 16 tile | Medium; use only as an explicit model assumption |
| AIC FP32 Cube | **unknown** | FP32 MMAD is supported, but no B1 per-cycle rate was found | Fail closed; do not infer from tile width |
| AIC HF32 Cube | **unknown** | Ascend calls this mode HF32, not NVIDIA TF32 | Fail closed until a version-matched rate is measured or published |
| AIC INT8 Cube | **727.4496 TOPS** | `24 × 1.85 GHz × 16 × 32 × 16 × 2` | High CANN-derived ceiling; the installed profiler model confirms work per active cycle |
| AIC INT4 Cube | **1454.8992 TOPS if one MMAD issues per cycle** | `24 × 1.85 GHz × 16 × 64 × 16 × 2` | Medium; tile shape is configured, issue rate is not explicitly published |
| AIV FP16 add | **11.3664 tera-add/s** | `48 × 1.85 GHz × 128 adds` | High for plain add; do not use as a Cube ceiling |
| AIV FP32 add | **5.6832 tera-add/s if one repeat issues per cycle** | `48 × 1.85 GHz × 64 adds` | Medium; repeat width is documented, issue rate is an assumption |
| AIV FMA / transcendental | **not asserted** | Per-intrinsic issue/latency is not public | Profile or simulator result required |
| MIX kernel | **no scalar aggregate** | AIC and AIV execute different work concurrently | Model the Cube and Vector portions separately |

CSET's secondary analysis estimates **386.4576 TFLOP/s** for the 24-core,
1.85 GHz B1 variant by counting Cube and both Vector units together. That is a
useful explanation for conflicting 910B headline numbers, but it is not the
Cube peak used here: it mixes execution paths and depends on an aggregate
accounting convention that does not describe an arbitrary AscendC kernel.

These are dense values. Although the platform advertises sparsity support, no
sparse multiplier is applied until the exact sparsity contract, dtype, and
CANN API path are proven for the candidate.

The machine-readable provider is `ascend910b1-cann-8.5.0-v1@1.2.0` in
`tools/compute_utilization.py`. It requires the execution path instead of
silently combining AIC and AIV. For example:

```bash
# Directly evidenced FP16 Cube ceiling.
python3 tools/compute_utilization.py \
  --gpu ascend910b1 --dtype fp16 --execution-kind aic \
  --flops 1000000000000 --bytes 1000000000 --time-ms 10

# Conditional BF16 model: an explicit opt-in is required.
python3 tools/compute_utilization.py \
  --gpu ascend910b1 --dtype bf16 --execution-kind aic --allow-modeled-peak \
  --bandwidth-kind measured_copy_256m \
  --flops 1000000000000 --bytes 1000000000 --time-ms 10
```

FP32/HF32, AIV FMA, and a scalar MIX peak remain fail-closed even with that
flag; `--allow-modeled-peak` only enables rows with explicit conditions.

### HBM and L2 ceilings

Bandwidth has several distinct ceilings on this target. They must not be
collapsed into one unlabeled `bandwidth` number:

| Memory evidence | Value | Confidence / semantics |
|---|---:|---|
| CANN 8.5 msProf `GM Read + Write` Roofline model | **1.8 TB/s** | Medium-high; the model embedded in a successful target Roofline artifact, not a physical-interface rating |
| CANN 8.5 msProf `L2 Read + Write` Roofline model | **8.0 TB/s** | Medium-high; same profiler model semantics |
| Independent 910B1 maximum HBM report | **1.6 TB/s** | Secondary evidence from CSET; do not present as a target-exposed Huawei/CANN field |
| Same-device 256 MiB FP16 copy | **1.26710 TB/s median**, **1.26770 TB/s max** | Practical aggregate read+write ceiling for this size and runtime |
| Installed `AICoreMemoryRates.ddr_rate` | **32, unit unspecified** | Compiler/platform cost-model field; its header does not define unit or card aggregation |
| Target-exposed physical HBM-interface rating | **unknown** | `npu-smi` reports capacity, clock, and usage percentage, but not bus width or rated GB/s |

In particular, do **not** multiply the raw `ddr_rate=32` by 24 cores and 1.85
GHz and call the result a device peak. The field's core ownership, direction,
unit, and aggregation semantics are undocumented, and that conditional result
would disagree with msProf's own 1.8 TB/s GM Roofline model.

On 2026-08-11, a 256 MiB NPU-to-NPU FP16 copy on container device 0 (mapped to
host device 2), counting one read and one write, produced seven 50-copy samples
with a median of **1.26710 TB/s** and a maximum of **1.26770 TB/s**. This is a
reproducible practical copy ceiling, not a replacement name for the physical
HBM-pin peak. Use the 1.8 TB/s msProf ceiling when matching CANN's theoretical
Roofline classification, and use the same-size measured ceiling for practical
bandwidth utilization.

Reproduce that practical ceiling in the target evaluator environment with:

```bash
python3 tools/bench_ascend_bandwidth.py \
  --device-index 0 --size-mib 256 --warmup 10 --copies 50 --trials 7
```

## Per-core local memory

| Memory | Capacity | Primary use |
|---|---:|---|
| L0A | 64 KiB | Cube A operand |
| L0B | 64 KiB | Cube B operand |
| L0C | 128 KiB | Cube accumulator/output |
| L1 | 512 KiB | Cube staging and reuse |
| Unified Buffer (UB) | 192 KiB | Vector input/output/temp queues |
| Bias Table (BT) | 1 KiB | Cube bias data |

These are physical/configured capacities, not an allocation promise. Queue
metadata, double buffering, API workspaces, and `ReserveLocalMemory` can reduce
the usable amount. Host tiling should query `PlatformAscendC::GetCoreMemSize`
for the selected core type instead of embedding 192 KiB as a universal budget.

## Transfer alignment and layouts

| Destination | Required/basic alignment on A2 |
|---|---:|
| AIV UB | 32 B |
| AIC L1 | 32 B |
| AIC L0A / L0B | 512 B |
| AIC L0C | 64 B |
| Bias Table | 64 B |
| FixPipe output | 64 B |

The common Cube layouts are `FRACTAL_ZZ` for L0A, `FRACTAL_ZN` for L0B,
`FRACTAL_NZ` for L0C, and `FRACTAL_NZ` for L1. UB does not impose one matrix
layout. A normal Vector repeat spans 256 B (eight 32-byte datablocks), but each
intrinsic has its own repeat, stride, overlap, and data-type limits.

`DataCopy` rounds an unsupported non-aligned continuous length down rather than
making the tail correct automatically. Use a legal padded tile, `DataCopyPad`,
or an explicit tail path and verify every boundary shape.

## UB bank model

CANN 8.5 documents 48 physical UB banks. Each bank has 128 rows of 32 B
(4 KiB); the banks form 16 groups of three. Read/write access to one bank,
write/write access within one group, and some read/read group patterns can
conflict. Offset or pad independent tensors and confirm the result with
`ResourceConflictRatio` in msProf.

Some older platform files expose an internal `ubbank_num=64` field. Do not use
that field as the optimization bank model; it conflicts with the documented
CANN 8.5 physical access model even though total UB capacity remains 192 KiB.

## Runtime discovery gate

Before generating or comparing a kernel, record:

```bash
npu-smi info
python3 -c 'import torch, torch_npu; print(torch.npu.get_device_name(0))'
find "${ASCEND_HOME_PATH}" -path '*platform_config/Ascend910B*.ini' -print
```

Also record the CANN toolkit, driver, firmware, 910B ops package, BiSheng/ccec,
PyTorch, and torch-npu versions. If Python packages are not present in the
selected environment, that is a runtime blocker rather than permission to
fall back to CUDA.

## Evidence and limits

- [CANN 8.5 PlatformAscendC API](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_00059.html)
- [GetCoreMemSize](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_1034.html)
- [GetCoreMemBw](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_1035.html)
- [A2 memory and execution architecture](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html)
- [CANN 8.5 per-core Cube and Vector work/cycle](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0002.html)
- [CANN 8.5 Mmad fractal shapes and A2 dtype support](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0249.html)
- [CANN 8.5 UB bank-conflict guidance](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0025.html)
- [CSET Ascend 910B model comparison and HBM analysis](https://cset.georgetown.edu/publication/pushing-the-limits-huaweis-ai-chip-tests-u-s-export-controls/) — secondary evidence only
- [CANN 8.1.RC1 Ascend910B1 platform-data mirror](https://github.com/leideng/CANN-8.1.RC1/blob/551ed27c29284929bbdc2239e929ddfedfdf0b8b/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux/data/platform_config/Ascend910B1.ini) — secondary cross-check only

Public documentation still does not provide enough version-specific
per-intrinsic issue rates, latency, cache behavior, or physical HBM-interface
data for a complete static cost model. Use the asserted paths above only, and
use simulator/msProf plus same-device ABBA measurements for everything else.

## Related documents

- [AscendC programming model](../../ref-docs/ascendc/programming-model.md)
- [Ascend 910B AscendC checklist](../../kernel-opt/ascendc/generation-checklist.md)
- [Ascend 910B pitfalls](../../pitfalls/ascendc/ascend910b-pitfalls.md)
