# Reference Projects

This directory contains GPU kernel optimization reference projects managed as git submodules.

## Usage

```bash
# Initialize all optional reference projects
git submodule update --init

# Or initialize only the CANN 8.5 / AscendC corpus
git submodule update --init --depth 1 \
  reference-projects/ops-nn \
  reference-projects/vllm-ascend \
  reference-projects/cann-ops
```

## Included Projects

| Project | Description |
|---------|-------------|
| cutlass | NVIDIA CUTLASS - CUDA Templates for Linear Algebra Subroutines |
| cutex | CUDA Template Extensions |
| cuLA | inclusionAI CUDA Linear Algebra |
| flash-attention | Flash Attention |
| flashinfer | FlashInfer - Kernel Library for LLM Serving |
| FlyDSL | ROCm FlyDSL |
| triton | Triton Language and Compiler |
| DeepGEMM | DeepSeek DeepGEMM |
| LeetCUDA | LeetCUDA - CUDA Learning |
| FlashMLA | DeepSeek FlashMLA |
| composable_kernel | ROCm Composable Kernel |
| cute-gemm | CuTe GEMM Examples |
| hpc-ops | Tencent HPC Ops |
| aiter | ROCm AIter |
| quack | Dao-AILab Quack |
| tilelang | TileLang |
| ops-nn | CANN 8.5 maintenance-line operators; primary AscendC kernel + host-tiling reference (the current official project corresponding to the older `cann-nn` name) |
| vllm-ascend | vLLM Ascend v0.13.0; CANN 8.5.0 LLM custom-operator integration |
| cann-ops | Migrated historical CANN operator patterns; supplementary reference only, not the CANN 8.5 ABI baseline |

## Ascend reference pins and search order

The gitlinks, not moving remote branches, define the reproducible versions:

| Project | Target line | Pinned commit | Use |
|---|---|---|---|
| ops-nn | CANN 8.5.0 maintenance branch | `9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef` | Search first: paired `op_host/` and `op_kernel/`, plus `examples/add_example/` |
| vllm-ascend | v0.13.0 | `6281c1207a7a499e9f23a42b3a1e7027469f2b10` | Search second: LLM workload, Python/runtime, build, registration, and AscendC sources |
| cann-ops | migrated master snapshot | `35fcd12e27bc446aef29bdf801d2869c15685f84` | Search last: historical implementation patterns only |

`ops-nn` and `cann-ops` use CANN Open Software License Agreements with
Ascend/Huawei-processor use and redistribution restrictions. They are exposed
to the agent as read-only reference corpora: learn APIs and implementation
structure, and perform a separate license/provenance review before copying any
source. `vllm-ascend` is Apache-2.0.

The published vLLM Ascend v0.13.0 matrix pairs CANN 8.5.0 with PyTorch and
torch-npu 2.8. A worker using a different PyTorch/torch-npu line must treat this
checkout as source-only integration evidence and validate every API against its
installed headers; it is not a drop-in runtime dependency merely because the
CANN version matches.

Do not use `git submodule update --remote` in a reproducible campaign: it moves
the reference checkout away from the CANN version validated for the target.
