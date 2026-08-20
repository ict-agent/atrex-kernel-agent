# Reference Projects

This directory contains GPU kernel optimization reference projects managed as git submodules.

## Usage

```bash
# Initialize all optional reference projects
git submodule update --init

# Or initialize only the pinned AscendC and inference corpus
git submodule update --init --depth 1 \
  reference-projects/ops-math \
  reference-projects/ops-nn \
  reference-projects/ops-transformer \
  reference-projects/ops-cv \
  reference-projects/vllm-ascend \
  reference-projects/cann-recipes-infer
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
| ops-math | CANN 8.5 mathematical operators; primary reference for elementwise, reduction, indexing, and numerical kernels |
| ops-nn | CANN 8.5 neural-network operators; primary reference for NN kernels and paired host tiling |
| ops-transformer | CANN 8.5 Transformer operators and fused LLM kernels |
| ops-cv | CANN 8.5 computer-vision and detection operators |
| vllm-ascend | vLLM Ascend v0.13.0; CANN 8.5.0 LLM custom-operator integration |
| cann-recipes-infer | Pinned CANN LLM and multimodal inference optimization recipes; integration evidence rather than an 8.5 ABI baseline |

## Ascend reference pins and search order

The gitlinks, not moving remote branches, define the reproducible versions:

| Project | Target line | Pinned commit | Use |
|---|---|---|---|
| ops-math | CANN 8.5.0 maintenance branch | `b6031bda9848fb77c4124891a90ff83db5c40d21` | Mathematical and elementwise operators, including the closest references for add |
| ops-nn | CANN 8.5.0 maintenance branch | `9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef` | Neural-network kernels with paired `op_host/` and `op_kernel/` implementations |
| ops-transformer | CANN 8.5.0 maintenance branch | `8fff8f9279086404073e2b38e51474792cae9e7b` | Transformer and fused LLM operator implementations |
| ops-cv | CANN 8.5.0 maintenance branch | `aa2542dbe5f3efb8dd35ba9bab7dc7ec1952f230` | Computer-vision and detection operator implementations |
| vllm-ascend | v0.13.0 | `6281c1207a7a499e9f23a42b3a1e7027469f2b10` | LLM workload, Python/runtime, build, registration, and AscendC sources |
| cann-recipes-infer | master snapshot | `a9adc555442bb38cdffee043d70fdbc59fa49650` | End-to-end LLM and multimodal inference optimization recipes |

The four `ops-*` repositories use CANN Open Software License Agreement Version
2.0 with Ascend/Huawei-processor use and redistribution restrictions. They are
exposed to the agent as read-only reference corpora: learn APIs and
implementation structure, and perform a separate license/provenance review
before copying any source. `vllm-ascend` and `cann-recipes-infer` are
Apache-2.0.

The published vLLM Ascend v0.13.0 matrix pairs CANN 8.5.0 with PyTorch and
torch-npu 2.8. A worker using a different PyTorch/torch-npu line must treat this
checkout as source-only integration evidence and validate every API against its
installed headers; it is not a drop-in runtime dependency merely because the
CANN version matches. The recipes snapshot follows master rather than the CANN
8.5 maintenance line, so its APIs require the same validation.

Do not use `git submodule update --remote` in a reproducible campaign: it moves
the reference checkout away from the CANN version validated for the target.
