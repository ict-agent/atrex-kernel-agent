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

"""
Roofline bottleneck analysis plus compute and bandwidth utilization calculator.

Core idea:
  Roofline analysis is performed at tile/block granularity rather than over
  whole-kernel global FLOPs and bytes. A GPU schedules blocks; each block loads
  its tile from HBM, computes, and stores results independently. Different tile
  sizes produce different arithmetic intensity and may change the bottleneck.

Features:
  1. Build a tile-level Roofline model and classify each tile as compute-bound
     or memory-bound.
  2. Select the proper utilization metric from the bottleneck type:
     - Compute-bound: actual TFLOPS / peak TFLOPS
     - Memory-bound: actual bandwidth / bandwidth ceiling

Supported GPUs (built-in specs):
  NVIDIA Hopper: h100, h20, h200
  AMD CDNA3:     mi300x, mi308x
  AMD CDNA4:     mi355x
  Ascend A2:     ascend910b1 (versioned CANN 8.5 cycle-model ceilings; an
                 explicit AIC/AIV execution path is required)
  Any other GPU (e.g. Blackwell): pass gpu-wiki-sourced peaks via --peak-tflops and
  --peak-bandwidth-tb-s; the tool never fabricates specs it does not have.

Usage:
    python tools/compute_utilization.py         --gpu h20 --dtype bf16         --flops-expr "2*BM*BN*K" --bytes-expr "(BM*K + BN*K + BM*BN)*2"         --time-ms 0.5 --grid-blocks 64

    python tools/compute_utilization.py         --gpu h100 --dtype bf16         --flops-expr "2*BM*BN*K" --bytes-expr "(BM*K + BN*K + BM*BN)*2"         --time-ms 0.05 --grid-blocks 16         --measured-bandwidth-tb-s 2.8

    python tools/compute_utilization.py         --flops 134217728 --bytes 1212416         --time-ms 0.5 --grid-blocks 64         --gpu mi300x --dtype bf16
"""

import argparse
from copy import deepcopy
import math
import sys
import os
import shutil
import importlib.util

# ============================================================
# Versioned peak evidence providers
# ============================================================

PEAK_PROVIDER_SCHEMA_VERSION = 1
ASCEND_910B1_PEAK_PROVIDER_ID = "ascend910b1-cann-8.5.0-v1"

# A provider record separates a target identity/topology contract from numeric
# Roofline ceilings. Every metric carries source, confidence, derivation, and a
# type that says whether it is documented, modeled, or a physical-interface
# peak. Unknown values fail closed.
PEAK_PROVIDERS = {
    ASCEND_910B1_PEAK_PROVIDER_ID: {
        "schema_version": PEAK_PROVIDER_SCHEMA_VERSION,
        "provider_id": ASCEND_910B1_PEAK_PROVIDER_ID,
        "provider_version": "1.2.0",
        "target": {
            "vendor": "ascend",
            "soc_version": "Ascend910B1",
            "npu_arch": 2201,
            "cann_version": "8.5.0",
        },
        "sources": {
            "cann_850_platform_config": {
                "kind": "version_matched_toolchain_config",
                "title": "CANN 8.5.0 Ascend910B1 platform configuration",
                "locator": (
                    "<CANN_ROOT>/<architecture>-linux/data/platform_config/"
                    "Ascend910B1.ini"
                ),
                "validated_sha256": (
                    "6c8ba94d7d64186c8721ceeade3e9403cd33e8d5d55eb0e6a9ec7ab294fa7449"
                ),
                "fields": [
                    "version.SoC_version",
                    "version.NpuArch",
                    "SoCInfo.cube_core_cnt",
                    "SoCInfo.vector_core_cnt",
                    "SoCInfo.cube_vector_combine",
                    "AICoreSpec.cube_freq",
                    "DtypeMKN.Default",
                    "DtypeMKN.DT_INT8",
                    "DtypeMKN.DT_INT4",
                    "AICoreMemoryRates.ddr_rate",
                    "AICoreMemoryRates.l2_rate",
                ],
                "caveat": (
                    "AICoreMemoryRates fields do not state enough unit/aggregation "
                    "semantics to derive a device-wide bandwidth ceiling"
                ),
            },
            "cann_850_cube_vector_cycle_model": {
                "kind": "official_cann_documentation",
                "title": "CANN 8.5 Ascend C heterogeneous compute architecture",
                "locator": (
                    "https://www.hiascend.com/document/detail/en/canncommercial/850/"
                    "opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0002.html"
                ),
                "claims": [
                    "one Cube executes a 16x16x16 FP16 matrix MAC per cycle",
                    "one Vector executes 128 FP16 additions per cycle",
                ],
            },
            "cann_850_mmad_a2": {
                "kind": "official_cann_api_documentation",
                "title": "CANN 8.5 Ascend C Mmad API, Ascend AI Processor A2",
                "locator": (
                    "https://www.hiascend.com/document/detail/en/canncommercial/850/"
                    "API/ascendcopapi/atlasascendc_api_07_0249.html"
                ),
                "claims": [
                    "A fractal is 16 x (32B / sizeof(AType))",
                    "B fractal is (32B / sizeof(BType)) x 16",
                ],
            },
            "cann_850_vector_repeat": {
                "kind": "official_cann_documentation",
                "title": "CANN 8.5 Ascend C vector repeat calculation",
                "locator": (
                    "https://www.hiascend.com/document/detail/en/canncommercial/850/"
                    "opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0030.html"
                ),
                "claims": ["one vector repeat covers 256 bytes"],
            },
            "cann_850_msprof_roofline": {
                "kind": "version_matched_profiler_artifact",
                "title": "CANN 8.5 msprof Roofline metric model for Ascend910B1",
                "locator": "msprof op --aic-metrics=Roofline (target CANN 8.5 artifact)",
                "fields": [
                    "GM read+write ceiling = 1.8 TB/s",
                    "L2 read+write ceiling = 8.0 TB/s",
                ],
                "caveat": (
                    "profiler analysis-model ceilings; not physical interface ratings"
                ),
            },
            "cann_850_int8_profiler_formula": {
                "kind": "version_matched_installed_profiler_source",
                "title": "CANN 8.5 AI Core profiler INT8 active-cycle formula",
                "locator": (
                    "<CANN_ROOT>/tools/profiler (calculate_ai_core_data.py and "
                    "ai_core_config.py)"
                ),
                "claims": [
                    "Ascend910B1 INT8 work is 16x16x32 MACs per active cycle",
                    "one MAC is counted as two operations",
                ],
            },
            "target_copy_256m": {
                "kind": "target_measurement",
                "title": "Ascend910B1 256 MiB torch copy practical bandwidth run",
                "locator": "tools/bench_ascend_bandwidth.py on 2026-08-11",
                "environment": {
                    "container_device": 0,
                    "host_device": 2,
                    "torch": "2.6.0",
                    "torch_npu": "2.6.0.post5",
                },
                "method": {
                    "bytes": 268435456,
                    "traffic_convention": "one source read plus one destination write",
                    "warmup_copies": 10,
                    "timed_copies_per_run": 50,
                    "runs": 7,
                },
                "result": {
                    "median_tb_s": 1.267099,
                    "max_tb_s": 1.267702,
                    "samples_tb_s": [
                        1.266477,
                        1.266910,
                        1.267165,
                        1.267279,
                        1.267702,
                        1.267099,
                        1.266437,
                    ],
                },
            },
        },
        "topology": {
            "aic_units": {
                "status": "available",
                "value": 24,
                "unit": "core",
                "source_ids": ["cann_850_platform_config"],
                "confidence": {
                    "level": "high",
                    "basis": "direct field from the version-matched installed platform config",
                },
                "derivation": {
                    "kind": "direct",
                    "expression": "SoCInfo.cube_core_cnt",
                },
            },
            "aiv_units": {
                "status": "available",
                "value": 48,
                "unit": "core",
                "source_ids": ["cann_850_platform_config"],
                "confidence": {
                    "level": "high",
                    "basis": "direct field from the version-matched installed platform config",
                },
                "derivation": {
                    "kind": "direct",
                    "expression": "SoCInfo.vector_core_cnt",
                },
            },
            "mix_groups": {
                "status": "available",
                "value": 24,
                "unit": "1-AIC:2-AIV group",
                "source_ids": ["cann_850_platform_config"],
                "confidence": {
                    "level": "high",
                    "basis": "derived from the version-matched split-core topology",
                },
                "derivation": {
                    "kind": "integer_topology",
                    "expression": "min(aic_units, aiv_units // 2)",
                },
            },
            "clock_mhz": {
                "status": "available",
                "value": 1850,
                "unit": "MHz",
                "source_ids": ["cann_850_platform_config"],
                "confidence": {
                    "level": "high",
                    "basis": "direct field from the version-matched installed platform config",
                },
                "derivation": {
                    "kind": "direct",
                    "expression": "AICoreSpec.cube_freq",
                },
            },
        },
        "peaks": {
            "compute_tflops": {
                "aic": {
                    "fp16": {
                        "status": "available",
                        "value": 363.7248,
                        "unit": "TFLOPS",
                        "peak_type": "cann_cycle_model_dense_cube",
                        "execution_kind": "aic",
                        "operation_kind": "mac",
                        "source_ids": [
                            "cann_850_platform_config",
                            "cann_850_cube_vector_cycle_model",
                        ],
                        "confidence": {
                            "level": "high",
                            "basis": "version-matched core/clock data and documented Cube work per cycle",
                        },
                        "derivation": {
                            "kind": "dense_cycle_model",
                            "expression": "24 * 1.85e9 * (16 * 16 * 16) * 2 / 1e12",
                        },
                    },
                    "bf16": {
                        "status": "conditional",
                        "value": 363.7248,
                        "unit": "TFLOPS",
                        "peak_type": "modeled_if_one_issue_per_cycle",
                        "execution_kind": "aic",
                        "operation_kind": "mac",
                        "source_ids": [
                            "cann_850_platform_config",
                            "cann_850_mmad_a2",
                        ],
                        "confidence": {
                            "level": "medium",
                            "basis": "version-matched DtypeMKN shape with a one-Cube-issue-per-cycle assumption",
                        },
                        "conditions": [
                            "assumes one 16x16x16 BF16 Cube issue per AIC cycle"
                        ],
                        "requires_model_opt_in": True,
                        "derivation": {
                            "kind": "modeled_if_one_issue_per_cycle",
                            "expression": "24 * 1.85e9 * (16 * 16 * 16) * 2 / 1e12",
                        },
                    },
                    "fp32": {
                        "status": "unknown",
                        "value": None,
                        "unit": "TFLOPS",
                        "peak_type": "unknown",
                        "execution_kind": "aic",
                        "operation_kind": "mac",
                        "source_ids": [],
                        "confidence": {
                            "level": "unknown",
                            "basis": "no Ascend910B1 FP32 Cube issue rate was found",
                        },
                        "derivation": {
                            "kind": "not_derived",
                            "expression": None,
                        },
                        "reason": "Mmad fractal shape does not establish the Ascend910B1 FP32 issue rate",
                    },
                    "hf32": {
                        "status": "unknown",
                        "value": None,
                        "unit": "TFLOPS",
                        "peak_type": "unknown",
                        "execution_kind": "aic",
                        "operation_kind": "mac",
                        "source_ids": [],
                        "confidence": {
                            "level": "unknown",
                            "basis": "no Ascend910B1 HF32 Cube issue rate was found",
                        },
                        "derivation": {"kind": "not_derived", "expression": None},
                        "reason": "no version-matched HF32 per-cycle rate is encoded",
                    },
                    "int8": {
                        "status": "available",
                        "value": 727.4496,
                        "unit": "TOPS",
                        "peak_type": "cann_profiler_cycle_model",
                        "execution_kind": "aic",
                        "operation_kind": "mac",
                        "source_ids": [
                            "cann_850_platform_config",
                            "cann_850_int8_profiler_formula",
                        ],
                        "confidence": {
                            "level": "high",
                            "basis": "version-matched installed profiler formula, core count, and clock; not an official SKU rating",
                        },
                        "derivation": {
                            "kind": "cann_profiler_active_cycle_model",
                            "expression": "24 * 1.85e9 * (16 * 32 * 16) * 2 / 1e12",
                        },
                    },
                    "int4": {
                        "status": "conditional",
                        "value": 1454.8992,
                        "unit": "TOPS",
                        "peak_type": "modeled_if_one_issue_per_cycle",
                        "execution_kind": "aic",
                        "operation_kind": "mac",
                        "source_ids": ["cann_850_platform_config"],
                        "confidence": {
                            "level": "medium",
                            "basis": "version-matched DT_INT4 MKN with a one-Cube-issue-per-cycle assumption; not an official SKU rating",
                        },
                        "conditions": [
                            "assumes one 16x64x16 INT4 Cube issue per AIC cycle"
                        ],
                        "requires_model_opt_in": True,
                        "derivation": {
                            "kind": "modeled_if_one_issue_per_cycle",
                            "expression": "24 * 1.85e9 * (16 * 64 * 16) * 2 / 1e12",
                        },
                    },
                    "default": {
                        "status": "unknown",
                        "value": None,
                        "unit": "TFLOPS",
                        "peak_type": "unknown",
                        "source_ids": [],
                        "confidence": {
                            "level": "unknown",
                            "basis": "no version-matched AIC cycle model is encoded for this dtype",
                        },
                        "derivation": {"kind": "not_derived", "expression": None},
                        "reason": "unsupported AIC dtype for the versioned cycle model",
                    },
                },
                "aiv": {
                    "fp16": {
                        "add": {
                            "status": "available",
                            "value": 11.3664,
                            "unit": "TFLOPS",
                            "peak_type": "documented_cann_cycle_model",
                            "execution_kind": "aiv",
                            "operation_kind": "add",
                            "source_ids": [
                                "cann_850_platform_config",
                                "cann_850_cube_vector_cycle_model",
                            ],
                            "confidence": {
                                "level": "medium-high",
                                "basis": "documented 128 FP16 adds per AIV cycle and version-matched core/clock data",
                            },
                            "derivation": {
                                "kind": "cycle_model",
                                "expression": "48 * 1.85e9 * 128 / 1e12",
                            },
                        },
                        "fma": {
                            "status": "unknown",
                            "value": None,
                            "unit": "TFLOPS",
                            "peak_type": "unknown",
                            "execution_kind": "aiv",
                            "operation_kind": "fma",
                            "source_ids": [],
                            "confidence": {
                                "level": "unknown",
                                "basis": "AIV FMA issue throughput is not documented",
                            },
                            "derivation": {
                                "kind": "not_derived",
                                "expression": None,
                            },
                            "reason": "128 FP16 add/cycle does not establish an AIV FMA issue rate",
                        },
                    },
                    "fp32": {
                        "add": {
                            "status": "conditional",
                            "value": 5.6832,
                            "unit": "TFLOPS",
                            "peak_type": "derived_cann_cycle_model",
                            "execution_kind": "aiv",
                            "operation_kind": "add",
                            "source_ids": [
                                "cann_850_platform_config",
                                "cann_850_vector_repeat",
                            ],
                            "confidence": {
                                "level": "medium",
                                "basis": "64 FP32 lanes are inferred from a 256-byte repeat divided by element width",
                            },
                            "conditions": [
                                "infers 64 FP32 lanes from 256 bytes per repeat",
                                "assumes one FP32 add issue per AIV cycle",
                            ],
                            "requires_model_opt_in": True,
                            "derivation": {
                                "kind": "type_width_cycle_model",
                                "expression": "48 * 1.85e9 * (256 / 4) / 1e12",
                            },
                        },
                        "fma": {
                            "status": "unknown",
                            "value": None,
                            "unit": "TFLOPS",
                            "peak_type": "unknown",
                            "execution_kind": "aiv",
                            "operation_kind": "fma",
                            "source_ids": [],
                            "confidence": {
                                "level": "unknown",
                                "basis": "AIV FP32 FMA lanes and issue throughput are not documented",
                            },
                            "derivation": {
                                "kind": "not_derived",
                                "expression": None,
                            },
                            "reason": "the FP32 add lane model does not establish an AIV FMA issue rate",
                        },
                    },
                    "default": {
                        "status": "unknown",
                        "value": None,
                        "unit": "TFLOPS",
                        "peak_type": "unknown",
                        "source_ids": [],
                        "confidence": {
                            "level": "unknown",
                            "basis": "no version-matched AIV cycle model is encoded for this dtype",
                        },
                        "derivation": {"kind": "not_derived", "expression": None},
                        "reason": "unsupported AIV dtype for the versioned cycle model",
                    },
                },
                "mix": {
                    "status": "unknown",
                    "value": None,
                    "unit": "TFLOPS",
                    "peak_type": "unknown",
                    "source_ids": [],
                    "confidence": {
                        "level": "unknown",
                        "basis": "AIC and AIV execute different operation classes",
                    },
                    "derivation": {"kind": "not_derived", "expression": None},
                    "reason": (
                        "MIX has no scalar compute peak; select AIC or AIV and its operation model"
                    ),
                },
            },
            "memory_bandwidth_tb_s": {
                "default_kind": "hbm_cycle_model",
                "hbm_cycle_model": {
                    "status": "available",
                    "value": 1.8,
                    "unit": "TB/s",
                    "peak_type": "cann_msprof_roofline_model",
                    "memory_level": "hbm",
                    "physical_interface_peak": False,
                    "source_ids": ["cann_850_msprof_roofline"],
                    "confidence": {
                        "level": "medium-high",
                        "basis": "direct ceiling embedded in the target CANN 8.5 msprof Roofline model",
                    },
                    "derivation": {
                        "kind": "direct_profiler_model_value",
                        "expression": "msprof Roofline GM read+write ceiling",
                    },
                },
                "l2_cycle_model": {
                    "status": "available",
                    "value": 8.0,
                    "unit": "TB/s",
                    "peak_type": "cann_msprof_roofline_model",
                    "memory_level": "l2",
                    "physical_interface_peak": False,
                    "source_ids": ["cann_850_msprof_roofline"],
                    "confidence": {
                        "level": "medium-high",
                        "basis": "direct ceiling embedded in the target CANN 8.5 msprof Roofline model",
                    },
                    "derivation": {
                        "kind": "direct_profiler_model_value",
                        "expression": "msprof Roofline L2 read+write ceiling",
                    },
                },
                "measured_copy_256m": {
                    "status": "available",
                    "value": 1.267099,
                    "unit": "TB/s",
                    "peak_type": "measured_practical_ceiling",
                    "memory_level": "hbm",
                    "physical_interface_peak": False,
                    "source_ids": ["target_copy_256m"],
                    "confidence": {
                        "level": "medium-high",
                        "basis": "median of seven repeated target runs for the stated 256 MiB copy setup",
                    },
                    "derivation": {
                        "kind": "measured_median",
                        "expression": "median(7 runs of 50 timed 256 MiB copies after 10 warmups)",
                    },
                    "scope": "practical ceiling for the recorded target and transfer size",
                },
                "physical_hbm_interface_peak": {
                    "status": "unknown",
                    "value": None,
                    "unit": "TB/s",
                    "peak_type": "physical_hbm_interface_peak",
                    "memory_level": "hbm",
                    "physical_interface_peak": True,
                    "source_ids": [],
                    "confidence": {
                        "level": "unknown",
                        "basis": "no version-matched physical HBM interface peak is encoded",
                    },
                    "derivation": {"kind": "not_derived", "expression": None},
                    "reason": "CANN profiler/CSET model ceilings are not a physical HBM interface rating",
                },
            },
        },
    },
}


# ============================================================
# Unified accelerator hardware spec table
# ============================================================
HARDWARE_SPECS = {
    # ── NVIDIA Hopper ──
    "h100": {
        "fp64_tensor": 33.5,
        "fp32_cuda": 67.0,
        "tf32": 494.7,
        "fp16": 989.4,
        "bf16": 989.4,
        "fp8": 1978.9,
        "int8": 1978.9,
        "memory_bandwidth_tb_s": 3.35,
        "num_units": 132,
        "unit_type": "SM",
        "description": "NVIDIA H100 SXM (sm_90, Hopper)",
    },
    "h20": {
        "fp16": 148.0,
        "bf16": 148.0,
        "fp8": 296.0,
        "int8": 296.0,
        "fp32_cuda": 39.6,
        "memory_bandwidth_tb_s": 4.0,
        "num_units": 78,
        "unit_type": "SM",
        "description": "NVIDIA H20 (sm_90, Hopper)",
    },
    "h200": {
        "fp64_tensor": 33.5,
        "fp32_cuda": 67.0,
        "tf32": 494.7,
        "fp16": 989.4,
        "bf16": 989.4,
        "fp8": 1978.9,
        "int8": 1978.9,
        "memory_bandwidth_tb_s": 4.8,
        "num_units": 132,
        "unit_type": "SM",
        "description": "NVIDIA H200 (sm_90, Hopper, HBM3e)",
    },
    # ── AMD CDNA3 ──
    "mi300x": {
        "fp64_vector": 81.7,
        "fp64_matrix": 163.4,
        "fp32": 163.4,
        "tf32": 653.7,
        "fp16": 1307.4,
        "bf16": 1307.4,
        "fp8": 2614.9,
        "int8": 2614.9,
        "memory_bandwidth_tb_s": 5.3,
        "num_units": 304,
        "unit_type": "CU",
        "description": "AMD Instinct MI300X (gfx942, CDNA3)",
    },
    "mi308x": {
        "fp16": 232.0,
        "bf16": 232.0,
        "fp8": 465.0,
        "int8": 465.0,
        "memory_bandwidth_tb_s": 5.3,
        "num_units": 80,
        "unit_type": "CU",
        "description": "AMD Instinct MI308X (gfx942, CDNA3)",
    },
    # ── AMD CDNA4 ──
    "mi355x": {
        "fp64": 78.6,
        "fp32": 157.3,
        "fp16": 5033.2,
        "bf16": 5033.2,
        "fp8": 10066.4,
        "int8": 10066.4,
        "fp6": 20132.6,
        "fp4": 20132.6,
        "memory_bandwidth_tb_s": 8.0,
        "num_units": 256,
        "unit_type": "CU",
        "description": "AMD Instinct MI355X (gfx950, CDNA4, HBM3e)",
    },
    # Ascend ceilings live in the provider because their execution path and
    # cycle-model/physical distinction cannot be represented by this flat table.
    "ascend910b1": {
        "peak_provider": ASCEND_910B1_PEAK_PROVIDER_ID,
        "unit_type": "AIC/AIV",
        "description": "Huawei Ascend 910B1 (NPU Arch 2201, CANN 8.5.0 contract)",
    },
}

# Map dtype names to compute-capability keys across NVIDIA and AMD.
# For dtypes with multiple metrics, such as fp64 tensor/vector/matrix,
# prefer the highest-throughput path by default.
# If a GPU has no mapped key, fall back to the dtype name itself.
DTYPE_TO_COMPUTE = {
    "fp64": ["fp64_tensor", "fp64_matrix", "fp64"],      # NVIDIA tensor > AMD matrix > generic
    "fp32": ["fp32_cuda", "fp32"],                        # NVIDIA CUDA cores > generic
    "tf32": ["tf32"],
    "fp16": ["fp16"],
    "bf16": ["bf16"],
    "fp8": ["fp8"],
    "fp6": ["fp6"],
    "fp4": ["fp4"],
    "int8": ["int8"],
    "int4": ["int4"],
}

# Non-compute fields used when listing supported compute types
_META_KEYS = (
    "memory_bandwidth_tb_s",
    "num_units",
    "unit_type",
    "description",
    "peak_provider",
)


def _normalize_gpu_key(gpu: str) -> str:
    """Normalize only aliases whose product identity is unambiguous."""
    raw = str(gpu).strip().lower()
    compact = "".join(character for character in raw if character.isalnum())
    if compact in {"910b1", "ascend910b1", "huaweiascend910b1"}:
        return "ascend910b1"
    return raw


_ASCEND_EXECUTION_ALIASES = {
    "aic": "aic",
    "aic_only": "aic",
    "aiv": "aiv",
    "aiv_only": "aiv",
    "mix": "mix",
}

_ASCEND_BANDWIDTH_KIND_ALIASES = {
    "hbm": "hbm_cycle_model",
    "ddr": "hbm_cycle_model",
    "hbm_cycle_model": "hbm_cycle_model",
    "cann_hbm_cycle_model": "hbm_cycle_model",
    "l2": "l2_cycle_model",
    "l2_cycle_model": "l2_cycle_model",
    "cann_l2_cycle_model": "l2_cycle_model",
    "measured_copy_256m": "measured_copy_256m",
    "copy_256m": "measured_copy_256m",
    "physical": "physical_hbm_interface_peak",
    "physical_hbm": "physical_hbm_interface_peak",
    "physical_hbm_interface_peak": "physical_hbm_interface_peak",
}


def _normalize_execution_kind(execution_kind: str | None) -> str:
    normalized = _ASCEND_EXECUTION_ALIASES.get(
        str(execution_kind or "").strip().lower().replace("-", "_")
    )
    if normalized is None:
        raise ValueError(
            "Ascend910B1 has split AIC/AIV execution; pass execution_kind as "
            "aic, aiv, or mix"
        )
    return normalized


def _normalize_bandwidth_kind(bandwidth_kind: str | None, default_kind: str) -> str:
    if bandwidth_kind is None:
        return default_kind
    normalized = _ASCEND_BANDWIDTH_KIND_ALIASES.get(
        str(bandwidth_kind).strip().lower().replace("-", "_")
    )
    if normalized is None:
        raise ValueError(
            "unsupported Ascend910B1 bandwidth_kind; use hbm_cycle_model, "
            "l2_cycle_model, measured_copy_256m, or physical_hbm_interface_peak"
        )
    return normalized


def get_peak_provider(gpu: str) -> dict | None:
    """Return a defensive copy of a GPU's versioned peak-evidence provider."""
    gpu_key = _normalize_gpu_key(gpu)
    provider_id = HARDWARE_SPECS.get(gpu_key, {}).get("peak_provider")
    if provider_id is None:
        return None
    provider = PEAK_PROVIDERS.get(provider_id)
    if not isinstance(provider, dict):
        raise ValueError(
            f"GPU {gpu_key} references missing peak provider {provider_id!r}"
        )
    if provider.get("schema_version") != PEAK_PROVIDER_SCHEMA_VERSION:
        raise ValueError(
            f"GPU {gpu_key} peak provider {provider_id!r} has unsupported schema "
            f"version {provider.get('schema_version')!r}"
        )
    return deepcopy(provider)


def get_peak_record(
    gpu: str,
    metric: str,
    dtype: str | None = None,
    execution_kind: str | None = None,
    operation_kind: str | None = None,
    bandwidth_kind: str | None = None,
) -> dict | None:
    """Return one provenance-bearing provider ceiling without inventing a value."""
    provider = get_peak_provider(gpu)
    if provider is None:
        return None
    peaks = provider.get("peaks")
    if not isinstance(peaks, dict):
        raise ValueError(f"peak provider {provider['provider_id']!r} has no peaks object")
    if metric == "compute_tflops":
        if not dtype:
            raise ValueError("dtype is required for a compute_tflops peak record")
        compute = peaks.get("compute_tflops")
        if not isinstance(compute, dict):
            raise ValueError(
                f"peak provider {provider['provider_id']!r} has no compute_tflops records"
            )
        execution = _normalize_execution_kind(execution_kind)
        execution_records = compute.get(execution)
        if not isinstance(execution_records, dict):
            raise ValueError(
                f"peak provider {provider['provider_id']!r} has no {execution} compute records"
            )
        if execution == "mix":
            record = execution_records
        elif execution == "aic":
            operation = str(operation_kind or "").strip().lower()
            if operation not in {"", "mac"}:
                raise ValueError(
                    "Ascend910B1 AIC uses the dense Cube MAC model; do not pass an "
                    "AIV operation_kind"
                )
            record = execution_records.get(dtype.lower(), execution_records.get("default"))
        else:
            operation = str(operation_kind or "").strip().lower()
            if operation not in {"add", "fma"}:
                raise ValueError(
                    "Ascend910B1 AIV requires operation_kind=add or operation_kind=fma"
                )
            dtype_records = execution_records.get(
                dtype.lower(), execution_records.get("default")
            )
            if not isinstance(dtype_records, dict):
                record = None
            elif "status" in dtype_records:
                record = dtype_records
            else:
                record = dtype_records.get(operation)
    elif metric == "memory_bandwidth_tb_s":
        bandwidth_records = peaks.get("memory_bandwidth_tb_s")
        if not isinstance(bandwidth_records, dict):
            raise ValueError(
                f"peak provider {provider['provider_id']!r} has no bandwidth records"
            )
        kind = _normalize_bandwidth_kind(
            bandwidth_kind, bandwidth_records.get("default_kind", "")
        )
        record = bandwidth_records.get(kind)
    else:
        raise ValueError(f"unsupported peak-provider metric: {metric}")
    if not isinstance(record, dict):
        raise ValueError(
            f"peak provider {provider['provider_id']!r} has no record for {metric}"
        )
    result = deepcopy(record)
    result["provider_id"] = provider["provider_id"]
    result["provider_version"] = provider["provider_version"]
    source_ids = result.get("source_ids", [])
    sources = provider.get("sources", {})
    result["source_records"] = {
        source_id: deepcopy(sources[source_id])
        for source_id in source_ids
        if isinstance(sources, dict) and source_id in sources
    }
    return result


def _available_provider_value(
    gpu: str,
    metric: str,
    record: dict,
    *,
    expected_unit: str,
    allow_modeled_peak: bool = False,
) -> float:
    """Resolve an available provider metric, failing closed on unknown/invalid data."""
    provider = get_peak_provider(gpu)
    assert provider is not None
    provider_name = f"{provider['provider_id']}@{provider['provider_version']}"
    status = record.get("status")
    if status not in {"available", "conditional"}:
        reason = record.get("reason") or "no reason recorded"
        raise ValueError(
            f"GPU {_normalize_gpu_key(gpu)} {metric} is unavailable in peak provider "
            f"{provider_name}: {reason}. Refusing to infer a numeric peak."
        )
    if (
        status == "conditional"
        and record.get("requires_model_opt_in") is True
        and not allow_modeled_peak
    ):
        conditions = "; ".join(record.get("conditions") or [])
        raise ValueError(
            f"GPU {_normalize_gpu_key(gpu)} {metric} is a conditional modeled ceiling "
            f"in peak provider {provider_name}: {conditions}. Pass "
            "allow_modeled_peak=True (CLI: --allow-modeled-peak) to opt in."
        )
    source_ids = record.get("source_ids")
    sources = provider.get("sources")
    confidence = record.get("confidence")
    derivation = record.get("derivation")
    if (
        not isinstance(source_ids, list)
        or not source_ids
        or not isinstance(sources, dict)
        or any(not isinstance(source_id, str) or source_id not in sources for source_id in source_ids)
        or not isinstance(confidence, dict)
        or confidence.get("level") not in {"high", "medium-high", "medium", "low"}
        or not confidence.get("basis")
        or not isinstance(derivation, dict)
        or derivation.get("kind") in {None, "not_derived"}
        or not derivation.get("expression")
        or (
            status == "conditional"
            and (
                not isinstance(record.get("conditions"), list)
                or not record["conditions"]
                or any(not str(condition).strip() for condition in record["conditions"])
            )
        )
    ):
        raise ValueError(
            f"GPU {_normalize_gpu_key(gpu)} {metric} in peak provider {provider_name} "
            "lacks complete source/confidence/derivation provenance"
        )
    value = record.get("value")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
        or record.get("unit") != expected_unit
    ):
        raise ValueError(
            f"GPU {_normalize_gpu_key(gpu)} {metric} is invalid in peak provider "
            f"{provider_name}; expected a positive finite {expected_unit} value"
        )
    return float(value)


def _resolve_compute_type(gpu_specs: dict, dtype: str) -> str:
    """Resolve a dtype to the compute key supported by this GPU."""
    candidates = DTYPE_TO_COMPUTE.get(dtype, [dtype])
    for candidate in candidates:
        if candidate in gpu_specs:
            return candidate
    return dtype  # fallback


def _flat_peak_record(value: float, unit: str, peak_type: str) -> dict:
    """Describe a legacy flat-table value without claiming provider provenance."""
    return {
        "status": "available",
        "value": value,
        "unit": unit,
        "peak_type": peak_type,
        "source_ids": ["HARDWARE_SPECS"],
        "source_records": {},
        "confidence": {
            "level": "legacy",
            "basis": "flat built-in or explicit caller override",
        },
        "derivation": {"kind": "flat_value", "expression": "HARDWARE_SPECS"},
    }


def _compute_unit(dtype: str) -> str:
    return "TOPS" if dtype.startswith("int") else "TFLOPS"


def _resolve_compute_peak(
    gpu: str,
    dtype: str,
    execution_kind: str | None = None,
    operation_kind: str | None = None,
    allow_modeled_peak: bool = False,
) -> tuple[float, dict]:
    gpu = _normalize_gpu_key(gpu)
    dtype = dtype.lower()
    if gpu not in HARDWARE_SPECS:
        raise ValueError(
            f"Unknown GPU: {gpu}. Supported GPUs: {list(HARDWARE_SPECS.keys())}"
        )

    specs = HARDWARE_SPECS[gpu]
    provider = get_peak_provider(gpu)
    provider_record = None
    normalized_execution = None
    if provider is not None:
        normalized_execution = _normalize_execution_kind(execution_kind)
        provider_record = get_peak_record(
            gpu,
            "compute_tflops",
            dtype,
            execution_kind=normalized_execution,
            operation_kind=operation_kind,
        )
        if normalized_execution == "mix":
            return (
                _available_provider_value(
                    gpu,
                    "MIX compute peak",
                    provider_record,
                    expected_unit=provider_record.get("unit", "TFLOPS"),
                    allow_modeled_peak=allow_modeled_peak,
                ),
                provider_record,
            )

    compute_type = _resolve_compute_type(specs, dtype)
    if compute_type in specs:
        value = specs[compute_type]
        return value, _flat_peak_record(value, _compute_unit(dtype), "flat_hardware_peak")

    if provider_record is not None:
        unit = provider_record.get("unit")
        if unit not in {"TFLOPS", "TOPS"}:
            raise ValueError(
                f"GPU {gpu} {dtype} compute record has unsupported unit {unit!r}"
            )
        value = _available_provider_value(
            gpu,
            f"{normalized_execution} {dtype} compute peak",
            provider_record,
            expected_unit=unit,
            allow_modeled_peak=allow_modeled_peak,
        )
        return value, provider_record

    raise ValueError(
        f"GPU {gpu} does not support {dtype} ({compute_type}). "
        f"Supported compute types: {[key for key in specs if key not in _META_KEYS]}"
    )


def get_peak_tflops(
    gpu: str,
    dtype: str,
    execution_kind: str | None = None,
    operation_kind: str | None = None,
    allow_modeled_peak: bool = False,
) -> float:
    """Return tera-operations/s; Ascend requires an explicit execution path."""
    value, _ = _resolve_compute_peak(
        gpu, dtype, execution_kind, operation_kind, allow_modeled_peak
    )
    return value


def _resolve_bandwidth_peak(
    gpu: str,
    bandwidth_kind: str | None = None,
) -> tuple[float, dict]:
    gpu = _normalize_gpu_key(gpu)
    if gpu not in HARDWARE_SPECS:
        raise ValueError(
            f"Unknown GPU: {gpu}. Supported GPUs: {list(HARDWARE_SPECS.keys())}"
        )

    specs = HARDWARE_SPECS[gpu]
    if "memory_bandwidth_tb_s" in specs and bandwidth_kind is None:
        value = specs["memory_bandwidth_tb_s"]
        record = _flat_peak_record(value, "TB/s", "physical_hardware_peak")
        record.update(memory_level="hbm", physical_interface_peak=True)
        return value, record

    provider_record = get_peak_record(
        gpu, "memory_bandwidth_tb_s", bandwidth_kind=bandwidth_kind
    )
    if provider_record is not None:
        value = _available_provider_value(
            gpu,
            f"{provider_record.get('peak_type', 'memory')} bandwidth ceiling",
            provider_record,
            expected_unit="TB/s",
        )
        return value, provider_record

    if "memory_bandwidth_tb_s" not in specs:
        raise ValueError(
            f"GPU {gpu} has no configured peak memory bandwidth (memory_bandwidth_tb_s). "
            f"Please add it to HARDWARE_SPECS."
        )
    value = specs["memory_bandwidth_tb_s"]
    record = _flat_peak_record(value, "TB/s", "physical_hardware_peak")
    record.update(memory_level="hbm", physical_interface_peak=True)
    return value, record


def get_peak_bandwidth(gpu: str, bandwidth_kind: str | None = None) -> float:
    """Return the selected bandwidth ceiling in TB/s.

    Ascend defaults to the CANN HBM cycle model. Its physical-interface HBM
    peak remains unavailable and fails closed when requested explicitly.
    """
    value, _ = _resolve_bandwidth_peak(gpu, bandwidth_kind)
    return value


def get_num_units(gpu: str, execution_kind: str | None = None) -> int:
    """Return schedulable units, requiring an execution kind for split Ascend cores."""
    gpu = _normalize_gpu_key(gpu)

    if gpu not in HARDWARE_SPECS:
        raise ValueError(
            f"Unknown GPU: {gpu}. Supported GPUs: {list(HARDWARE_SPECS.keys())}"
        )

    specs = HARDWARE_SPECS[gpu]
    if "num_units" not in specs:
        provider = get_peak_provider(gpu)
        if provider is not None:
            execution = _normalize_execution_kind(execution_kind)
            topology_key = {
                "aic": "aic_units",
                "aiv": "aiv_units",
                "mix": "mix_groups",
            }[execution]
            topology = provider.get("topology", {})
            record = topology.get(topology_key) if isinstance(topology, dict) else None
            value = record.get("value") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or record.get("status") != "available"
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(
                    f"GPU {gpu} provider has no valid {topology_key} topology value"
                )
            return value
        raise ValueError(
            f"GPU {gpu} has no configured compute unit count (num_units). "
            f"Please add it to HARDWARE_SPECS."
        )

    return specs["num_units"]


def get_unit_type(gpu: str) -> str:
    """Return the compute-unit type name."""
    gpu = _normalize_gpu_key(gpu)
    return HARDWARE_SPECS.get(gpu, {}).get("unit_type", "Unit")


def _record_source_label(record: dict) -> str:
    source_ids = record.get("source_ids") or []
    provider_id = record.get("provider_id")
    provider_version = record.get("provider_version")
    if provider_id:
        provider = f"{provider_id}@{provider_version}"
        return f"{provider} ({', '.join(source_ids)})" if source_ids else provider
    return ", ".join(source_ids) if source_ids else "unspecified"


def compute_ridge_point(
    gpu: str,
    dtype: str,
    execution_kind: str | None = None,
    operation_kind: str | None = None,
    bandwidth_kind: str | None = None,
    allow_modeled_peak: bool = False,
) -> float:
    """
    Compute the Roofline ridge point.

    Ridge Point = peak compute (FLOPS) / peak bandwidth (Bytes/s)
    Unit: FLOPs/Byte
    """
    peak_tflops = get_peak_tflops(
        gpu, dtype, execution_kind, operation_kind, allow_modeled_peak
    )
    peak_bandwidth_tb_s = get_peak_bandwidth(gpu, bandwidth_kind)
    # TFLOPS / (TB/s) = (1e12 FLOPS) / (1e12 Bytes/s) = FLOPs/Byte
    return peak_tflops / peak_bandwidth_tb_s


def roofline_analysis(
    flops: float,
    bytes_transferred: float,
    gpu: str,
    dtype: str,
    execution_kind: str | None = None,
    operation_kind: str | None = None,
    bandwidth_kind: str | None = None,
    allow_modeled_peak: bool = False,
) -> dict:
    """
    Run Roofline bottleneck analysis.

    Returns:
        dict containing arithmetic_intensity, ridge_point, and bottleneck ("compute" | "memory")
    """
    peak_compute, compute_record = _resolve_compute_peak(
        gpu, dtype, execution_kind, operation_kind, allow_modeled_peak
    )
    peak_bandwidth, bandwidth_record = _resolve_bandwidth_peak(gpu, bandwidth_kind)
    arithmetic_intensity = flops / bytes_transferred
    ridge_point = peak_compute / peak_bandwidth

    if arithmetic_intensity >= ridge_point:
        bottleneck = "compute"
    else:
        bottleneck = "memory"

    return {
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "bottleneck": bottleneck,
        "flops": flops,
        "bytes_transferred": bytes_transferred,
        "peak_compute": peak_compute,
        "peak_compute_unit": compute_record["unit"],
        "compute_ceiling_status": compute_record["status"],
        "compute_ceiling_type": compute_record["peak_type"],
        "compute_ceiling_source": _record_source_label(compute_record),
        "compute_peak_record": compute_record,
        "peak_bandwidth_tb_s": peak_bandwidth,
        "bandwidth_ceiling_type": bandwidth_record["peak_type"],
        "bandwidth_ceiling_source": _record_source_label(bandwidth_record),
        "bandwidth_physical_interface_peak": bandwidth_record.get(
            "physical_interface_peak"
        ),
        "bandwidth_peak_record": bandwidth_record,
    }


def compute_utilization(
    flops: float,
    time_ms: float,
    gpu: str,
    dtype: str,
    execution_kind: str | None = None,
    operation_kind: str | None = None,
    allow_modeled_peak: bool = False,
) -> dict:
    """Compute compute-throughput utilization for compute-bound cases."""
    peak_tflops, peak_record = _resolve_compute_peak(
        gpu, dtype, execution_kind, operation_kind, allow_modeled_peak
    )
    time_s = time_ms / 1000.0
    actual_tflops = flops / time_s / 1e12
    utilization = actual_tflops / peak_tflops * 100.0

    return {
        "flops": flops,
        "time_ms": time_ms,
        "actual_tflops": actual_tflops,
        "peak_tflops": peak_tflops,
        "peak_unit": peak_record["unit"],
        "peak_status": peak_record["status"],
        "peak_type": peak_record["peak_type"],
        "peak_source": _record_source_label(peak_record),
        "peak_record": peak_record,
        "utilization_pct": utilization,
        "gpu": gpu,
        "dtype": dtype,
    }


def compute_bandwidth_utilization(
    bytes_transferred: float,
    time_ms: float,
    gpu: str,
    measured_bandwidth_tb_s: float = None,
    bandwidth_kind: str | None = None,
) -> dict:
    """
    Compute bandwidth utilization for memory-bound cases.

    GPUs are high-latency, high-bandwidth devices. Small kernels may not
    have enough data movement to fill the memory pipeline, so they may never
    reach a device-wide ceiling. If measured_bandwidth_tb_s is provided, use it
    as the denominator. Ascend otherwise uses a typed CANN cycle-model ceiling,
    not a physical HBM-interface peak.
    """
    reference_bandwidth_tb_s, peak_record = _resolve_bandwidth_peak(
        gpu, bandwidth_kind
    )
    time_s = time_ms / 1000.0
    actual_bandwidth_tb_s = bytes_transferred / time_s / 1e12

    if measured_bandwidth_tb_s is not None:
        bandwidth_ceiling_tb_s = measured_bandwidth_tb_s
        ceiling_source = "measured bandwidth ceiling"
        ceiling_type = "measured_bandwidth_ceiling"
    else:
        bandwidth_ceiling_tb_s = reference_bandwidth_tb_s
        ceiling_type = peak_record["peak_type"]
        if peak_record.get("physical_interface_peak"):
            ceiling_source = "hardware theoretical peak"
        else:
            ceiling_source = f"CANN cycle-model ceiling: {_record_source_label(peak_record)}"

    utilization = actual_bandwidth_tb_s / bandwidth_ceiling_tb_s * 100.0

    return {
        "bytes_transferred": bytes_transferred,
        "time_ms": time_ms,
        "actual_bandwidth_tb_s": actual_bandwidth_tb_s,
        "bandwidth_ceiling_tb_s": bandwidth_ceiling_tb_s,
        "reference_bandwidth_tb_s": reference_bandwidth_tb_s,
        "hardware_peak_bandwidth_tb_s": (
            reference_bandwidth_tb_s
            if peak_record.get("physical_interface_peak")
            else None
        ),
        "ceiling_source": ceiling_source,
        "bandwidth_ceiling_type": ceiling_type,
        "bandwidth_ceiling_source": (
            "caller-provided measurement"
            if measured_bandwidth_tb_s is not None
            else _record_source_label(peak_record)
        ),
        "bandwidth_physical_interface_peak": (
            False
            if measured_bandwidth_tb_s is not None
            else peak_record.get("physical_interface_peak")
        ),
        "bandwidth_peak_record": peak_record,
        "utilization_pct": utilization,
        "gpu": gpu,
    }


def compute_theoretical_ceiling(
    tile_flops: float,
    tile_bytes: float,
    grid_blocks: int,
    num_units: int,
    gpu: str,
    dtype: str,
    measured_bandwidth_tb_s: float = None,
    execution_kind: str | None = None,
    operation_kind: str | None = None,
    bandwidth_kind: str | None = None,
    allow_modeled_peak: bool = False,
) -> dict:
    """
    Estimate the theoretical performance ceiling for the current configuration.

    Considers:
      - tile-level Roofline bound type
      - SM/CU utilization, based on grid_blocks vs num_units
      - bandwidth ceiling, measured or theoretical

    Principle:
      The GPU schedules blocks to SMs/CUs in waves. Each wave can run at most
      num_units blocks in parallel. The number of waves is
      ceil(grid_blocks / num_units).

      The minimum per-block time depends on the bottleneck:
        - Compute-bound: tile_time_min = tile_flops / peak_compute
        - Memory-bound:  tile_time_min = tile_bytes / bandwidth_ceiling

      The minimum kernel latency is num_waves * tile_time_min.
      The theoretical ceiling is total FLOPs / minimum kernel latency.
    """
    import math

    peak_tflops, compute_record = _resolve_compute_peak(
        gpu, dtype, execution_kind, operation_kind, allow_modeled_peak
    )
    peak_bandwidth_tb_s, bandwidth_record = _resolve_bandwidth_peak(
        gpu, bandwidth_kind
    )
    unit_type = get_unit_type(gpu)

    # Theoretical performance ceiling
    if measured_bandwidth_tb_s is not None:
        bandwidth_ceiling_tb_s = measured_bandwidth_tb_s
        bandwidth_source = "measured bandwidth ceiling"
        bandwidth_ceiling_type = "measured_bandwidth_ceiling"
    else:
        bandwidth_ceiling_tb_s = peak_bandwidth_tb_s
        bandwidth_ceiling_type = bandwidth_record["peak_type"]
        bandwidth_source = (
            "hardware theoretical peak"
            if bandwidth_record.get("physical_interface_peak")
            else f"CANN cycle-model ceiling: {_record_source_label(bandwidth_record)}"
        )

    # Roofline bound classification
    arithmetic_intensity = tile_flops / tile_bytes
    ridge_point = peak_tflops / peak_bandwidth_tb_s

    if arithmetic_intensity >= ridge_point:
        bottleneck = "compute"
        tile_time_min_s = tile_flops / (peak_tflops * 1e12)
    else:
        bottleneck = "memory"
        tile_time_min_s = tile_bytes / (bandwidth_ceiling_tb_s * 1e12)

    # SM/CU scheduling: number of waves
    num_waves = math.ceil(grid_blocks / num_units)
    unit_utilization_pct = min(grid_blocks / num_units, 1.0) * 100.0

    # Minimum theoretical kernel latency
    theoretical_kernel_time_s = num_waves * tile_time_min_s
    theoretical_kernel_time_ms = theoretical_kernel_time_s * 1000.0

    # Theoretical compute ceiling
    total_flops = tile_flops * grid_blocks
    theoretical_tflops = total_flops / theoretical_kernel_time_s / 1e12

    # Theoretical compute ceiling ()
    total_bytes = tile_bytes * grid_blocks
    theoretical_bandwidth_tb_s = total_bytes / theoretical_kernel_time_s / 1e12

    return {
        "bottleneck": bottleneck,
        "tile_flops": tile_flops,
        "tile_bytes": tile_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "grid_blocks": grid_blocks,
        "num_units": num_units,
        "unit_type": unit_type,
        "num_waves": num_waves,
        "unit_utilization_pct": unit_utilization_pct,
        "tile_time_min_ms": tile_time_min_s * 1000.0,
        "theoretical_kernel_time_ms": theoretical_kernel_time_ms,
        "theoretical_tflops": theoretical_tflops,
        "theoretical_bandwidth_tb_s": theoretical_bandwidth_tb_s,
        "peak_tflops": peak_tflops,
        "peak_unit": compute_record["unit"],
        "compute_ceiling_status": compute_record["status"],
        "compute_ceiling_type": compute_record["peak_type"],
        "compute_ceiling_source": _record_source_label(compute_record),
        "compute_peak_record": compute_record,
        "bandwidth_ceiling_tb_s": bandwidth_ceiling_tb_s,
        "bandwidth_source": bandwidth_source,
        "bandwidth_ceiling_type": bandwidth_ceiling_type,
        "bandwidth_ceiling_source": (
            "caller-provided measurement"
            if measured_bandwidth_tb_s is not None
            else _record_source_label(bandwidth_record)
        ),
        "bandwidth_physical_interface_peak": (
            False
            if measured_bandwidth_tb_s is not None
            else bandwidth_record.get("physical_interface_peak")
        ),
        "bandwidth_peak_record": bandwidth_record,
        "model_warning": (
            "Legacy generic tile-wave model applies a device-wide peak independently "
            "to concurrent tiles, so theoretical_tflops may exceed the device peak; "
            "treat it as an uncapped scheduling heuristic."
        ),
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "gpu": gpu,
        "dtype": dtype,
    }


def _print_theoretical_ceiling(ceiling: dict):
    """Print the theoretical performance ceiling."""
    desc = HARDWARE_SPECS[_normalize_gpu_key(ceiling["gpu"])].get("description", "")
    unit_type = ceiling["unit_type"]
    bottleneck_label = (
        "Compute Bound" if ceiling["bottleneck"] == "compute"
        else "Memory Bound"
    )

    print(f"\n{'='*64}")
    print(f"  Theoretical Performance Ceiling")
    print(f"{'='*64}")
    print(f"  GPU              : {ceiling['gpu'].upper()} ({desc})")
    print(f"  dtype          : {ceiling['dtype']}")
    print(f"  Bottleneck       : {bottleneck_label}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Tile AI          : {ceiling['arithmetic_intensity']:.2f} FLOPs/Byte")
    print(f"  Ridge Point      : {ceiling['ridge_point']:.2f} FLOPs/Byte")
    print(f"  ─────────────────────────────────────────")
    print(f"  Grid blocks      : {ceiling['grid_blocks']}")
    print(f"  {unit_type} count       : {ceiling['num_units']}")
    print(f"  Scheduling waves : {ceiling['num_waves']}")
    print(f"  {unit_type} utilization : {ceiling['unit_utilization_pct']:.1f}%")
    print(f"  ─────────────────────────────────────────")
    print(f"  Per-tile min time : {ceiling['tile_time_min_ms']:.6f} ms")
    print(f"  Kernel min time   : {ceiling['theoretical_kernel_time_ms']:.4f} ms")
    print(f"  ─────────────────────────────────────────")
    print(
        f"  Compute ceiling  : {ceiling['theoretical_tflops']:.2f} "
        f"{ceiling['peak_unit']} (reference {ceiling['peak_tflops']:.1f} "
        f"{ceiling['peak_unit']}; {ceiling['compute_ceiling_status']}; "
        f"{ceiling['compute_ceiling_type']})"
    )
    print(f"  Compute source   : {ceiling['compute_ceiling_source']}")
    print(f"  Bandwidth ceiling: {ceiling['theoretical_bandwidth_tb_s']:.2f} TB/s"
          f" ({ceiling['bandwidth_source']}: {ceiling['bandwidth_ceiling_tb_s']:.1f} TB/s; "
          f"{ceiling['bandwidth_ceiling_type']})")
    print(f"  Model warning    : {ceiling['model_warning']}")
    print(f"{'='*64}")


def _clear_triton_cache():
    cache_dir = os.path.expanduser("~/.triton")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)


def measure_kernel_time(kernel_file, wrapper_name, setup_name, warmup=25, rep=100):
    """Measure kernel latency in ms. Requires torch and triton."""
    import torch
    import triton

    spec = importlib.util.spec_from_file_location("kernel_module", kernel_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    wrapper_fn = getattr(module, wrapper_name)
    setup_fn = getattr(module, setup_name)

    captured = {}
    original_wrapper = wrapper_fn

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original_wrapper(*args, **kwargs)

    setattr(module, wrapper_name, spy)
    try:
        setup_fn()
    except Exception:
        pass

    if not captured:
        raise RuntimeError(f"Unable to capture call arguments for {wrapper_name} from {setup_name}")

    wrapper_fn = getattr(module, wrapper_name)
    torch.cuda.synchronize()
    _clear_triton_cache()
    ms, _, _ = triton.testing.do_bench(
        lambda: wrapper_fn(**captured),
        quantiles=[0.5, 0.2, 0.8],
        warmup=warmup,
        rep=rep,
    )
    return ms


def _eval_expr(expr: str, label: str) -> float:
    """Evaluate a numeric Python expression."""
    try:
        return float(eval(expr))
    except Exception as exc:
        print(f"Error: failed to evaluate {label} expression: {exc}")
        sys.exit(1)


def _print_roofline_result(roofline: dict, gpu: str, dtype: str):
    """Print the tile-level Roofline result."""
    desc = HARDWARE_SPECS[_normalize_gpu_key(gpu)].get("description", "")
    bottleneck_label = (
        "Compute Bound" if roofline["bottleneck"] == "compute"
        else "Memory Bound"
    )

    print(f"\n{'='*64}")
    print(f"  Tile-level Roofline Analysis")
    print(f"{'='*64}")
    print(f"  GPU              : {gpu.upper()} ({desc})")
    print(f"  dtype          : {dtype}")
    print(f"  Tile FLOPs       : {roofline['flops']:.2e}")
    print(f"  Tile Bytes       : {roofline['bytes_transferred']:.2e}")
    print(f"  Tile AI          : {roofline['arithmetic_intensity']:.2f} FLOPs/Byte")
    print(f"  Ridge Point      : {roofline['ridge_point']:.2f} FLOPs/Byte")
    print(
        f"  Compute ceiling  : {roofline['peak_compute']:.4f} "
        f"{roofline['peak_compute_unit']} [{roofline['compute_ceiling_status']}; "
        f"{roofline['compute_ceiling_type']}]"
    )
    print(f"  Compute source   : {roofline['compute_ceiling_source']}")
    print(
        f"  Bandwidth ceiling: {roofline['peak_bandwidth_tb_s']:.4f} TB/s "
        f"[{roofline['bandwidth_ceiling_type']}]"
    )
    print(f"  Bandwidth source : {roofline['bandwidth_ceiling_source']}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Bottleneck       : {bottleneck_label}")
    print(f"{'='*64}")


def _print_compute_utilization(result: dict):
    """Print the compute-utilization result (compute-bound case)."""
    desc = HARDWARE_SPECS[_normalize_gpu_key(result["gpu"])].get("description", "")

    print(f"\n{'='*64}")
    print(f"  Compute Utilization Analysis (Compute Bound)")
    print(f"{'='*64}")
    print(f"  GPU              : {result['gpu'].upper()} ({desc})")
    print(f"  dtype          : {result['dtype']}")
    print(f"  FLOPs            : {result['flops']:.2e}")
    print(f"  latency              : {result['time_ms']:.4f} ms")
    print(f"  Actual compute   : {result['actual_tflops']:.2f} {result['peak_unit']}")
    print(
        f"  Compute ceiling  : {result['peak_tflops']:.2f} {result['peak_unit']} "
        f"[{result['peak_status']}; {result['peak_type']}]"
    )
    print(f"  Ceiling source   : {result['peak_source']}")
    print(f"  Utilization      : {result['utilization_pct']:.1f}%")
    print(f"{'='*64}")


def _print_bandwidth_utilization(result: dict):
    """Print the bandwidth-utilization result (memory-bound case)."""
    desc = HARDWARE_SPECS[_normalize_gpu_key(result["gpu"])].get("description", "")

    print(f"\n{'='*64}")
    print(f"  Bandwidth Utilization Analysis (Memory Bound)")
    print(f"{'='*64}")
    print(f"  GPU              : {result['gpu'].upper()} ({desc})")
    print(f"  Bytes transferred: {result['bytes_transferred']:.2e}")
    print(f"  latency              : {result['time_ms']:.4f} ms")
    print(f"  Actual bandwidth : {result['actual_bandwidth_tb_s']:.2f} TB/s"
          f"  ({result['actual_bandwidth_tb_s'] * 1000:.1f} GB/s  <- record as bandwidth_gbps)")
    print(
        f"  Bandwidth ceiling: {result['bandwidth_ceiling_tb_s']:.2f} TB/s "
        f"[{result['bandwidth_ceiling_type']}]"
    )
    print(f"  Ceiling source   : {result['ceiling_source']}")
    if result["ceiling_source"] == "measured bandwidth ceiling":
        if result["hardware_peak_bandwidth_tb_s"] is not None:
            print(f"  Hardware peak    : {result['hardware_peak_bandwidth_tb_s']:.2f} TB/s")
        else:
            print(
                f"  CANN model ref   : {result['reference_bandwidth_tb_s']:.2f} TB/s "
                f"[{result['bandwidth_peak_record']['peak_type']}]"
            )
    print(f"  Utilization      : {result['utilization_pct']:.1f}%")
    print(f"{'='*64}")


def _print_exit_status(utilization_pct: float, bottleneck: str):
    """Print final status and return an exit code."""
    metric_name = "compute utilization" if bottleneck == "compute" else "bandwidth utilization"

    if utilization_pct >= 90:
        print(f"✅ {metric_name} reached {utilization_pct:.1f}% (>=90%); no further optimization is required")
        return 0
    else:
        print(f"⚠️  {metric_name} is {utilization_pct:.1f}% (<90%); instruction-level profiling is recommended")
        return 2  # exit code 2 means more optimization is recommended


def main():
    gpu_list = ", ".join(HARDWARE_SPECS.keys())
    parser = argparse.ArgumentParser(
        description=(
            "Roofline bottleneck analysis plus compute/bandwidth utilization calculation "
            "(NVIDIA + AMD + provenance-gated Ascend)"
        )
    )
    parser.add_argument("kernel", nargs="?", help="Kernel source file")
    parser.add_argument("--gpu", required=True, help=f"GPU model ({gpu_list})")
    parser.add_argument("--dtype", required=True, help="dtype (bf16, fp16, fp8, fp32, ...)")
    parser.add_argument(
        "--execution-kind",
        choices=("aic", "aiv", "mix"),
        help="Ascend execution path; required for Ascend910B1 compute analysis",
    )
    parser.add_argument(
        "--operation-kind",
        choices=("add", "fma"),
        help=(
            "AIV operation model; required for AIV (the FMA peak is currently unknown)"
        ),
    )
    parser.add_argument(
        "--bandwidth-kind",
        choices=(
            "hbm_cycle_model",
            "l2_cycle_model",
            "measured_copy_256m",
            "physical_hbm_interface_peak",
        ),
        help=(
            "Ascend bandwidth ceiling (default: CANN msprof HBM/GM Roofline model; "
            "the physical HBM interface peak is currently unavailable)"
        ),
    )
    parser.add_argument(
        "--allow-modeled-peak",
        action="store_true",
        help="Opt in to conditional modeled compute ceilings and their stated assumptions",
    )
    parser.add_argument("--wrapper-name", help="wrapper function name")
    parser.add_argument("--setup-name", help="setup function name")

    parser.add_argument("--flops-expr", help="FLOPs expression (Python expression)")
    parser.add_argument("--flops", type=float, help="FLOPs value")

    parser.add_argument("--bytes-expr", help="Bytes transferred expression (Python expression)")
    parser.add_argument("--bytes", type=float, help="Bytes transferred value")

    parser.add_argument("--measured-bandwidth-tb-s", type=float,
                        help="Measured same-size bandwidth ceiling in TB/s. Use a memcpy kernel with the same data volume as a practical memory-bandwidth baseline.")

    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="Peak compute throughput in TFLOPS for --dtype. REQUIRED for a GPU not in the built-in table (e.g. Blackwell); source it from gpu-wiki. Overrides the built-in value when the GPU is known.")
    parser.add_argument("--peak-bandwidth-tb-s", type=float, default=None,
                        help="Peak HBM bandwidth in TB/s. REQUIRED for a GPU not in the built-in table; source it from gpu-wiki. Overrides the built-in value when the GPU is known.")

    parser.add_argument("--grid-blocks", type=int,
                        help="Number of blocks in the grid. If provided, --time-ms is treated as whole-kernel latency and divided by grid-blocks to derive per-tile latency. If omitted, --flops, --bytes, and --time-ms are assumed to be tile-level values.")

    parser.add_argument("--num-units", type=int, default=None,
                        help="Number of GPU compute units, SM for NVIDIA or CU for AMD. If omitted, infer it from --gpu.")

    parser.add_argument("--time-ms", type=float, help="Latency in ms")
    parser.add_argument("--warmup", type=int, default=25, help="number of warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="number of repetitions")
    parser.add_argument("--list-gpus", action="store_true", help="List supported GPUs and their specs")
    args = parser.parse_args()

    # ──  GPU  ──
    if args.list_gpus:
        print("Supported GPU models:")
        for gpu_name, specs in HARDWARE_SPECS.items():
            print(f"\n  {gpu_name}: {specs.get('description', '')}")
            for key, val in specs.items():
                if key == "memory_bandwidth_tb_s":
                    print(f"    peak memory bandwidth: {val} TB/s")
                elif key == "num_units":
                    unit_type = specs.get("unit_type", "Unit")
                    print(f"    {unit_type} count: {val}")
                elif key not in ("description", "unit_type"):
                    if key != "peak_provider":
                        print(f"    {key}: {val} TFLOPS")
            provider = get_peak_provider(gpu_name)
            if provider is not None:
                print(
                    f"    peak provider: {provider['provider_id']} "
                    f"(provider version {provider['provider_version']})"
                )
                for topology_name, record in provider["topology"].items():
                    print(
                        f"    {topology_name}: {record['value']} {record['unit']} "
                        f"(confidence {record['confidence']['level']})"
                    )
                compute_records = provider["peaks"]["compute_tflops"]
                for execution_name in ("aic", "aiv", "mix"):
                    records = compute_records[execution_name]
                    if records.get("status") == "unknown":
                        print(f"    {execution_name} compute: unknown ({records['reason']})")
                    else:
                        supported = sorted(key for key in records if key != "default")
                        print(f"    {execution_name} compute dtypes: {', '.join(supported)}")
                bandwidth_records = provider["peaks"]["memory_bandwidth_tb_s"]
                for kind in (
                    "hbm_cycle_model",
                    "l2_cycle_model",
                    "measured_copy_256m",
                ):
                    record = bandwidth_records[kind]
                    print(
                        f"    {kind}: {record['value']} {record['unit']} "
                        f"[{record['peak_type']}; not physical rated]"
                    )
                physical = bandwidth_records["physical_hbm_interface_peak"]
                print(f"    physical_hbm_interface_peak: {physical['status']} ({physical['reason']})")
        sys.exit(0)

    # ── latency ──
    if args.time_ms is not None:
        time_ms = args.time_ms
    elif args.kernel and args.wrapper_name and args.setup_name:
        print("Measuring kernel latency...")
        time_ms = measure_kernel_time(
            args.kernel, args.wrapper_name, args.setup_name,
            args.warmup, args.rep
        )
        print(f"  latency: {time_ms:.4f} ms")
    else:
        print("Error: provide --time-ms or provide kernel with --wrapper-name and --setup-name")
        sys.exit(1)

    # ──  FLOPs ──
    if args.flops is not None:
        flops = args.flops
    elif args.flops_expr:
        flops = _eval_expr(args.flops_expr, "FLOPs")
    else:
        print("Error: provide --flops or --flops-expr")
        sys.exit(1)

    # ──  Bytes transferred（） ──
    bytes_transferred = None
    if args.bytes is not None:
        bytes_transferred = args.bytes
    elif args.bytes_expr:
        bytes_transferred = _eval_expr(args.bytes_expr, "Bytes")

    # ──  ──
    gpu = _normalize_gpu_key(args.gpu)
    dtype = args.dtype.lower()
    grid_blocks = args.grid_blocks

    # Architecture-agnostic peaks: for a GPU not in the built-in table (e.g. Blackwell
    # sm_100/sm_103) the caller supplies gpu-wiki-sourced peaks via --peak-tflops /
    # --peak-bandwidth-tb-s. The tool never invents specs it does not have (no fabrication);
    # when the GPU IS known these flags override the built-in value.
    if args.peak_tflops is not None or args.peak_bandwidth_tb_s is not None:
        spec = dict(HARDWARE_SPECS.get(gpu, {}))
        if args.peak_tflops is not None:
            spec[dtype] = args.peak_tflops
        if args.peak_bandwidth_tb_s is not None:
            spec["memory_bandwidth_tb_s"] = args.peak_bandwidth_tb_s
        if args.num_units is not None:
            spec["num_units"] = args.num_units
        spec.setdefault("unit_type", "SM")
        spec.setdefault("description", f"{args.gpu} (peaks provided via CLI; source: gpu-wiki)")
        HARDWARE_SPECS[gpu] = spec
    elif gpu not in HARDWARE_SPECS:
        print(
            f"Error: unknown GPU '{args.gpu}' and no peaks provided. Either pick one of "
            f"{list(HARDWARE_SPECS.keys())}, or pass gpu-wiki-sourced peaks via "
            f"--peak-tflops and --peak-bandwidth-tb-s (recommended for new architectures)."
        )
        sys.exit(1)

    # ──  ──
    num_units = args.num_units
    if num_units is None:
        try:
            num_units = get_num_units(gpu, args.execution_kind)
        except ValueError:
            num_units = None

    unit_type = get_unit_type(gpu)

    # ──  per-tile latency ──
    if grid_blocks is not None and grid_blocks > 0:
        per_tile_time_ms = time_ms / grid_blocks
        print(f"\n  Tile : kernel latency {time_ms:.4f} ms / {grid_blocks} blocks "
              f"= {per_tile_time_ms:.6f} ms per tile")
    else:
        per_tile_time_ms = time_ms

    measured_bw = args.measured_bandwidth_tb_s

    if bytes_transferred is not None:
        # Analysis inputs Tile  Roofline 
        roofline = roofline_analysis(
            flops,
            bytes_transferred,
            gpu,
            dtype,
            execution_kind=args.execution_kind,
            operation_kind=args.operation_kind,
            bandwidth_kind=args.bandwidth_kind,
            allow_modeled_peak=args.allow_modeled_peak,
        )
        _print_roofline_result(roofline, gpu, dtype)

        # ── ceiling ──
        if grid_blocks is not None and num_units is not None:
            ceiling = compute_theoretical_ceiling(
                tile_flops=flops,
                tile_bytes=bytes_transferred,
                grid_blocks=grid_blocks,
                num_units=num_units,
                gpu=gpu,
                dtype=dtype,
                measured_bandwidth_tb_s=measured_bw,
                execution_kind=args.execution_kind,
                operation_kind=args.operation_kind,
                bandwidth_kind=args.bandwidth_kind,
                allow_modeled_peak=args.allow_modeled_peak,
            )
            _print_theoretical_ceiling(ceiling)

        if roofline["bottleneck"] == "compute":
            result = compute_utilization(
                flops,
                per_tile_time_ms,
                gpu,
                dtype,
                execution_kind=args.execution_kind,
                operation_kind=args.operation_kind,
                allow_modeled_peak=args.allow_modeled_peak,
            )
            _print_compute_utilization(result)
        else:
            result = compute_bandwidth_utilization(
                bytes_transferred,
                per_tile_time_ms,
                gpu,
                measured_bw,
                bandwidth_kind=args.bandwidth_kind,
            )
            _print_bandwidth_utilization(result)

        exit_code = _print_exit_status(result["utilization_pct"], roofline["bottleneck"])

        # Analysis inputs: , Output
        if roofline["bottleneck"] == "memory":
            compute_result = compute_utilization(
                flops,
                per_tile_time_ms,
                gpu,
                dtype,
                execution_kind=args.execution_kind,
                operation_kind=args.operation_kind,
                allow_modeled_peak=args.allow_modeled_peak,
            )
            print(f"\n  Compute (reference) : {compute_result['actual_tflops']:.2f} / "
                  f"{compute_result['peak_tflops']:.2f} {compute_result['peak_unit']} = "
                  f"{compute_result['utilization_pct']:.1f}%")

        # Analysis inputs: , Output
        if roofline["bottleneck"] == "compute":
            bw_result = compute_bandwidth_utilization(
                bytes_transferred,
                per_tile_time_ms,
                gpu,
                measured_bw,
                bandwidth_kind=args.bandwidth_kind,
            )
            print(f"\n  Bandwidth (reference) : {bw_result['actual_bandwidth_tb_s']:.2f} / "
                  f"{bw_result['bandwidth_ceiling_tb_s']:.2f} TB/s = "
                  f"{bw_result['utilization_pct']:.1f}%")

        # Theoretical compute ceiling, Output vs 
        if grid_blocks is not None and num_units is not None:
            actual_kernel_tflops = flops * grid_blocks / (time_ms / 1000.0) / 1e12
            efficiency = actual_kernel_tflops / ceiling["theoretical_tflops"] * 100.0
            print(f"\n  Actual vs ceiling : {actual_kernel_tflops:.2f} / "
                  f"{ceiling['theoretical_tflops']:.2f} TFLOPS = {efficiency:.1f}%")

        sys.exit(exit_code)

    else:
        # No bytes provided: skip Roofline classification; run compute-utilization only.
        print("\n⚠️  No --bytes/--bytes-expr provided; skipping Roofline classification, "
              "running compute-utilization only.")
        result = compute_utilization(
            flops,
            per_tile_time_ms,
            gpu,
            dtype,
            execution_kind=args.execution_kind,
            operation_kind=args.operation_kind,
            allow_modeled_peak=args.allow_modeled_peak,
        )
        _print_compute_utilization(result)
        exit_code = _print_exit_status(result["utilization_pct"], "compute")
        sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
