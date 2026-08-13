#!/usr/bin/env python3
"""Discover an Ascend target and emit generation-relevant CANN capabilities.

The installed platform configuration is versioned with CANN and is more precise
than a short scheduler label such as ``910B``. This tool is read-only: it finds
the exact long SoC, parses the matching INI, and records the active compiler and
Python runtime so an AscendC campaign can freeze its target contract.
"""

from __future__ import annotations

import argparse
import configparser
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SOC_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:Ascend[ \t_-]*)?"
    r"(910B(?:[ \t_-]*\d(?:[ \t_-]*\d)?))(?![A-Za-z0-9])",
    re.I,
)


def normalize_soc(value: object) -> str:
    token = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    match = re.fullmatch(r"(?:ASCEND)?(910B\d(?:\d)?)", token)
    if not match:
        return ""
    suffix = match.group(1)[4:]
    if len(suffix) == 2:
        suffix = f"{suffix[0]}-{suffix[1]}"
    return f"Ascend910B{suffix}"


def _torch_soc() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return ""
    try:
        import torch_npu  # type: ignore
    except Exception:
        torch_npu = None
    for api in (
        getattr(torch, "npu", None),
        getattr(torch_npu, "npu", None) if torch_npu is not None else None,
    ):
        getter = getattr(api, "get_device_name", None)
        if not callable(getter):
            continue
        for args in ((0,), ()):
            try:
                soc = normalize_soc(getter(*args))
            except Exception:
                continue
            if soc:
                return soc
    return ""


def _npu_smi_soc() -> str:
    executable = shutil.which("npu-smi")
    if not executable:
        return ""
    try:
        result = subprocess.run(
            [executable, "info"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for match in SOC_RE.finditer(f"{result.stdout}\n{result.stderr}"):
        if soc := normalize_soc(match.group(0)):
            return soc
    return ""


def discover_soc(explicit: str = "") -> str:
    return normalize_soc(explicit) or _torch_soc() or _npu_smi_soc()


def _candidate_roots(explicit: str = "") -> list[Path]:
    values = [
        explicit,
        os.environ.get("ASCEND_HOME_PATH", ""),
        os.environ.get("ASCEND_TOOLKIT_HOME", ""),
    ]
    roots: list[Path] = []
    for value in values:
        if not value:
            continue
        candidate = Path(value).expanduser()
        if candidate not in roots:
            roots.append(candidate)
    installed = Path("/usr/local/Ascend")
    if installed.is_dir():
        roots.extend(path for path in sorted(installed.glob("cann-*")) if path not in roots)
        latest = installed / "ascend-toolkit" / "latest"
        if latest not in roots:
            roots.append(latest)
    return roots


def find_platform_config(soc: str, cann_root: str = "") -> Path:
    filename = f"{soc}.ini"
    checked: list[Path] = []
    for root in _candidate_roots(cann_root):
        candidates = [
            root / "data" / "platform_config" / filename,
            root / f"{platform.machine()}-linux" / "data" / "platform_config" / filename,
            root / "x86_64-linux" / "data" / "platform_config" / filename,
            root / "aarch64-linux" / "data" / "platform_config" / filename,
        ]
        for candidate in candidates:
            if candidate in checked:
                continue
            checked.append(candidate)
            if candidate.is_file():
                return candidate.resolve()
    locations = ", ".join(str(path) for path in checked) or "no CANN roots discovered"
    raise FileNotFoundError(f"platform config {filename} not found; checked: {locations}")


def _integer(section: configparser.SectionProxy, key: str) -> int | None:
    value = section.get(key, "").strip()
    try:
        return int(value, 0) if value else None
    except ValueError:
        return None


def _mkn(section: configparser.SectionProxy, key: str) -> tuple[int, int, int] | None:
    """Parse one CANN ``DtypeMKN`` entry without guessing missing types."""

    parts = [part.strip() for part in section.get(key, "").split(",")]
    if len(parts) != 3:
        return None
    try:
        m, k, n = (int(part, 0) for part in parts)
    except ValueError:
        return None
    if min(m, k, n) <= 0:
        return None
    return m, k, n


def _dense_ops_per_second(
    cores: int | None,
    mhz: int | None,
    mkn: tuple[int, int, int] | None,
) -> float | None:
    """Return dense MAC throughput, counting multiply and add as two ops."""

    if not cores or not mhz or not mkn:
        return None
    m, k, n = mkn
    return float(cores * mhz * 1_000_000 * m * k * n * 2)


def _tera(value: float | None) -> float | None:
    return round(value / 1e12, 6) if value is not None else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cann_version(config_path: Path) -> str | None:
    for directory in (config_path.parent, *config_path.parents):
        if directory.name.startswith("cann-") and len(directory.name) > len("cann-"):
            return directory.name[len("cann-") :]
    return None


def read_capabilities(config_path: Path) -> dict[str, Any]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(config_path, encoding="utf-8")
    version = parser["version"]
    soc = parser["SoCInfo"]
    core = parser["AICoreSpec"]
    dtype = parser["DtypeMKN"] if parser.has_section("DtypeMKN") else {}
    memory_rates = (
        parser["AICoreMemoryRates"]
        if parser.has_section("AICoreMemoryRates")
        else None
    )

    aic = _integer(soc, "cube_core_cnt")
    aiv = _integer(soc, "vector_core_cnt")
    nominal_mhz = _integer(core, "cube_freq")
    default_mkn = _mkn(dtype, "Default") if dtype else None
    int8_mkn = _mkn(dtype, "DT_INT8") if dtype else None
    int4_mkn = _mkn(dtype, "DT_INT4") if dtype else None
    vector_fp16_lanes = _integer(core, "vec_calc_size")
    vector_fp32_lanes = vector_fp16_lanes // 2 if vector_fp16_lanes else None
    mix_groups = min(aic, aiv // 2) if aic is not None and aiv is not None else None
    toolchain = {
        name: shutil.which(name) for name in ("ccec", "bisheng", "msprof", "npu-smi")
    }
    toolchain.update(
        cann=_cann_version(config_path),
        python=sys.version.split()[0],
        torch=_package_version("torch"),
        torch_npu=_package_version("torch-npu"),
    )
    return {
        "schema_version": 1,
        "identity": {
            "soc_version": version.get("SoC_version"),
            "short_soc_version": version.get("Short_SoC_version"),
            "npu_arch": _integer(version, "NpuArch"),
            "aic_version": version.get("AIC_version"),
            "compiler_aic": version.get("CCEC_AIC_version"),
            "compiler_aiv": version.get("CCEC_AIV_version"),
        },
        "topology": {
            "aic_cores": aic,
            "aiv_cores": aiv,
            "ai_cpu_cores": _integer(soc, "ai_cpu_cnt"),
            "core_types": [
                item for item in soc.get("core_type_list", "").split(",") if item
            ],
            "cube_vector_mode": soc.get("cube_vector_combine"),
            "mix_ratio": "1:2" if aic and aiv == 2 * aic else None,
            "mix_block_dim": mix_groups if aic and aiv == 2 * aic else None,
        },
        "memory_bytes": {
            "device": _integer(soc, "memory_size"),
            "l2": _integer(soc, "l2_size"),
            "l0a_per_core": _integer(core, "l0_a_size"),
            "l0b_per_core": _integer(core, "l0_b_size"),
            "l0c_per_core": _integer(core, "l0_c_size"),
            "l1_per_core": _integer(core, "l1_size"),
            "ub_per_core": _integer(core, "ub_size"),
            "ub_datablock": _integer(core, "ubblock_size"),
            "ub_bank_size": _integer(core, "ubbank_size"),
            "platform_internal_ubbank_num": _integer(core, "ubbank_num"),
            "ub_bank_groups": _integer(core, "ubbank_group_num"),
        },
        "compute": {
            "nominal_mhz": nominal_mhz,
            "vector_fp16_elements_per_cycle": vector_fp16_lanes,
            "dtype_mkn": dict(dtype),
            "dense_peak": {
                "method": (
                    "installed CANN topology x nominal clock x documented or "
                    "version-matched profiler work per cycle; one MAC counts as two operations"
                ),
                "fp16_cube_tflops": _tera(
                    _dense_ops_per_second(aic, nominal_mhz, default_mkn)
                ),
                "bf16_cube_tflops_if_fp16_issue_rate": _tera(
                    _dense_ops_per_second(aic, nominal_mhz, default_mkn)
                ),
                "fp32_cube_tflops": None,
                "int8_cube_tops": _tera(
                    _dense_ops_per_second(aic, nominal_mhz, int8_mkn)
                ),
                "int4_cube_tops_if_one_mmad_per_cycle": _tera(
                    _dense_ops_per_second(aic, nominal_mhz, int4_mkn)
                ),
                "fp16_vector_add_tera_ops_s": _tera(
                    float(aiv * nominal_mhz * 1_000_000 * vector_fp16_lanes)
                    if aiv and nominal_mhz and vector_fp16_lanes
                    else None
                ),
                "fp32_vector_add_tera_ops_s_if_one_repeat_per_cycle": _tera(
                    float(aiv * nominal_mhz * 1_000_000 * vector_fp32_lanes)
                    if aiv and nominal_mhz and vector_fp32_lanes
                    else None
                ),
                "confidence": {
                    "fp16_cube_tflops": "high: CANN 8.5 documents one FP16 16x16x16 MAC per Cube core per clock",
                    "int8_cube_tops": "high: version-matched CANN profiler model uses 16x16x32 per active Cube cycle",
                    "bf16_cube_tflops_if_fp16_issue_rate": "medium: dtype and tile are supported, but the BF16 issue rate is not explicitly published",
                    "int4_cube_tops_if_one_mmad_per_cycle": "medium: tile is configured, but the INT4 issue rate is not explicitly published",
                    "fp32_cube_tflops": "unknown: no version-matched per-cycle FP32 Cube rate",
                    "fp16_vector_add_tera_ops_s": "high: CANN 8.5 documents 128 FP16 additions per Vector core per clock",
                    "fp32_vector_add_tera_ops_s_if_one_repeat_per_cycle": "medium: 64-element repeat width is documented, but one repeat per cycle is an assumption",
                    "vector_fma": "unknown: intrinsic-specific issue rate is not published",
                },
            },
        },
        "memory_bandwidth": {
            "raw_platform_rate_fields": dict(memory_rates) if memory_rates else {},
            "raw_platform_rate_semantics": (
                "CANN compiler/platform cost-model fields; the installed header "
                "does not document unit, direction aggregation, or card-level semantics"
            ),
            "cann_msprof_roofline_gm_rw_tb_s": None,
            "cann_msprof_roofline_l2_rw_tb_s": None,
            "physical_hbm_interface_peak_tb_s": None,
            "physical_hbm_interface_peak_note": (
                "not exposed by the platform metadata; capture the CANN msprof "
                "Roofline model and a same-size streaming measurement separately"
            ),
        },
        "toolchain": toolchain,
        "source": {"platform_config": str(config_path)},
        "warnings": [
            "Configured local-memory capacities are not guaranteed usable capacities; query PlatformAscendC at tiling time.",
            "platform_internal_ubbank_num is not the CANN 8.5 physical UB bank model; use official version-matched guidance.",
            "Dense compute peaks are cycle-model ceilings, not guaranteed sustained application throughput.",
            "AICoreMemoryRates values lack public unit and aggregation semantics; do not multiply them into a card bandwidth peak.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit installed Ascend/CANN capabilities as JSON")
    parser.add_argument("--soc", default="", help="Long SoC override, for example Ascend910B1")
    parser.add_argument("--cann-root", default="", help="Explicit CANN toolkit root")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args(argv)

    soc = discover_soc(args.soc)
    if not soc:
        parser.error("unable to identify an exact Ascend 910B product; pass --soc Ascend910B1")
    try:
        config_path = find_platform_config(soc, args.cann_root)
        payload = read_capabilities(config_path)
    except (FileNotFoundError, KeyError, configparser.Error) as exc:
        parser.error(str(exc))
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
