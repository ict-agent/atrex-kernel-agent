from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from tools import compute_utilization as utilization


REPO_ROOT = Path(__file__).resolve().parents[1]


class Ascend910B1PeakProviderTests(unittest.TestCase):
    def test_provider_is_versioned_and_targets_exact_soc_and_cann(self) -> None:
        provider = utilization.get_peak_provider("Ascend 910B1")

        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider["schema_version"], utilization.PEAK_PROVIDER_SCHEMA_VERSION)
        self.assertEqual(provider["provider_id"], "ascend910b1-cann-8.5.0-v1")
        self.assertEqual(provider["provider_version"], "1.2.0")
        self.assertEqual(provider["target"]["soc_version"], "Ascend910B1")
        self.assertEqual(provider["target"]["npu_arch"], 2201)
        self.assertEqual(provider["target"]["cann_version"], "8.5.0")

    def test_sources_capture_exact_fields_urls_and_bandwidth_caveat(self) -> None:
        provider = utilization.get_peak_provider("910B1")
        assert provider is not None
        platform = provider["sources"]["cann_850_platform_config"]
        self.assertEqual(platform["kind"], "version_matched_toolchain_config")
        self.assertEqual(
            platform["validated_sha256"],
            "6c8ba94d7d64186c8721ceeade3e9403cd33e8d5d55eb0e6a9ec7ab294fa7449",
        )
        self.assertIn("DtypeMKN.DT_INT8", platform["fields"])
        self.assertIn("AICoreMemoryRates.ddr_rate", platform["fields"])
        self.assertIn("do not state enough unit/aggregation", platform["caveat"])
        cycle_source = provider["sources"]["cann_850_cube_vector_cycle_model"]
        self.assertTrue(
            cycle_source["locator"].endswith(
                "atlas_ascendc_best_practices_10_0002.html"
            )
        )
        msprof = provider["sources"]["cann_850_msprof_roofline"]
        self.assertIn("GM read+write ceiling = 1.8 TB/s", msprof["fields"])
        self.assertIn("not physical interface ratings", msprof["caveat"])
        measured = provider["sources"]["target_copy_256m"]
        self.assertEqual(
            measured["method"]["traffic_convention"],
            "one source read plus one destination write",
        )
        self.assertEqual(len(measured["result"]["samples_tb_s"]), 7)

    def test_topology_requires_explicit_split_execution_kind(self) -> None:
        self.assertEqual(utilization.get_num_units("910B1", "aic"), 24)
        self.assertEqual(utilization.get_num_units("ascend_910b1", "aiv_only"), 48)
        self.assertEqual(utilization.get_num_units("Huawei Ascend 910B1", "mix"), 24)

        with self.assertRaisesRegex(ValueError, "split AIC/AIV execution"):
            utilization.get_num_units("ascend910b1")
        with self.assertRaisesRegex(ValueError, "pass execution_kind"):
            utilization.get_num_units("ascend910b1", "generic")

    def test_aic_requires_execution_and_returns_auditable_cycle_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "pass execution_kind"):
            utilization.get_peak_tflops("ascend910b1", "fp16")

        direct = {"fp16": 363.7248, "int8": 727.4496}
        conditional = {"bf16": 363.7248, "int4": 1454.8992}
        for dtype, value in direct.items():
            with self.subTest(dtype=dtype):
                self.assertAlmostEqual(
                    utilization.get_peak_tflops(
                        "ascend910b1", dtype, execution_kind="aic"
                    ),
                    value,
                )
                record = utilization.get_peak_record(
                    "ascend910b1",
                    "compute_tflops",
                    dtype,
                    execution_kind="aic",
                )
                assert record is not None
                self.assertEqual(record["execution_kind"], "aic")
                self.assertIn(record["status"], {"available", "conditional"})
                self.assertTrue(record["source_ids"])
                self.assertTrue(record["source_records"])
                self.assertIn("kind", record["derivation"])

        for dtype, value in conditional.items():
            with self.subTest(dtype=dtype, status="conditional"):
                with self.assertRaisesRegex(ValueError, "allow_modeled_peak=True"):
                    utilization.get_peak_tflops(
                        "ascend910b1", dtype, execution_kind="aic"
                    )
                self.assertAlmostEqual(
                    utilization.get_peak_tflops(
                        "ascend910b1",
                        dtype,
                        execution_kind="aic",
                        allow_modeled_peak=True,
                    ),
                    value,
                )

        fp16 = utilization.get_peak_record(
            "ascend910b1", "compute_tflops", "fp16", execution_kind="aic"
        )
        bf16 = utilization.get_peak_record(
            "ascend910b1", "compute_tflops", "bf16", execution_kind="aic"
        )
        int8 = utilization.get_peak_record(
            "ascend910b1", "compute_tflops", "int8", execution_kind="aic"
        )
        assert fp16 is not None and bf16 is not None and int8 is not None
        self.assertEqual(fp16["status"], "available")
        self.assertEqual(fp16["confidence"]["level"], "high")
        self.assertEqual(bf16["status"], "conditional")
        self.assertEqual(bf16["confidence"]["level"], "medium")
        self.assertEqual(int8["status"], "available")
        self.assertEqual(int8["confidence"]["level"], "high")
        self.assertEqual(int8["unit"], "TOPS")

    def test_aic_fp32_and_hf32_fail_closed(self) -> None:
        for dtype in ("fp32", "hf32"):
            with self.subTest(dtype=dtype):
                record = utilization.get_peak_record(
                    "ascend910b1",
                    "compute_tflops",
                    dtype,
                    execution_kind="aic",
                )
                assert record is not None
                self.assertEqual(record["status"], "unknown")
                self.assertIsNone(record["value"])
                with self.assertRaisesRegex(
                    ValueError, "Refusing to infer a numeric peak"
                ):
                    utilization.get_peak_tflops(
                        "ascend910b1", dtype, execution_kind="aic"
                    )

    def test_aiv_requires_operation_and_fma_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires operation_kind"):
            utilization.get_peak_tflops(
                "ascend910b1", "fp16", execution_kind="aiv"
            )

        self.assertAlmostEqual(
            utilization.get_peak_tflops(
                "ascend910b1",
                "fp16",
                execution_kind="aiv",
                operation_kind="add",
            ),
            11.3664,
        )
        fp16_add = utilization.get_peak_record(
            "ascend910b1",
            "compute_tflops",
            "fp16",
            execution_kind="aiv",
            operation_kind="add",
        )
        assert fp16_add is not None
        self.assertEqual(fp16_add["status"], "available")
        self.assertEqual(fp16_add["confidence"]["level"], "medium-high")

        fp32_add = utilization.get_peak_record(
            "ascend910b1",
            "compute_tflops",
            "fp32",
            execution_kind="aiv",
            operation_kind="add",
        )
        assert fp32_add is not None
        self.assertEqual(fp32_add["status"], "conditional")
        with self.assertRaisesRegex(ValueError, "allow_modeled_peak=True"):
            utilization.get_peak_tflops(
                "ascend910b1",
                "fp32",
                execution_kind="aiv",
                operation_kind="add",
            )
        self.assertAlmostEqual(
            utilization.get_peak_tflops(
                "ascend910b1",
                "fp32",
                execution_kind="aiv",
                operation_kind="add",
                allow_modeled_peak=True,
            ),
            5.6832,
        )

        for dtype in ("fp16", "fp32"):
            with self.subTest(dtype=dtype, operation="fma"):
                fma = utilization.get_peak_record(
                    "ascend910b1",
                    "compute_tflops",
                    dtype,
                    execution_kind="aiv",
                    operation_kind="fma",
                )
                assert fma is not None
                self.assertEqual(fma["status"], "unknown")
                with self.assertRaisesRegex(
                    ValueError, "Refusing to infer a numeric peak"
                ):
                    utilization.get_peak_tflops(
                        "ascend910b1",
                        dtype,
                        execution_kind="aiv",
                        operation_kind="fma",
                    )

    def test_mix_and_unsupported_dtype_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "MIX has no scalar compute peak"):
            utilization.get_peak_tflops(
                "ascend910b1", "fp16", execution_kind="mix"
            )
        with self.assertRaisesRegex(ValueError, "unsupported AIV dtype"):
            utilization.get_peak_tflops(
                "ascend910b1",
                "bf16",
                execution_kind="aiv",
                operation_kind="add",
            )

    def test_bandwidth_defaults_to_msprof_model_and_physical_fails_closed(self) -> None:
        self.assertAlmostEqual(utilization.get_peak_bandwidth("ascend910b1"), 1.8)
        self.assertAlmostEqual(
            utilization.get_peak_bandwidth("ascend910b1", "l2_cycle_model"),
            8.0,
        )
        self.assertAlmostEqual(
            utilization.get_peak_bandwidth("ascend910b1", "measured_copy_256m"),
            1.267099,
        )
        hbm = utilization.get_peak_record(
            "ascend910b1", "memory_bandwidth_tb_s"
        )
        assert hbm is not None
        self.assertEqual(hbm["peak_type"], "cann_msprof_roofline_model")
        self.assertEqual(hbm["memory_level"], "hbm")
        self.assertFalse(hbm["physical_interface_peak"])
        self.assertEqual(hbm["source_ids"], ["cann_850_msprof_roofline"])

        physical = utilization.get_peak_record(
            "ascend910b1",
            "memory_bandwidth_tb_s",
            bandwidth_kind="physical_hbm_interface_peak",
        )
        assert physical is not None
        self.assertEqual(physical["status"], "unknown")
        with self.assertRaisesRegex(ValueError, "Refusing to infer a numeric peak"):
            utilization.get_peak_bandwidth(
                "ascend910b1", "physical_hbm_interface_peak"
            )

    def test_roofline_and_utilization_expose_ceiling_type_and_source(self) -> None:
        roofline = utilization.roofline_analysis(
            1024,
            1024,
            "ascend910b1",
            "fp16",
            execution_kind="aic",
        )
        self.assertEqual(roofline["compute_ceiling_type"], "cann_cycle_model_dense_cube")
        self.assertEqual(
            roofline["bandwidth_ceiling_type"], "cann_msprof_roofline_model"
        )
        self.assertIn("cann_850_msprof_roofline", roofline["bandwidth_ceiling_source"])
        self.assertFalse(roofline["bandwidth_physical_interface_peak"])

        compute = utilization.compute_utilization(
            363.7248e9,
            1.0,
            "ascend910b1",
            "fp16",
            execution_kind="aic",
        )
        self.assertAlmostEqual(compute["utilization_pct"], 100.0)
        self.assertEqual(compute["peak_unit"], "TFLOPS")
        self.assertIn("ascend910b1-cann-8.5.0-v1", compute["peak_source"])

        bandwidth = utilization.compute_bandwidth_utilization(
            1.8e9, 1.0, "ascend910b1"
        )
        self.assertAlmostEqual(bandwidth["utilization_pct"], 100.0)
        self.assertEqual(
            bandwidth["bandwidth_ceiling_type"], "cann_msprof_roofline_model"
        )
        self.assertIsNone(bandwidth["hardware_peak_bandwidth_tb_s"])
        self.assertFalse(bandwidth["bandwidth_physical_interface_peak"])
        self.assertNotIn("hardware theoretical peak", bandwidth["ceiling_source"])

    def test_cli_errors_are_actionable_without_tracebacks(self) -> None:
        common = [
            sys.executable,
            str(REPO_ROOT / "tools" / "compute_utilization.py"),
            "--gpu",
            "ascend910b1",
            "--dtype",
            "fp16",
            "--flops",
            "1024",
            "--bytes",
            "1024",
            "--time-ms",
            "1",
        ]
        missing_execution = subprocess.run(
            common,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(missing_execution.returncode, 1)
        self.assertIn("pass execution_kind", missing_execution.stderr)
        self.assertNotIn("Traceback", missing_execution.stderr)

        missing_operation = subprocess.run(
            common + ["--execution-kind", "aiv"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(missing_operation.returncode, 1)
        self.assertIn("requires operation_kind", missing_operation.stderr)
        self.assertNotIn("Traceback", missing_operation.stderr)

    def test_cli_labels_cann_bandwidth_as_model_not_physical_peak(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "compute_utilization.py"),
                "--gpu",
                "ascend910b1",
                "--dtype",
                "fp16",
                "--execution-kind",
                "aic",
                "--flops",
                "1024",
                "--bytes",
                "1024",
                "--time-ms",
                "1",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("cann_msprof_roofline_model", completed.stdout)
        self.assertIn("cann_850_msprof_roofline", completed.stdout)
        self.assertNotIn("Hardware peak", completed.stdout)
        self.assertNotIn("Traceback", completed.stderr)

    def test_provider_result_is_a_defensive_copy(self) -> None:
        first = utilization.get_peak_provider("ascend910b1")
        assert first is not None
        first["topology"]["aic_units"]["value"] = 999

        second = utilization.get_peak_provider("ascend910b1")
        assert second is not None
        self.assertEqual(second["topology"]["aic_units"]["value"], 24)

    def test_available_provider_value_without_provenance_fails_closed(self) -> None:
        provider = utilization.get_peak_provider("ascend910b1")
        assert provider is not None
        record = provider["peaks"]["compute_tflops"]["aic"]["fp16"]
        record["source_ids"] = []
        with mock.patch.dict(
            utilization.PEAK_PROVIDERS,
            {utilization.ASCEND_910B1_PEAK_PROVIDER_ID: provider},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "lacks complete.*provenance"):
                utilization.get_peak_tflops(
                    "ascend910b1", "fp16", execution_kind="aic"
                )

    def test_conditional_provider_value_without_conditions_fails_closed(self) -> None:
        provider = utilization.get_peak_provider("ascend910b1")
        assert provider is not None
        provider["peaks"]["compute_tflops"]["aic"]["bf16"]["conditions"] = []
        with mock.patch.dict(
            utilization.PEAK_PROVIDERS,
            {utilization.ASCEND_910B1_PEAK_PROVIDER_ID: provider},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "lacks complete.*provenance"):
                utilization.get_peak_tflops(
                    "ascend910b1",
                    "bf16",
                    execution_kind="aic",
                    allow_modeled_peak=True,
                )

    def test_explicit_numeric_override_still_requires_ascend_execution_kind(self) -> None:
        overridden = dict(utilization.HARDWARE_SPECS["ascend910b1"])
        overridden.update(bf16=321.0, memory_bandwidth_tb_s=1.25)
        with mock.patch.dict(
            utilization.HARDWARE_SPECS,
            {"ascend910b1": overridden},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "pass execution_kind"):
                utilization.get_peak_tflops("910B1", "bf16")
            self.assertEqual(
                utilization.get_peak_tflops(
                    "910B1", "bf16", execution_kind="aic"
                ),
                321.0,
            )
            self.assertEqual(utilization.get_peak_bandwidth("910B1"), 1.25)
            self.assertAlmostEqual(
                utilization.compute_ridge_point(
                    "910B1", "bf16", execution_kind="aic"
                ),
                256.8,
            )


class ExistingHardwareCompatibilityTests(unittest.TestCase):
    def test_existing_nvidia_and_amd_flat_table_interfaces_are_unchanged(self) -> None:
        self.assertEqual(utilization.get_peak_tflops("h100", "bf16"), 989.4)
        self.assertEqual(utilization.get_peak_bandwidth("h100"), 3.35)
        self.assertEqual(utilization.get_num_units("h100"), 132)
        self.assertEqual(utilization.get_unit_type("h100"), "SM")

        self.assertEqual(utilization.get_peak_tflops("mi300x", "fp16"), 1307.4)
        self.assertEqual(utilization.get_peak_bandwidth("mi300x"), 5.3)
        self.assertEqual(utilization.get_num_units("mi300x"), 304)
        self.assertEqual(utilization.get_unit_type("mi300x"), "CU")

    def test_legacy_bandwidth_result_still_uses_hardware_peak_label(self) -> None:
        result = utilization.compute_bandwidth_utilization(3.35e9, 1.0, "h100")
        self.assertEqual(result["hardware_peak_bandwidth_tb_s"], 3.35)
        self.assertEqual(result["ceiling_source"], "hardware theoretical peak")
        self.assertTrue(result["bandwidth_physical_interface_peak"])

    def test_generic_tile_wave_formula_remains_uncapped_and_warns(self) -> None:
        result = utilization.compute_theoretical_ceiling(
            tile_flops=1e9,
            tile_bytes=1.0,
            grid_blocks=2,
            num_units=132,
            gpu="h100",
            dtype="bf16",
        )

        self.assertEqual(result["bottleneck"], "compute")
        self.assertGreater(result["theoretical_tflops"], result["peak_tflops"])
        self.assertIn("may exceed the device peak", result["model_warning"])


if __name__ == "__main__":
    unittest.main()
