from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.agent_runtime.process import dependency_process_violation
from orchestrator.hardware import (
    hardware_vendor,
    normalize_arch_token,
    supported_frameworks,
)
from orchestrator.session_io import _RUNTIME_ARCH_PROBE, _arch_from_probe_output, detect_arch


class HardwareIdentityTests(unittest.TestCase):
    def test_ascend_910b_aliases_preserve_family_and_b1_product(self) -> None:
        family_aliases = (
            "910B",
            "Ascend 910B",
        )
        for alias in family_aliases:
            with self.subTest(alias=alias):
                self.assertEqual(normalize_arch_token(alias), "ascend910b")

        product_aliases = (
            "910B1",
            "Ascend910B1",
            "ascend_910b1",
            "Huawei Ascend 910B1",
        )
        for alias in product_aliases:
            with self.subTest(alias=alias):
                self.assertEqual(normalize_arch_token(alias), "ascend910b1")

    def test_ascend_platform_and_arch_aliases_select_ascendc(self) -> None:
        for platform in ("910B", "Ascend 910B", "Ascend910B1", "Huawei Ascend 910B1"):
            with self.subTest(platform=platform):
                self.assertEqual(hardware_vendor(platform), "ascend")
                self.assertEqual(supported_frameworks(platform), ("AscendC",))

        self.assertEqual(hardware_vendor("REMOTE_ACCELERATOR", "910B"), "ascend")
        self.assertEqual(
            supported_frameworks("REMOTE_ACCELERATOR", "Ascend910B1"),
            ("AscendC",),
        )

    def test_runtime_arch_remains_authoritative_over_platform_name(self) -> None:
        self.assertEqual(hardware_vendor("Ascend910B1", "sm_90"), "nvidia")
        self.assertEqual(hardware_vendor("H20", "ascend910b1"), "ascend")

    def test_existing_nvidia_amd_and_unknown_dispatch_do_not_regress(self) -> None:
        self.assertEqual(normalize_arch_token("sm103"), "sm_103")
        self.assertEqual(normalize_arch_token("gfx942"), "gfx942")
        self.assertEqual(hardware_vendor("H20", "sm_90"), "nvidia")
        self.assertEqual(hardware_vendor("MI308X", "gfx942"), "amd")
        self.assertEqual(supported_frameworks("B200"), ("Triton", "CuteDSL", "Cuda"))
        self.assertEqual(supported_frameworks("MI300X"), ("Triton", "FlyDSL"))
        self.assertEqual(supported_frameworks("mystery accelerator"), ("Triton",))


class ArchitectureProbeTests(unittest.TestCase):
    def test_runtime_probe_checks_torch_npu_before_cuda(self) -> None:
        self.assertIn("import torch_npu", _RUNTIME_ARCH_PROBE)
        self.assertLess(
            _RUNTIME_ARCH_PROBE.index('getattr(torch, "npu", None)'),
            _RUNTIME_ARCH_PROBE.index("torch.cuda.get_device_properties"),
        )

    def test_family_name_does_not_join_a_next_line_device_index(self) -> None:
        self.assertEqual(
            _arch_from_probe_output("Model: Ascend 910B\n1 device online\n"),
            "ascend910b",
        )

    @mock.patch("orchestrator.session_io._sandbox_command")
    def test_sandbox_torch_npu_result_is_normalized(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="torch_npu initialized\nAscend910B1\n",
            stderr="",
        )

        self.assertEqual(detect_arch("ASCEND_WORKER"), "ascend910b1")
        self.assertEqual(run.call_count, 1)
        probe_command = run.call_args.args[5]
        self.assertEqual(probe_command[:2], ["python", "-c"])
        self.assertIn("import torch_npu", probe_command[2])

    @mock.patch("orchestrator.session_io._sandbox_command")
    def test_sandbox_falls_back_to_npu_smi_when_torch_is_missing(
        self, run: mock.Mock
    ) -> None:
        run.side_effect = (
            subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'torch'",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
            stdout="| NPU  Name | Ascend 910B1 | Health OK |\n",
                stderr="",
            ),
        )

        self.assertEqual(detect_arch("ASCEND_WORKER"), "ascend910b1")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[1].args[5], ["npu-smi", "info"])

    @mock.patch("orchestrator.session_io.subprocess.run")
    def test_local_probe_uses_npu_smi_after_python_probes_fail(
        self, run: mock.Mock
    ) -> None:
        def result_for(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if command == ["npu-smi", "info"]:
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout="Chip Name: 910B1\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(
                args=command,
                returncode=1,
                stdout="",
                stderr="No module named torch",
            )

        run.side_effect = result_for
        self.assertEqual(detect_arch(), "ascend910b1")
        self.assertEqual(run.call_args_list[-1].args[0], ["npu-smi", "info"])

    @mock.patch("orchestrator.session_io._sandbox_command")
    def test_existing_cuda_and_rocm_probe_tokens_are_preserved(self, run: mock.Mock) -> None:
        for output, expected in (("sm_103\n", "sm_103"), ("gfx942\n", "gfx942")):
            with self.subTest(output=output):
                run.reset_mock()
                run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=output, stderr=""
                )
                self.assertEqual(detect_arch("GPU_WORKER"), expected)
                self.assertEqual(run.call_count, 1)


class AscendHostGuardTests(unittest.TestCase):
    def test_blocks_direct_ascend_build_profile_probe_and_wrapper(self) -> None:
        commands = (
            ["ccec", "kernel.cpp", "-o", "kernel.o"],
            ["bisheng", "kernel.cpp", "-o", "kernel.o"],
            ["ascendc_pack_kernel", "device.o"],
            ["msprof", "op", "python", "profile_driver.py"],
            ["mskpp", "--model", "kernel"],
            ["npu-smi", "info"],
            ["bash", "tools/profile_ascend.sh", "profile_driver.py"],
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNotNone(dependency_process_violation(command))

    def test_blocks_direct_torch_npu_import_but_allows_sandbox(self) -> None:
        self.assertIsNotNone(
            dependency_process_violation(
                ["python3", "-c", "import torch_npu; print(torch_npu.npu.is_available())"]
            )
        )
        self.assertIsNone(
            dependency_process_violation(
                ["python3", "tools/sandbox.py", "--hardware", "local", "--", "npu-smi", "info"]
            )
        )


if __name__ == "__main__":
    unittest.main()
