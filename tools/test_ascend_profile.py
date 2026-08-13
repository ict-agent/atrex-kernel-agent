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

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference import profile_driver
from tools import local_gateway, sandbox


def _profile_payload(profiler: str = "msprof") -> dict[str, object]:
    return {
        "spec": {"target_hardware": ["local"]},
        "candidate": "class Model:\n    pass\n",
        "reference": {
            "operator": "test",
            "reference_py": "def reference(*args):\n    return args\n",
            "input_py": "def _make_inputs(**kwargs):\n    return {}\n",
            "shapes": {"0": {"init_kwargs": {}, "input_kwargs": {}}},
        },
        "profiler": profiler,
        "options": {"timeout_s": 60},
    }


class SandboxRouteTests(unittest.TestCase):
    def test_ascend_wrapper_routes_to_profile_and_is_uploaded(self) -> None:
        command = ["bash", "tools/profile_ascend.sh", "profile_driver.py"]
        self.assertTrue(sandbox._is_profile_command(command))
        self.assertEqual(sandbox._requested_gateway_kind("auto", command), "profile")
        self.assertEqual(sandbox._profile_command_profiler(command), "msprof")
        self.assertEqual(sandbox._sandbox_telemetry_category(command), "profile")
        with tempfile.TemporaryDirectory() as temporary:
            selected = sandbox._command_input_paths(Path(temporary), command)
        self.assertIn("tools/profile_ascend.sh", selected)

    def test_cli_accepts_msprof(self) -> None:
        args = sandbox.build_parser().parse_args(
            ["--hardware", "local", "--kind", "profile", "--profiler", "msprof", "--", "true"]
        )
        self.assertEqual(args.profiler, "msprof")


class GatewayRouteTests(unittest.TestCase):
    def test_gateway_worker_path_preserves_venv_launcher_directory(self) -> None:
        source = Path(local_gateway.__file__).read_text(encoding="utf-8")
        self.assertIn("python_bin = str(Path(sys.executable).parent)", source)
        self.assertNotIn("Path(sys.executable).resolve().parent", source)

    def test_typed_profile_allowlist_accepts_msprof(self) -> None:
        request = local_gateway._validate_typed_request(_profile_payload(), "profile")
        self.assertEqual(request["profiler"], "msprof")

    def test_find_and_auto_select_msprof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "msprof"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o700)
            env = {"PATH": temporary}
            self.assertEqual(local_gateway._find_profile_tool("msprof", env), str(executable))

            with mock.patch.object(
                local_gateway.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, [sys.executable]),
            ), mock.patch.object(local_gateway, "_find_npu_smi", return_value="/usr/bin/npu-smi"):
                self.assertEqual(local_gateway._auto_profile_tool(env), "msprof")

    def test_msprof_argv_uses_cann_op_and_npu_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            (root / "scripts" / "run_eval.py").write_text("# runner\n", encoding="utf-8")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            msprof = bin_dir / "msprof"
            msprof.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            msprof.chmod(0o700)
            workdir = root / "work"
            workdir.mkdir()
            scheduler = object.__new__(local_gateway.LocalScheduler)
            scheduler.atrex_bench_root = root
            request = {"profiler": "msprof", "level": "sol"}
            env = {"PATH": str(bin_dir)}

            argv = scheduler._profile_argv(request, workdir, env)

            self.assertEqual(argv[:2], [str(msprof), "op"])
            self.assertEqual(argv[2], f"--output={workdir / 'profile_output'}")
            self.assertEqual(argv[-2], sys.executable)
            self.assertEqual(env["PROFILE_DEVICE"], "npu:0")
            self.assertEqual(request["_resolved_profiler"], "msprof")

    def test_msprof_artifacts_are_retained_without_ncu_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "PROF_000001" / "device_0" / "op_summary.csv"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("Task Start Time(us),Task Duration(us)\n1,2\n", encoding="utf-8")

            result = local_gateway._parse_msprof_artifacts(root, level="sol")

        self.assertEqual(result["profiler"], "msprof")
        self.assertEqual(result["metrics_status"], "unavailable")
        self.assertEqual(result["artifact_count"], 1)
        self.assertEqual(result["artifacts"][0]["path"], "PROF_000001/device_0/op_summary.csv")
        for ncu_field in ("compute_sol_pct", "mem_sol_pct", "dram_pct", "occupancy_pct", "bound"):
            self.assertNotIn(ncu_field, json.dumps(result))

    def test_completed_msprof_job_uses_artifact_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            output = workdir / "profile_output"
            output.mkdir()
            (output / "raw.json").write_text("{}\n", encoding="utf-8")
            scheduler = object.__new__(local_gateway.LocalScheduler)
            scheduler.store = mock.Mock()

            scheduler._complete_job(
                "pf_test",
                "profile",
                {"_resolved_profiler": "msprof", "level": "sol"},
                workdir,
                {"exit_code": 0, "stdout": "", "stderr": ""},
            )

        completed = scheduler.store.complete.call_args.kwargs
        self.assertEqual(completed["status"], "succeeded")
        self.assertEqual(completed["result"]["profiler"], "msprof")
        self.assertEqual(completed["result"]["metrics_status"], "unavailable")


class EnvironmentProbeTests(unittest.TestCase):
    NPU_SMI_OUTPUT = """
| NPU   Name        | Health | Power(W) | Temp(C) | Memory-Usage(MB) |
| 0     910B1       | OK     | 91.0     | 41      | 300 / 65536     |
Driver Version: 24.1.rc3
"""

    def test_npu_smi_table_parser(self) -> None:
        result = local_gateway._parse_npu_smi_info(self.NPU_SMI_OUTPUT)
        self.assertEqual(result["gpu_model"], "Ascend 910B1")
        self.assertEqual(result["arch"], "ascend910b1")
        self.assertEqual(result["total_memory_mb"], 65536)
        self.assertEqual(result["driver_version"], "24.1.rc3")

    def test_npu_smi_910b1_live_header_variant(self) -> None:
        output = """
| npu-smi 25.3.rc1                 Version: 25.3.rc1 |
| 0     910B1               | OK   | 91.5 | 37 | 0 / 0 |
| 0                         | bus  | 0    | 0 / 0 | 61768/ 65536 |
"""
        result = local_gateway._parse_npu_smi_info(output)
        self.assertEqual(result["arch"], "ascend910b1")
        self.assertEqual(result["total_memory_mb"], 65536)
        self.assertEqual(result["driver_version"], "25.3.rc1")

    def test_environment_falls_back_to_npu_smi_without_torch(self) -> None:
        def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if command[0] == "nvidia-smi":
                raise FileNotFoundError("nvidia-smi")
            if command == ["/usr/local/bin/npu-smi", "info"]:
                return subprocess.CompletedProcess(command, 0, self.NPU_SMI_OUTPUT, "")
            raise subprocess.CalledProcessError(1, command)

        with mock.patch.object(local_gateway, "_find_npu_smi", return_value="/usr/local/bin/npu-smi"), mock.patch.object(
            local_gateway.subprocess, "run", side_effect=run
        ):
            result = local_gateway._probe_environment()

        self.assertEqual(result["vendor"], "ascend")
        self.assertEqual(result["gpu_model"], "Ascend 910B1")
        self.assertEqual(result["arch"], "ascend910b1")

    def test_torch_probe_prefers_torch_npu_device_name(self) -> None:
        self.assertIn("import torch_npu", local_gateway._TORCH_DEVICE_PROBE)
        self.assertIn("get_device_name", local_gateway._TORCH_DEVICE_PROBE)
        self.assertLess(
            local_gateway._TORCH_DEVICE_PROBE.index('"vendor": "ascend"'),
            local_gateway._TORCH_DEVICE_PROBE.index("torch.cuda.get_device_properties"),
        )

    def test_environment_consumes_torch_npu_probe_result(self) -> None:
        torch_result = {
            "vendor": "ascend",
            "model": "Ascend910B1",
            "arch": "Ascend910B1",
            "sm_count": 48,
            "total_memory_mb": 65536,
            "torch": "2.6.0",
            "runtime": "2.6.0.post5",
        }

        def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if command[0] == "nvidia-smi":
                raise FileNotFoundError("nvidia-smi")
            return subprocess.CompletedProcess(command, 0, json.dumps(torch_result), "")

        with mock.patch.object(local_gateway, "_find_npu_smi", return_value=None), mock.patch.object(
            local_gateway.subprocess, "run", side_effect=run
        ):
            result = local_gateway._probe_environment()

        self.assertEqual(result["vendor"], "ascend")
        self.assertEqual(result["gpu_model"], "Ascend910B1")
        self.assertEqual(result["arch"], "ascend910b1")
        self.assertEqual(result["toolchain"]["runtime"], "2.6.0.post5")


class ProfileDriverTests(unittest.TestCase):
    def test_npu_synchronize_uses_torch_npu_api(self) -> None:
        npu = mock.Mock()
        fake_torch = types.ModuleType("torch")
        fake_torch.npu = npu
        fake_torch.cuda = mock.Mock()
        fake_torch_npu = types.ModuleType("torch_npu")
        fake_torch_npu.npu = npu

        with mock.patch.dict(sys.modules, {"torch": fake_torch, "torch_npu": fake_torch_npu}):
            profile_driver._synchronize("npu:0")

        npu.synchronize.assert_called_once_with()
        fake_torch.cuda.synchronize.assert_not_called()

    def test_cuda_synchronize_remains_supported(self) -> None:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = mock.Mock()
        fake_torch.cuda.is_available.return_value = True

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            profile_driver._synchronize("cuda:0")

        fake_torch.cuda.synchronize.assert_called_once_with()


class WrapperTests(unittest.TestCase):
    def test_wrapper_runs_msprof_op_and_preserves_raw_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.py"
            target.write_text(
                "import os\nprint('target-ran', os.environ['PROFILE_DEVICE'])\n",
                encoding="utf-8",
            )
            fake_msprof = root / "msprof"
            fake_msprof.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == "op" ]]
shift
output="${1#--output=}"
shift
mkdir -p "$output/PROF_000001/device_0"
printf 'raw-msprof-artifact\\n' > "$output/PROF_000001/device_0/op_summary.csv"
"$@"
""",
                encoding="utf-8",
            )
            fake_msprof.chmod(0o700)
            output = root / "profiles" / "v1"
            env = os.environ.copy()
            env.pop("PROFILE_DEVICE", None)
            env.update(MSPROF_BIN=str(fake_msprof), PROFILE_PYTHON=sys.executable)

            completed = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "tools" / "profile_ascend.sh"),
                    str(target),
                    "--output-dir",
                    str(output),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("target-ran npu:0", completed.stdout)
            self.assertTrue((output / "msprof" / "PROF_000001" / "device_0" / "op_summary.csv").is_file())
            summary = json.loads((output / "profile_summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["metrics_status"], "unavailable")
            paths = {entry["path"] for entry in manifest["artifacts"]}
            self.assertIn("msprof/PROF_000001/device_0/op_summary.csv", paths)
            self.assertIn("msprof.log", paths)


if __name__ == "__main__":
    unittest.main()
