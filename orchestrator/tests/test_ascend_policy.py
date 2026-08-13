from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from orchestrator.campaign import Campaign
from orchestrator.optimization_policy import (
    declared_solution_source_paths,
    optimization_mode_directive,
    production_kernel_violations,
)
from orchestrator.session_io import _dependency_review_candidate_paths


KERNEL = """\
import torch
import torch_npu
from torch_npu.utils.cpp_extension import load

extension = load(name="atrex_ascendc_add", sources=["ascend/add.cpp"])

def run(x, y, out):
    return torch.ops.atrex_ascendc.add(x, y, out)
"""

ASCENDC_SOURCE = """\
#include "kernel_operator.h"
#include "add.h"

extern "C" __global__ __aicore__ void add_kernel(
    GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
  AscendC::TPipe pipe;
}
"""


def _write_candidate(workspace: Path, *, kernel: str = KERNEL, source: str = ASCENDC_SOURCE) -> None:
    (workspace / "ascend").mkdir(parents=True, exist_ok=True)
    (workspace / "kernel.py").write_text(kernel, encoding="utf-8")
    (workspace / "ascend" / "add.cpp").write_text(source, encoding="utf-8")
    (workspace / "ascend" / "add.h").write_text("#pragma once\n", encoding="utf-8")
    (workspace / "solution.json").write_text(
        json.dumps(
            {
                "spec": {
                    "languages": ["pytorch", "ascendc"],
                    "dependencies": ["torch", "torch-npu", "CANN", "pybind11"],
                },
                "sources": [
                    {"path": "kernel.py"},
                    {"path": "ascend/add.cpp"},
                    {"path": "ascend/add.h"},
                ],
            }
        ),
        encoding="utf-8",
    )


class AscendCProductionPolicyTests(unittest.TestCase):
    def test_directive_explains_self_authored_ascendc_boundary(self) -> None:
        directive = optimization_mode_directive("production", "Ascend-C")
        self.assertIn("self-authored AscendC AI Core kernel", directive)
        self.assertIn("ACLNN", directive)

    def test_accepts_declared_ascendc_sources_and_narrow_launch_glue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_candidate(workspace)
            self.assertEqual(production_kernel_violations(workspace, "AscendC"), [])

    def test_accepts_direct_checked_cmake_build_plumbing(self) -> None:
        kernel = KERNEL.replace(
            "import torch\n",
            "import torch\nimport subprocess\nfrom pathlib import Path\n",
        ).replace(
            'extension = load(name="atrex_ascendc_add", sources=["ascend/add.cpp"])',
            'subprocess.run(["cmake", "-S", str(Path(__file__).parent), '
            '"-B", str(Path(__file__).parent / "build")], check=True)\n'
            'subprocess.run(["cmake", "--build", '
            'str(Path(__file__).parent / "build")], check=True)',
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_candidate(workspace, kernel=kernel)
            self.assertEqual(production_kernel_violations(workspace, "AscendC"), [])

    def test_rejects_shell_or_unchecked_ascendc_subprocess(self) -> None:
        bad_calls = (
            'subprocess.run(["bash", "-c", "cmake ."], check=True)',
            'subprocess.run("cmake .", check=True, shell=True)',
            'subprocess.run(["cmake", "."])',
        )
        for call in bad_calls:
            with self.subTest(call=call), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                kernel = KERNEL.replace("import torch\n", "import torch\nimport subprocess\n")
                kernel = kernel.replace(
                    'extension = load(name="atrex_ascendc_add", sources=["ascend/add.cpp"])',
                    call,
                )
                _write_candidate(workspace, kernel=kernel)
                violations = production_kernel_violations(workspace, "AscendC")
                self.assertTrue(any("subprocess" in item.lower() for item in violations))

    def test_recovery_prompt_forbids_host_npu_import_and_shell_cmake(self) -> None:
        campaign = Campaign(
            name="ascend_recovery_prompt",
            kernel_demo="unused/reference.py",
            platform="Ascend910B1",
            framework="AscendC",
        )
        prompt = campaign._framework_baseline_recovery_constraints()
        self.assertIn("Do not probe the host", prompt)
        self.assertIn("importlib.util.find_spec('torch_npu')", prompt)
        self.assertIn('subprocess.run(["cmake"', prompt)
        self.assertIn("any non-`cmake` subprocess executable", prompt)

    def test_rejects_torch_npu_prebuilt_compute(self) -> None:
        kernel = KERNEL.replace(
            "torch.ops.atrex_ascendc.add(x, y, out)",
            "torch_npu.npu_rms_norm(x)",
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_candidate(workspace, kernel=kernel)
            violations = production_kernel_violations(workspace, "ascendc")
            self.assertTrue(any("Torch-NPU prebuilt compute" in item for item in violations))

    def test_rejects_pytorch_prebuilt_compute(self) -> None:
        kernel = KERNEL.replace(
            "torch.ops.atrex_ascendc.add(x, y, out)",
            "torch.add(x, y, out=out)",
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_candidate(workspace, kernel=kernel)
            violations = production_kernel_violations(workspace, "ascendc")
            self.assertTrue(any("PyTorch prebuilt compute" in item for item in violations))

    def test_rejects_aclnn_prebuilt_compute_in_auxiliary_source(self) -> None:
        source = ASCENDC_SOURCE + "\nvoid bypass() { aclnnMatmul(nullptr); }\n"
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_candidate(workspace, source=source)
            violations = production_kernel_violations(workspace, "ascendc")
            self.assertTrue(any("ACLNN prebuilt operator" in item for item in violations))

    def test_accepts_run_op_api_wrapping_a_self_authored_launch(self) -> None:
        source = ASCENDC_SOURCE + r"""
void launch(uint32_t block_dim, void *stream, GM_ADDR x, GM_ADDR y, GM_ADDR out) {
  auto launch_candidate = [=]() -> int {
    add_kernel<<<block_dim, nullptr, stream>>>(x, y, out, nullptr, nullptr);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("atrex_add", launch_candidate);
}
"""
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_candidate(workspace, source=source)
            self.assertEqual(production_kernel_violations(workspace, "ascendc"), [])

    def test_rejects_other_torch_npu_native_entry_points(self) -> None:
        source = ASCENDC_SOURCE + (
            "\nvoid bypass() { at_npu::native::NPUNativeFunctions::matmul(); }\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_candidate(workspace, source=source)
            violations = production_kernel_violations(workspace, "ascendc")
            self.assertTrue(any("Torch-NPU native prebuilt" in item for item in violations))

    def test_existing_triton_and_cuda_markers_remain_accepted(self) -> None:
        candidates = {
            "Triton": "import torch\nimport triton\n@triton.jit\ndef kernel(x):\n    return x\n",
            "Cuda": (
                "import torch\nfrom torch.utils.cpp_extension import load_inline\n"
                "source = '''extern \"C\" __global__ void kernel(float *x) {}'''\n"
                "extension = load_inline(name='atrex_cuda', cpp_sources='', "
                "cuda_sources=source, functions=None)\n"
            ),
        }
        for framework, kernel in candidates.items():
            with self.subTest(framework=framework), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                (workspace / "kernel.py").write_text(kernel, encoding="utf-8")
                self.assertEqual(production_kernel_violations(workspace, framework), [])


class DeclaredSourceBoundaryTests(unittest.TestCase):
    def test_dependency_reviewer_receives_declared_auxiliary_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_candidate(workspace)
            paths = {
                path.relative_to(workspace).as_posix()
                for path in _dependency_review_candidate_paths(workspace)
            }
        self.assertEqual(
            paths,
            {"kernel.py", "solution.json", "ascend/add.cpp", "ascend/add.h"},
        )

    def test_rejects_traversal_absolute_missing_and_symlink_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside.cpp"
            outside.write_text(ASCENDC_SOURCE, encoding="utf-8")

            bad_paths = ("../outside.cpp", str(outside), "missing.cpp")
            for source_path in bad_paths:
                (workspace / "solution.json").write_text(
                    json.dumps({"sources": [{"path": source_path}]}),
                    encoding="utf-8",
                )
                with self.subTest(source_path=source_path):
                    with self.assertRaises(ValueError):
                        declared_solution_source_paths(workspace)

            link = workspace / "linked.cpp"
            link.symlink_to(outside)
            (workspace / "solution.json").write_text(
                json.dumps({"sources": [{"path": "linked.cpp"}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                declared_solution_source_paths(workspace)

            (workspace / "solution.json").write_text(
                json.dumps({"sources": []}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "non-empty"):
                declared_solution_source_paths(workspace)

    def test_framework_baseline_commits_declared_sources_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = Campaign(
                name="ascend_policy_test",
                kernel_demo="unused/reference.py",
                platform="Ascend910B1",
                framework="AscendC",
                work_dir=str(root),
            )
            workspace = campaign.workspace
            workspace.mkdir()
            subprocess.run(["git", "init"], cwd=workspace, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(
                ["git", "config", "user.email", "policy-test@example.invalid"],
                cwd=workspace,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Policy Test"],
                cwd=workspace,
                check=True,
            )
            (workspace / "kernel.py").write_text("def run(*args):\n    return None\n", encoding="utf-8")
            (workspace / "solution.json").write_text(
                json.dumps({"sources": [{"path": "kernel.py"}]}),
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "--", "kernel.py", "solution.json"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "commit", "-m", "V0"],
                cwd=workspace,
                check=True,
                stdout=subprocess.DEVNULL,
            )

            _write_candidate(workspace)
            scratch = workspace / "scratch.txt"
            scratch.write_text("must stay out of the accepted commit\n", encoding="utf-8")
            subprocess.run(["git", "add", "--", "scratch.txt"], cwd=workspace, check=True)

            kernel_commit = campaign._commit_framework_baseline(
                1,
                {
                    "all_pass": True,
                    "latency_us_geomean": 1.0,
                    "latency_us_arith_mean": 1.0,
                    "latency_us_by_shape": {"shape": 1.0},
                },
            )
            for source in ("kernel.py", "ascend/add.cpp", "ascend/add.h"):
                shown = subprocess.run(
                    ["git", "cat-file", "-e", f"{kernel_commit}:{source}"],
                    cwd=workspace,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.assertEqual(shown.returncode, 0, source)
            scratch_in_commit = subprocess.run(
                ["git", "cat-file", "-e", f"{kernel_commit}:scratch.txt"],
                cwd=workspace,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.assertNotEqual(scratch_in_commit.returncode, 0)
            self.assertTrue(scratch.is_file())

    def test_framework_baseline_pins_commit_after_supervisor_stages_aux_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = Campaign(
                name="ascend_precommitted_kernel",
                kernel_demo="unused/reference.py",
                platform="Ascend910B1",
                framework="AscendC",
                work_dir=str(root),
            )
            workspace = campaign.workspace
            workspace.mkdir()
            subprocess.run(["git", "init"], cwd=workspace, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(
                ["git", "config", "user.email", "policy-test@example.invalid"],
                cwd=workspace,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Policy Test"],
                cwd=workspace,
                check=True,
            )
            (workspace / "kernel.py").write_text(
                "def run(*args):\n    return None\n", encoding="utf-8"
            )
            subprocess.run(["git", "add", "--", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "commit", "-m", "V0"],
                cwd=workspace,
                check=True,
                stdout=subprocess.DEVNULL,
            )

            _write_candidate(workspace)
            subprocess.run(["git", "add", "--", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "commit", "-m", "agent committed kernel only"],
                cwd=workspace,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            kernel_only_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            canonical_commit = campaign._commit_framework_baseline(
                1,
                {
                    "all_pass": True,
                    "latency_us_geomean": 1.0,
                    "latency_us_arith_mean": 1.0,
                    "latency_us_by_shape": {"shape": 1.0},
                },
            )

            self.assertNotEqual(canonical_commit, kernel_only_commit)
            for source in ("kernel.py", "solution.json", "ascend/add.cpp", "ascend/add.h"):
                shown = subprocess.run(
                    ["git", "cat-file", "-e", f"{canonical_commit}:{source}"],
                    cwd=workspace,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.assertEqual(shown.returncode, 0, source)

    def test_framework_baseline_without_solution_keeps_legacy_gpu_compatibility(self) -> None:
        cases = (("H20", "Triton"), ("MI308X", "FlyDSL"))
        for platform, framework in cases:
            with self.subTest(platform=platform, framework=framework), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                campaign = Campaign(
                    name="legacy_policy_test",
                    kernel_demo="unused/reference.py",
                    platform=platform,
                    framework=framework,
                    work_dir=str(root),
                )
                workspace = campaign.workspace
                workspace.mkdir()
                subprocess.run(
                    ["git", "init"], cwd=workspace, check=True, stdout=subprocess.DEVNULL
                )
                subprocess.run(
                    ["git", "config", "user.email", "policy-test@example.invalid"],
                    cwd=workspace,
                    check=True,
                )
                subprocess.run(
                    ["git", "config", "user.name", "Policy Test"],
                    cwd=workspace,
                    check=True,
                )
                (workspace / "kernel.py").write_text(
                    "def run(*args):\n    return None\n", encoding="utf-8"
                )
                subprocess.run(["git", "add", "--", "kernel.py"], cwd=workspace, check=True)
                subprocess.run(
                    ["git", "commit", "-m", "V0"],
                    cwd=workspace,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                (workspace / "kernel.py").write_text(
                    f"# {framework} candidate\ndef run(*args):\n    return None\n",
                    encoding="utf-8",
                )

                canonical_commit = campaign._commit_framework_baseline(
                    1,
                    {
                        "all_pass": True,
                        "latency_us_geomean": 1.0,
                        "latency_us_arith_mean": 1.0,
                        "latency_us_by_shape": {"shape": 1.0},
                    },
                )

                self.assertTrue(canonical_commit)
                self.assertFalse((workspace / "solution.json").exists())
                shown = subprocess.run(
                    ["git", "cat-file", "-e", f"{canonical_commit}:kernel.py"],
                    cwd=workspace,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.assertEqual(shown.returncode, 0)


if __name__ == "__main__":
    unittest.main()
