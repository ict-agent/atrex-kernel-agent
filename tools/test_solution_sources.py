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
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference import sol_finalize, test_kernel
from tools import sandbox


class SolutionSourceSchemaTests(unittest.TestCase):
    def _workspace(self, root: Path) -> Path:
        workspace = root / "workspace"
        (workspace / "ascend").mkdir(parents=True)
        (workspace / "kernel.py").write_text("def run():\n    return 1\n", encoding="utf-8")
        (workspace / "ascend" / "add.cpp").write_text(
            "extern \"C\" void add_custom() {}\n", encoding="utf-8"
        )
        (workspace / "reference.py").write_text("def reference():\n    pass\n", encoding="utf-8")
        (workspace / "input.py").write_text("def inputs():\n    pass\n", encoding="utf-8")
        (workspace / "shapes.json").write_text('{"0": {}}\n', encoding="utf-8")
        return workspace

    @staticmethod
    def _write_solution(workspace: Path, sources: object) -> None:
        (workspace / "solution.json").write_text(
            json.dumps({"spec": {"languages": ["Python"]}, "sources": sources}),
            encoding="utf-8",
        )

    def test_sandbox_accepts_both_forms_and_detects_string_auxiliary_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self._workspace(Path(temporary))
            sources = ["kernel.py", {"path": "ascend/add.cpp"}]
            self._write_solution(workspace, sources)

            self.assertEqual(
                sandbox._solution_source_paths(workspace, {"sources": sources}),
                ("kernel.py", "ascend/add.cpp"),
            )
            self.assertEqual(
                sandbox._typed_workspace_limitation(
                    workspace, ["python", "test_kernel.py"]
                ),
                "solution.json declares auxiliary candidate sources",
            )
            self.assertEqual(
                sandbox._declared_candidate_sources(workspace),
                {"kernel.py", "ascend/add.cpp"},
            )

    def test_typed_route_allows_string_kernel_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self._workspace(Path(temporary))
            self._write_solution(workspace, ["kernel.py"])

            self.assertIsNone(
                sandbox._typed_workspace_limitation(
                    workspace, ["python", "test_kernel.py"]
                )
            )

    def test_submission_builders_inline_and_normalize_both_forms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self._workspace(Path(temporary))
            self._write_solution(
                workspace,
                [
                    "kernel.py",
                    {
                        "path": "ascend/add.cpp",
                        "language": "AscendC",
                        "content": "stale",
                    },
                ],
            )

            submission_path = test_kernel._write_submission(workspace)
            self.assertIsNotNone(submission_path)
            harness_submission = json.loads(
                submission_path.read_text(encoding="utf-8")
            )
            finalized_submission = sol_finalize.build_submission(workspace)

        self.assertEqual(harness_submission, finalized_submission)
        self.assertEqual(
            harness_submission["sources"],
            [
                {
                    "path": "kernel.py",
                    "content": "def run():\n    return 1\n",
                },
                {
                    "path": "ascend/add.cpp",
                    "language": "AscendC",
                    "content": 'extern "C" void add_custom() {}\n',
                },
            ],
        )

    def test_invalid_entries_fail_closed_in_all_consumers(self) -> None:
        invalid_sources = (
            [None],
            [{"content": "missing path"}],
            [{"path": 3}],
            [""],
            [" kernel.py"],
            ["./kernel.py"],
            ["../kernel.py"],
            ["/tmp/kernel.py"],
            ["ascend\\add.cpp"],
            ["missing.cpp"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self._workspace(Path(temporary))
            for sources in invalid_sources:
                with self.subTest(sources=sources):
                    self._write_solution(workspace, sources)
                    solution = json.loads(
                        (workspace / "solution.json").read_text(encoding="utf-8")
                    )
                    with self.assertRaises(ValueError):
                        sandbox._solution_source_paths(workspace, solution)
                    with self.assertRaises(ValueError):
                        sandbox._typed_workspace_limitation(
                            workspace, ["python", "test_kernel.py"]
                        )
                    with self.assertRaises(SystemExit):
                        test_kernel._write_submission(workspace)
                    with self.assertRaises(SystemExit):
                        sol_finalize.build_submission(workspace)

    def test_invalid_container_and_empty_sources_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self._workspace(Path(temporary))
            for sources in (None, {}, "kernel.py"):
                with self.subTest(sources=sources):
                    self._write_solution(workspace, sources)
                    with self.assertRaises(ValueError):
                        sandbox._declared_candidate_sources(workspace)
                    with self.assertRaises(SystemExit):
                        test_kernel._write_submission(workspace)
                    with self.assertRaises(SystemExit):
                        sol_finalize.build_submission(workspace)

            self._write_solution(workspace, [])
            with self.assertRaises(ValueError):
                sandbox._declared_candidate_sources(workspace)
            with self.assertRaises(SystemExit):
                test_kernel._write_submission(workspace)
            with self.assertRaises(SystemExit):
                sol_finalize.build_submission(workspace)

    def test_invalid_solution_document_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self._workspace(Path(temporary))
            for document in ("[]", "{not-json"):
                with self.subTest(document=document):
                    (workspace / "solution.json").write_text(document, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        sandbox._declared_candidate_sources(workspace)
                    with self.assertRaises(SystemExit):
                        test_kernel._write_submission(workspace)
                    with self.assertRaises(SystemExit):
                        sol_finalize.build_submission(workspace)

    def test_symlink_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self._workspace(Path(temporary))
            (workspace / "linked.cpp").symlink_to(workspace / "ascend" / "add.cpp")
            self._write_solution(workspace, ["linked.cpp"])

            with self.assertRaisesRegex(ValueError, "symlink"):
                sandbox._declared_candidate_sources(workspace)
            with self.assertRaisesRegex(SystemExit, "symlink"):
                test_kernel._write_submission(workspace)
            with self.assertRaisesRegex(SystemExit, "symlink"):
                sol_finalize.build_submission(workspace)


if __name__ == "__main__":
    unittest.main()
