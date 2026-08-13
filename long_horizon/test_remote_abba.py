from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from long_horizon import remote_abba
from long_horizon.verifier import (
    _is_loopback_url,
    _required_allocation_timeout,
    _uses_cmake_candidate,
)
from tools import sandbox


class RemoteABBATests(unittest.TestCase):
    def test_revisions_use_stable_isolated_build_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshots = root / "verification_artifacts" / "case"
            snapshots.mkdir(parents=True)
            (snapshots / "incumbent.bin").write_text("incumbent", encoding="utf-8")
            (snapshots / "candidate.bin").write_text("candidate", encoding="utf-8")
            (root / "revision.txt").write_text("candidate", encoding="utf-8")
            (root / "test_kernel.py").write_text(
                """\
import json
from pathlib import Path

revision = Path('revision.txt').read_text().strip()
identity = Path('build_identity.txt')
if identity.exists() and identity.read_text().strip() != revision:
    raise SystemExit('build tree crossed revisions')
identity.write_text(revision)
latency = 20.0 if revision == 'incumbent' else 10.0
print('[test_kernel] RESULT_JSON=' + json.dumps({
    'all_pass': True,
    'latency_us_geomean': latency,
}))
""",
                encoding="utf-8",
            )
            request = {
                "schema_version": 1,
                "schedule": [
                    {"revision": "incumbent", "repeat": 0},
                    {"revision": "candidate", "repeat": 0},
                    {"revision": "candidate", "repeat": 1},
                    {"revision": "incumbent", "repeat": 1},
                ],
                "manifests": {
                    "incumbent": {"revision.txt": "incumbent.bin"},
                    "candidate": {"revision.txt": "candidate.bin"},
                },
                "command": [sys.executable, "test_kernel.py"],
                "run_timeout_seconds": 5,
                "cold_start_grace_seconds": 1,
            }
            request_path = snapshots / "request.json"
            result_path = snapshots / "result.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            previous = Path.cwd()
            timeouts: list[int] = []
            actual_run = remote_abba.subprocess.run

            def recording_run(*args: object, **kwargs: object):
                timeouts.append(int(kwargs["timeout"]))
                return actual_run(*args, **kwargs)

            try:
                import os

                os.chdir(root)
                with mock.patch.object(
                    remote_abba.subprocess, "run", side_effect=recording_run
                ):
                    self.assertEqual(remote_abba.run(request_path, result_path), 0)
            finally:
                os.chdir(previous)

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertIsNone(result["error"])
            self.assertEqual([row["exit_code"] for row in result["runs"]], [0, 0, 0, 0])
            self.assertEqual(
                [row["result"]["latency_us_geomean"] for row in result["runs"]],
                [20.0, 10.0, 10.0, 20.0],
            )
            self.assertEqual(timeouts, [66, 66, 66, 66])

    def test_local_cmake_verifier_budget_includes_per_run_build_grace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "solution.json").write_text(
                json.dumps({"sources": ["kernel.py", {"path": "CMakeLists.txt"}]}),
                encoding="utf-8",
            )
            self.assertTrue(_uses_cmake_candidate(root))
        self.assertTrue(_is_loopback_url("http://127.0.0.1:8000"))
        self.assertTrue(_is_loopback_url("http://[::1]:8000"))
        self.assertFalse(_is_loopback_url("https://gateway.example"))
        self.assertEqual(_required_allocation_timeout(4, 120, 240), 1710)
        self.assertEqual(sandbox.LEGACY_LOCAL_ABBA_TIMEOUT, 1710)


if __name__ == "__main__":
    unittest.main()
