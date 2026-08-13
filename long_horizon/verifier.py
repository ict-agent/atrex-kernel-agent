from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from . import main_adapter
from .git_episode import _git
from .models import VerificationResult, VerificationRun
from .protocol import atomic_write_json
from .store import VERIFY_DIR


ABBA_RESULT_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="
COLD_BUILD_GRACE_SECONDS = 240
RUN_TEARDOWN_GRACE_SECONDS = 60
VERIFIER_TEARDOWN_GRACE_SECONDS = 30
MAX_LOCAL_ALLOCATION_TIMEOUT = 1800


def _is_loopback_url(value: str) -> bool:
    if not value:
        return False
    try:
        hostname = urlsplit(value).hostname
    except ValueError:
        return False
    return hostname in {"127.0.0.1", "::1", "localhost"}


def _uses_cmake_candidate(workspace: Path) -> bool:
    solution_path = workspace / "solution.json"
    try:
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    sources = solution.get("sources") if isinstance(solution, dict) else None
    if not isinstance(sources, list):
        return False
    paths = {
        entry if isinstance(entry, str) else entry.get("path")
        for entry in sources
        if isinstance(entry, (str, dict))
    }
    return "CMakeLists.txt" in paths


def _required_allocation_timeout(
    schedule_size: int,
    per_run_timeout: int,
    cold_start_grace: int,
) -> int:
    # Every timed evaluator gets both the remote driver's fixed teardown grace
    # and the CANN compiler-discovery allowance. AscendC can repeat that
    # discovery in each evaluator process even with a warm CMake directory.
    return (
        schedule_size
        * (per_run_timeout + RUN_TEARDOWN_GRACE_SECONDS + cold_start_grace)
        + VERIFIER_TEARDOWN_GRACE_SECONDS
    )


def _payload_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract the long-horizon ABBA payload from ordinary sandbox stdout."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith(ABBA_RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(ABBA_RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise ValueError("malformed ABBA result sentinel") from exc
        if not isinstance(payload, dict):
            raise ValueError("ABBA result sentinel must contain a JSON object")
        return payload
    raise ValueError("missing ABBA result sentinel")


def verification_schedule(repeats: int) -> list[dict[str, int | str]]:
    schedule: list[dict[str, int | str]] = []
    for repeat in range(max(1, repeats)):
        revisions = ("incumbent", "candidate") if repeat % 2 == 0 else ("candidate", "incumbent")
        schedule.extend({"revision": revision, "repeat": repeat} for revision in revisions)
    return schedule


def _geomean(values: list[float]) -> float | None:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _result_latency(result: object) -> float | None:
    if not isinstance(result, dict) or not result.get("all_pass"):
        return None
    value = result.get("latency_us_geomean")
    if not isinstance(value, (int, float)):
        performance = result.get("performance")
        value = performance.get("latency_us") if isinstance(performance, dict) else None
    if not isinstance(value, (int, float)) or value <= 0 or not math.isfinite(float(value)):
        return None
    return float(value)


def score_verification_payload(
    payload: object,
    *,
    schedule: list[dict[str, int | str]],
    repeats: int,
    min_improvement_pct: float,
    artifact: str = "",
) -> VerificationResult:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return VerificationResult("ERROR", None, None, None, error="unsupported result schema")
    rows = payload.get("runs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return VerificationResult("ERROR", None, None, None, error="runs must be a list of objects")
    runs = [
        VerificationRun(
            revision=str(row.get("revision", "")),
            repeat=int(row.get("repeat", -1)),
            exit_code=int(row.get("exit_code", -1)),
            result=row.get("result") if isinstance(row.get("result"), dict) else None,
            stdout_tail=str(row.get("stdout_tail", "")),
            stderr_tail=str(row.get("stderr_tail", "")),
        )
        for row in rows
    ]
    if payload.get("error"):
        return VerificationResult(
            "ERROR", None, None, None, runs=runs, error=str(payload["error"]), artifact=artifact
        )
    actual = [{"revision": run.revision, "repeat": run.repeat} for run in runs]
    if actual != schedule:
        return VerificationResult(
            "ERROR", None, None, None, runs=runs,
            error="remote verifier did not execute the exact ABBA schedule", artifact=artifact,
        )
    candidate_values = [
        value for run in runs if run.revision == "candidate" and run.exit_code == 0
        if (value := _result_latency(run.result)) is not None
    ]
    incumbent_values = [
        value for run in runs if run.revision == "incumbent" and run.exit_code == 0
        if (value := _result_latency(run.result)) is not None
    ]
    candidate = _geomean(candidate_values)
    incumbent = _geomean(incumbent_values)
    if len(candidate_values) != repeats or len(incumbent_values) != repeats:
        return VerificationResult(
            "FAIL", candidate, incumbent, None, runs=runs,
            error="not every authoritative ABBA run passed", artifact=artifact,
        )
    improvement = ((incumbent - candidate) / incumbent * 100.0) if candidate and incumbent else None
    if improvement is None or improvement <= min_improvement_pct:
        return VerificationResult(
            "FAIL", candidate, incumbent, improvement, runs=runs,
            error=(
                f"candidate improvement {improvement if improvement is not None else 'unknown'} "
                f"did not exceed {min_improvement_pct:.3f}%"
            ),
            artifact=artifact,
        )
    return VerificationResult(
        "PASS", candidate, incumbent, improvement, runs=runs, artifact=artifact
    )


def _git_blob(workspace: Path, revision: str, relative: str) -> bytes | None:
    result = _git(workspace, "show", f"{revision}:{relative}", check=False, binary=True)
    return result.stdout if result.returncode == 0 else None


def _snapshot_revision(
    workspace: Path,
    revision: str,
    changed_paths: list[str],
    destination: Path,
    label: str,
) -> dict[str, str | None]:
    manifest: dict[str, str | None] = {}
    for index, relative in enumerate(changed_paths):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe candidate path: {relative}")
        content = _git_blob(workspace, revision, relative)
        if content is None:
            manifest[relative] = None
            continue
        snapshot_relative = f"snapshots/{label}/{index:04d}.bin"
        target = destination / snapshot_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest[relative] = snapshot_relative
    return manifest


class GatewayABBAValidator:
    def __init__(
        self,
        *,
        hardware: str,
        profile: str = "",
        url: str = "",
        timeout: int = 600,
        repeats: int = 2,
        per_run_timeout: int = 120,
        min_improvement_pct: float = 0.0,
        queue_wait_grace: int = 14_400,
    ):
        self.hardware = hardware
        self.profile = profile
        self.url = url
        self.timeout = timeout
        self.repeats = max(1, repeats)
        self.per_run_timeout = per_run_timeout
        self.min_improvement_pct = min_improvement_pct
        self.queue_wait_grace = queue_wait_grace

    def verify(
        self,
        workspace: Path,
        *,
        base_commit: str,
        candidate_commit: str,
        changed_paths: list[str],
    ) -> VerificationResult:
        schedule = verification_schedule(self.repeats)
        cold_start_grace = COLD_BUILD_GRACE_SECONDS if _uses_cmake_candidate(workspace) else 0
        required_timeout = _required_allocation_timeout(
            len(schedule), self.per_run_timeout, cold_start_grace
        )
        allocation_timeout = self.timeout
        if required_timeout > allocation_timeout and _is_loopback_url(self.url):
            allocation_timeout = min(required_timeout, MAX_LOCAL_ALLOCATION_TIMEOUT)
        if required_timeout > allocation_timeout:
            return VerificationResult(
                "ERROR", None, None, None,
                error=(
                    "ABBA schedule cannot fit in one gateway allocation timeout: "
                    f"requires {required_timeout}s including cold-build allowance, "
                    f"gateway budget is {allocation_timeout}s"
                ),
            )
        verification_id = uuid.uuid4().hex
        relative_dir = f"{VERIFY_DIR}/{verification_id}"
        directory = workspace / relative_dir
        directory.mkdir(parents=True, exist_ok=False)
        # Naming the driver test_kernel.py deliberately selects the sandbox's
        # immutable-evaluator payload route. The runtime bundler includes the verification
        # driver and snapshots from their dedicated artifact directory.
        driver = directory / "test_kernel.py"
        shutil.copy2(Path(__file__).with_name("remote_abba.py"), driver)
        manifests = {
            "incumbent": _snapshot_revision(
                workspace, base_commit, changed_paths, directory, "incumbent"
            ),
            "candidate": _snapshot_revision(
                workspace, candidate_commit, changed_paths, directory, "candidate"
            ),
        }
        request_relative = f"{relative_dir}/request.json"
        result_relative = f"{relative_dir}/result.json"
        request = {
            "schema_version": 1,
            "schedule": schedule,
            "manifests": manifests,
            "command": ["python3", "test_kernel.py", "--version", "vlong", "--no-memory"],
            "run_timeout_seconds": self.per_run_timeout,
            "cold_start_grace_seconds": cold_start_grace,
        }
        atomic_write_json(workspace / request_relative, request)
        try:
            process = main_adapter.run_sandbox(
                workspace,
                self.hardware,
                self.profile,
                self.url,
                allocation_timeout,
                [
                    "python3",
                    f"{relative_dir}/test_kernel.py",
                    request_relative,
                    result_relative,
                ],
                # The result is emitted with a long-horizon-only stdout
                # sentinel.  No artifact is synchronized, avoiding the
                # gateway's elided-payload/base64 transport failure.
                sync=(),
                wall_timeout=allocation_timeout + self.queue_wait_grace + 120,
                gateway_kind="dev",
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                "ERROR", None, None, None, error="gateway ABBA verifier timed out"
            )
        if process.returncode != 0:
            return VerificationResult(
                "ERROR", None, None, None,
                error=f"gateway ABBA command exited {process.returncode}: "
                + (process.stdout + "\n" + process.stderr)[-3000:],
            )
        try:
            payload = _payload_from_stdout(process.stdout)
        except ValueError as exc:
            return VerificationResult(
                "ERROR", None, None, None,
                error=f"gateway returned no valid ABBA result: {exc}",
            )
        result_path = workspace / result_relative
        try:
            atomic_write_json(result_path, payload)
        except (OSError, TypeError, ValueError) as exc:
            return VerificationResult(
                "ERROR", None, None, None,
                error=f"cannot persist gateway ABBA result: {type(exc).__name__}: {exc}",
            )
        return score_verification_payload(
            payload,
            schedule=schedule,
            repeats=self.repeats,
            min_improvement_pct=self.min_improvement_pct,
            artifact=str(result_path),
        )
