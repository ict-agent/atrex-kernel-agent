"""Coding-agent session execution and sandbox I/O.

Owns session spawning and accounting, the independent dependency review, sandbox command
construction, and evaluator result parsing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import agent_runtime as _agent_runtime
from .constants import (
    DEFAULT_SANDBOX_TIMEOUT,
    DEPENDENCY_REVIEW_SCHEMA_VERSION,
    HUMANIZE_DIR,
    REPO_ROOT,
    SANDBOX_DIRECTIVE_PROMPT,
    SANDBOX_TOOL,
    TEST_RESULT_PREFIX,
)
from .hardware import normalize_arch_token
from .optimization_policy import DependencyReviewSignal, declared_solution_source_paths


def _status_is(value: object, expected: str) -> bool:
    """Accept a status even when a CLI accidentally stored it as a JSON-quoted string."""
    current = value
    for _ in range(2):
        if current == expected:
            return True
        if not isinstance(current, str):
            return False
        try:
            decoded = json.loads(current)
        except json.JSONDecodeError:
            return current.strip() == expected
        if decoded == current:
            return False
        current = decoded
    return current == expected


@dataclass
class SessionResult:
    exit_status: int
    timed_out: bool
    tokens: int
    stdout_tail: str
    stderr_tail: str
    session_id: str = ""
    terminal_usage: _agent_runtime.TokenUsage | None = None
    events: tuple[_agent_runtime.NormalizedAgentEvent, ...] = ()
    capabilities: _agent_runtime.AgentRuntimeCapabilities | None = None
    observation_errors: tuple[str, ...] = ()


def _render(template_path: Path, **kw: str) -> str:
    text = template_path.read_text(encoding="utf-8")
    mode_policy = kw.pop("MODE_POLICY", "")
    for key, val in kw.items():
        text = text.replace("{{" + key + "}}", str(val))
    if mode_policy:
        text = str(mode_policy).rstrip() + "\n\n" + text
    return text


def _find_jq() -> Optional[str]:
    found = shutil.which("jq")
    if found:
        return found
    adjacent = Path(sys.executable).resolve().parent / "jq"
    if adjacent.is_file() and os.access(adjacent, os.X_OK):
        return str(adjacent)
    return None


def ensure_jq() -> str:
    """Install jq with an available package manager when the runtime lacks it."""
    found = _find_jq()
    if found:
        return found

    privileged_prefix: list[str] | None
    if getattr(os, "geteuid", lambda: 1)() == 0:
        privileged_prefix = []
    elif shutil.which("sudo"):
        privileged_prefix = ["sudo"]
    else:
        privileged_prefix = None

    installers: list[tuple[str, list[str], dict[str, str] | None]] = []
    system_commands = (
        ("apt-get", ["apt-get", "install", "-y", "jq"]),
        ("dnf", ["dnf", "install", "-y", "jq"]),
        ("yum", ["yum", "install", "-y", "jq"]),
        ("apk", ["apk", "add", "jq"]),
        ("zypper", ["zypper", "--non-interactive", "install", "jq"]),
    )
    if privileged_prefix is not None:
        for manager, command in system_commands:
            if shutil.which(manager):
                installers.append((manager, [*privileged_prefix, *command], None))
    if shutil.which("brew"):
        installers.append(("brew", ["brew", "install", "jq"], None))
    if shutil.which("conda"):
        conda_env = os.environ.copy()
        conda_env["CONDA_SOLVER"] = "classic"
        installers.append(
            (
                "conda",
                ["conda", "install", "-y", "-c", "conda-forge", "jq"],
                conda_env,
            )
        )

    failures: list[str] = []
    for manager, command, environment in installers:
        print(f"[orchestrator] jq not found; installing with {manager}", flush=True)
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"{manager}: {exc}")
            continue
        found = _find_jq()
        if completed.returncode == 0 and found:
            jq_dir = str(Path(found).resolve().parent)
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            if jq_dir not in path_parts:
                os.environ["PATH"] = os.pathsep.join([jq_dir, *path_parts])
            print(f"[orchestrator] jq installed with {manager}", flush=True)
            return found
        output_lines = (completed.stdout or "").strip().splitlines()
        detail = output_lines[-1] if output_lines else f"exit {completed.returncode}"
        failures.append(f"{manager}: {detail}")

    detail = "; ".join(failures) if failures else "no supported package manager found"
    raise RuntimeError(f"jq is required and automatic installation failed: {detail}")


def ensure_submodules() -> None:
    """Initialize submodules and host tools required by the optimization pipeline.

    Covers: gpu-wiki/3rdparty (KernelWiki), 3rdparty/ncu-report-skill, 3rdparty/humanize.
    Skips reference-projects (large, optional — only needed for L2 search).
    Idempotent: already-initialized submodules are untouched.
    """
    needed = [
        ("gpu-wiki/3rdparty/", REPO_ROOT / "gpu-wiki" / "3rdparty" / "KernelWiki" / "README.md"),
        ("3rdparty/ncu-report-skill", REPO_ROOT / "3rdparty" / "ncu-report-skill" / "SKILL.md"),
        ("3rdparty/humanize", HUMANIZE_DIR / "skills" / "humanize-gen-plan" / "SKILL.md"),
    ]
    to_init = [path for path, marker in needed if not marker.exists()]
    if to_init:
        print(f"[orchestrator] initializing submodules: {to_init}", flush=True)
        cmd = ["git", "submodule", "update", "--init", "--depth", "1", "--"] + to_init
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
        # verify
        for path, marker in needed:
            if not marker.exists():
                raise RuntimeError(
                    f"submodule init failed for {path} — {marker} not found. "
                    "Run `git submodule update --init` manually."
                )
        print("[orchestrator] all submodules ready", flush=True)
    ensure_jq()


def run_session(
    workspace: Path,
    prompt: str,
    timeout: int,
    agent_cli: str = "claude",
    sandbox_hardware: str = "",
    sandbox_profile: str = "",
    sandbox_url: str = "",
    sandbox_timeout: int = DEFAULT_SANDBOX_TIMEOUT,
    reasoning_effort: str = "max",
    extra_environment: Optional[dict[str, str]] = None,
    agent_plugins: bool = True,
) -> SessionResult:
    """Run one clean coding-agent session with no conversational memory from prior iterations."""
    session_id = str(uuid.uuid4())
    runtime = _agent_runtime.build_agent_runtime(
        agent_cli,
        process_runner=_agent_runtime.run_bounded,
        humanize_dir=(
            HUMANIZE_DIR
            if agent_plugins
            else workspace / ".atrex-disabled-agent-plugins"
        ),
    )
    result = runtime.run(
        _agent_runtime.AgentRunRequest(
            workspace=workspace,
            prompt=prompt,
            timeout_s=timeout,
            reasoning_effort=reasoning_effort,
            sandbox_hardware=sandbox_hardware,
            sandbox_profile=sandbox_profile,
            sandbox_url=sandbox_url,
            sandbox_timeout_s=sandbox_timeout,
            session_id=session_id,
            extra_environment=extra_environment,
        )
    )
    return SessionResult(
        exit_status=result.exit_status,
        timed_out=result.timed_out,
        tokens=result.tokens,
        stdout_tail=result.stdout_tail,
        stderr_tail=result.stderr_tail,
        session_id=result.session_id,
        terminal_usage=result.terminal_usage,
        events=result.events,
        capabilities=result.capabilities,
        observation_errors=result.observation_errors,
    )


_DEPENDENCY_ALLOW_CATEGORIES = {
    "toolchain_plumbing",
    "framework_runtime",
    "support_utility",
}
_DEPENDENCY_REJECT_CATEGORIES = {
    "prebuilt_compute",
    "alternate_framework",
    "hidden_dispatch",
    "external_code",
    "unresolved",
}


def _dependency_review_candidate_paths(workspace: Path) -> list[Path]:
    """Return the complete, bounded source set shown to the dependency reviewer."""
    paths = [
        workspace / "kernel.py",
        workspace / "solution.json",
    ]
    try:
        declared = declared_solution_source_paths(workspace)
    except ValueError:
        # The mechanical policy pass reports the invalid manifest. Do not follow
        # untrusted paths while collecting evidence for the independent reviewer.
        declared = ()
    paths.extend(workspace / relative for relative in declared)
    paths = list(dict.fromkeys(paths))
    return [path for path in paths if path.is_file()]


def _dependency_review_digest(
    workspace: Path,
    framework: str,
    signals: tuple[DependencyReviewSignal, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"dependency-review-v{DEPENDENCY_REVIEW_SCHEMA_VERSION}\0".encode())
    digest.update(framework.encode("utf-8", errors="replace"))
    for review_signal in signals:
        digest.update(
            json.dumps(
                {
                    "id": review_signal.id,
                    "kind": review_signal.kind,
                    "value": review_signal.value,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
    for path in _dependency_review_candidate_paths(workspace):
        relative = path.relative_to(workspace).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_dependency_review(
    payload: object,
    signals: tuple[DependencyReviewSignal, ...],
) -> tuple[list[str], str]:
    """Validate an agent verdict and translate rejected items into policy errors."""
    if not isinstance(payload, dict):
        raise ValueError("dependency review must be a JSON object")
    if payload.get("schema_version") != DEPENDENCY_REVIEW_SCHEMA_VERSION:
        raise ValueError("dependency review has an unsupported schema_version")
    verdict = payload.get("verdict")
    if verdict not in {"allow", "reject"}:
        raise ValueError("dependency review verdict must be allow or reject")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("dependency review summary must be non-empty")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("dependency review items must be a list")

    expected = {review_signal.id for review_signal in signals}
    reviewed: dict[str, dict] = {}
    rejected: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("dependency review item must be an object")
        signal_id = item.get("id")
        if not isinstance(signal_id, str) or signal_id not in expected:
            raise ValueError(f"dependency review returned unexpected signal id: {signal_id!r}")
        if signal_id in reviewed:
            raise ValueError(f"dependency review duplicated signal id: {signal_id}")
        decision = item.get("decision")
        category = item.get("category")
        reason = item.get("reason")
        evidence = item.get("evidence")
        if decision not in {"allow", "reject"}:
            raise ValueError(f"dependency review decision is invalid for {signal_id}")
        categories = (
            _DEPENDENCY_ALLOW_CATEGORIES
            if decision == "allow"
            else _DEPENDENCY_REJECT_CATEGORIES
        )
        if category not in categories:
            raise ValueError(
                f"dependency review category {category!r} is inconsistent with "
                f"decision {decision!r} for {signal_id}"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"dependency review reason is empty for {signal_id}")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(value, str) and value.strip() for value in evidence
        ):
            raise ValueError(f"dependency review evidence is invalid for {signal_id}")
        reviewed[signal_id] = item
        if decision == "reject":
            rejected.append(
                "third-party dependency rejected by independent agent: "
                f"{signal_id}: {reason.strip()}"
            )

    missing = sorted(expected - set(reviewed))
    if missing:
        raise ValueError(
            "dependency review omitted signal ids: " + ", ".join(missing)
        )
    expected_verdict = "reject" if rejected else "allow"
    if verdict != expected_verdict:
        raise ValueError(
            f"dependency review verdict {verdict!r} disagrees with item decisions"
        )
    return rejected, summary.strip()


def sandbox_directive(hardware: str, profile: str = "", url: str = "") -> str:
    """Mandatory execution boundary injected into every optimization session."""
    if url:
        endpoint = f" using gateway URL `{url}`"
    elif profile:
        endpoint = f" using gateway profile `{profile}`"
    else:
        endpoint = " using agate's configured gateway"
    return _render(
        SANDBOX_DIRECTIVE_PROMPT, HARDWARE=hardware, ENDPOINT=endpoint
    )


def _sandbox_command(
    workspace: Path,
    hardware: str,
    profile: str,
    url: str,
    timeout: int,
    command: list[str],
    *,
    sync: tuple[str, ...] = (),
    wall_timeout: Optional[int] = None,
    gateway_kind: str = "auto",
) -> subprocess.CompletedProcess[str]:
    """Run one command through tools/sandbox.py and capture its user-visible output."""
    cmd = [
        sys.executable, str(SANDBOX_TOOL),
        "--kind", gateway_kind,
        "--hardware", hardware,
        "--workspace", str(workspace),
        "--timeout", str(timeout),
    ]
    if url:
        cmd += ["--url", url]
    elif profile:
        cmd += ["--gateway-profile", profile]
    if sync:
        for path in sync:
            cmd += ["--sync", path]
    else:
        cmd.append("--no-sync")
    cmd += ["--", *command]
    return subprocess.run(
        cmd,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        # Gateway execution timeout starts only after a worker claims the job.
        # The local wait must additionally tolerate time spent in a shared queue.
        timeout=wall_timeout if wall_timeout is not None else timeout + 240,
    )


def _test_result_from_stdout(stdout: str) -> dict:
    """Read the structured result emitted by the active sandbox harness."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(TEST_RESULT_PREFIX):
            result = json.loads(line[len(TEST_RESULT_PREFIX):])
            if isinstance(result, dict):
                return result
    raise RuntimeError("sandbox test output has no structured RESULT_JSON line")


def _record_local_test_result(workspace: Path, version: str, result: dict) -> Path:
    """Merge a remote --no-memory test result into local optimizer memory."""
    mem_dir = workspace / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = mem_dir / f"{version}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    data.setdefault("version", version)
    data.setdefault("masked", False)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    perf = data.setdefault("performance", {})
    perf["latency_us"] = result.get("latency_us_geomean", 0.0)
    perf["latency_us_geomean"] = result.get("latency_us_geomean", 0.0)
    perf["latency_us_arith_mean"] = result.get("latency_us_arith_mean", 0.0)
    perf["latency_us_by_shape"] = result.get("latency_us_by_shape", {})
    perf["speedup_vs_ref_geomean"] = result.get("speedup_vs_ref_geomean", 0.0)
    all_pass = bool(result.get("all_pass"))
    corr = data.setdefault("correctness", {})
    corr["status"] = "PASS" if all_pass else "FAIL"
    corr["max_abs_err"] = result.get("max_abs_err", 0.0)
    corr["max_rel_err"] = result.get("max_rel_err", 0.0)
    gate = data.setdefault("quality_gate", {})
    gate["result"] = "PASS" if all_pass else "FAIL"
    failures = result.get("failures") or []
    gate["failure_reason"] = None if all_pass else "; ".join(map(str, failures))[:2000]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


_RUNTIME_ARCH_PROBE = """\
import torch

try:
    import torch_npu
except Exception:
    torch_npu = None


def npu_device_name(api):
    if api is None:
        return ""
    getter = getattr(api, "get_device_name", None)
    if callable(getter):
        for args in ((0,), ()):
            try:
                value = getter(*args)
            except Exception:
                continue
            if value:
                return str(value)
    getter = getattr(api, "get_device_properties", None)
    if callable(getter):
        try:
            properties = getter(0)
        except Exception:
            properties = None
        for attribute in ("name", "device_name"):
            value = getattr(properties, attribute, "")
            if value:
                return str(value)
    return ""


for npu_api in (
    getattr(torch, "npu", None),
    getattr(torch_npu, "npu", None) if torch_npu is not None else None,
):
    device_name = npu_device_name(npu_api)
    if device_name:
        print(device_name)
        raise SystemExit(0)

properties = torch.cuda.get_device_properties(0)
if getattr(torch.version, "hip", None):
    print(getattr(properties, "gcnArchName", "").split(":")[0])
else:
    capability = torch.cuda.get_device_capability(0)
    print("sm_%d%d" % (capability[0], capability[1]))
"""

_ASCEND_910B_OUTPUT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:(?:Huawei[ \t_-]*)?Ascend[ \t_-]*)?"
    r"910B(?:[ \t_-]*1)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _arch_from_probe_output(output: str) -> str:
    """Extract and normalize a supported architecture from probe output."""
    for line in reversed(output.splitlines()):
        if normalized := normalize_arch_token(line):
            return normalized
    if match := _ASCEND_910B_OUTPUT_RE.search(output):
        return normalize_arch_token(match.group(0))
    return ""


def _completed_process_arch(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode != 0:
        return ""
    return _arch_from_probe_output(result.stdout) or _arch_from_probe_output(result.stderr)


def detect_arch(
    sandbox_hardware: str = "",
    sandbox_profile: str = "",
    sandbox_url: str = "",
) -> str:
    """Return the normalized runtime accelerator architecture, or ``''``.

    NVIDIA/CUDA -> ``sm_<cap>``; AMD/ROCm -> ``gfx...``; an exact Ascend 910B1
    name -> ``ascend910b1`` (a family-only name remains ``ascend910b``). The
    PyTorch runtime API is authoritative: Ascend probes
    ``torch.npu``/``torch_npu.npu`` before CUDA/ROCm. ``npu-smi info`` is used only
    as an Ascend fallback when the Python runtime probe is unavailable.
    """
    commands = (
        ("PyTorch runtime", ["python", "-c", _RUNTIME_ARCH_PROBE]),
        ("npu-smi fallback", ["npu-smi", "info"]),
    )
    if sandbox_hardware:
        failures: list[str] = []
        with tempfile.TemporaryDirectory(prefix="atrex-arch-") as temp_dir:
            for label, command in commands:
                try:
                    result = _sandbox_command(
                        Path(temp_dir), sandbox_hardware, sandbox_profile, sandbox_url, 120,
                        command,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    failures.append(f"{label}: {exc}")
                    continue
                if arch := _completed_process_arch(result):
                    return arch
                detail = (result.stderr or result.stdout).strip()[-1000:]
                failures.append(f"{label}: {detail or f'exit {result.returncode}'}")
        print(
            f"[orchestrator] WARNING: sandbox arch detection failed on {sandbox_hardware}: "
            + "; ".join(failures),
            file=sys.stderr,
            flush=True,
        )
        return ""

    seen_python: set[str] = set()
    for python_executable in ("python", "python3", sys.executable):
        if python_executable in seen_python:
            continue
        seen_python.add(python_executable)
        try:
            result = subprocess.run(
                [python_executable, "-c", _RUNTIME_ARCH_PROBE],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if arch := _completed_process_arch(result):
            return arch

    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return _completed_process_arch(result)
