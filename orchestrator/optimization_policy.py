"""Optimization-mode policy and mechanical production-kernel enforcement."""

from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable


OPTIMIZATION_MODE_CHOICES = ("leaderboard", "production")
MODE_STATE_FILE = ".orchestrator_mode.json"
POLICY_BEGIN = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_BEGIN -->"
POLICY_END = "<!-- ATREX_OPTIMIZATION_MODE_POLICY_END -->"


@dataclass(frozen=True)
class DependencyReviewSignal:
    """One non-mechanical dependency signal for independent agent review."""

    id: str
    kind: str
    value: str


DependencyReviewer = Callable[
    [Path, str, tuple[DependencyReviewSignal, ...]],
    list[str],
]


def _framework_key(framework: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "", framework.strip().lower())
    aliases = {
        "triton": "triton",
        "gluon": "gluon",
        "tritongluon": "gluon",
        "cutedsl": "cutedsl",
        "cute": "cutedsl",
        "cuda": "cuda",
        "cudac": "cuda",
        "flydsl": "flydsl",
        "fly": "flydsl",
        "ascendc": "ascendc",
        "cannascendc": "ascendc",
    }
    return aliases.get(token, token)


def _code_without_prose(source: str) -> str:
    """Blank comments and docstrings so textual scans judge code, not description.

    A docstring reading "vLLM-style paged attention" describes the algorithm; it is not a
    dependency. Ordinary string literals are preserved because a Cuda candidate embeds its
    C++ source (with ``#include`` and ``__global__``) in one.
    """
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            # Any bare string statement is discarded at runtime, so it can only be
            # documentation. This also covers a "docstring" that Python does not count as
            # one because `from __future__ import annotations` precedes it.
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and node.end_lineno is not None
            ):
                for index in range(node.lineno, node.end_lineno + 1):
                    lines[index - 1] = ""
    blanked = "\n".join(lines)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(blanked).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return blanked
    out = blanked.splitlines()
    for token in tokens:
        if token.type == tokenize.COMMENT:
            row, column = token.start
            out[row - 1] = out[row - 1][:column]
    return "\n".join(out)


def source_uses_gluon(source: str) -> bool:
    """Return whether Python source imports the Triton experimental Gluon DSL.

    Parse imports instead of searching for the word ``gluon`` so comments, strings,
    and failure notes cannot accidentally satisfy the mandatory conversion gate.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "triton.experimental.gluon"
                or alias.name.startswith("triton.experimental.gluon.")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "triton.experimental" and any(
                alias.name == "gluon" for alias in node.names
            ):
                return True
            if module == "triton.experimental.gluon" or module.startswith(
                "triton.experimental.gluon."
            ):
                return True
    return False


def optimization_mode_directive(mode: str, framework: str) -> str:
    """Self-contained policy block injected into every coding-agent prompt."""
    if mode == "leaderboard":
        return (
            "## Optimization mode: leaderboard\n\n"
            "Follow the workspace `CLAUDE.md` exactly. Its existing framework guidance remains "
            "unchanged: the requested framework is a recommended direction, compatible mixed/alternate "
            "implementations are allowed when evidence supports them, and third-party helper/kernel "
            "libraries may be used.\n"
        )
    if mode != "production":
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    framework_key = _framework_key(framework)
    if framework_key == "triton":
        framework_rule = (
            "- The initial implementation framework is exactly **Triton**. After the orchestrator "
            "enters its mandatory Triton-to-Gluon conversion phase, a direct implementation in "
            "`triton.experimental.gluon` is allowed and becomes the required framework for later "
            "iterations. Do not switch early, switch back, mix Triton and Gluon compute kernels, "
            "or use any other DSL.\n"
        )
        candidate_framework = "the active Triton/Gluon phase"
        compute_kind = "GPU"
    elif framework_key == "ascendc":
        framework_rule = (
            "- The implementation framework is exactly **AscendC**. The accelerator computation "
            "must live in a self-authored AscendC AI Core kernel declared in `solution.json` "
            "`sources`. Python/PyTorch-NPU, CANN compiler bindings, ACL runtime launch calls, and "
            "a narrowly named custom `torch.ops` namespace (`atrex_*`, `ascendc_*`, `custom`, "
            "or `custom_*`) "
            "may only build or launch that declared kernel. ACLNN, built-in Torch-NPU operators, "
            "and other prebuilt compute paths are forbidden.\n"
        )
        candidate_framework = "AscendC"
        compute_kind = "accelerator"
    else:
        framework_rule = (
            f"- The implementation framework is exactly **{framework}**. It is a hard constraint, "
            "not a recommendation. Do not switch to another DSL, mix another kernel framework "
            "into the candidate, or replace the implementation with a prebuilt operator.\n"
        )
        candidate_framework = framework
        compute_kind = "GPU"
    return (
        "## Optimization mode: production (hard gate)\n\n"
        "This generated section overrides any conflicting permissive framework or third-party-library "
        "guidance elsewhere in `CLAUDE.md`.\n\n"
        f"{framework_rule}"
        "- The V0 PyTorch reference wrapper is the only baseline exception. Every optimized candidate "
        f"committed after V0 must implement the {compute_kind} computation directly in **{candidate_framework}**.\n"
        "- Third-party dependencies are reviewed by a separate, read-only policy agent based on how "
        "they are actually used. Compiler bindings, header discovery, ABI/launch plumbing, and ordinary "
        "non-compute support utilities may be accepted when they only build or launch the candidate's "
        "self-authored kernel. Prebuilt kernels/operators/math implementations, alternate DSLs, hidden "
        "dispatch, and external implementation loading remain forbidden. Do not assume that either an "
        "unfamiliar package name or a familiar vendor package is automatically accepted or rejected.\n"
        "- Update `solution.json` so its languages and dependencies contain only PyTorch/evaluator "
        "plumbing plus the selected framework. Before committing, inspect `kernel.py` and "
        "`solution.json` against these rules. The orchestrator will mechanically reject and revert a "
        "kernel-changing commit that violates them, even if it is faster and correct.\n"
    )


def workspace_policy_block(mode: str, framework: str) -> str:
    directive = optimization_mode_directive(mode, framework).rstrip()
    return f"{POLICY_BEGIN}\n\n{directive}\n\n{POLICY_END}\n"


def install_workspace_policy(
    workspace: Path,
    mode: str,
    framework: str,
    *,
    agent_runtime: str | None = None,
) -> None:
    """Persist immutable mode, framework, and optional campaign runtime identity.

    Existing workspaces without ``agent_runtime`` remain readable. Their first
    explicit post-upgrade runtime is adopted before a session starts; later
    attempts to resume with another backend fail closed.
    """
    if mode not in OPTIMIZATION_MODE_CHOICES:
        raise ValueError(f"unsupported optimization mode: {mode!r}")
    requested_runtime = (
        str(agent_runtime).strip() if agent_runtime is not None else None
    )
    if agent_runtime is not None and not requested_runtime:
        raise ValueError("agent_runtime must be a non-empty runtime id")

    workspace.mkdir(parents=True, exist_ok=True)
    state_path = workspace / MODE_STATE_FILE
    state_changed = False
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid optimization-mode state: {state_path}") from exc
        existing_mode = state.get("mode")
        existing_framework = state.get("framework")
        if existing_mode != mode or existing_framework != framework:
            raise RuntimeError(
                "workspace policy mismatch: "
                f"recorded mode/framework={existing_mode}/{existing_framework}, "
                f"requested={mode}/{framework}"
            )
        existing_runtime = str(state.get("agent_runtime") or "").strip()
        if requested_runtime and existing_runtime and existing_runtime != requested_runtime:
            raise RuntimeError(
                "workspace agent runtime mismatch: "
                f"recorded={existing_runtime}, requested={requested_runtime}; "
                "use a fresh campaign workspace to change backend"
            )
        if requested_runtime and not existing_runtime:
            state["agent_runtime"] = requested_runtime
            state_changed = True
    else:
        state = {"mode": mode, "framework": framework}
        if requested_runtime:
            state["agent_runtime"] = requested_runtime
        state_changed = True

    if state_changed:
        state_path.write_text(
            json.dumps(state, indent=2) + "\n",
            encoding="utf-8",
        )

    claude_path = workspace / "CLAUDE.md"
    current = claude_path.read_text(encoding="utf-8") if claude_path.exists() else ""
    generated = workspace_policy_block(mode, framework)
    if POLICY_BEGIN in current and POLICY_END in current:
        before, remainder = current.split(POLICY_BEGIN, 1)
        _, after = remainder.split(POLICY_END, 1)
        current = before.rstrip() + "\n\n" + generated + after.lstrip("\n")
    else:
        current = current.rstrip() + ("\n\n" if current.strip() else "") + generated
    claude_path.write_text(current, encoding="utf-8")

    gitignore = workspace / ".gitignore"
    ignored = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    entry = f"/{MODE_STATE_FILE}"
    if entry not in ignored.splitlines():
        with gitignore.open("a", encoding="utf-8") as handle:
            if ignored and not ignored.endswith("\n"):
                handle.write("\n")
            handle.write("\n# orchestrator optimization-mode identity (local policy state)\n")
            handle.write(entry + "\n")


_STDLIB_IMPORTS = frozenset(sys.stdlib_module_names) | {"__future__"}
_ALLOWED_IMPORTS = {
    "triton": {"torch", "triton", "sol_execbench"},
    "gluon": {"torch", "triton", "sol_execbench"},
    "cutedsl": {"torch", "cutlass", "cuda", "sol_execbench"},
    "cuda": {"torch", "cuda", "sol_execbench"},
    "flydsl": {"torch", "flydsl", "sol_execbench"},
    # AscendC is a C++ device DSL.  Python is launch/build glue only; torch_npu's
    # cpp_extension and stream/device helpers are the narrow official integration route.
    "ascendc": {"torch", "torch_npu", "sol_execbench"},
}
_ALLOWED_DEPENDENCY_TOKENS = {
    "triton": {"torch", "triton"},
    "gluon": {"torch", "triton"},
    "cutedsl": {"torch", "cutlass", "nvidiacutlassdsl", "cuda", "cudapython"},
    "cuda": {"torch", "cuda", "cudapython"},
    "flydsl": {"torch", "flydsl"},
    "ascendc": {"torch", "torchnpu", "cann", "ascendc", "pybind11"},
}


def declared_solution_source_paths(workspace: Path) -> tuple[str, ...]:
    """Return existing, regular source files declared by ``solution.json``.

    Source paths are part of the candidate's trust boundary: they are uploaded to an
    evaluator and later staged into the accepted baseline commit.  Keep the contract
    deliberately small and portable (canonical POSIX paths beneath the workspace) and
    reject traversal, absolute paths, symlink indirection, and missing files.
    """
    solution_path = workspace / "solution.json"
    try:
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid solution.json: {exc}") from exc
    if not isinstance(solution, dict):
        raise ValueError("solution.json must contain an object")
    sources = solution.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("solution.json sources must be a list")
    if not sources:
        raise ValueError("solution.json sources must be a non-empty list")

    root = workspace.resolve()
    selected: list[str] = []
    for index, source in enumerate(sources):
        if isinstance(source, str):
            source_text = source
        elif isinstance(source, dict) and isinstance(source.get("path"), str):
            source_text = source["path"]
        else:
            raise ValueError(
                f"solution.json sources[{index}] must be a path string or path object"
            )
        if (
            not source_text
            or source_text != source_text.strip()
            or "\\" in source_text
            or "\x00" in source_text
        ):
            raise ValueError(
                f"solution.json sources[{index}] is not a canonical workspace-relative path"
            )
        relative = PurePosixPath(source_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != source_text
            or not relative.parts
            or relative.parts[0] == ".git"
        ):
            raise ValueError(
                f"solution.json sources[{index}] escapes or targets workspace metadata: "
                f"{source_text!r}"
            )
        candidate = workspace.joinpath(*relative.parts)
        if not candidate.is_file():
            raise ValueError(
                f"solution.json sources[{index}] does not name an existing file: "
                f"{source_text!r}"
            )
        resolved = candidate.resolve()
        expected = root.joinpath(*relative.parts)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"solution.json sources[{index}] escapes the workspace: {source_text!r}"
            ) from exc
        if resolved != expected:
            raise ValueError(
                f"solution.json sources[{index}] uses symlink indirection: {source_text!r}"
            )
        canonical = relative.as_posix()
        if canonical not in selected:
            selected.append(canonical)
    return tuple(selected)


def _import_roots(tree: ast.AST) -> tuple[set[str], bool]:
    roots: set[str] = set()
    relative = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = True
            if node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots, relative


def _normalized_dependency(value: object) -> str:
    text = re.split(r"[<>=!~\[; ]", str(value).strip(), maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]+", "", text.lower())


_BANNED_TORCH_COMPUTE = {
    "addmm", "amax", "amin", "bmm", "conv1d", "conv2d", "conv3d", "cumprod", "cumsum",
    "einsum", "exp", "gelu", "layer_norm", "log", "log_softmax", "matmul", "max", "mean",
    "min", "mm", "rms_norm", "scaled_dot_product_attention", "sigmoid", "silu", "softmax",
    "sort", "sum", "topk",
}


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _ascendc_custom_torch_op(suffix: list[str]) -> bool:
    """Whether ``torch.<suffix>`` is narrow AscendC launch glue.

    ``torch.ops.load_library`` loads a candidate-built extension.  Calls into a
    deliberately custom namespace may launch that extension, while standard namespaces
    such as aten/npu remain rejected as prebuilt compute.
    """
    if suffix == ["ops", "load_library"]:
        return True
    if len(suffix) < 3 or suffix[0] != "ops":
        return False
    namespace = suffix[1].lower()
    return bool(re.match(r"^(?:atrex(?:_|$)|ascendc(?:_|$)|custom(?:_|$))", namespace))


_ASCENDC_TORCH_ALLOCATION_GLUE = {
    "empty",
    "empty_like",
    "empty_strided",
    "new_empty",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
}


def _ascendc_torch_glue_call(suffix: list[str]) -> bool:
    if _ascendc_custom_torch_op(suffix):
        return True
    if suffix and suffix[0] == "library":
        return True
    if suffix[:2] == ["utils", "cpp_extension"]:
        return True
    if len(suffix) == 1 and suffix[0] in _ASCENDC_TORCH_ALLOCATION_GLUE | {
        "device",
        "from_dlpack",
    }:
        return True
    return (
        len(suffix) == 2
        and suffix[0] == "npu"
        and suffix[1] in _ASCENDC_NPU_GLUE_CALLS
    )


def _torch_compute_violations(
    tree: ast.AST,
    *,
    allow_ascendc_glue: bool = False,
) -> list[str]:
    torch_aliases = {"torch"}
    functional_aliases: set[str] = set()
    direct_functional_calls: set[str] = set()
    direct_torch_calls: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    torch_aliases.add(alias.asname or "torch")
                elif alias.name == "torch.nn.functional":
                    functional_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "torch.nn.functional":
            direct_functional_calls.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            direct_torch_calls.update(
                (alias.asname or alias.name, alias.name) for alias in node.names
            )

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            violations.append("Python/PyTorch matrix multiplication is not the selected kernel framework")
            continue
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        if not dotted:
            continue
        parts = dotted.split(".")
        if parts[0] in direct_functional_calls:
            violations.append(f"PyTorch functional call is forbidden in production candidate: {dotted}")
            continue
        if allow_ascendc_glue and parts[0] in direct_torch_calls:
            original = direct_torch_calls[parts[0]]
            if not _ascendc_torch_glue_call([original, *parts[1:]]):
                violations.append(
                    f"PyTorch prebuilt compute call is forbidden for AscendC: {dotted}"
                )
            continue
        if any(dotted == alias or dotted.startswith(alias + ".") for alias in functional_aliases):
            violations.append(f"PyTorch functional call is forbidden in production candidate: {dotted}")
            continue
        if parts[0] not in torch_aliases or len(parts) < 2:
            continue
        suffix = parts[1:]
        if suffix[0] == "ops":
            if not (allow_ascendc_glue and _ascendc_custom_torch_op(suffix)):
                violations.append(
                    f"torch.ops dispatch is forbidden in production candidate: {dotted}"
                )
        elif suffix[:2] == ["nn", "functional"]:
            violations.append(f"PyTorch functional call is forbidden in production candidate: {dotted}")
        elif suffix[-1] in _BANNED_TORCH_COMPUTE:
            violations.append(f"PyTorch compute call is forbidden in production candidate: {dotted}")
        elif allow_ascendc_glue and not _ascendc_torch_glue_call(suffix):
            violations.append(
                f"PyTorch prebuilt compute call is forbidden for AscendC: {dotted}"
            )
    return list(dict.fromkeys(violations))


_ASCENDC_NPU_GLUE_CALLS = {
    "current_device",
    "current_stream",
    "device_count",
    "get_device_name",
    "get_device_properties",
    "is_available",
    "set_device",
    "synchronize",
}

_ASCENDC_BUILD_EXECUTABLES = {"cmake"}


def _ascendc_subprocess_violations(tree: ast.AST) -> list[str]:
    """Allow only direct, non-shell CMake invocations for declared AscendC sources.

    CANN's supported fast-kernel-launch examples use CMake to drive Bisheng.  The
    evaluator may therefore build a candidate on first load, but production code
    must not gain a general shell or process-execution escape hatch.  Requiring a
    literal argv, ``check=True``, and no shell keeps the executable mechanically
    auditable while still permitting dynamic path arguments after ``cmake``.
    """
    module_aliases = {"subprocess"}
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            violations.append(
                "AscendC build plumbing must import the subprocess module, not symbols from it"
            )

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or node.id not in module_aliases:
            continue
        parent = parents.get(node)
        if (
            isinstance(parent, ast.Attribute)
            and parent.value is node
            and parent.attr == "run"
        ):
            continue
        violations.append(
            "AscendC subprocess plumbing may only call subprocess.run directly"
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id not in module_aliases:
            continue
        if node.func.attr != "run":
            violations.append(
                f"AscendC subprocess method is forbidden: {node.func.attr}"
            )
            continue
        command = node.args[0] if node.args else None
        if not isinstance(command, (ast.List, ast.Tuple)) or not command.elts:
            violations.append(
                "AscendC subprocess.run requires a literal argv list beginning with cmake"
            )
            continue
        executable = command.elts[0]
        executable_name = (
            Path(executable.value).name.lower()
            if isinstance(executable, ast.Constant) and isinstance(executable.value, str)
            else ""
        )
        if executable_name not in _ASCENDC_BUILD_EXECUTABLES:
            violations.append(
                "AscendC subprocess.run may execute only cmake with a literal argv list"
            )
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        check = keywords.get("check")
        if not isinstance(check, ast.Constant) or check.value is not True:
            violations.append("AscendC cmake subprocess.run must set check=True")
        shell = keywords.get("shell")
        if shell is not None and (
            not isinstance(shell, ast.Constant) or shell.value is not False
        ):
            violations.append("AscendC cmake subprocess.run must not enable a shell")
        if "executable" in keywords:
            violations.append(
                "AscendC cmake subprocess.run must not override the executable"
            )
    return list(dict.fromkeys(violations))


def _ascendc_python_glue_violations(tree: ast.AST) -> list[str]:
    """Reject Torch-NPU compute while retaining build/device/stream launch glue."""
    module_aliases = {"torch_npu"}
    npu_aliases: set[str] = set()
    forbidden_direct_calls: set[str] = set()
    allowed_direct_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch_npu":
                    module_aliases.add(alias.asname or "torch_npu")
                elif alias.name == "torch_npu.npu":
                    npu_aliases.add(alias.asname or "torch_npu.npu")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "torch_npu":
                for alias in node.names:
                    name = alias.asname or alias.name
                    if alias.name == "npu":
                        npu_aliases.add(name)
                    else:
                        forbidden_direct_calls.add(name)
            elif module == "torch_npu.npu":
                for alias in node.names:
                    name = alias.asname or alias.name
                    if alias.name in _ASCENDC_NPU_GLUE_CALLS:
                        allowed_direct_calls.add(name)
                    else:
                        forbidden_direct_calls.add(name)

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        if not dotted or dotted in allowed_direct_calls:
            continue
        if dotted in forbidden_direct_calls:
            violations.append(
                f"Torch-NPU prebuilt compute call is forbidden for AscendC: {dotted}"
            )
            continue
        parts = dotted.split(".")
        if parts[0] in module_aliases:
            suffix = parts[1:]
            if suffix[:2] == ["utils", "cpp_extension"]:
                continue
            if (
                len(suffix) == 2
                and suffix[0] == "npu"
                and suffix[1] in _ASCENDC_NPU_GLUE_CALLS
            ):
                continue
            violations.append(
                f"Torch-NPU prebuilt compute call is forbidden for AscendC: {dotted}"
            )
        elif parts[0] in npu_aliases and (
            len(parts) != 2 or parts[1] not in _ASCENDC_NPU_GLUE_CALLS
        ):
            violations.append(
                f"Torch-NPU prebuilt compute call is forbidden for AscendC: {dotted}"
            )
    return list(dict.fromkeys(violations))


def _c_like_code_without_comments(source: str) -> str:
    """Blank C/C++ comments while retaining includes and string literals."""
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def _has_ascendc_marker(source: str) -> bool:
    has_header = bool(
        re.search(r"#\s*include\s*[<\"][^>\"]*kernel_operator\.h[>\"]", source)
    )
    has_device_code = "__aicore__" in source or "AscendC::" in source
    return has_header and has_device_code


def _has_cuda_marker(source: str) -> bool:
    if "__global__" not in source:
        return False
    cuda_specific = bool(
        re.search(r"\b(?:blockIdx|threadIdx|cuda_runtime|cudaLaunchKernel|nvrtc)\b", source)
    )
    return cuda_specific or "__aicore__" not in source


def production_kernel_violations(
    workspace: Path,
    framework: str,
    *,
    require_gluon: bool = False,
    dependency_reviewer: DependencyReviewer | None = None,
) -> list[str]:
    """Return production-policy violations for the current candidate.

    Mechanically provable rules stay local. Ambiguous dependency provenance is
    delegated through ``dependency_reviewer`` and fails closed when no reviewer
    is supplied. Runtime correctness and performance still use the normal sandbox.
    """
    key = _framework_key(framework)
    errors: list[str] = []
    if key not in _ALLOWED_IMPORTS:
        return [f"unsupported production framework: {framework}"]
    kernel_path = workspace / "kernel.py"
    if not kernel_path.is_file():
        return ["kernel.py is missing"]
    source = kernel_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(kernel_path))
    except SyntaxError as exc:
        return [f"kernel.py is not valid Python: {exc.msg} (line {exc.lineno})"]

    solution: dict | None = None
    declared_sources: tuple[str, ...] = ()
    solution_path = workspace / "solution.json"
    if solution_path.is_file():
        try:
            decoded_solution = json.loads(solution_path.read_text(encoding="utf-8"))
            if not isinstance(decoded_solution, dict):
                raise ValueError("solution.json must contain an object")
            solution = decoded_solution
            declared_sources = declared_solution_source_paths(workspace)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"solution.json is invalid: {exc}")

    roots, has_relative_import = _import_roots(tree)
    if has_relative_import:
        errors.append("relative/local-module imports are not self-contained")
    policy_source = _code_without_prose(source)
    candidate_source_parts = [policy_source]
    declared_source_parts: list[str] = []
    for relative in declared_sources:
        if relative == "kernel.py":
            declared_source_parts.append(policy_source)
            continue
        auxiliary_path = workspace / relative
        auxiliary = auxiliary_path.read_text(encoding="utf-8", errors="replace")
        if auxiliary_path.suffix.lower() == ".py":
            auxiliary_code = _code_without_prose(auxiliary)
        else:
            auxiliary_code = _c_like_code_without_comments(auxiliary)
        candidate_source_parts.append(auxiliary_code)
        declared_source_parts.append(auxiliary_code)
    candidate_source = "\n".join(candidate_source_parts)
    declared_source = "\n".join(declared_source_parts)
    # A production Triton campaign may enter the orchestrator-controlled Gluon phase.
    # Once a Gluon marker is present, validate the candidate as Gluon (which necessarily
    # imports the Triton package) rather than rejecting it as an alternate framework.
    effective_key = key
    has_gluon_import = source_uses_gluon(source)
    if key == "triton" and has_gluon_import:
        effective_key = "gluon"
    if require_gluon and effective_key != "gluon":
        errors.append("switching back from the accepted Gluon phase to Triton is forbidden")

    allowed = _STDLIB_IMPORTS | _ALLOWED_IMPORTS[effective_key]
    dependency_signals: list[DependencyReviewSignal] = [
        DependencyReviewSignal(
            id=f"import:{root}",
            kind="import",
            value=root,
        )
        for root in sorted(roots - allowed)
    ]
    dynamic_roots = roots & {"ctypes", "importlib", "pkgutil", "runpy", "subprocess"}
    if effective_key == "ascendc" and "subprocess" in dynamic_roots:
        errors.extend(_ascendc_subprocess_violations(tree))
        dynamic_roots.remove("subprocess")
    for root in sorted(dynamic_roots):
        errors.append(f"dynamic external-code loading is forbidden in production candidate: {root}")
    errors.extend(_torch_compute_violations(tree, allow_ascendc_glue=effective_key == "ascendc"))
    if effective_key == "ascendc":
        errors.extend(_ascendc_python_glue_violations(tree))

    ascendc_marker = _has_ascendc_marker(candidate_source)
    declared_ascendc_marker = _has_ascendc_marker(declared_source)
    cuda_marker = _has_cuda_marker(candidate_source)

    marker_checks = {
        "triton": (
            bool(re.search(r"(?:^|\n)\s*(?:import|from)\s+triton\b", policy_source)),
            "missing Triton implementation/import",
        ),
        "gluon": (has_gluon_import, "missing Gluon implementation"),
        "cutedsl": ("cutlass.cute" in policy_source or "@cute.kernel" in policy_source, "missing CuteDSL implementation"),
        "cuda": (
            cuda_marker
            and bool(re.search(r"load_inline|cpp_extension|CUDAExtension|nvrtc|cuda\.bindings", policy_source)),
            (
                "missing self-authored CUDA kernel/loader in kernel.py; use an in-process "
                "loader such as cuda.bindings/NVRTC"
            ),
        ),
        "flydsl": (bool(re.search(r"(?:^|\n)\s*(?:import|from)\s+flydsl\b", policy_source)), "missing FlyDSL implementation"),
        "ascendc": (
            declared_ascendc_marker,
            (
                "missing self-authored AscendC kernel; declare an existing source containing "
                "kernel_operator.h and AscendC AI Core device code in solution.json"
            ),
        ),
    }
    marker_ok, marker_error = marker_checks[effective_key]
    if not marker_ok:
        errors.append(marker_error)

    foreign_markers = {
        "triton": bool(re.search(r"(?:^|\n)\s*(?:import|from)\s+triton\b", policy_source)),
        "gluon": has_gluon_import,
        "cutedsl": "cutlass.cute" in policy_source or "@cute.kernel" in policy_source,
        "cuda": cuda_marker,
        "flydsl": bool(re.search(r"(?:^|\n)\s*(?:import|from)\s+flydsl\b", policy_source)),
        "ascendc": ascendc_marker,
    }
    compatible_markers = {effective_key}
    if effective_key == "gluon":
        compatible_markers.add("triton")  # Gluon is distributed under triton.experimental.
    for other, present in foreign_markers.items():
        if present and other not in compatible_markers:
            errors.append(f"mixed/alternate framework marker is forbidden: {other}")

    banned_source_patterns = {
        r"\btorch\.nn\.functional\b": "torch.nn.functional is not the selected kernel framework",
        r"\btorch\.(?:linalg|_scaled_mm)\b": "PyTorch compute fallback is not the selected kernel framework",
    }
    if effective_key != "ascendc":
        banned_source_patterns[
            r"\btorch\.ops\b"
        ] = "torch.ops dispatch is a prebuilt/custom operator call"
    for pattern, message in banned_source_patterns.items():
        if re.search(pattern, policy_source, flags=re.IGNORECASE):
            errors.append(message)

    if effective_key == "ascendc":
        ascendc_prebuilt_patterns = {
            r"#\s*include\s*[<\"][^>\"]*aclnn[^>\"]*[>\"]": (
                "ACLNN headers expose prebuilt operators, not a self-authored AscendC kernel"
            ),
            r"\baclnn[A-Za-z0-9_]*\s*\(": (
                "ACLNN prebuilt operator calls are forbidden for AscendC candidates"
            ),
            r"\baclop(?:CompileAndExecute|ExecuteV2?)\s*\(": (
                "ACL op execution is a prebuilt operator path, not AscendC launch glue"
            ),
            r"\b(?:EXEC_NPU_CMD|EXEC_NPU_NO_FORMAT_CHECK|OP_EXEC)\s*\(": (
                "Torch-NPU prebuilt operator macros are forbidden for AscendC candidates"
            ),
            # Torch-NPU's official fast-launch integration uses RunOpApi only as
            # stream/error-accounting glue around a caller-supplied lambda that
            # launches the candidate's own AscendC kernel.  Keep every other
            # at_npu::native entry point fail-closed; the ACLNN/aclop/macro rules
            # above still reject a prebuilt operator hidden inside that lambda.
            r"\bat_npu\s*::\s*native\s*::\s*(?!OpCommand\s*::\s*RunOpApi(?:V2)?\s*\()": (
                "Torch-NPU native prebuilt operators are forbidden for AscendC candidates"
            ),
        }
        for pattern, message in ascendc_prebuilt_patterns.items():
            if re.search(pattern, candidate_source):
                errors.append(message)

    review_source_patterns = {
        "kernel_library_reference": (
            r"\b(?:flashinfer|flash_attn|xformers|vllm|sglang|bitsandbytes)\b",
            "third-party kernel/operator library reference",
        ),
        "cuda_library_reference": (
            r"\b(?:cublas|cudnn)[A-Za-z0-9_]*\b",
            "CUDA math/operator library reference",
        ),
        "cutlass_header_reference": (
            r"#\s*include\s*[<\"]cutlass/",
            "CUTLASS header reference",
        ),
    }
    for marker, (pattern, description) in review_source_patterns.items():
        if re.search(pattern, policy_source, flags=re.IGNORECASE) and not (
            effective_key == "cutedsl" and marker == "cutlass_header_reference"
        ):
            dependency_signals.append(
                DependencyReviewSignal(
                    id=f"source:{marker}",
                    kind="source_reference",
                    value=description,
                )
            )

    if solution is not None:
        spec = solution.get("spec") or {}
        if not isinstance(spec, dict):
            errors.append("solution.json spec must be an object")
            spec = {}
        dependencies = spec.get("dependencies") or []
        if not isinstance(dependencies, list):
            errors.append("solution.json spec.dependencies must be a list")
            dependencies = []
        allowed_dependencies = _ALLOWED_DEPENDENCY_TOKENS[effective_key]
        for dependency in dependencies:
            token = _normalized_dependency(dependency)
            if token and token not in allowed_dependencies:
                dependency_signals.append(
                    DependencyReviewSignal(
                        id=f"solution_dependency:{dependency}",
                        kind="solution_dependency",
                        value=str(dependency),
                    )
                )

    dependency_signals = list(dict.fromkeys(dependency_signals))
    if dependency_signals:
        if dependency_reviewer is None:
            errors.append(
                "third-party dependency requires independent agent review: "
                + ", ".join(signal.id for signal in dependency_signals)
            )
        else:
            try:
                review_errors = dependency_reviewer(
                    workspace,
                    framework,
                    tuple(dependency_signals),
                )
            except Exception as exc:
                errors.append(
                    "independent dependency review failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                if not isinstance(review_errors, list) or not all(
                    isinstance(item, str) and item.strip() for item in review_errors
                ):
                    errors.append(
                        "independent dependency review returned an invalid result"
                    )
                else:
                    errors.extend(review_errors)
    return list(dict.fromkeys(errors))


def reject_production_commit(
    workspace: Path,
    version: int,
    pre_head: str,
    violations: list[str],
) -> Path:
    """Revert a violating kernel commit and preserve an actionable local record."""
    memory_path = workspace / "memory" / f"v{version}.json"
    try:
        memory = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        memory = {}
    if pre_head:
        subprocess.run(
            ["git", "reset", "--hard", pre_head],
            cwd=str(workspace),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory["version"] = f"v{version}"
    memory["masked"] = False
    memory["git_commit_hash"] = None
    memory["quality_gate"] = {
        "result": "FAIL",
        "failure_reason": "production policy violation: " + "; ".join(violations),
    }
    memory["optimization"] = {
        "action_category": "production_policy_rejection",
        "action_description": "reverted candidate that used a forbidden dependency or wrong framework",
    }
    pitfalls = memory.setdefault("pitfalls_and_fixes", [])
    pitfalls.append({
        "error_type": "production_policy",
        "error_message": "; ".join(violations),
        "lesson": "implement the candidate directly and exclusively in the selected framework",
    })
    memory_path.write_text(json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return memory_path
