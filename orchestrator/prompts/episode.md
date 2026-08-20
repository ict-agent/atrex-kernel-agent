# Kernel optimization episode {{EPISODE}}

Own one complete engineering direction in this isolated Git worktree. Continue through as many
profile, research, plan, edit, compile, correctness, benchmark, autotune, and repair cycles as the
direction needs. Do not stop after one edit, one failed compile, or one benchmark while a concrete
next engineering step remains.

The supervisor owns the incumbent branch, authoritative ABBA verification, canonical memory, and
final squash promotion. You own only this episode branch and its structured evidence.

## Context

- Workspace: `{{WORKSPACE}}`
- Canonical version produced by the supervisor: `v{{VERSION}}`
- Platform: `{{PLATFORM}}`
- Framework: `{{FRAMEWORK}}`
- Incumbent commit: `{{BASE_COMMIT}}`
- Episode branch: `{{EPISODE_BRANCH}}`
- Journal: `{{JOURNAL_PATH}}`
- Handoff: `{{HANDOFF_PATH}}`
- Additional constraints: {{NOTES}}
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are linked into the worktree.
{{AGENT_RUNTIME}}

Never switch branches, push, merge, rebase, or alter refs. Private checkpoint commits on the episode
branch are allowed. Never edit evaluator or ground-truth files, including `test_kernel.py`,
`profile_driver.py`, `definition.json`, `reference.py`, `workload.jsonl`, `input.py`, `shapes.json`,
`metadata.json`, `roofline.json`, `CLAUDE.md`, or `README.md`. Do not write canonical `memory/vN.json`;
the supervisor creates it after terminal validation.

For an AscendC episode (`vendor=ascend`, `arch=ascend910b1`, `framework=AscendC`), the candidate may
consist of the evaluator-facing `kernel.py` plus self-authored `.cpp`, `.h`, and `.asc` files. Every such
file must be listed in `solution.json.sources`; undeclared files, prebuilt custom operators, and external
implementations are outside candidate ownership. Before changing an AscendC PyTorch binding, host tiling,
or Cube/Matmul launch contract, use the repository-local `ascendc-custom-pytorch-op` skill and its pinned
`cann-skills/` knowledge root.

An AscendC candidate using CANN's CMake/Bisheng fast-launch route must also declare `CMakeLists.txt` in
`solution.json.sources`. Candidate Python may run only direct, checked, literal-argv CMake commands:
`subprocess.run(["cmake", ...], check=True)`. Never execute `bash`/`sh`, enable `shell=True`, build a
command string, source a setup script, or execute a non-`cmake` subprocess. The gateway already supplies
the CANN environment. Keep all compilation and candidate imports behind `tools/sandbox.py`.

{{MODE_POLICY}}

{{EVALUATOR}}

{{HARDWARE}}

{{SANDBOX}}

## Non-negotiable execution boundary

- Never run `python test_kernel.py`, `python kernel.py`, or import GPU/JIT kernel packages directly
  on the host. Route every compile, correctness, benchmark, and profiling command through
  `python tools/sandbox.py ... --`.
- Never start, stop, restart, signal, replace, or mutate the shared gateway service, its screen
  session, state directory, database, log, or jobs. Report infrastructure failure instead.
- Never install dependencies with pip, uv, conda, setup.py, or package-manager commands. Use only the
  immutable campaign environment. AscendC may use the preinstalled CANN compiler/CMake/ninja path through
  the sandbox only to build declared, self-authored candidate sources; building third-party code is forbidden.
- Static source inspection is allowed. Imports or probes that may initialize CUDA/ROCm/NPU/JIT code must
  run through the sandbox.
- Never use a host `import torch_npu` merely to locate its package. Static discovery, when unavoidable,
  is `python -c "import importlib.util; print(importlib.util.find_spec('torch_npu').submodule_search_locations)"`;
  runtime candidate code may inspect `torch_npu.__file__` inside the gateway.

On Ascend, claim profiler evidence only when the sandbox exposes a working `msprof` path and the command
produces inspectable output for the committed candidate. If that integration is unavailable, record the
profiling gap explicitly and continue only with honestly labeled static, compile, correctness, and benchmark
evidence; never invent `msprof` metrics or translate NCU/rocprof fields into Ascend results.

For AscendC research, keep L1 architecture-scoped with `--arch ascend910b1 --vendor ascend --dsl ascendc`;
do not drop those filters to manufacture a hit. If L1 is insufficient, start L2 with the CANN 8.5
repository matching the operator class (`reference-projects/ops-math`, `ops-nn`, `ops-transformer`,
or `ops-cv`), then use `reference-projects/vllm-ascend` and `cann-recipes-infer` for runtime and
inference-integration patterns. Repositories carrying the Ascend Open Source Software License
Agreement (OSLA) are reference-only: derive API/design patterns, but do not copy code verbatim, load
their implementation, or make them a candidate dependency.

## Framework escalation state

{{CONVERSION_DIRECTIVE}}

When conversion is mandatory, treat the whole episode as a Triton-to-Gluon lowering direction:

1. Read only the conversion sheet matching the authoritative runtime architecture:
   - `sm_100`/`sm_103`: `gpu-wiki/docs/nvidia/blackwell/converter/blackwell.md`
   - `sm_90`: `gpu-wiki/docs/nvidia/hopper/converter/hopper.md`
   - `gfx94*`: `gpu-wiki/docs/amd/cdna3/converter/cdna3.md`
   - `gfx95*`: `gpu-wiki/docs/amd/cdna4/converter/cdna4.md`
2. Extract TTGIR before writing Gluon and derive layouts from the real kernel; never fabricate them.
3. Preserve algorithm, tiling, signatures, and evaluator behavior. Fix compile/correctness/parity
   defects inside this episode rather than handing off the first translation attempt.
4. A terminal candidate must be committed Gluon, correctness-passing in development, and plausibly
   within 5% of the incumbent. The supervisor independently enforces parity.

This Triton-to-Gluon conversion table is GPU-backend-specific. It does not create a conversion path for
an AscendC/Ascend 910B1 campaign: keep AscendC and use its CANN toolchain unless the authoritative
`{{CONVERSION_DIRECTIVE}}` explicitly selects another supported framework.

## Prior episode evidence

```json
{{HISTORY}}
```

Historical attempts are evidence, not orders. Do not repeat a rejected direction unless new evidence
or a materially different implementation changes the expected result.

## Engineering loop

`skills/gpu-kernel-episode-loop/SKILL.md` defines the binding evidence loop for this episode:
reconstruct the incumbent, profile and localize, research progressively, plan one coherent direction,
implement and repair, validate development correctness and performance, record every decisive
experiment, and mark the phase telemetry. **Read that file now and execute its loop**; it is a
requirement, not background reading.

Bind its placeholders to this episode:

| Skill placeholder | This episode |
| --- | --- |
| `<PROFILE_DIR>` | `profiles/episode_{{EPISODE}}` |
| `<PLAN_DRAFT>` | `plans/v{{VERSION}}_draft.md` |
| `<PLAN_FILE>` | `plans/v{{VERSION}}_plan.md` |
| `<JOURNAL_CLI>` | `{{JOURNAL_COMMAND}}` |
| `<JOURNAL_PATH>` | `{{JOURNAL_PATH_SHELL}}` |

`<PLAN_GENERATOR>` is the backend-native plan generator for this session:

{{PLAN_GENERATOR}}

As soon as one coherent candidate passes the full development correctness check and has credible
performance evidence, publish the terminal handoff. Do not hold a promotable candidate while pursuing
secondary tweaks; those belong to a later episode and version.

## Terminal contract

Reach exactly one evidence-backed terminal state:

1. `candidate_ready`: a mature candidate is committed, the worktree is clean, and development
   correctness/performance supports independent verification.
2. `pivot`: the engineering direction is exhausted and a fresh episode should pursue another one.
3. `blocked`: infrastructure or missing authority prevents meaningful progress.

For `candidate_ready`, commit the exact candidate, append final evidence, then finalize the journal:

```bash
candidate_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state candidate_ready \
  --candidate-commit "$candidate_commit" \
  --outcome-json '{"summary":"...","next_directions":["..."]}'
```

For `pivot` or `blocked`, finalize with that state and omit `--candidate-commit`. The journal must
contain at least one structured experiment and a non-empty outcome summary.

Only after finalizing, atomically publish the control handoff by writing complete JSON to
`{{HANDOFF_PATH}}.tmp` and renaming it to `{{HANDOFF_PATH}}`:

```json
{
  "status": "candidate_ready | pivot | blocked",
  "candidate_commit": "required only for candidate_ready",
  "last_trial_commit": "optional checkpoint for pivot or blocked"
}
```

Chat text is not a handoff. A missing or invalid handoff causes bounded same-session recovery. Do not
claim a speedup merely to terminate; a well-supported pivot is a valid outcome.
