# Atrex compatibility contract

The upstream CANN-SKILLS catalog and CANN reference projects describe general operator development.
An Atrex production candidate has a narrower, self-contained execution contract.

## Allowed adaptation

- Learn file organization, CANN APIs, tiling invariants, registration patterns, stream usage, and
  launch structure from the upstream material.
- Reimplement the required kernel, binding, and tiling logic in candidate-owned files.
- Compile declared sources with CMake/BiSheng inside the sandbox.
- Use a narrow candidate-owned PyTorch custom namespace or a candidate-owned PyBind module only to
  launch the declared kernel.

## Do not carry over literally

- Do not run upstream `pip install`, editable install, wheel packaging, vendor deployment, `build.sh`,
  setup-script sourcing, or system installation steps.
- Do not call prebuilt `aclnnMatmul`, `aclnnLinear`, other ACLNN compute, or installed torch_npu compute
  as the candidate implementation.
- Do not use the upstream `npu_ops_transformer_ext` package or namespace as a dependency.
- Do not copy OSLA reference implementation code verbatim.
- Do not add undeclared side files or binaries.

Candidate Python may run only direct, checked, literal-argv CMake commands allowed by the campaign.
All accelerator execution and compilation probes remain behind `tools/sandbox.py`.

## Upstream catalog routing

The campaign exposes the pinned upstream catalog at `cann-skills/`. The upstream Claude marketplace
manifest is not used: its current bundle paths omit the actual `skills/tutorial/` component, and
registering every generated operator card would make discovery unnecessarily expensive.

Use these release-matched tutorial paths directly:

- `cann-skills/skills/tutorial/cann-op-scaffold/SKILL.md`
- `cann-skills/skills/tutorial/cann-tiling/SKILL.md`
- `cann-skills/skills/tutorial/cann-kernel/SKILL.md`
- `cann-skills/skills/tutorial/cann-debugging/SKILL.md`
- `cann-skills/skills/tutorial/cann-profiling/SKILL.md`

The catalog targets CANN 8.5/Ascend 910B but also contains auto-generated cards derived from other
header snapshots. Verify every selected API against the campaign's installed CANN headers and runtime.
