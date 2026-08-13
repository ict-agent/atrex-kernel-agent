# AscendC Add smoke kernel

This is a deliberately small, generated AscendC AIV kernel for validating the
repository's 910B1 knowledge/toolchain path. It adds 16,384 contiguous FP16
elements with eight blocks and four 512-element tiles per block. Every transfer
is a multiple of the A2 32-byte datablock.

This is a compile/run smoke test, not an Atrex/SOL performance benchmark and not
a substitute for a real operator's host tiling, registration, multi-shape
correctness suite, or ABBA verification.

## Reproduce on CANN 8.5.0 / Ascend 910B1

The verified run used the official Ascend samples kernel-launch scaffold at
commit `1a01baa8007f59a503eafcd4c5bce331f546430f` (tag
`v1.11-8.5.0.alpha001`). Copy this repository's `add_custom.cpp` over the
scaffold's file, then run:

```bash
source <python-env-with-numpy>/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=<free-device-id>
bash run.sh -r npu -v Ascend910B1 -i <cann-8.5.0-root>
```

Success requires the generated binary to execute on the NPU and
`verify_result.py` to report `error ratio: 0.0000` and `test pass`.

The scaffold remains an external validation fixture; it is not added as a
runtime or candidate dependency. Production candidates must declare their own
device, host-tiling, registration, and launch sources in `solution.json`.

