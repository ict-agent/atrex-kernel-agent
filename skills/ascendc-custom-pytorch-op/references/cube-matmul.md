# Cube/Matmul contract

Use this checklist when implementing a self-authored Cube path rather than calling an installed Matmul
operator.

## Tiling ownership

Choose exactly one supported model:

1. Host-generated runtime tiling: configure the release-matched matmul tiler, obtain a complete
   `TCubeTiling`, serialize/pass it through the declared kernel ABI, and call `Matmul::Init` with the
   valid device-visible tiling object; or
2. A documented constant/static-tiling path whose API explicitly guarantees which runtime fields may
   be omitted.

Do not reinterpret `MatmulApiStaticTiling` as `TCubeTiling`, pass a null pointer merely to satisfy an
overload, or reconstruct a runtime tiling by copying a convenient subset of fields.

## Scheduler agreement

The host launch, tiling object, and device kernel must agree on:

- physical `blockDim` and tiling `usedCoreNum`;
- single-core M/N/K ranges and tail ownership;
- whether the high-level Matmul object or user code owns global tile scheduling;
- A/B transpose and leading dimensions;
- C/FixPipe output layout and row stride.

Avoid two independent schedulers. If user code loops over global tiles with `GetBlockIdx()`, prove that
the selected Matmul API treats its inputs as one local tile and does not apply a second global block
offset internally.

## Bring-up sequence

1. Start with one block and one full base tile.
2. Validate A/B/C pointers without bias, then add transpose, full output stride, and bias separately.
3. Validate the exact serialized tiling bytes received by the device before expanding block count.
4. Expand to the full scheduler only after the one-tile path is correct.
5. Treat MPU/MTE/FixPipe faults as address/layout/tiling evidence. Inspect `cube error`, MTE, and FixPipe
   fields separately; a zero Cube arithmetic error does not validate the surrounding data path.
6. Confirm the compiled or profiled candidate actually contains/executes the Cube path when Cube use is
   a requirement.

The `ops-transformer/experimental/posembedding/rope_matrix` reference demonstrates the important
separation: a shared tiling definition, host tiling generation, PyTorch stream-aware launch, explicit
tiling arguments, and device-side tiling copy before Cube execution. Derive the contract; do not copy
its implementation.
