# Ascend 910B1 torch.add smoke operator

This is a minimal native Atrex-Bench-compatible evaluator fixture for a
one-dimensional, 4096-element FP16 `torch.add` workload on `npu:0`.

The immutable V0 reference is:

```python
torch.add(x, y)
```

The evaluator runs correctness with fresh seeded inputs and measures only the
candidate invocation after NPU synchronization. It is intentionally small and
is meant for AscendC campaign smoke tests, not for publishing benchmark scores.

