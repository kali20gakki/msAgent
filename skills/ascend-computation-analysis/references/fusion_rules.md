# Ascend Computation Fusion Rules

Use this reference only when the main workflow finds repeated operator sequences, small-operator overhead, vector-heavy expressions, redundant data movement, or advisor fusion findings.

Do not recommend fusion by name alone. Fusion is useful only when it changes a meaningful bottleneck.

## Core Principles

### Generality Principle

Prioritize fusion patterns that appear across multiple mainstream models or repeated hot paths in the current workload.

Avoid recommending a custom fusion just because a one-off sequence exists.

### Benefit Principle

Evaluate whether fusion changes the bottleneck.

Possible benefit layers:

- Host benefit: fewer operators, lower dispatch overhead, less graph scheduling overhead.
- Device benefit: fewer intermediate tensors, less load/store traffic, better producer-consumer pipeline behavior.

If the fused operators remain bound by the same pipeline with little movement or dispatch overhead removed, expected benefit may be low.

## Deep Fusion Vs Shallow Fusion

Deep fusion:

- Producer and consumer can work in block-level or tile-level handoff.
- Intermediate data may avoid full materialization.
- More likely to improve device-side efficiency.

Shallow fusion:

- Multiple logical operations are packaged together.
- Intermediate data or pipeline behavior may not change much.
- More likely to help host dispatch than device execution.

Use this distinction when explaining expected benefit.

## Fusion Opportunity Signals

Look for:

- Repeated short operator sequences with meaningful cumulative time.
- Many small elementwise or vector operators around AI Core kernels.
- Repeated `Cast`, `TransData`, or `Transpose` around a stable compute pattern.
- Host-bound behavior from many small operators.

Do not rely on:

- A single occurrence.
- Low cumulative time.
- Pattern name without measured impact.

## Fusion Patterns To Check

When these appear in hot repeated paths, check whether the official fused operator or replacement path is applicable:

- `RotaryMul` / `RotaryMulGrad`
- `RmsNorm` / `RmsNormGrad`
- `ScaledMaskedSoftmax` / `ScaledMaskedSoftmaxGrad`
- `MatmulAllReduce`
- `FlashAttentionScore`
- `SwiGlu`

Treat the list as a prompt for investigation, not as an automatic recommendation.

## Cube-Vector, Vector-Vector, And Small-Operator Fusion

Cube-Vector fusion:

- Combines compute and vector/data-movement style work.
- Benefit is more likely when it reduces materialization or improves cache/locality.
- Low data volume may limit benefit.

Vector-Vector fusion:

- Combines vector-style operations.
- Often helps when it removes repeated load, store, or cast overhead.

Small-operator fusion:

- Useful when launch/dispatch overhead dominates.
- Especially relevant for high-frequency tiny operators.

Always tie the recommendation to measured count and cumulative time.

## When Not To Recommend Fusion

Avoid strong fusion recommendations when:

- The sequence has low cumulative duration.
- The sequence is not on the critical path.
- The fused pattern would still be bound by the same dominant pipeline.
- The input data lacks enough operator sequence evidence.
- The recommendation would require model-code changes that are not justified by measured impact.

## Output Guidance

When recommending fusion, state:

- The repeated sequence or pattern.
- Its count and cumulative time when available.
- The expected benefit layer: host dispatch, memory movement, pipeline handoff, or dtype/format cleanup.
- The concrete experiment to try.
- The metric that should improve.

Do not generate replacement code without concrete model code context.
