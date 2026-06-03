# Ascend Computation Failure Handbook

Use this reference after the main workflow has collected overview, operator, sequence, and advisor evidence.

Do not force a single pattern. Computation issues often combine multiple contributing factors. For each finding, report the evidence strength and the expected time impact.

## AI Core Pipeline Inefficiency

### Signals

- AI Core operators dominate runtime but ratio metrics are not strong.
- Same operator shape has unstable or unexpectedly poor average duration.
- Cube, MatMul, or FA operators appear in the Top operators but do not look compute-saturated.

### Evidence To Check

- Same-shape average, max, and variance.
- `mac_ratio`, `mte2_ratio`, and similar ratio fields when available.
- Block dim and AI Core count.
- Input/output dtype and format.
- Whether the operator is surrounded by conversion or synchronization tasks.

### Common False Positives

- Very short kernels where ratio metrics are noisy.
- Dynamic-shape paths where shape grouping is incomplete.
- Frequency drops that are actually a secondary runtime effect.

### Optimization Directions

- Improve shape affinity.
- Adjust batch, head, tile, sequence length, or tensor-parallel partitioning.
- Reduce adjacent data movement.
- Consider fusion only if the surrounding sequence shows repeated overhead.

### Confidence Notes

Confidence is high only when same-shape evidence, ratio fields, and time impact agree.

## Shape Affinity Issue

### Signals

- Hot AI Core operators have shape clusters with consistently worse duration.
- `mte2_ratio` is high or data movement looks dominant.
- Tensor-parallel partitioning creates unfriendly inner dimensions or tail axes.

### Evidence To Check

- Input and output shapes.
- Input and output formats.
- Whether `NZ` or other internal-friendly formats are used.
- Tail-axis alignment and partitioned dimensions.
- Hot shape frequency and cumulative time.

### Common False Positives

- One-off shapes with low cumulative time.
- Shape data missing because profiler level is too low.
- Slowdown caused by adjacent `TransData` rather than the AI Core kernel itself.

### Optimization Directions

- Adjust shape choices or partitioning.
- Improve layout or format compatibility.
- Reduce repeated conversion around the hot operator.

### Confidence Notes

Treat detailed alignment rules as clues unless the data explicitly supports them.

## Data Movement Pressure

### Signals

- `mte2_ratio` or equivalent memory-movement ratio is high.
- `TransData`, `Transpose`, or `Cast` appears around hot operators.
- Small-M inference MatMul/Cube is dominated by repeated weight movement.

### Evidence To Check

- Ratio fields on hot operators.
- Repeated conversion sequences.
- Whether weights or activations are repeatedly loaded.
- Shape and format compatibility.

### Common False Positives

- High total duration from compute-saturated MatMul misread as memory pressure.
- Conversion operators that are present but low cumulative time.

### Optimization Directions

- Reduce repeated format conversion.
- Avoid unnecessary dtype oscillation.
- Revisit micro-batch and partitioning choices.
- Reduce repeated weight movement in decode or small-M paths.

### Confidence Notes

Confidence increases when operator ratios and sequence evidence both point to movement pressure.

## AI Vector-Heavy Expression

### Signals

- AI Vector time share is high.
- Top vector operators are frequent or high cumulative time.
- Known expensive vector patterns appear, such as `IndexPutV2`, `NonZero`, `where`, or `SelectV2`.

### Evidence To Check

- AI Vector total time share.
- Per-operator count, total duration, and average duration.
- Whether vector operators appear in repeated hot sequences.
- Whether the surrounding model expression can be rewritten.

### Common False Positives

- A vector operator appears often but has low total duration.
- The vector-heavy section is not on the critical path.

### Optimization Directions

- Replace sparse or index-heavy expressions when possible.
- Rewrite toward AI Core-friendly computation.
- Fuse or remove repeated vector sequences.

### Confidence Notes

Priority depends on time share, not operator name alone.

## AICPU Fallback

### Signals

- `AI_CPU` tasks have meaningful cumulative duration.
- One or more AICPU operators have high average or max duration.
- AICPU appears on the critical path rather than as a negligible side task.

### Evidence To Check

- AICPU total time share.
- AICPU operator names and counts.
- Dtype, shape, and framework source op when available.
- Whether the operator is a communication helper.

### Common False Positives

- Communication-related AICPU kernels such as `allgatherAicpuKernel` or `allreduceAicpuKernel`.
- Very short AICPU tasks with low total impact.

### Optimization Directions

- Replace unsupported or unfriendly operators.
- Change dtype or shape path when safe.
- Trace back to the framework op and rewrite the expression.

### Confidence Notes

Treat AICPU as primary only when cumulative time is material.

## Dynamic Shape Or Graph Overhead

### Signals

- Dynamic-shape operators appear in hot paths.
- The same logical operator appears with many shape variants.
- High operator count and repeated shape inference or graph overhead are suspected.

### Evidence To Check

- Op state or dynamic-shape fields.
- Shape clusters and their counts.
- Hot path operator counts.
- Advisor dynamic-shape findings.

### Common False Positives

- Shape fields unavailable in `level0` data.
- Dynamic shape present but low time impact.

### Optimization Directions

- Converge hot-path shapes.
- Prefer static-shape paths where possible.
- Reduce shape variability before micro-tuning kernels.

### Confidence Notes

Confidence depends heavily on profiler level and shape visibility.

## Block Dim Or Insufficient Parallelism

### Signals

- Block dim is low or poorly aligned with available AI Core count.
- Same shape repeatedly shows poor parallelism.
- Hot operators underutilize cores.

### Evidence To Check

- `block_dim`, `mix_block_dim`, AI Core count, and shape.
- Repeated shape patterns.
- Batch, head, tile, sequence length, and tensor-parallel partitioning.

### Common False Positives

- Very small operators where high parallelism is not expected.
- Missing hardware/core-count fields.

### Optimization Directions

- Adjust model dimensions or partition strategy.
- Improve batch or tile choices.
- Avoid partitioning that creates small or irregular shards.

### Confidence Notes

Map the profiler symptom back to model dimensions before recommending changes.

## Redundant Cast / TransData / Transpose

### Signals

- `Cast`, `TransData`, or `Transpose` has meaningful cumulative time.
- These operators appear repeatedly around hot compute kernels.
- dtype or format oscillates across adjacent operations.
- Paired `TransData` operators appear around a compute operator, suggesting automatic format conversion into and out of an internal layout.
- Many `Transpose` or non-contiguous tensor conversion operations appear before consumers of the same tensor.

### Evidence To Check

- Sequence context before and after hot compute operators.
- dtype and format transitions.
- Whether tensors are non-contiguous and consumed repeatedly.
- Whether the framework is moving between standard `ND` format and NPU-private `NZ` format.
- Whether the target compute operator benefits enough from `NZ` to offset the inserted `TransData` cost.
- Whether one non-contiguous tensor is consumed by multiple downstream operators, causing repeated non-contiguous-to-contiguous conversion on NPU.

### Common False Positives

- A conversion exists but is cheap and not repeated.
- A single `Transpose` is required by model semantics.
- `NZ` improves the compute kernel itself, but the end-to-end sequence still slows down because two surrounding `TransData` operators dominate the saved compute time.

### Optimization Directions

- Reduce format churn.
- Avoid repeated dtype conversion.
- Use contiguous tensors only when repeated non-contiguous reuse causes repeated layout work.
- Align adjacent operators to compatible formats.
- If automatic internal-format conversion creates expensive paired `TransData` operators, test disabling it with `torch.npu.config.allow_internal_format = false`.
- Alternatively, choose a more suitable `data_format` or layout so adjacent operators agree on format without repeated conversion.
- If a non-contiguous tensor is reused by multiple downstream operators, test converting it to contiguous once immediately after the layout-changing operation.

### Confidence Notes

Sequence evidence is more important than individual operator presence.

`ND` is the standard format. NPU-private `NZ` format can better utilize Cube compute and is often faster for Cube-friendly operators, but not every operator supports or benefits from `NZ`. Always evaluate the full sequence cost, not just the compute kernel duration.

On NPU, the runtime may need to insert non-contiguous-to-contiguous conversion before consumers. If the same non-contiguous tensor is consumed multiple times, one early `.contiguous()` can replace repeated implicit conversions.

Example:

```python
# Non-contiguous reuse: each consumer may trigger extra conversion on NPU.
x = input_.permute(0, 1)
y1 = norm1(x)
y2 = norm2(x)

# Convert once before repeated reuse.
x = input_.permute(0, 1).contiguous()
y1 = norm1(x)
y2 = norm2(x)
```

## Cross-Stream Wait

### Signals

- Main compute stream waits on auxiliary stream work.
- `EVENT_WAIT` or wait time is material.
- Short auxiliary tasks serialize the main flow.

### Evidence To Check

- Stream IDs and wait relationships.
- Wait duration and waited-on task.
- Nearby AICPU, conversion, synchronization, or communication tasks.

### Common False Positives

- Expected synchronization with low total impact.
- Communication wait that belongs to a communication analysis workflow.

### Optimization Directions

- Remove unnecessary synchronization.
- Reduce auxiliary-stream conversion work.
- Move or fuse small blocking tasks when possible.

### Confidence Notes

Only prioritize this pattern when waits are on the critical path.

## Frequency Or Runtime-State Issue

### Signals

- Advisor or profiler data reports AI Core frequency drops.
- Frequency issues correlate with long-running hot kernels.

### Evidence To Check

- Frequency samples or advisor frequency findings.
- Correlation with hot operator windows.
- Whether code-path inefficiencies have already been ruled out.

### Common False Positives

- Frequency drop is secondary to idle/wait behavior.
- The main problem is still shape, data movement, or host dispatch.

### Optimization Directions

- Treat system-state tuning as later-stage unless evidence is strong.
- First address obvious code, shape, movement, and fusion issues.

### Confidence Notes

Do not make frequency the primary cause unless supported by strong correlation and time impact.

## No-Bound Operator Pattern

### Signals

- No single resource ratio dominates.
- Hot operator has unclear bottleneck characteristics.
- Advisor flags operator no-bound issues.

### Evidence To Check

- Ratio fields across compute and movement pipelines.
- Adjacent sequences.
- Operator count and duration.
- Presence of conversion, fusion, or launch overhead around the operator.

### Common False Positives

- Short kernels with noisy ratios.
- Incomplete ratio fields.

### Optimization Directions

- Switch from single-operator tuning to sequence analysis.
- Check fusion opportunities.
- Check surrounding `Cast`, `TransData`, `Transpose`, and wait tasks.

### Confidence Notes

This pattern is often a prompt to change analysis angle, not a final root cause.
