---
name: ascend-computation-analysis
description: Analyze Ascend NPU computation-side profiling data for single-card runs or a selected rank from multi-card runs. Use this skill when the user asks to diagnose computation bottlenecks, AI Core / AI Vector / AICPU hotspots, dynamic shape overhead, block dim issues, redundant TransData/Transpose/Cast, cross-stream waits, frequency/runtime-state issues, or fusion opportunities.
---

# Ascend Computation Analysis

This skill diagnoses Ascend NPU computation-side performance problems and turns profiling evidence into prioritized optimization directions.

Keep the main workflow evidence-first. Computation problems often have multiple contributing factors, so do not force every case into one root-cause category. Use the failure handbook as a reference after evidence has been collected.

## Scope

Use this skill for:

- Single-card computation profiling analysis.
- Multi-card profiling when the task is to select one computation-heavy rank and analyze it.
- AI Core, AI Vector, AICPU, dynamic shape, block dim, frequency, fusion, data movement, and cross-stream wait analysis.
- Interpreting computation-related `msprof-analyze advisor` findings.

Do not use this skill as the primary workflow for:

- Communication matrix, slow link, bandwidth, or HCCL root-cause analysis.
- Low-level DB schema exploration or ad hoc SQL design.
- Full cluster fast/slow rank diagnosis beyond selecting the computation-heavy rank for local drill-down.

## Input Priority

Prefer evidence in this order:

1. `ascend_pytorch_profiler_{rank_id}.db`
2. `kernel_details.csv`
3. `op_statistic.csv`
4. `profiler_info.json`
5. `step_trace_time.csv` or `analysis.db` / `StepTraceTime`
6. `msprof-analyze advisor` stdout, HTML, or XLSX artifacts

Use a DB-first, CSV-fallback, advisor-cross-check approach.

If both DB and CSV are present, prefer DB for flexible queries and use CSV for quick summaries or schema-compatible fallback. If only CSV is present, continue with degraded analysis instead of blocking.

## Tooling Boundary

This skill owns the reasoning workflow:

- Input routing.
- Evidence selection.
- Hypothesis building.
- Contributing-factor analysis.
- Recommendation priority.

Use `msprof_mcp` for stable data access when available, especially:

- `get_profiler_config`
- `analyze_kernel_details`
- `analyze_op_statistic`
- `get_operator_details`
- `execute_sql`
- `execute_sql_to_csv`
- `msprof_analyze_advisor`

Do not create narrowly scoped scripts for every issue type. If a reusable capability is missing, prefer direct DB/CSV analysis in the current task and note the missing reusable capability as future `msprof_mcp` work.

## Workflow Overview

Follow this order:

1. Validate and route input.
2. Select the analysis target.
3. Build the computation overview.
4. Drill down into operators and sequences.
5. Use advisor as a cross-check.
6. Consult the failure handbook and fusion rules.
7. Prioritize findings and produce recommendations.

Do not jump directly to a single Top operator before building the overview. Do not force a single root cause when evidence points to multiple contributing factors.

## Step 1: Validate And Route Input

Identify the available data and decide the analysis mode.

Check:

- Whether the input is single-card or multi-card.
- Whether `ascend_pytorch_profiler_{rank_id}.db` exists.
- Whether `kernel_details.csv` or `op_statistic.csv` exists.
- Whether `profiler_info.json` exists.
- Whether advisor artifacts already exist.

Read `profiler_info.json` when available and determine profiler level.

Interpret profiler level this way:

- `level1` or higher: shape, dtype, format, op type, block dim, and ratio-style fields may be available; detailed evidence review is possible.
- `level0`: analysis is limited. Op type, shape, dtype, format, block dim, and ratio fields may be missing. State which checks are degraded.

Output of this step:

- Data source selected.
- Analysis mode selected.
- Field limitations.
- Whether conclusions need lower confidence because of missing fields.

## Step 2: Select The Analysis Target

For single-card input, analyze that card directly.

For multi-card input, keep the selection lightweight:

- Use cluster-level `ClusterStepTraceTime` table in `cluster_analysis_output/cluster_analysis.db`, or aggregate rank-level `step_trace_time.csv` or `StepTraceTime` table in `analysis.db` when available.
- Select the rank with the longest computation time as the main analysis target.
- If computation time is unavailable, select the rank with the largest device compute operator time.

Do not perform full cluster fast/slow rank diagnosis in this skill. Once the target rank is selected, continue as a single-rank computation analysis.

Output of this step:

- Target rank.
- Selection evidence.
- Missing rank-selection evidence, if any.

## Step 3: Build The Computation Overview

Before drilling into any operator, build an overview from the selected data source.

Required summaries:

1. Operator type summary:
   total duration, count, average duration, and Top types.
2. Accelerator core summary:
   time share of `AI_CORE`, `AI_VECTOR`, `AI_CPU`, `MIX_AIC`, `MIX_AIV`, or equivalent categories.
3. Top operator summary:
   total duration, count, average duration, and max duration.
4. Stream or task category summary when available:
   compute streams, communication streams, auxiliary streams, waits, and `OTHER` tasks.

The overview should produce observations, not final classification.

Output of this step:

- Computation time distribution.
- Initial observations.
- Drill-down candidates.

## Step 4: Drill Down Into Operators And Sequences

Use two complementary drill-down modes.

### Operator Drill-Down

Use this for dominant single operators or operator types.

Inspect when available:

- Name and op type.
- Total, count, average, max duration.
- Shape clusters.
- Input/output dtype and format.
- Accelerator core.
- Block dim.
- Ratio fields such as `mac_ratio` and `mte2_ratio`.
- PMU-like metrics if already available.

For AI Core operators:

- A high `mac_ratio` may suggest compute-side saturation.
- A high `mte2_ratio` may suggest data movement pressure.
- Low ratios across the board may indicate shape, launch, fusion, or sequence-level problems.

For small-M MatMul/Cube cases:

- Do not reason only from FLOPs.
- Weight movement and repeated loading may dominate.
- Consider micro-batch, partitioning, and repeated-weight-movement changes.

### Sequence Drill-Down

Use this for repeated small operators, no-bound operators, conversion-heavy paths, or fusion opportunities.

For computation-side operator sequences, first group by the same `stream_id`. Do not infer a compute sequence by globally sorting timestamps across multiple streams; operators that are close in time but run on different streams may be parallel, independent, or connected only through explicit wait/sync edges.

Look for:

- `Cast -> TransData -> Compute`
- `Transpose -> Compute`
- repeated `Cast`
- many short vector operators
- format conversion before and after AI Core kernels
- synchronization or wait around auxiliary streams

Sequence-level findings often produce better recommendations than isolated single-op tuning.

Output of this step:

- Operator-level findings.
- Sequence-level findings.
- Candidate contributing factors.
- Confidence notes.

## Step 5: Use Advisor As Cross-Check

If advisor artifacts are not already available and `msprof-analyze` can be executed, run it on the selected single-card profiling directory:

```bash
msprof-analyze advisor all -d <single_card_profiling_path> -o <advisor_output_path>
```

Advisor execution rules:

- Always pass `-o`. Advisor creates new deliverables on every execution, so do not rely on the default output location.
- Make `<advisor_output_path>` descriptive enough to identify the analyzed data. Include useful features such as rank id, scenario, for example `advisor_rank3_compute`, `advisor_rank0_prefill`, or `advisor_rank7_decode`.
- For multi-card input, `<single_card_profiling_path>` must be the selected target-rank profiling directory, not the cluster root.
- Record the exact command and output path used.

Advisor commonly emits both HTML and XLSX artifacts. Prefer XLSX as the primary machine-readable source. Use HTML only when the XLSX file is missing, incomplete, or unreadable.

Use advisor to:

- Find known rule-based problems.
- Provide additional evidence for optimization direction.

Do not use advisor to:

- Replace the overview.
- Decide priority without checking time impact.
- Copy generated text into the final answer.

Relevant computation-related advisor sheets or issue names include:

- `AICPU Issues`
- `Operator Dynamic Shape Issues`
- `AI Core Performance Analysis`
- `Cube Operator Perf Analysis`
- `FA Operator Perf Analysis`
- `Vector Operator Perf Analysis`
- `Block Dim Issues`
- `Operator No Bound Issues`
- `AI Core Frequency Issues`
- `Affinity API Issues`

Advisor reading rules:

- Read the overview or problem summary first, the sheet name should be `问题综述` or `problems`. It also contains suggestions for every problem.
- List actual sheet names before assuming a sheet exists.
- Read only triggered computation-related sheets by default.
- Re-rank advisor findings by measured time impact.

Output of this step:

- Advisor findings that support existing evidence.
- Advisor findings that add new hypotheses.
- Advisor findings that appear low-impact or unsupported.

## Step 6: Consult References

After operator and sequence drill-down, consult references as needed:

- Read `references/failure_handbook.md` to map evidence to likely contributing factors and to avoid common false positives.
- Read `references/fusion_rules.md` only when repeated operator sequences, small-operator overhead, vector-heavy sequences, or fusion opportunities are observed.

Use them as a handbook for interpreting evidence. A finding may match multiple patterns, and some cases may remain inconclusive.

## Step 7: Prioritize Findings And Recommend Next Actions

Prioritize by:

1. Time impact.
2. Evidence strength.
3. Likelihood that the proposed change is actionable.
4. Risk or invasiveness of the change.

Prefer recommendations framed as next experiments:

- What to change.
- What metric should improve.
- What evidence would confirm or reject the hypothesis.

## Output Format

The final answer should follow this shape:

1. Summary:
   main performance story and confidence.
2. Evidence:
   key timings, counts, ratios, ranks, streams, operators, or advisor findings.
3. Likely contributing factors:
   multiple factors are allowed; avoid forcing one root cause.
4. Recommendations:
   concrete next experiments or code/model changes, ordered by expected impact.
5. Limitations:
   missing fields, degraded profiler level, unavailable advisor sheets, or incomplete rank data.

Avoid:

- Full raw tables.
- SQL dumps.
- Advisor text replay.
- Recommendations without time-impact evidence.
- Overstating conclusions when fields are missing.

## Degraded Modes

If ideal data is missing, continue with a clear limitation.

- No DB but CSV exists: use CSV summaries.
- `level0` data: avoid strong shape, dtype, format, ratio, and block-dim conclusions.
- No advisor output: continue with raw profiling evidence.

Never block the analysis solely because one preferred artifact is missing.
