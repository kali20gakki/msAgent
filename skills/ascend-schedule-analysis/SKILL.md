---
name: ascend-schedule-analysis
description: Analyze Ascend NPU schedule, operator dispatch, operator launch, and Host Bound profiling issues in Ascend profiling data. Use when need to diagnose device Free time, framework/operator dispatch latency, launch latency, PYTORCH_API/CANN_API launch gaps, aclrtSynchronizeStream stalls, task queue behavior, CPU scheduling interference, GC/lock pauses, CPU affinity, or schedule-side optimization actions.
---

# Ascend Schedule Analysis

Analyze NPU scheduling issues from evidence first. Do not start with tuning suggestions. Establish whether the workload is actually Host Bound, then separate model-side dispatch pressure from CPU runtime scheduling interference and environment/configuration issues.

## Core Workflow

1. Validate the profiling format and available artifacts.
2. Establish whether Host Bound exists.
3. Split Free time into preparation, in-step gaps, and post-dispatch queueing behavior.
4. Identify the dominant cause category.
5. Map each finding to the smallest practical optimization experiment.

Prefer DB profiling data with `ascend_pytorch_profiler_{rank_id}.db`. For DB data, first try to create or use the dispatch view through `msprof_mcp-create_dispatch_view`. If the view is unavailable, fall back to direct DB queries only for the fields required by the current question.

For cluster profiling data, rank candidates by device Free time first. Prefer analyzing the rank with the longest Free time, because it is the most likely Host Bound suspect. When the evidence is ambiguous or when a fast/slow-rank explanation is needed, compare dispatch APIs between a long-Free rank and a short-Free rank in the same step range.

## Evidence Standard

Treat a schedule issue as Host Bound only when the evidence shows device starvation:

- Device-side `Free` time is greater than 10% of the step or is much larger than neighboring/rank baseline.
- The Free interval contains neither compute nor communication tasks.
- Dispatch or host-side gaps line up with the Free interval.

Do not infer Host Bound from high API latency alone. A slow API can still be device-bound if the device already has queued work.

Always report:

- the rank and step analyzed
- the Free-time ratio or duration
- the specific interval or API sequence causing the conclusion
- whether the conclusion is high-confidence or degraded because fields/artifacts are missing

## Split Free Time

For one step, split device Free time into three parts:

Before manually reconstructing gaps, first look for the step-level summary. Both device Free time and Preparation time are usually available in `analysis.db/StepTraceTime` or `step_trace_time.csv`.

1. Preparation time: from step start to first device task. Prefer `analysis.db/StepTraceTime` or `step_trace_time.csv`.
2. In-step Free gaps: from the first device kernel/task to the last kernel/task, identify large empty intervals. Use `msprof-analyze -m free_analysis` when the input path is a cluster path or a single-card parent directory.
3. Tail or boundary effects: time after the last relevant device task. Treat this separately from dispatch bottlenecks unless it blocks the next step.

For `free_analysis`, remember the path limitation: do not pass the single-card directory directly; pass the cluster data path or the single-card parent directory.

Typical command:

```bash
msprof-analyze -m free_analysis -d <cluster_or_parent_path> -o <output_dir> --agent
```

## Interpret Dispatch Geometry

Use the relationship between framework call, kernel launch, and device execution:

```text
call ops -> kernel launch -> kernel run
```

For PyTorch on Ascend, `PYTORCH_API` usually represents framework/operator calls that enqueue work into a task queue. An async `acl_thread` in the CANN layer dequeues and launches concrete `launch` work onto the device.

When looking at dispatch lines or paired API/task timing:

- A nearly vertical dispatch-to-run relationship means the task starts soon after launch. If this coincides with device Free time, classify it as Host Bound.
- A delayed device start after launch can be healthy device-side queueing when other device work is already running.
- Pay special attention to intervals where the slope suddenly becomes more vertical; these are often where host dispatch stops hiding behind device work.

## Check Abnormally Slow APIs

Before assigning the gap to CPU scheduling or model structure, check whether any same-kind API has abnormal latency.

Scope:

- Prefer `PYTORCH_API` and `CANN_API` records in the target step.
- Focus on `depth = 1` records when depth is available. `depth = 0` is often the outer `ProfilerStep#` wrapper and is usually too coarse for root cause.
- Compare like with like: `aten::` against other `aten::` calls of the same name, `npu::` against the same `npu::` API, `launch` against other launch APIs, and `Enqueue@` / `Dequeue@` against the same queue operation.

Procedure:

1. Group APIs by normalized name or prefix plus exact name when available.
2. Compute count, total duration, average duration, p95/p99, max duration, and the timestamp of the max event.
3. Flag an API when its max or high percentile is much larger than the same API on neighboring steps, the same rank outside the gap, or a short-Free rank in cluster data.
4. Correlate the flagged API timestamp with device Free intervals and dispatch-to-run geometry.

Interpretation:

- `launch` time is included inside `Dequeue@` time. Interpret abnormal `launch` latency as part of the dequeue-side launch path, not as an independent interval to add again.
- A very long `Dequeue@` often means the secondary pipeline `acl_thread` was interfered with or delayed.
- A very long `Enqueue@` can mean the secondary pipeline `task_queue` is full and the framework dispatch thread is waiting for dequeue consumption; it can also mean the framework dispatch thread itself was interfered with.
- Abnormally long `aten::` or `npu::` APIs can also indicate framework dispatch thread interference. Judge this by the frequency, time interval pattern, and whether the slow events align with device Free intervals.
- If the API is slow but the device already has queued work, treat it as an optimization opportunity rather than proof of Host Bound.

## Root Cause Buckets

Classify findings into one or more buckets.

### Model-side dispatch pressure

Look for:

- many high-frequency small operators where launch overhead is comparable to device runtime
- sequences with repeated tiny kernels and low device occupancy
- too many individual launches that could be fused or batched

Actions:

- recommend operator fusion or model/code restructuring first
- prefer reducing launch count over micro-optimizing every tiny kernel
- use computation-side analysis for repeated `Cast`, `TransData`, `Transpose`, or other fusible sequences

### Synchronization breaks

Look for:

- `aclrtSynchronizeStream` or equivalent sync calls in the gap
- PyTorch operations such as `aten::item`, `.cpu()`, scalar extraction, host-side logging, or data-dependent control flow around the gap

Actions:

- ask whether the synchronization is semantically required
- move sync out of the hot path, batch it, or keep values on device where possible
- distinguish explicit sync from natural stream dependency waits

### CPU-only work

Look for:

- long intervals where the main dispatch thread performs CPU work without issuing device tasks
- Python preprocessing, dataloader handling, logging, metric formatting, shape/control logic, or host tensor manipulation

Actions:

- reduce hot-path Python work
- precompute or cache shape/control results
- overlap CPU preparation with device execution when possible

### CPU scheduling interference

Look for:

- dispatch thread descheduled during the gap
- CPU contention, IRQ interference, NUMA mismatch, or thread migration
- unexplained empty bubbles where the launch thread has no recorded API activity

Actions:

- use ftrace or OS scheduling evidence when available
- inspect hot threads: main forward dispatch thread, backward dispatch thread, and secondary pipeline `acl_thread`
- consider CPU affinity only after evidence shows scheduling interference or strong fast/slow-rank symptoms

### GC, locks, and allocator effects

Look for:

- periodic pauses
- lock contention around enqueue/dequeue or Python runtime activity
- memory allocation churn around the gap

Actions:

- correlate gap periodicity with GC or allocator events
- consider high-performance allocator replacement only when host allocation overhead is visible

## Optimization Map

Use the least invasive action that matches the evidence.

### TASK_QUEUE_ENABLE

Use when the evidence shows host dispatch work is not sufficiently overlapped with device execution, especially when CANN/aclnn launch work appears on the critical path and device Free time is waiting for the next task.

Official behavior: `TASK_QUEUE_ENABLE` configures whether the `task_queue` operator dispatch queue is enabled and which optimization level is used.

Values:

- `0`: disable `task_queue` operator dispatch queue optimization.
- `1` or unset: enable Level 1 optimization. This splits operator dispatch into two pipeline stages; part of the work, mainly aclnn operator calls, is placed on the secondary pipeline. The primary and secondary pipelines pass tasks through an operator queue and run in parallel, hiding part of dispatch latency.
- `2`: enable Level 2 optimization. This includes Level 1 and further balances primary/secondary pipeline load, mainly by moving workspace-related work to the secondary pipeline. This gives stronger hiding and is the recommended value when the binary scenario supports it.

Recommended experiment:

```bash
export TASK_QUEUE_ENABLE=2
```

Interpretation rules:

- If `ASCEND_LAUNCH_BLOCKING=1`, task queue is disabled and `TASK_QUEUE_ENABLE` does not take effect.
- Level 2 can increase NPU memory peak because of memory concurrency. Mention this risk when recommending it.
- Do not recommend `TASK_QUEUE_ENABLE=2` as a generic cure-all. Tie it to evidence that first-level dispatch or aclnn/workspace-related host work is on the critical path.
- After changing it, compare step time, device Free ratio, and the dispatch-to-run geometry in the same rank/step range.

### CPU affinity

Use when dispatch threads are preempted, migrate across cores/NUMA nodes, or fast/slow-rank symptoms suggest CPU scheduling instability.

Official environment variable:

```bash
export CPU_AFFINITY_CONF=<mode>,npu<card_id>:<start_core>-<end_core>
```

Modes:

- `0` or unset: disable affinity
- `1`: coarse-grained affinity, bind all threads related to one NPU card into the configured CPU core range
- `2`: fine-grained affinity, bind major threads related to one NPU card to isolated cores

Examples:

```bash
export CPU_AFFINITY_CONF=1
export CPU_AFFINITY_CONF=2
export CPU_AFFINITY_CONF=1,npu0:0-1,npu1:2-5,npu3:6-6
```

Before recommending custom core ranges, check NPU/NUMA affinity with:

```bash
npu-smi info -t topo
```

Warn that cross-NUMA memory access can hide the benefit of binding.

### Compile optimization

Use when the host-side framework/runtime overhead is broad rather than tied to one API. Ascend PyTorch documentation describes compiler-based optimization with Bisheng, including LTO and PGO, across Python, PyTorch, and torch_npu compatibility combinations.

Treat this as a larger environment experiment, not a first-line code fix. Recommend it after high-frequency small ops, sync points, and CPU scheduling are checked.

### High-performance memory allocator

Use when host memory allocation overhead or lock contention is visible. The official PyTorch tuning doc recommends `tcmalloc` through `LD_PRELOAD`.

Example:

```bash
LD_PRELOAD="/usr/local/lib/libtcmalloc.so" python train_script.py
```

Verify with:

```bash
ldd $(which python)
```

Do not recommend allocator replacement solely because a workload is Host Bound.

### ACLNN cache

Use when the dispatch bottleneck appears in first-level pipeline or dynamic-shape operator API execution.

Official behavior: `ACLNN_CACHE_LIMIT` configures the number of single-operator Host-side cache entries. The cache stores information such as workspace size, executor, and tiling data. The documented range is `[1,10000000]`, and the default is `10000`.

Example:

```bash
export ACLNN_CACHE_LIMIT=100000
```

Warn that increasing the cache raises Host memory usage. This is most plausible for dynamic-shape workloads with a large shape range.


## Search Policy

If more external documentation is needed, first use the links already present in this skill or the official Ascend documentation. Strongly prefer Tavily only when the user has configured `TAVILY_API_KEY`. If the key is unavailable and the user did not explicitly ask for web search, do not proactively search.

Useful official references:

- `TASK_QUEUE_ENABLE`: https://www.hiascend.com/document/detail/zh/Pytorch/720/comref/Envvariables/Envir_007.html
- CPU affinity optimization: https://www.hiascend.com/document/detail/zh/mindstudio/830/practicalcases/GeneralPerformanceIssue/toolsample6_058.html?framework=mindspore
- Compile optimization: https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/comp_opt_intro.md
- High-performance memory allocator: https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/hi_perf_mem_pool_sub.md
- Other common optimization operations and `ACLNN_CACHE_LIMIT`: https://www.hiascend.com/document/detail/zh/Pytorch/2600/ptmoddevg/trainingmigrguide/FrameworkPTAdapter/26.0.0/zh/pytorch_model_migration_fine_tuning/other_common_opt_ops.md
- Ascend docs source tree: https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/zh/

## Output Shape

Write the final analysis in this order:

1. Conclusion: Host Bound, device-bound queueing, mixed, or inconclusive.
2. Evidence: rank/step, Free ratio, key intervals, API/task names, dispatch geometry.
3. Cause bucket: small-op launch pressure, sync break, CPU-only work, CPU scheduling, GC/lock/allocator, or mixed.
4. Optimization experiments: list only actions supported by evidence.
5. Missing evidence: state exactly what would improve confidence.

Keep recommendations concrete and testable. Prefer "try `ACLNN_CACHE_LIMIT=100000` for this dynamic-shape first-level dispatch bottleneck and compare Free time" over broad statements like "optimize scheduling".
