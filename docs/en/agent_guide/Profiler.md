# Profiler Performance Tuning

`Profiler` is an agent for Ascend Profiling and performance tuning scenarios. It converts complex performance data into structured conclusions, root-cause analysis, and actionable optimization recommendations.

## 1. Agent Positioning

- Targets Ascend performance analysis scenarios, such as single-device, multi-device, and cluster environments.
- Focuses on Profiling data interpretation, bottleneck identification, and tuning recommendations.
- Applies to the analysis of issues such as fast and slow devices, slow nodes, MFU, communication bottlenecks, operator hot spots, and launch scheduling.

## 2. Core Capabilities

- Profiling data inspection and data quality verification
- MFU calculation, formula explanation, and result interpretation
- Analysis of fast and slow devices, slow nodes, and load imbalance in clusters
- Identification of communication bottlenecks, operator hot spots, and host launch and scheduling issues
- Structured analysis and export of deliverables such as DB, CSV, and Trace files
- Breakdown of the durations of stages such as RL and inference based on **DB format** deliverables
- Fact verification of Ascend official documentation, community resources, API and tool semantics, and sample and open-source code (the `ascend-knowledge` subagent is delegated to collect evidence and returns evidence with sources and version or branch information)

## 3. Recommended Usage

- Provide the Profiling data directory path directly and describe the problem you want to solve.
- For cluster or multi-device issues, describe the anomaly, the involved rank, or the training stage wherever possible.
- If your goal is data extraction or export, provide the DB or CSV file paths and the target format directly.
- If your question involves the use of msprof, CANN, or Ascend tools, API semantics, or repository documentation, ask directly. Profiler verifies the official documentation and repository materials through the `ascend-knowledge` subagent (ascend-doc-mcp channel).

## 4. Typical Results

| Scenario | Example Prompt | Sample Output |
|---|---|---|
| MFU calculation | `Based on path/to/kernel_details.csv, calculate the MFU (10B3) of matmul and explain the basis of each calculation.` | <img src="../figures/kernel-details-mfu-file.png" alt="MFU calculation example" width="800"> |
| Fast and slow device diagnosis | `Analyze whether /path/to/cluster_profiling/ contains fast and slow device issues, locate the abnormal rank, and provide possible causes.` | <img src="../figures/slow-rank-detect.png" alt="Fast and slow device diagnosis example" width="800"> |
| Profiling data inspection | `Analyze whether the data in /path/to/xxx_ascend_pt/ was collected properly.` | <img src="../figures/profiler-data-check.png" alt="Data integrity verification example" width="800"> |
| `msprof` tool usage consultation | `How do I compile the run package with msprof?` | <img src="../figures/msprof-build.png" alt="Tool consultation example" width="800"> |
| DB custom content export to CSV | `Based on ascend_pytorch_profiler_0.db, extract the total duration of each operator type and output them to a CSV file in descending order.` | <img src="../figures/db-export.png" alt="Data export example" width="800"> |
| VERL reinforcement learning duration breakdown | `Based on ascend_pytorch_profiler_0.db, help me analyze the duration of each RL stage and output a breakdown report` | <img src="../figures/profiling_breakdown.png" alt="Duration breakdown example" width="800"> |
