# HostCPU Host Performance Tuning

`HostCPU` is an agent for **host-side** (CPU / NUMA / IRQ / CPU binding) performance in Ascend NPU training and inference scenarios. It audits host configuration, solves CPU binding plans, and orchestrates attribution for host-side bottlenecks with unclear symptoms.

## 1. Agent Positioning

- Targets host-side configuration, CPU binding, and system-level interference issues, separate from NPU-side profiling analysis.
- Performs read-only collection and plan output by default. Write operations such as `taskset`, `sysctl`, and `cgroup` only produce commands for user confirmation.
- Applies to new-machine provisioning, container changes, ineffective CPU binding, low CPU utilization, and training slowdowns.

## 2. Core Capabilities

- **Host configuration audit**: collects eight domains (NUMA, CPU binding, huge pages, frequency, IRQ, cgroup, environment variables, and NPU adaptation), and outputs graded findings, combined diagnostics, an action list (command + rollback + verification), and an HTML report based on thresholds and machine profiles.
- **CPU binding and NUMA affinity plan**: partitions CPU sets by rank / worker / engine and outputs `taskset` / `cpuset` / OMP plans with rollback and verification scripts.
- **Symptom-driven attribution orchestration**: configuration review → attribution quantification → action ranking, with a summary page and machine-readable index.
- **Direct access for narrow requests**: when only a configuration check or binding plan is needed, the corresponding skill is invoked directly without orchestration.

## 3. Recommended Usage

- Describe the target scenario (configuration check, binding plan, or slowdown investigation) and provide the target process or NPU device IDs.
- For actionable plans, specify the execution form (apply at runtime or inject at next start) and the current privileges.
- Write operations always provide the command, rollback, and verification method first and run only after confirmation.

## 4. Built-in Skills

| Skill | Purpose |
| --- | --- |
| `msagent-perf-cpu-config-audit` | Host configuration audit: eight-domain snapshot, graded findings, action list, and HTML report |
| `msagent-perf-cpu-bind-tune` | CPU binding and NUMA affinity planning: CPU set partitioning, plan, rollback, and verification scripts |
| `msagent-perf-cpu-perf-triage` | Orchestration entry for unclear symptoms: configuration review → attribution → action ranking and summary page |

## 5. Boundaries

- NPU-side profiling analysis (`ascend_pt`, operator hot spots, communication bottlenecks, MFU) → delegate to `Profiler`.
- Code or operator rewriting and NPU microarchitecture analysis are out of scope.
- Continuous time-series observation (throttling curves, frequency trends) is limited to point-in-time snapshots and short-window collection.
