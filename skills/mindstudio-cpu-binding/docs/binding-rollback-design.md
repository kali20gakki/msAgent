# 自动绑核执行与回滚设计

## 1. 目标

本文定义 `mindstudio-cpu-binding` 如何接入内部经过验证的绑核脚本，以及如何保存、验证和执行回滚。核心原则是：执行后端可以替换，但执行前校验、状态保存、审计记录和回滚语义必须由 `mindstudio-cpu-binding` 统一管理。

## 2. 执行后端

第一版支持两类执行后端：

| 后端 | 说明 | 推荐用途 |
|------|------|----------|
| `taskset` | 直接调用系统 `taskset -cp <cpu-list> <pid>` | 简单临时绑核、实验验证 |
| `internal-script` | 调用内部成熟绑核脚本 | 团队标准化绑核流程、生产前验证 |

Skill 不应把内部脚本逻辑硬编码到诊断规则中，而应通过执行适配器接入。

## 3. 内部脚本接入要求

内部脚本建议提供明确的 apply / rollback 或 apply / query 能力。若脚本没有 rollback 子命令，`mindstudio-cpu-binding` 仍可基于保存的原始 affinity 生成回滚动作。

建议接口：

```bash
internal-bind --apply --pid <pid> --cpu-list <cpu-list> [--reason <text>]
internal-bind --rollback --pid <pid> --cpu-list <original-cpu-list> [--state <rollback-state.json>]
internal-bind --query --pid <pid>
```

最低要求：

1. 能对指定 PID 应用 CPU affinity。
2. 能返回成功/失败退出码。
3. 失败时输出可读错误信息。
4. 不隐式 kill 或 restart 进程。
5. 不修改 cgroup、IRQ affinity、CPU governor、kernel 参数等中高风险系统配置，除非后续单独设计权限边界。

## 4. 执行适配器

`mindstudio-cpu-binding` 应通过 `BindingExecutor` 统一封装执行：

```text
BindingExecutor
├── TasksetExecutor
└── InternalScriptExecutor
```

无论使用哪种后端，都必须经过相同前置流程：

```text
读取当前 affinity
  -> 校验 PID 仍存在
  -> 校验目标 CPU 在 cpuset_cpus_effective 内
  -> 保存 rollback-state.json
  -> 展示 current -> target diff
  -> 用户确认
  -> 调用执行后端
  -> 重新读取 affinity 验证结果
  -> 记录执行日志
```

## 5. rollback-state.json

回滚状态必须在执行前保存。建议结构：

```json
{
  "schema_version": "0.1.0",
  "created_at": "2026-06-01T10:30:00+08:00",
  "executor_backend": "taskset",
  "snapshot_ref": "snapshot.json",
  "plan_ref": "plan.json",
  "actions": [
    {
      "action_id": "bind-pid-12345",
      "pid": 12345,
      "process_start_time": "optional-proc-starttime-or-null",
      "command": "python train.py --local_rank=0",
      "before": {
        "cpus_allowed_list": "0-127",
        "mems_allowed_list": "0-1"
      },
      "target": {
        "cpus_allowed_list": "0-31"
      },
      "after": {
        "cpus_allowed_list": null
      },
      "status": "pending",
      "apply_command": "taskset -cp 0-31 12345",
      "rollback_command": "taskset -cp 0-127 12345"
    }
  ]
}
```

字段说明：

| 字段 | 说明 |
|------|------|
| `pid` | 原目标 PID。 |
| `process_start_time` | 可选，用于避免 PID 复用误回滚。Linux 可来自 `/proc/<pid>/stat` starttime。 |
| `before.cpus_allowed_list` | 执行前原始 affinity，回滚核心依据。 |
| `target.cpus_allowed_list` | 计划应用的 affinity。 |
| `after.cpus_allowed_list` | 执行后重新读取到的 affinity。 |
| `status` | `pending` / `applied` / `failed` / `rolled_back` / `rollback_failed`。 |

## 6. 回滚流程

回滚不是简单执行一条命令，必须先确认当前对象仍然是原进程。

```text
读取 rollback-state.json
  -> 检查 action.status 是否为 applied
  -> 检查 PID 是否存在
  -> 可选检查 process_start_time 是否一致
  -> 读取当前 affinity
  -> 展示 current -> original diff
  -> 用户确认
  -> 调用回滚后端
  -> 重新读取 affinity 验证结果
  -> 更新 rollback-state.json 状态
```

如果 PID 已退出：

```text
不执行回滚；标记为 no_target，并提示原进程已不存在。
```

如果 PID 被复用且 `process_start_time` 不一致：

```text
不执行回滚；标记为 pid_reused，并提示需要人工确认。
```

## 7. 多 PID 原子性

多 rank 或多实例场景中，多个 PID 的绑核可能部分成功。第一版不承诺跨 PID 原子性，但必须记录每个 action 的状态。

推荐策略：

1. 顺序执行每个 action。
2. 任一 action 失败时停止后续动作。
3. 已成功的 action 保留可回滚状态。
4. 报告中明确哪些 PID 已应用、哪些失败、哪些未执行。

后续可以实验是否支持失败自动回滚已成功动作，但第一版建议由用户确认后再回滚。

## 8. 实验计划

回滚机制需要在 Linux 环境单独实验。

### 实验 1：单 PID taskset 应用和回滚

1. 启动一个长生命周期测试进程。
2. 记录原始 affinity。
3. 应用目标 affinity。
4. 验证 `/proc/<pid>/status` 中 `Cpus_allowed_list` 变化。
5. 执行回滚。
6. 验证 affinity 恢复。

### 实验 2：PID 退出

1. 保存 rollback-state 后让目标进程退出。
2. 尝试回滚。
3. 期望结果：不执行回滚，提示 no_target。

### 实验 3：PID 复用保护

1. 保存 `process_start_time`。
2. 模拟或等待 PID 复用。
3. 尝试回滚。
4. 期望结果：检测 starttime 不一致，拒绝回滚。

### 实验 4：多 PID 部分失败

1. 准备两个测试 PID。
2. 让第二个 PID 在执行前退出。
3. 应用多 PID 计划。
4. 期望结果：第一个 action applied，第二个 failed，后续停止，并保留第一个 action 的回滚能力。

### 实验 5：内部脚本后端

1. 使用内部脚本 apply 一个测试 PID。
2. 读取实际 affinity 验证结果。
3. 使用内部脚本或 `taskset` 回滚。
4. 验证状态文件和实际 affinity 一致。

## 9. 验收标准

1. 任意执行动作前都能保存完整 rollback-state。
2. 回滚不会误作用到已退出或 PID 复用后的其他进程。
3. 执行失败不会丢失已成功 action 的回滚信息。
4. taskset 后端和内部脚本后端共享同一套状态文件与回滚流程。
5. HTML 报告能展示执行后端、当前状态、应用命令、回滚命令和实验风险。
