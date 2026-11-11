# 进程/线程角色分类规则

> 与 bind-tune 的 process-roles.md 保持口径一致（同构副本，各自维护）。
> 权威锚点：`npu-smi info` 卡↔进程映射，优先于 comm 猜测。

## 角色分类表

| 类型 | 识别方式 | 诊断用途 |
| ---- | -------- | -------- |
| 训练主进程 | npu-smi 卡占用 + cmdline（torchrun/msrun/python） | 确定业务目标与 rank 集合 |
| 推理进程 | npu-smi 卡占用 + cmdline（vllm serve/mindie） | 确定业务目标与实例集合 |
| DataLoader worker | 主进程 fork 的子进程，不占卡，comm 常为 python | B 域冲突判定 |
| HCCL 通信线程 | 进程内线程 comm 含 hccl* | B 域冲突判定 |
| NPU runtime 线程 | 进程内线程，跟随卡 | B 域亲和现状采集（node 偏差判定归 bind-tune） |
| 监控/采集 | comm 白名单：npu-smi/node_exporter/fluentd/telescope/dcgm-exporter | 干扰判定 |
| 容器 runtime | comm 白名单：containerd-shim/runc/dockerd | 干扰判定（不可迁移） |
| 日志/agent | comm 白名单：fluentd/filebeat/rsyslogd | 干扰判定 |
| 编译/交互 | comm 白名单：gcc/g++/pip/jupyter | 告警项（建议移出） |

## 分类优先级

1. npu-smi 卡占用（最强证据）；
2. PPID 进程树（同祖先 = 同作业实例）；
3. comm/cmdline 模式匹配；
4. 白名单命中即终止（不重复匹配）。

## 输出格式

分类结果写入 targets 节（static-snapshot.json）：

```json
{"pid": 12345, "comm": "python", "role": "train_main",
 "npus": [0, 1], "affinity": "0-127", "source": "npu_occupancy"}
```

role 取值：train_main / infer_main / dataloader / monitor / container_runtime /
logger / compiler / other。
