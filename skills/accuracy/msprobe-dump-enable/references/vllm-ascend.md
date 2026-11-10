# vLLM 推理场景 dump 使能（vllm-ascend / 社区原生 vLLM）

本文覆盖本技能范围内的 vLLM 推理服务 dump 使能。vllm-ascend 部分依据官方文档
[vLLM Ascend msProbe Debugging Guide](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/msprobe_guide.html)；
社区原生 vLLM（GPU）部分依据 msprobe 官方文档
[vLLM采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/vllm_dump_instruct.md)。
配置字段含义与输出目录结构见 [config-json.md](config-json.md)（与常规 config.json 采集相同），本文不重复。

## 两种实现的核心区别

官方对两种实现的模式支持与使用方式：

| 实现 | 硬件 | Eager | 图模式 | 使用方式 |
| --- | --- | --- | --- | --- |
| `vllm-ascend` | NPU | ✅ | ✅ | 已内置 msProbe 接入，**无需插桩**，经 `--additional-config` 传配置 |
| 社区原生 vLLM | GPU | ✅ | ❌ | **需要在 `GPUModelRunner` 中接入 `PrecisionDebugger`（插桩）**，启动须加 `--enforce-eager` |

- **vllm-ascend 免插桩**：不需要 `seed_all()`、不需要实例化 `PrecisionDebugger`、不需要 `start()`/`stop()`/`step()`。
  每个推理请求由框架内部自动驱动 msprobe 的 `start → forward → stop → step` 序列，且 runner 在每个代码路径
  都会调用 `step()`，即使推理提前返回也能得到完整数据。
- **社区原生 vLLM（GPU）需要插桩**：框架没有内置 msProbe 接入，按官方示例在 `GPUModelRunner` 中插入
  `PrecisionDebugger` 打点（见下文），且仅支持 eager 模式采集，启动服务必须指定 `--enforce-eager`。

## vllm-ascend 使能方式（官方，免插桩）

方式一：直接传配置内容：

```shell
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --additional-config '{
    "dump_config": {
      "task": "statistics",
      "level": "L1",
      "dump_path": "/data/msprobe_dump",
      "statistics": {
        "list": []
      }
    }
  }' &
```

方式二：传配置文件路径（兼容模式，legacy）：

```shell
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --additional-config '{"dump_config_path": "/data/msprobe_config.json"}' &
```

启动后照常发送推理请求（如 `curl /v1/completions`），采集自动进行，无需其他操作。

## 社区原生 vLLM（GPU）插装（官方）

1. 创建标准 config.json（与常规采集相同，见 [config-json.md](config-json.md)；官方示例 level 为 `"mix"`）。
2. 找到 `vllm/v1/worker/gpu_model_runner.py` 的 `GPUModelRunner` 类：

   - `__init__` 方法中、模型执行逻辑开始前（尽量靠前）初始化：

     ```python
     class GPUModelRunner:
         def __init__(
             self,
             vllm_config: VllmConfig,
             device: torch.device,
         ):
             ################################ msprobe ################################
             from msprobe.pytorch import PrecisionDebugger, seed_all
             seed_all(mode=True)
             self.debugger = PrecisionDebugger(config_path="./config.json")
             ################################ msprobe ################################
             self.vllm_config = vllm_config
             self.device = device
             ...
     ```

   - `execute_model` 方法中加 `start`，并用 `try...finally` 统一在 `finally` 中调用 `stop` 和 `step`
     （`execute_model` 通常较长，这样最省事且不易遗漏返回分支）：

     ```python
     @torch.inference_mode()
     def execute_model(
         self,
         scheduler_output: "SchedulerOutput",
         intermediate_tensors: Optional[IntermediateTensors] = None,
     ) -> Union[ModelRunnerOutput, torch.Tensor]:
         ################################ msprobe ################################
         if hasattr(self, "debugger"):
             self.debugger.start(model=self.model)
         ################################ msprobe ################################

         try:
             ...
             return output
         finally:
             ################################ msprobe ################################
             if hasattr(self, "debugger"):
                 self.debugger.stop()
                 self.debugger.step()
             ################################ msprobe ################################
     ```

     不方便整体包裹时，在各 `return` 前补充：

     ```python
     if hasattr(self, "debugger"):
         self.debugger.stop()
         self.debugger.step()
     return output
     ```

   要点（官方）：L0/mix 级别 `start` 需传 `model` 参数，仅 L1 统计可直接 `start()`；不调用 `stop`/`step`
   可能导致 dump.json 落盘不完整。

3. 启动服务时开启 eager 模式：

   ```shell
   #!/bin/bash
   export TORCHDYNAMO_DISABLE=1

   vllm serve Qwen/Qwen2.5-0.5B-Instruct \
     --dtype float16 \
     --enforce-eager \
     --host 0.0.0.0 \
     --port 8000
   ```

## vllm-ascend 配置要点

- 配置内容与 `PrecisionDebugger` 的 config.json **完全相同**（task/level/dump_path/rank/step 等字段
  取值与含义见 [config-json.md](config-json.md)）。
- eager 与图模式支持的字段不同（官方字段表）：

| 字段 | 必选 | eager | 图模式 |
| --- | --- | --- | --- |
| `task`（statistics/tensor 等） | 是 | ✅ | ✅ |
| `level`（L0/L1/mix） | 是 | ✅ | ✅ |
| `dump_path` | 否 | ✅ | ✅ |
| `rank`（单卡必须 `[]`） | 否 | ✅ | ✅ |
| `list` | 否 | ✅ | ✅ |
| `step` | 否 | ✅ | ❌ |
| `async_dump` | 否 | ✅ | ❌ |
| `scope` | 否 | ✅ | ❌ |
| `dump_enable` | 否 | ✅ | ❌ |

官方 eager 模式配置示例：

```json
{
  "task": "statistics",
  "dump_path": "/home/data_dump",
  "rank": [],
  "step": [],
  "level": "L1",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

## 输出与检查

- 输出目录结构与常规 config.json 采集相同（见 [config-json.md](config-json.md) §5）：`dump_path/step*/rank{ID}/`
  下 `dump.json`、`stack.json`、`construct.json`（L0/mix 时非空）、`dump_tensor_data/`、`dump_error_info.log`
  （仅报错时存在）；非分布式场景目录名为 `rank`。图模式输出为 `L0_dump/step*/rank*/dump.json`。
- 检查 dump 是否 ok：发送请求后 `dump_path` 下出现对应 `step*/rank*/dump.json` 且内容非空。

## 常见问题排查（官方）

| 现象 | 处理 |
| --- | --- |
| `RuntimeError: Please enforce eager mode` | 重启 vLLM 并加 `--enforce-eager` |
| 无 dump 文件 | 确认配置路径正确、每个节点有写权限；分布式场景设置 `keep_all_ranks` 使每个 rank 都写自己的 dump |
| dump 数据过大 | 先用 statistics 任务定位异常 tensor，再用 `scope`/`list`/`tensor_list`、`token_range` 等缩小范围 |
| `TypeError: PrecisionDebugger.start() got an unexpected keyword argument 'scheduled_tokens'` | 升级 msProbe 至 v26.2.0 或更高 |

## 边界

本技能只负责把 dump 使能到 vLLM 服务并确认落盘。
