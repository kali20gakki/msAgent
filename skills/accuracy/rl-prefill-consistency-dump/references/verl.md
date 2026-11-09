# verl 架构适配线索卡(v0.6 SPMD / v0.7–0.8 异步 / v0.9 V1)

verl 版本演进彻底改变了架构,裁剪/接入/request_id 位置随版本变。**先识别版本和架构形态,再看对应节;references 只是线索,改前现场 `grep` 确认符号,不存在就按新增处理并记入改动清单。**

## 0. 版本 → 架构 → 官方文档

| verl 版本 | 架构形态 | 推理与训练资源 | 官方文档 |
|---|---|---|---|
| ≤ v0.6 | SPMD(Single Program Multiple Data) | 共享引擎 | `dump/verl_fsdp_consistency_preprocess_dump.md` / `verl_megatron_...` |
| v0.7–0.8 | Hybrid AgentLoop(默认)/ Fully Async | Hybrid 共享 NPU;Fully Async 独立 NPU 池 | `dump/verl_async_consistency_preprocess_dump.md` |
| v0.9.dev | V1 Trainer(Sync / Colocate Async / Separate Async) | TransferQueue 传数据,jagged TensorDict | `dump/verl_v1_trainer_consistency_preprocess_dump.md` |

能否用官方 `msprobe compare --consistent_check` 自动比对属分析侧判断(由 **rl-consistency-dump-analysis** 处理);采集姿势按上表对应版本的官方 `dump/verl_*_consistency_preprocess_dump.md`。

## 1. 识别架构形态

- SPMD:训练侧在 `verl/workers/actor/{dp_actor,megatron_actor}.py`,无 request_id 贯穿机制。
- Async:`ray_trainer.py` / `fully_async_trainer.py` 有 Hybrid/Fully Async 分支;有 `DispatchLogger` + request_id 链路。
- V1:启动参数 `trainer.use_v1=True` + `trainer.v1.trainer_mode`;有 `agent_loop_tq.py` / `engine_workers.py`。

## 2. 公共前置约束(三版共用)

`mini_batch_num=1`、`gac=1`、`use_remove_padding=True`、`balance_batch=False`、`DUMP_ON=1`、`PROMPTS_ONLY=1`、`enforce_eager=True`、`TORCHDYNAMO_DISABLE=1`、`val_before_train=False`(防 generate 污染 dump)。推理侧 `additional_config` 加 `dump_config_path`。

## 3. SPMD(≤v0.6)

**裁剪位置**:FSDP 在 `dp_actor.py` 的 `_forward_micro_batch` / `compute_log_prob` / `update_policy` 三处按 `PROMPTS_ONLY=1` 裁 input_ids/attention_mask/position_ids,responses/rollout_log_probs/response_mask 置空;Megatron 在 `megatron_actor.py` `compute_log_prob` 开头裁,涉及 `compute_log_probs_fn`、`recompute_old_log_prob`、`forward_backward_batch` 的 `loss_func` 三处,response 为空时用全 True mask 兜底。`metrics.py` 与 `rollout_corr_helper.py` 都要 `PROMPTS_ONLY` 早退。

**接入**:`verl/workers/{fsdp,megatron}_workers.py` 实例化 debugger;`update_actor`(训练)与 `generate_sequences`(推理)包前向,`start(model=...) → 前向 → stop() → step()`。

**关联**:SPMD 无 request_id,靠**逐卡对应的样本**比对(训练/推理各 rank 的 dump 路径按 rank 对应)。

## 4. Async(v0.7–0.8)

**两种模式**:

| 模式 | 资源 | 特征 |
|---|---|---|
| Hybrid AgentLoop(默认) | 共享 NPU(`hybrid_engine`) | `LLMServerClient` + `GlobalRequestLoadBalancer` |
| Fully Async | 独立 NPU 池 | `FullyAsyncRollouter` + `MessageQueue` + `ParameterSynchronizer` |

**裁剪位置**:Hybrid 在 `ray_trainer.py` `fit()` 的 `bypass_recomputing_logprobs` 之前;Fully Async 在 `fully_async_trainer.py` `_fit_generate()` 的 `_get_samples_from_queue()` 之后。两者都可能需新增 `get_prompts_only_batch`。

**推理接入**:`vllm_ascend/worker/dispatch_logger.py`(新增)+ `model_runner_v1.py` 每处 `debugger.stop()` 前插 `log_step()`;dump 路径挂 PID 子目录。

**request_id 链路**:`LLMServerClient.generate()` 生成 vllm_request_id 同时写 `extra_fields["request_id"]` → `TokenOutput` → `AgentLoopOutput` → `DataProto.non_tensor_batch` → micro_batch → `update_actor_log.jsonl`。Fully Async 的 client 必须 `final_output.extra_fields.update(output.extra_fields)` 继承。

## 5. V1(v0.9.dev)

**三种模式**:`sync`(串行)/ `colocate_async`(共享资源)/ `separate_async`(独立资源),rollout 数据经 TransferQueue。

**裁剪位置**:V1 是 jagged TensorDict,按每条样本实际 prompt 长度裁;`engine_workers.py` `update_actor()` 里浅复制后 `prompts.offsets()` 取各样本长度,把 responses/rollout_log_probs/response_mask/loss_mask 等裁成 `[:0]`(nested tensor 同理)。

**request_id**:`agent_loop_tq.py` `AgentLoopWorkerTQ._agent_loop_postprocess()` 把 `extra_fields.request_id` 提升为顶层字段(即使 None 也保留,避免 TransferQueue 字段不一致)→ TransferQueue → actor TensorDict。partial rollout 的 client 要保留第一段 request_id 和全部分段 ID。

## 6. 数据关联(Async / V1 共用)

1. `dispatch_log.jsonl` 挑 `phase=prefill` 且单请求的 step 和 request_id。
2. **vllm ≥0.14 给 request_id 追加 8 位 hex 后缀**,匹配前剥掉。
3. `update_actor_log.jsonl` 搜同一 request_id,得训练侧 step 和 rank。
4. 按 `step_N/rank_M` 读两侧 `dump.json`。

## 7. 结果文件

```text
{dump_generate_path}/{pid}/step_N/rank_M/dump.json   + dispatch_log.jsonl
{dump_actor_path}/step_N/rank_M/dump.json            + {pid}/update_actor_log.jsonl
```

> 完整三份旧文档已合并。精确行号/大段代码历史留档在 session 产物里,这里只留定位线索。官方对应版本文档见 §0 表。
