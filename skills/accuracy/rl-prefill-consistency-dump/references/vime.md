# vime 框架适配线索卡(Megatron-Bridge 训练 + vLLM-ascend 推理)

vime 是基于 verl 封装的训推框架:训练侧是 Megatron-Bridge(`/root/vime/vime/backends/megatron_utils/actor.py`),推理侧是 vLLM-ascend(**由 vime rollout 自动拉起**,不是单独起服务)。actor(训练)+ rollout(推理)按卡位分角色。基于 2026-09-10 GLM-4.7 真实排查写成。

> **地图不是坐标**:路径/函数名/签名基于某次真实核对,可能随版本不同。改任何文件前先在本机仓库 `grep` 确认符号,差异按真实签名适配并记入改动清单。

## 设备分配机制(最高优先级,踩坑重灾区)

vime 的 vLLM 设备分配**原生正确、支持各卡位**(生产 0-3 / 分卡 4,5 / 6,7 均验证正常)。不要凭函数名表象断言"非 0 基址缺陷",更不要在设备分配处加"映射 patch"。

机制链(vLLM 引擎启动前,ray 已把卡分给各 actor):

```text
rollout.py  base_gpu_id = int(reordered_gpu_ids[gpu_index])   # ★ 来自 ray placement group,真实物理卡号
            → 显式传给 VLLMEngine(RolloutRayActor)
vllm_engine.py  _compute_server_args: visible_devices = base_gpu_id + i
                build_vllm_subprocess_env: env["ASCEND_RT_VISIBLE_DEVICES"] = server_args["visible_devices"]
```

**真实事故(09-10)**:误把已是物理卡号的 `base_gpu_id` 当"可见索引",对 `ASCEND_RT_VISIBLE_DEVICES="4,5"` 二次取模,base=5 越界 → 空列表 → `visible_devices=''` → 子进程拿不到卡 → EngineCore 崩,反复 7 次才定位。回滚成原生后各卡位都正常。

**"引擎起不来"三步排查**(按序,别急着改代码):
1. 看 vLLM 子进程**实际拿到**的 `ASCEND_RT_VISIBLE_DEVICES`(启动日志 / 容器内 `ps e -p <pid>`);为空 = 设备传递链断了。
2. 沿 `_compute_server_args → build_vllm_subprocess_env` 确认它在哪被置空。
3. **回滚自己改动 → 原生跑一次对照**(只改卡号)。原生能起,就是你的 patch 引入的 bug,不是框架/卡缺陷。

崩溃点:启动链路 `VLLMEngine → _compute_server_args → build_vllm_cmd_and_env → build_vllm_subprocess_env → launch_server_process`。`is_npu()` 在可见卡为空时要么 True 要么 raise,报错是 `RuntimeError: torch_npu detected, but NPU device is not available or visible.`,不是"没走 NPU 分支"。

## 三件事的定位(现场 grep 确认符号)

| 事项 | 位置(线索) | 现场验证 |
|---|---|---|
| PROMPTS_ONLY 裁剪 | 训练侧 `actor.py` `train_actor`,`get_data_iterator(rollout_data)` 之前:`DUMP_ON=1 && PROMPTS_ONLY=1` 时构造 prompt-only 副本——tokens 截到 prompt_length(`total_length - response_length`),response_lengths 置 0,loss_masks 置空,total_lengths 改 prompt_length | 裁剪副本只喂 log_prob 前向,不替换整个 rollout_data;采完 `git checkout actor.py` |
| 训练侧接入 | 主前向 `actor.py` `compute_log_prob(store_prefix="")` → `model.py` `forward_only`。**官方做法:直接改源码,不用 vime 预留 hook 入口(`--custom-megatron-before-log-prob-hook-path`,勿用)**——actor 构造时 `DUMP_ON=1` 实例化 `PrecisionDebugger`(env-gated,否则 `None`),`compute_log_prob` 主前向 `start(model=...) → 前向 → stop() → step()`,只采 actor 主前向(ref 引擎 `forward_only` 跳过) | 改动记入清单,采完恢复 |
| 推理侧 msprobe | vLLM-ascend 0.22.1 `model_runner_v1.py` 原生集成 PrecisionDebugger,传 config 即启用:`--vllm-enforce-eager`(★ cudagraph_mode==NONE 才走 PrecisionDebugger,必须带)+ `--vllm-additional-config '{"dump_config_path": "<session>/msprobe_infer_config.json"}'` | config 示例 `{"task":"tensor","dump_path":"<session>/dumps/infer","rank":[],"step":[0],"level":"mix"}`(两侧统一 level=mix) |
| request_id 关联 | 通用方法:`new-framework.md` §2③ | vllm ≥0.14 的 8 位 hex 后缀剥除 |

## 训练侧指标与碰头点(vime 特有)

- **response 段 E2E**:vime 里叫 `train_rollout_logprob_abs_diff`(`loss.py`,定义 `(old_log_probs − rollout_log_probs).abs()`),锚定 response 段;PROMPTS_ONLY 下被裁,metrics 缺字段主动返回 0 值(valid=0/mean=0 是"没算"不是"一致",见 `probs-diff-monitoring.md`)。
- **PROMPTS_ONLY 下使其可算(两处,缺一不可)**:① `model.py` 训练前向 `get_batch` 的 keys 列表**必须加 `"prompt_logprobs"`**,否则 `policy_loss_function` 的 `batch.get("prompt_logprobs")` 恒 None,推理侧 prompt 段 logprob 进不了 loss;② `loss.py` 的 `train_rollout_logprob_abs_diff` 分支将 `old_log_probs`(train prompt 段)与 `batch["prompt_logprobs"]`(推理侧同一段)对齐,`torch.cat` 后算 `(old_log_probs − rollout_log_probs).abs().mean()`——**绕过 `sum_of_sample_mean`**(它是个 0 stub),直接 `.mean()` 按 prompt token 归约。修复细节见 `pitfalls.md` P17。
  - 数值口径:raw logprob 空间的 prompt 段均值绝对差。prompt 段大量低概率 token,小概率差被放大成几个 nats;与标准训练 response 段(高概率 token,log 差小)数量级不同属预期,须同口径对比。
- **prefill logprob_diff 碰头点**:`log_rollout_data` / `compute_advantages_and_returns` 附近(此时 actor 主前向已把 `log_probs` update 进 rollout_data;推理侧 prompt 段 logprob 在 train_data 的 `prompt_logprobs`)。实测实现:`actor.py` 新增 `_print_prefill_logprob_diff(rollout_id, rollout_data)`,数据源 `rollout_data["lp_train_prompt"]`(训练侧 log_softmax 后的 prompt 段)+ `rollout_data["prompt_logprobs"]`(vLLM prompt_logprobs 解析),逐 token `|exp(lp_train)-exp(lp_infer)|` + mean/max/std 打印。`lp_train_prompt` 在 `model.py` `forward_only` 按 `log_softmax` 取 `[:, :-1]` 存入 rollout_data(DUMP_ON gate)。通用方法见 `prefill-logprob-diff.md`。
- **`rollout_log_probs` 写入条件**:`samples[0].rollout_log_probs is not None` 才写入 train_data(`ray/rollout.py:732`);未返回则 probs_diff 静默空 dict 不报错——须验证日志确实出现输出。
- **response 段 mask 字段**:vime 用 `loss_masks`(非 verl 的 `response_mask`)。

## Megatron-Bridge 接入坑(TE 硬依赖假阻塞 / F.silu 引用缓存)

- **TE 硬依赖假阻塞**:孤立环境(无 transformer_engine)`import megatron.bridge` 崩 `ModuleNotFoundError`;连锁假象是 AutoBridge 报"架构 not yet supported"(注册语句因导入期崩从未执行)。修复:TE import 改 `try/except`(te=None)+ `TELinearAdapter` 基类按 HAVE_TE 切换 + 文件顶 `from __future__ import annotations`(Python 3.12 无 future 时类体注解在定义期求值会崩)。详细见 **`pitfalls.md` P21**。
- **F.silu 引用缓存连锁**:msprobe(level=mix)实例化时全局替换 `F.silu` 为包装函数;Megatron-Bridge 在模块导入期缓存原生 silu 引用(`ACTIVATION_MAPPING`),模型构造时与动态 `F.silu` 做引用相等校验 → 崩(报 `Only support fusion of gelu and swiglu` 或 `bias_activation_fusion` 校验)。修复:`hf_to_megatron_activation` 对 silu/gelu **调用期动态求值**,反向 `megatron_to_hf_activation` 加 `is F.silu` 兜底。详细见 **`pitfalls.md` P20**。
- 共通:这类改动是**采集态补丁,采完用 bridge_backup 还原**。

## 启动脚本约束(分卡模式)

- 卡位由 `VIME_DEVICES` + `VIME_NUM_NPUS` 定义(生成 ray 的 NPU 资源),`ASCEND_RT_VISIBLE_DEVICES` 与 `VIME_DEVICES` 保持一致;`RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`。
- ray head:`ray start --head ... --num-gpus=0 --resources='{"NPU": N}'`;作业经 `ray job submit` 跑 `train.py`。
- **分卡模式**(actor 卡 + rollout 卡分开):`--actor-num-gpus-per-node 1 --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1`(每 engine 单卡即 TP=1)。生产全量是 actor 2 + rollout 2、`--rollout-num-gpus-per-engine 2`。
- 确定性单样本:不传 `--rollout-shuffle`;`--num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 1 --rollout-max-response-len 4 --rollout-temperature 0.0 --global-batch-size 1`;关 eval(MTP 关、不传 `--eval-interval` 或 `--skip-eval-before-train`);`--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-* 1`。
- 无 pad:`--data-pad-size-multiplier 1`,TP=1 时无需填充,天然对齐。

## 已知坑(训练侧,与引擎无关)

采集验证 job 即使引擎/设备全正常,也可能在 actor 训练反向报 `found NaN in local grad norm for bucket #0`——这是**训练侧问题**(如 5 层截断权重 + 单样本确定性配置),不代表引擎或采集失败。判断时把"引擎起来/rollout 成功"与"训练反向 NaN"分开看。

> **运行环境不预设容器形态**:部署环境(容器/裸机、PID namespace、容器名、docker exec 方式)因机器而异,不属于框架本身。清卡/杀残留/确认可见卡一律按 SKILL 阶段 1 现场确认,勿照抄本文档或记忆里的容器名。

## 采集环境变量

| 变量 | 作用 |
|---|---|
| `DUMP_ON=1` / `PROMPTS_ONLY=1` | 开启裁剪 + 采集(采集代码读) |
| `VIME_TRAIN_DUMP_CONFIG` / `VIME_TRAIN_MSPROBE_DUMP_PATH` | 训练侧 msprobe config 路径 / dump 根目录(debugger 模块导入期读取) |
| `ASCEND_RT_VISIBLE_DEVICES` / `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` | 卡位 + 禁止 ray 覆盖可见卡 |
| `VLLM_ADD_CONFIG`(或 `--vllm-additional-config`) | 推理侧 msprobe config(dump_config_path) |

## 结果目录与比对

```text
{session}/dumps/train/step0/rank{M}/{dump.json,...}    # 训练侧(路径来自 VIME_TRAIN_MSPROBE_DUMP_PATH)
{session}/dumps/infer/step0/rank{M}/{dump.json,...}    # 推理侧(来自 dump_config_path)
```

两侧 dump 落盘后,语义映射、逐边界比对、首差异定位、ULP 量化、放大链、排除假设、结论与报告由 **rl-consistency-dump-analysis** skill 承担(本 skill 只采集不分析)。
