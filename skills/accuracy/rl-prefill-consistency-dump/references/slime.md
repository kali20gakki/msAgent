# slime 框架适配线索卡(v0.2.2,Megatron 训练 + SGLang 推理)

slime 是"训练侧 Megatron + 推理侧 SGLang"的训推分离架构。训练侧 `old_log_prob` 默认输入是 prompt+response,推理侧 prefill 只输入 prompt,两侧 token 不一致无法直接比。

> **地图不是坐标**:本卡基于某次真实核对写成,签名/文件位置可能随版本不同。接入锚点以本机 `grep` 为准,改前先确认,差异按真实签名适配并记入改动清单。

## 与 verl 的主要差异(决定比对方式)

- **训推分卡运行**(不是共享引擎),两侧 rank 目录通常不对齐 → **只能逐 rank 单卡比对**,不能多卡批量。
- 推理侧 SGLang ≥0.5.11 **原生内置 msprobe**,传参即可不用改码;<0.5.11 才侵入式改 `ModelRunner`。
- 有 padding 对齐约束(TP>1 时),训练侧 Megatron `pad_size = tensor_model_parallel_world_size * data_pad_size_multiplier`。

## 三件事的定位(现场 grep 确认符号)

| 事项 | 位置(线索) | 现场验证 |
|---|---|---|
| PROMPTS_ONLY 裁剪 | 训练侧 `slime/backends/megatron_utils/actor.py` 的 `train_actor` 入口:构造 prompt-only 副本(deepcopy + 按 `prompt_length = total_length - response_length` 截 tokens,response_lengths 置 0,loss_masks 置空)后喂 `get_data_iterator` | `get_data_iterator` 签名随版本:msprobe 文档示例 3 参,slime v0.2.2 实测**单参**(data.py:256)。改前 grep |
| 训练侧接入 | `MegatronTrainRayActor.init` 实例化 debugger(**必须在 monkey_patch_torch_dist 之前**,避免 bias_swiglu_fusion 冲突);`train_actor` 里 debugger 包住 `compute_log_prob`(只喂 prompt-only 副本),完成后 `save_debug_train_data` | 只采 `old_log_prob` 一个阶段;`step()` 在 `stop()` 后调用一次。官方做法是裁剪 + 改 loss/metrics 兜底、训练照常走完(见 SKILL 阶段 6、`pitfalls.md` P6b),不要 save 后 return 跳过训练 |
| 推理侧接入 | SGLang ≥0.5.11:传 `--sglang-msprobe-dump-config` 即可;原生 start/stop/step 包住每次 `forward()`,warmup 走 `self.model.forward()` 绕过 → **step0 就是首个真实 prefill**;原生自动 `disable_cuda_graph=True` + `skip_server_warmup=True`。<0.5.11:改 `ModelRunner.__init__` 实例化 + `forward()` 里仅对 `is_extend` 且 token 数 ≥ 阈值(默认 2)的采集,跳过启动约 1 token 的 dummy EXTEND | 须加 `--sglang-disable-cuda-graph` 否则 msprobe 钩不住 |
| request_id 关联 | 通用方法:`new-framework.md` §2③;rank 对齐用 `rank_id=self.gpu_id`(SGLang DP 各 worker `get_rank()` 重复) | vllm ≥0.14 的 8 位 hex 后缀剥除 |

## 预处理约束

| 目标侧 | 效果 | 约束 |
|---|---|---|
| 训练侧 | `old_log_prob` 输入截为 prompt-only | 裁剪副本**只喂 log_prob 前向**,不替换整个 rollout_data;adv/correction 路径早退 + 改 loss/metrics 兜底,训练照常走完(官方口径,见 `pitfalls.md` P6b) |
| 推理侧 | SGLang 第一次 EXTEND(prefill)输入为 prompt | 见下列约束 |

**序列长度与 padding**:`--data-pad-size-multiplier 1`(默认 128),让序列只对齐到 TP 的倍数。推理侧按相同 pad_size 在传入 SGLang 前对齐长度:

| 配置 | pad_size | 说明 |
|---|---|---|
| TP=1, multiplier=1(推荐) | 1 | 无需填充,天然对齐 |
| TP>1, multiplier=1 | TP | prompt 长度补到 TP 整数倍 |
| TP>1, multiplier=128 | TP×128 | 填充过多,不推荐用于精度比对 |

**SGLang DP 与 chunked prefill**:`--sglang-dp-size > 1` 时须 `sglang-chunked-prefill-size / sglang-dp-size > 实际 prefill prompt 长度`。

**并行切分一致**(dump 才能按 rank 对应):TP 训练侧 `--tensor-model-parallel-size` / 推理侧 `--rollout-num-gpus-per-engine`;PP/CP 通常 1;DP 训练侧 world_size/TP/PP/CP / 推理侧 `--sglang-dp-size`。

**单条 prompt 与 DP**:`--rollout-batch-size 1`、`--n-samples-per-prompt ${DP}`、`--global-batch-size ${DP}`。global_batch_size 不能整除 DP 会报错。

**禁用干扰**:不传 `--balance-data`(自动均衡重排 batch)、`--use-dynamic-batch-size`;关闭 `--rollout-shuffle`、`--over-sampling-batch-size`。

## 启动参数建议

```shell
--num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt ${DP} --global-batch-size ${DP}
--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
--data-pad-size-multiplier 1 --no-gradient-accumulation-fusion
# 勿传 --balance-data、--use-dynamic-batch-size
--sglang-disable-cuda-graph --sglang-chunked-prefill-size 16384
```

## 两侧 msprobe 配置

训练/推理各一份 config。训练侧 dump_path 指向根目录(代码会拼 `old_log_prob` 子目录),推理侧指向 `generate` 子目录。`step` 通常只采 step0。`task=statistics` 轻量;要真实张量改 `tensor`。模板见 `references/msprobe-config.md`。

## 环境变量

- **训练侧**(ray job submit 的 `runtime-env-json.env_vars`):`DUMP_ON=1`、`PROMPTS_ONLY=1`、`TORCHDYNAMO_DISABLE=1`、`MSPROBE_SEED=1234`、`MSPROBE_CONFIG_PATH=<config_actor.json>`。
- **推理侧**(<0.5.11):`SGLANG_MSPROBE_DUMP=1`、`MSPROBE_GENERATE_CONFIG=<config_generate.json>`、`MSPROBE_MIN_DUMP_TOKENS=2`。

## 结果目录与比对

```text
{dump}/generate/step0/rank{ID}/{dump.json, stack.json, construct.json}
{dump}/old_log_prob/step0/rank{ID}/{dump.json, stack.json, construct.json}
```

两侧 dump 落盘后,语义映射、逐边界比对、首差异定位、结论与报告由 **rl-consistency-dump-analysis** skill 承担(本 skill 只采集不分析)。
