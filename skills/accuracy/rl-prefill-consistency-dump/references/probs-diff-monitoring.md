# E2E 指标 `rollout_probs_diff`(官网逐 token probs_diff 监控)

训推一致性 E2E 指标 `training/rollout_probs_diff_mean` 的**实测**来源是官网 msprobe 文档 `verl_token_level_probs_diff_monitoring.md`。**训练侧必须能打印这个指标**,否则 E2E 只能标 `evidence_missing`,不能推断数值。

> **口径(官方做法,现场确认)**:本指标(`rollout_probs_diff`,vime 里叫 `train_rollout_logprob_abs_diff`,见 `vime.md`)锚定 **response 段**,不是给 prefill 训推一致性采集用的。官方 `metrics.py` 在 `responses`/`rollout_log_probs`/`old_log_probs` 不在 batch 时,**主动返回 `{rollout_probs_diff_valid:0, _max/_mean/_std:0.0, ...}` 不崩**。因此 PROMPTS_ONLY 裁剪下训练日志里的该指标是 0 值——**这是"没算",不是"一致"**。要实测 E2E,须单独补跑 `PROMPTS_ONLY=0` 标准训练 step,从标准日志抓;没抓到标 `evidence_missing`。判断训推一致的**数值指标**用 `prefill-logprob-diff.md` 的 **prefill logprob_diff**(训练循环内直接算出并打印);首差异定位走两侧 dump 逐边界比对(rl-consistency-dump-analysis)。**本指标不提供这两类证据**。

## 指标含义

一条 response token 序列会经过两条路径分别计算 logprob:

| 路径 | 引擎 | 参数来源 |
|---|---|---|
| rollout | vLLM 推理生成 response 时记录 `rollout_log_probs` | 上一个同步周期的 base model 权重 |
| actor 前向 | 训练时 actor 重算 `log_probs`(verl 里即 `old_log_probs`) | 当前 actor model 权重 |

```text
diff = | exp(rollout_log_probs) - exp(actor_log_probs) |     # response 段,按 response_mask/loss_mask 取有效位置
```

同一批 response token,两侧各自算概率再相减。权重一致 + 算子一致时 diff 应为 0;非 0 来自:权重同步滞后、算子精度差异(如融合 vs 独立 GEMM ±1 ULP)、dropout 等训练/推理模式差异。

## 官网启用方式(verl 原生)

**前提(必须,否则 vLLM 不返回 `rollout_log_probs`,diff 无法计算)**:

```bash
actor_rollout_ref.rollout.calculate_log_probs=True
```

**开关与参数**(带 `+` 前缀是 Hydra 未预定义项,须加 `+`):

```bash
+trainer.enable_token_level_prob_diff=True
+trainer.prob_diff_save_dir="/path/to/save"     # npy 落盘;None 不存
+trainer.prob_diff_token_max_print=10           # 每 sample 输出的 position 数
+trainer.prob_diff_sample_max_print=8           # 输出的 sample 数
```

**实现落点**(verl 原生):
- `verl/utils/debug/metrics.py`:`calculate_token_level_prob_diff(data, save_dir, step, token_max_print, sample_max_print) -> dict`。输入 `data.batch` 需含 `rollout_log_probs`、`old_log_probs`、`responses`、`response_mask`(或 `attention_mask` 尾部);缺任一返回空 dict,不报错。
- `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit()`:在 `calculate_debug_metrics` 调用之后条件调用(`enable_token_level_prob_diff` 为真时)。

**输出**:

```text
training/rollout_probs_diff/s0_p0000    0.0012    # 逐 token 网格 ≤ sample_max_print × token_max_print,仅 mask=1 位
training/rollout_probs_diff_mean       0.0012    # 聚合
training/rollout_probs_diff_max        0.0234
training/rollout_probs_diff_std        0.0031
training/rollout_actor_probs_pearson_corr 0.9987
```

**磁盘文件**(指定 `prob_diff_save_dir` 时):
- `prob_diff_step_{step}.npy`:`[batch, response_length]` float32 diff 矩阵,padding 位置为 0
- `prob_mask_step_{step}.npy`:`[batch, response_length]` bool mask,1=有效位置
- npy 只在单 driver 进程保存一次,无多副本问题

## 各框架落点

- **verl**(SPMD ≤0.6 / Async v0.7–0.8 / V1 v0.9):官网原文落点(`metrics.py` + `ray_trainer.py fit()`),见 `verl.md`。不同版本现场 `grep` 确认符号——V1 结构可能不同,别照抄行号。
- **vime**(Megatron-Bridge,自研 train loop):数据源与插入点见 `vime.md`(数据源已齐、缺 `calculate_token_level_prob_diff` 移植 + env 开关;插入点 `log_rollout_data` 附近;response 段 mask 用 `loss_masks`);PROMPTS_ONLY 下可算的修复见 `pitfalls.md` P17。此处仅保留通用方法。
- **slime / 其他框架**:按阶段 3 现场确认 actor 重算 logprobs 与 rollout logprobs 的存放字段,移植官网函数;框架无原生开关就 env-gated 插入。

## 通用落地方法(现场写码,不预置死代码)

1. **找数据汇合点**:训练循环里 actor 前向重算的 log_probs 与 vLLM 带回的 rollout_log_probs **碰头**的位置——verl 在 `calculate_debug_metrics` 附近;其他框架 `grep rollout_log_probs / old_log_probs / log_probs` 定位(框架落点见对应适配文档,如 vime 见 `vime.md`)。
2. **算**:同一批 response token,`|exp(rollout_log_probs) − exp(actor_log_probs)|`(先还原概率再相减),按 response 段 mask 取有效位置;输出**逐 token 明细**(前 sample_max × token_max)+ **mean/max/std 聚合**。数据缺失或形状对不上时返回空、不编。
3. **gate**:包在环境变量开关里(如 `*_PRINT_PROBS_DIFF=1`),默认关闭,生产零影响。
4. **打出来**:写到训练日志或终端,带统一关键词(`training/rollout_probs_diff`),保证能被 grep 到;ray 系框架的 actor `print` 会进 job 终端日志。
5. **验证 + 恢复**:阶段 7 确认日志出现输出且非空;采完随阶段 9 一起移除/还原。

## 关键坑

1. **PROMPTS_ONLY 裁剪下算不出(且不是 bug)**:probs_diff 是 **response 段**逐 token 的。裁剪把 response 裁掉后,actor 重算的 `log_probs` 只有 prompt 段,与完整 `rollout_log_probs`(response 段)形状/语义对不上——官方 `metrics.py` 缺字段时**主动返回 0 值,不崩**。**日志里 valid=0/mean=0 是"没算",不是"一致"**,别当"训推一致"证据。要实测 E2E 必须补跑一步不裁剪的标准训练(`PROMPTS_ONLY=0`),从标准日志抓数值,没抓到标 `evidence_missing`。裁剪采集下硬走 loss 需按官方改 loss/metrics 兜底,**训练照常走完**(见 `pitfalls.md` P6b),不要跳过训练或把字段设 `None` 绕过。
2. **前提是 vLLM 返回 `rollout_log_probs`**:部分框架仅在返回非空时才写入 train_data(如 vime `samples[0].rollout_log_probs is not None`,见 `vime.md`);未返回则函数静默返回空 dict 不报错——**须验证训练日志确实出现输出**,不能只看未报错。
3. **mask 别把 padding 算进 mean**:用 response_mask / loss_mask 的有效位置,离线聚合时 `valid_diffs = diff[mask]`。
4. **这是 E2E 指标证据,不是差异位置**:probs_diff 非零只说明"训推不一致"这一事实,首差异由 rl-consistency-dump-analysis 靠两侧 tensor dump 逐边界定位。两者互补:probs_diff 是"是否一致"的标量/矩阵,prefill dump 是"第一个独立差异在哪"。
