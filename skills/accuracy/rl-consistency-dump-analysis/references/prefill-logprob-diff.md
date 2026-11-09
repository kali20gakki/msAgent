# prefill logprob_diff:训练侧 prefill logprob vs 推理侧 prompt 段 logprob

训推一致性数值指标:两侧输入对齐到 prompt-only 后,训练侧 prefill 算出的逐 token logprob 与推理侧 prefill 算出的逐 token logprob 的差异。计算时机在**训练进行中**(采集侧接入计算,打印训练日志),不是训推结束后的 dump 离线分析。本 skill 只**引用训练日志实测数值**做解读,不负责采集。

## 是什么 / 不是什么

- **是**:同一段 prompt,训练侧 prefill 输出 logits 的逐 token logprob(裁剪后 `log_probs[:, :-1]`)与推理侧 prompt 段逐 token logprob,取 `|exp(lp_train) − exp(lp_infer)|`,聚合 `_mean/_max/_std`。训推一致的 prefill 级数值证据,**训练过程中直接产出**。
- **不是** response 段 E2E 指标(`rollout_probs_diff`/`train_rollout_logprob_abs_diff`):那个锚定 **response 段**,prompt-only 裁剪下被裁,metrics 缺字段主动返回 0 值(valid=0/mean=0 是"没算"不是"一致")。**两个指标不同层面,不能互相冒充**。

## 通用计算口径(采集侧现场落地)

1. **找碰头点**:训练循环里 actor 前向重算的 log_probs 与 rollout 带回的推理侧 logprob **同时可见**的位置(verl 在 `calculate_debug_metrics` 附近;其他框架 `grep rollout_log_probs / prompt_log_probs / log_probs` 定位)。
2. **取两侧 logprob**:
   - `lp_train`:训练侧裁剪后 prefill 输出 logits → `log_softmax(dim=-1)` → 取 prompt 段 `log_probs[:, :-1]`(去末位无 next-token 预测位);训练侧可能是 padding 序列,按真实 prompt token 数截取。
   - `lp_infer`:rollout 返回的 prompt 段 logprob(注意是 logprob 还是 logits;dtype/精度与 `lp_train` 对齐)。**推理侧不返回 prompt 段 logprob 就算不了**;别拿 response 段 `rollout_log_probs` 顶替。
3. **按 token 对齐**:两侧按真实 prompt token 数截齐(去 padding、训练侧去末位),覆盖同一组 (position, next_token) 对。
4. **算 diff**:逐 token `|exp(lp_train) − exp(lp_infer)|`(先还原概率再相减);输出逐 token 明细 + `_mean/_max/_std`(**只算有效 token 位,别把 padding 算进 mean**)。
5. **打印**:带统一关键词(如 `prefill_logprob_diff_mean`),保证 grep 到;env-gated,默认关。

**兜底(仅当训练循环内算不出时)**:读两侧 dump 的 final 输出边界(训练 output_layer/lm_head、推理对应 logits)离线算——确认语义等价(shape/vocab/并行切分对得上;对不上先语义映射,映射不了标 `evidence_missing`)→ `log_softmax` → 按 token 对齐 → 逐 token `|exp(lp_train) − exp(lp_infer)|` + 聚合。这只是兜底,不替代训练循环内直接算。

## 解读陷阱(本 skill 引用数值时注意)

1. **推理侧没返回 prompt 段 logprob**:训练日志里没有 `lp_infer`,算不了;日志缺失 → `evidence_missing`,不编。
2. **对齐陷阱**:训练侧带 padding,推理侧可能没有;按真实 token 数对齐,`mean` 只用有效位。
3. **末位处理**:prompt 最后一个 token 没有 next-token 预测,两侧都要排除或对齐处理,别把形状差当 diff。
4. **`|exp(lp_train)−exp(lp_infer)|` 是绝对概率差,会掩盖大相对差**:prompt 段绝大多数 token 概率≈0(exp(−20)≈2e−9),绝对差只有 1e−11 级别(看着"接近一致"),但**相对差可达几十 %、lp 空间差 1.5+ nats**。解读 mean 前先看 lp 空间明细(逐 token `lp_train` vs `lp_infer`)。mean≈0(如 <1e−5)才是真一致;个别 position 概率高(如 0.02 vs 0.09)就能把 mean 抬到 1e−4 量级——这种"稀疏大差"正是首差异(如融合 vs 独立 GEMM ±1 ULP)放大后的正常形态,首差异定位靠逐边界 tensor 门禁(本 skill),别只凭 mean 非零就断言权重/配置错误。
5. **别与 response 段 E2E 混淆**:prompt-only 裁剪下 response 段 probs_diff 日志里是 0 值("没算"不是"一致");本指标锚定 prefill 段,不同层。
6. **不泄露**:原始 tensor/dump/prompt 不进报告与仓库,只留聚合数值。

## 与阶段的关系

- 阶段 7 结论:该 diff 数值(训练日志实测)作为 prefill 训推一致的 logprob 证据填入;response 段 E2E 按官方口径处理(补跑含 response 的标准训练才有实测,否则 `evidence_missing`)。
