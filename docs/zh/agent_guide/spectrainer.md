# SpecTrainer 投机解码 on-policy 重采样

`SpecTrainer` 是面向投机解码（speculative decoding）**训练数据准备**场景的 Agent：把已有多轮对话数据里每条 assistant 回答丢弃，用 verifier 模型（vLLM 服务）逐轮**重新生成**（on-policy，框架语境称“重采样”），产出可直接进训练的**预分词样本**（每行含 `input_ids` / `loss_mask`）。用户用自然语言描述目标，Agent 编排「数据预检 → 输入归一化 → 调用重生成 → 产物校验」全流程。

> **与开源 speculators 的关系**：本 Agent 技能的 `scripts/` 是**开源 speculators 仓脚本的编排封装**（仅做参数组装与调用，不修改上游源码）。版本基线为 **speculators v0.6.0**（[vllm-project/speculators](https://github.com/vllm-project/speculators)）。若上游脚本接口有变更，需同步更新封装脚本的调用参数。

## 1. Agent 定位

- 只做**数据重采样/重生成**这一段：不负责拉起 verifier 服务，不做训练，也不做接受率/精度验收
- 输入可以是原始多轮对话 jsonl，也可以是已符合重生成格式的 jsonl
- 关键决策（endpoint、数据、条数/并发/采样参数、输出目录）会先与你确认，不擅自占资源

## 2. 核心能力

- **预检**：对原始多轮对话做 DFX 预检，按 high / medium / low 分级上报（高危先停下与你确认）
- **归一化**：把原始对话转成重生成所需格式 `{id, conversations:[{from,value}]}`
- **重生成**：转发给 speculators 仓的 `response_regeneration` 脚本，用 verifier 服务逐轮重生成 assistant 回答（on-policy）
- **交付**：预分词 jsonl（每行 `id` / `primary_id` / `input_ids` / `loss_mask` / `text`）+ 错误行文件 + ok/errors/truncated 统计

## 3. 上手使用

**只需用自然语言说出目标**，`path/to/...` 换成你机器上的实际路径，例如：

| 想做什么 | 示例提示词 |
| --- | --- |
| 重采样（最小用法） | `用 path/to/Qwen3-8B 对 path/to/conversations.jsonl 做重采样` |
| 指定条数与并发 | `对这个数据集做重采样，只跑 500 条，并发 8` |
| 先质检数据 | `先帮我看看 path/to/raw.jsonl 这批数据能不能直接用于重采样` |
| 指定输出目录 | `用这个模型对这份数据做重采样，产物放到 path/to/out` |

缺关键信息时 Agent 会逐项问你（不猜、不硬跑）。**建议提前想清楚**：

| 信息 | 说明 | 缺省行为 |
| --- | --- | --- |
| verifier 服务 endpoint | OpenAI 兼容 `/v1/chat/completions`，须返回 `prompt_token_ids` 与 `choices[0].token_ids` | 问你要（本 Agent 不代你拉起服务） |
| 模型 | verifier 模型本地路径（作 `--model` 传入） | 问你要 |
| 数据 | 原始多轮对话 jsonl，或已符合 `{id, conversations}` 格式的 jsonl | 问你要 |
| 条数 / 并发 | 本次跑多少条（可先小样本冒烟）、并发数 | 问你要 |
| 采样参数 | 可选，如 `temperature` 等 JSON 片段 | 不给则用上游默认 |
| 输出目录 | 产物与日志统一存放位置 | 问你要 |

## 4. 流程是怎样的

1. **对齐**：Agent 复述你的 endpoint / 数据 / 条数 / 输出目录；缺的问、错的改。
2. **预检**（原始对话数据时）：DFX 预检并出报告，high 风险先停下与你确认。
3. **归一化**：需要时把原始对话转成重生成输入格式，保持行序与内容不变。
4. **服务校验**：先探活 `/v1/models`，再发一次最小 chat 验证 `return_token_ids` 生效；不通过则不进入下一步。
5. **重生成**：按你的条数/并发执行；大批量后台化并轮询日志。
6. **校验交付**：校验产物字段与长度一致性，汇报 ok/errors/truncated、产物与日志路径。

你只需在数据风险与参数确认等关键节点拍板；其余由 Agent 执行并汇报每步产物路径。

## 5. 需要提前准备

- Ascend 推理环境与可用 NPU（**服务由你自己拉起**）。
- speculators 代码仓（v0.6.0，重生成脚本来自该仓）。
- 已就绪的 verifier vLLM 服务 endpoint，且支持返回 token ids。
- 待处理的多轮对话数据。
- 缺任何一项，Agent 会在动手前提示你补齐。

## 6. 交付物

- 预分词样本 jsonl（`input_ids` / `loss_mask`，可直接进训练）。
- 错误行文件（如 `<outfile>.errors.jsonl`）与日志。
- 若做了预检：DFX 预检报告与分级结论。
- 执行参数与统计（endpoint、条数、并发、ok/errors/truncated）。

## 7. 注意事项

- 服务未通过 `return_token_ids` 验证前，Agent 不会执行重生成。
- 不代你拉起服务、不擅自占卡；大批量任务会后台化轮询，日志无进展并不代表卡死。
- 只重生成，不改动输入数据内容与顺序，也不做训练与验收。
- 上游 speculators 接口若升级（高于 v0.6.0），封装脚本可能需要同步调整参数。
