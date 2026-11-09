# 新框架 / 未知框架适配(通用方法论)

verl / slime / vime 的适配文档(`verl.md` / `slime.md` / `vime.md`)只是**速查线索**,不是权威坐标。适配文档未覆盖、或已过时/错误的框架,按本文现场完成。

## 0. 核心立场

- **现场证据优先于任何文档**:改代码前必须在目标代码里 `grep` 确认符号/函数/日志真实存在。
- **reference 可能是错的**:它基于某次真实核对写成,但框架会升级、文件会重构、路径会变。reference 说"在 X 文件改 Y",现场找不到 Y 或行为对不上 → **以现场为准**,把偏差记入改动清单,采完后修正或标注已过时。
- **没有 reference 不阻塞**:新框架直接走本文 1–3 节,现场找三件事,不需要等适配文档。

## 1. 现场识别清单(找什么)

| 要识别的 | 现场怎么找 |
|---|---|
| 训练框架及版本 | `pip show <框架>` / `git log` / 训练启动脚本里的 import |
| 训练后端 | 并行原语:FSDP 有 `shard_model_optimizer`;Megatron 有 `tensor_model_parallel`;DeepSpeed 有 `ds_config` |
| 推理引擎 | vLLM / SGLang / 自研 Transformer 推理;决定原生 msprobe 集成是否存在 |
| 架构形态 | 训练主循环是**显式 micro_batch 循环**(包循环体)还是**连续前向**(插 forward 内部) |
| 环境 | msprobe 装在哪个容器 / 哪个 conda env(查错环境 → 静默空 dump) |
| 设备传递链 | 卡号来源(常见变量名 `base_gpu_id`)与 `visible_devices` 传递路径 |

## 2. 三件事的通用找法

### ① PROMPTS_ONLY 裁剪位置
- **目标**:找到「构造 log_prob 前向输入」的那一处(不是训练前向)。
- **现场找**:`grep -rn "compute_log_prob\|get_data_iterator\|rollout_data\|RolloutBatch" <训练主目录>`,顺数据流定位。
- **裁剪数学框架无关**:`prompt_length = total_length - response_length`;`loss_masks` 裁空(`m[:0]`)靠 `F.pad(mask, (prompt_length-1, 1))` 展开成 prompt_length。
- **副作用处理通用原则**:裁剪副本**只喂 log_prob 前向**,不替换整个 rollout_data;`rollout_corr_helper` 在 `PROMPTS_ONLY=1` 时直接 `return batch, {}` 早退,loss/metrics 同步兜底,**训练照常走完**(官方口径,不跳过训练;见 `pitfalls.md` P6b)。字段名不同就按新框架 rollout_data 结构改写,数学不变。

### ② 接入位置
- msprobe **框架无关**(经 PyTorch module 钩子采集算子输入/输出),难点只在 `start → forward → stop → step()` 生命周期与框架 step 语义对齐。
- **显式 micro_batch 循环**(FSDP 型)→ 包循环体,每轮 start/stop,step 对应一轮。
- **连续前向**(Megatron 型)→ 插在 forward 内部 per-micro-batch 粒度。
- **主前向可能被缓存跳过**:框架常复用已有 log_probs(如 verl `can_reuse_log_probs_in_loss`)。现场找这个条件,采不到数据先怀疑它,再怀疑环境。
- 统一按官方做法**直接改源码**:手写 `debugger.start(model) → 前向 → stop() → step()` 包住目标前向;不另写 hook 模块、不用框架预留 hook 入口。

### ③ request_id 关联
- 找推理侧调度日志的 request_id ↔ 训练侧 rollout 数据对应字段(现场 `grep request_id`)。
- 没有现成日志:在接入处自己把 request_id 打到训练侧日志。
- vllm ≥0.14 会给 request_id 追加 8 位 hex 后缀,匹配前剥掉。

### ④ 并行 / 设备 / 确定性(框架无关硬约束)
- 训练与推理的 TP/PP/CP/DP 必须一致,dump 才能按 rank 一一对应。
- 确定性:enforce_eager、固定 seed、关 shuffle、关 eval、单样本小 batch。
- 设备传递:确认子进程实际拿到的可见卡(启动日志 / `ps e -p <pid>`),拿错先怀疑自己的改动,再怀疑框架。

## 3. 如何验证 reference 是否正确(自我纠错)

改任何代码前,对 reference 里的每个坐标跑一遍:

| reference 说 | 现场验证 | 判定 |
|---|---|---|
| "在 `xxx.py` 改 `yyy()`" | `grep -n "def yyy\|yyy" <路径>` | 找不到 → reference 已过时,按现场改 |
| "框架版本 v0.8,接入在 A" | `pip show` / `git log`,对照代码 | 版本不符 → 按现场代码结构重找 |
| "request_id 从 X 到 Y" | 走一遍数据流,看日志 | 流不通 → 现场加日志重找关联 |
| "PROMPTS_ONLY 在 C 位置裁" | 看该处是否真的构造 log_prob 输入 | 不是 → 顺数据流找真位置 |

**冲突处理流程**:以现场为准 → 偏差记入「改动清单」→ 采完修正 reference 或标注 outdated。

## 4. 新框架适配坑清单

- 推理引擎非 vLLM/SGLang:原生 msprobe 集成**不存在**,需手写推理侧接入(model_runner 注入 debugger + 调度日志)——最重的工作。
- 训练数据流结构不同:裁剪数学不变,但 rollout_data 字段/嵌套按新框架重写。
- step 语义:msprobe `step()` 对应哪个训练步骤要现场摸,采错就是空 dump。
- 多容器多 env:msprobe 装错环境 → 静默空 dump,不报错。
- 设备分配:不要凭记忆加"物理映射 patch";先确认卡号来源,做原生对照。

## 5. 何时更新 references

- 现场验证确认某 reference 错误 → **采完后修正它**(本 skill 自维护,避免下次再踩)。
- 新框架适配成功后 → 可选地把「该框架的三件事」沉淀成新的 `references/<framework>.md`,标注基于哪次核对,保持"现场验证优先"的口径。
