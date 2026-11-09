---
name: rl-prefill-consistency-dump
description: 训推一致性(prefill)采集侧:为训练/推理两侧采集可比的 prefill dump(输入对齐 PROMPTS_ONLY → 参数对齐校验(dump 前前置:校验两侧 config 超参/结构/权重,不一致输出参数不一致点并对齐)→ msprobe 接入采集 → 采集验证 → 恢复代码)。按用户需求判断用法:① 仅采集 dump——用户只要 dump 数据,走采集主流程(0→7→9),直接产出两侧可比 dump;② 配合分析——用户要求找所有独立差异点/差异点隔离验证/配合 rl-consistency-dump-analysis 分析时,才按分析侧打桩指令用 msprobe `load` 覆盖边界二次采集(阶段 8)再恢复代码。差异分析、首差异定位、打桩指令、整体结论由 rl-consistency-dump-analysis 承担。触发词:训推一致、训推一致性、prefill 比对、msprobe dump/采集、精度比对数据采集、打桩二次采集、参数对齐校验、参数不一致点、mHC 配置不一致、权重不同步、清卡验环境后采数据;已采 dump 的分析转 rl-consistency-dump-analysis。
---

# Prefill 训推一致性数据采集

核心思路:训练输入是 prompt+response,推理 prefill 只有 prompt → 训练侧先裁成 prompt-only(`PROMPTS_ONLY`),两侧各用 msprobe 采集 prefill 前向,用 `request_id` 把两侧 dump 钉在一起,产出两侧可比的 prefill dump。**对齐硬约束:训练端 padding 必须去掉,两侧输入 shape 对齐到真实 prompt 长度(如 190 vs 256 须去 padding 统一到 190),否则 dump 形状天然不等、逐 token 比对无从谈起(P18)。**

## 做什么 / 不做什么

- **做**:预检清卡、输入对齐、接入采集、request_id 关联、采集验证、打桩二次采集、恢复代码。
- **不做**:差异分析、首差异定位、输出打桩指令、整体结论(那是 **rl-consistency-dump-analysis** 的职责,本 skill 只采集 + 按指令打桩);修复差异。
- **纪律**:采集/打桩要临时改代码,预期内;但**采集完成后必须恢复代码,不留任何诊断残留**。恢复是流程的一部分,不是可选项。

## 执行顺序

```
0 锁定目标 → 1 预检清卡 → 2 授权 → 3 识别现场
  → 4 输入对齐 → 5 参数对齐校验(dump 前前置)
  → 6 接入采集 → 7 采集验证
  → 8 打桩二次采集(与分析侧循环:拿打桩指令 → load 覆盖边界 → 再产 dump)
  → 9 恢复代码
→ dump 交接给 rl-consistency-dump-analysis(差异点/打桩指令/整体结论)
```

阶段 7→8→7 是**与分析侧的循环**,直到 analysis 判定"所有独立差异点已找到";恢复代码只在全部轮次结束后执行一次。1 和 2 可合并(先看环境再一次性问清);阶段 5 是 dump 前前置(参数层),与阶段 4 输入对齐(数据层)、阶段 8 tensor 打桩(输入覆盖)互补、各管一层。

**硬性执行顺序(必须遵守)**:严格按 0→9 逐阶段推进,不得跳步。每一阶段的产物未产出并验证通过前,不得开始下一阶段——**尤其阶段 5 参数对齐校验(`param_alignment_report.json` 全部 EXACT)必须先于阶段 6 接入采集/改代码**;若发现已越过前置阶段,必须先回退补做该阶段,再继续。

**单独使用出口(只要 dump,不分析)**:本 skill 可单独使用,也可配合分析侧使用。
- **单独使用**:阶段 7 采集验证通过即达停止条件(拿到两侧可比的 prefill dump),**跳过阶段 8**(打桩二次采集/分析交接),直接**阶段 9 恢复代码**——恢复仍是强制项,不留诊断残留。
- **配合使用**:阶段 7 后交接 `rl-consistency-dump-analysis`,按打桩指令走阶段 8 循环,全部轮次结束后阶段 9 恢复。
"0→9 不得跳步"指所选用法路径内逐阶段推进;单独使用是流程终点提前到阶段 7,不是跳阶段。

## 0 锁定目标

把下列信息记入 `references/task_card.json`:模型 / 训练后端(Megatron|FSDP|其他)/ 推理引擎及版本(vLLM|SGLang|其他)/ 硬件 / 证据工具(msprobe PrecisionDebugger 或等价)/ 采集范围(确定性单样本 prefill)/ 停止条件(采到两侧可比 prefill dump)。

**通用输入模板**(框架/后端/硬件无关;用户给得越全,后续越少反问):

```text
训练框架及版本 / 训练后端(FSDP/Megatron/DeepSpeed/自研)
推理引擎及版本(vLLM/SGLang/Transformers/自研) / 硬件后端(Ascend/CUDA/ROCm)
执行环境(SSH目标/容器/conda或venv激活) / 工作目录(绝对路径) / 基线命令 / 日志位置
已知环境约束:占卡残留服务 / PID namespace 隔离 / 宿主无 root kill 权限 / 端口占用等
模型与样本:权重路径 / 固定样本 ID / prompt 来源
证据工具:msprobe PrecisionDebugger(不确定就先检查并提出适配方案)
采集范围:确定性单样本 prefill(默认) / 停止条件:采到两侧可比的 prefill dump(默认)
目标 E2E 指标(选填,如 rollout_probs_diff_mean == 0)
授权与纪律:一次问清、采完恢复代码、全程中文
```

只记事实,未知字段留空或标 `unknown`,不编默认值。框架识别按阶段 3 现场完成。

## 1 预检清卡

目的:目标卡空闲、残留进程不干扰、环境可用。**只动本任务相关进程**,不宽泛清理。

1. **查卡**:`npu-smi info`(或等价)确认占用与占用者。
2. **识别残留**:占卡的 vLLM/ray/训练服务。注意 PID namespace 隔离:宿主进程列表与容器内不同,杀进程须进对应容器;宿主通常无 root kill 权限。
3. **停残留**:精确 PID/PGID 或容器内正常退出,禁止宽泛 `pkill` 或无界 `/tmp` glob。
4. **验环境**:确认容器/conda/venv 激活方式;确认 msprobe 已装(`pip show mindstudio-probe`)且**在哪个容器/env**——一台机器常多容器多 env,查错环境会得出相反结论,训练/推理环境分开确认。SGLang ≥0.5.11 原生集成在 `ImportError` 时静默跳过,msprobe 装错环境 → dump 静默为空、不报错。
5. **确认资产**:权重存在、基线命令可执行、日志可写、产物根目录已规划。
6. **记录**:清卡前后卡状态、停掉的进程,写进 task_card。

## 2 授权

一次性问清下列动作,记录后流程中不再打断:

- `AUTH_STOP_RUNNING_SERVICES` — 停占卡残留服务
- `AUTH_EDIT_TARGET_CODE` — 临时改训练/推理代码(接入采集,采后恢复)
- `AUTH_RUN_DUMP_JOB` — 启停采集作业
- `AUTH_CLEAN_APPROVED_DIRS` — 删指定的产物/临时目录
- `AUTH_RESTORE_CODE` — 采后恢复代码

同时确认 SSH 目标、环境激活、工作目录、基线命令、日志位置、产物根目录。凭据绝不写入 skill/工作区/报告。

## 3 识别现场(框架无关)

接入位置强依赖框架版本/后端。**先现场识别再动手;references 只是线索,不是坐标**。

1. **识别形态**:训练框架及版本、训练后端(FSDP|Megatron|DeepSpeed|自研)、推理引擎(vLLM|SGLang|其他,决定原生 msprobe 集成是否存在)、架构形态(显式 micro_batch 循环 FSDP 型 → 包循环体 vs 连续前向 Megatron 型 → 插 forward 内部)。
2. **查线索**:对应框架有 `verl.md`/`slime.md`/`vime.md`;新框架或文档过时 → 按 `new-framework.md` 通用方法现场搞定。
3. **现场验证三件事**(必须):① PROMPTS_ONLY 裁剪位置(`grep compute_log_prob / get_data_iterator / rollout_data`,顺数据流找到构造 log_prob 输入处);② 接入位置(找到 log_prob 主前向,确认是否被缓存跳过如 `can_reuse_log_probs_in_loss`——采不到数据先怀疑它);③ request_id 关联(找推理调度日志与训练 rollout 数据的对应字段,没有就自己打)。
4. **冲突处理**:以现场为准,偏差记入改动清单,采完修正 reference 或标注 outdated。
5. **官方采集姿势**:模型/框架是否在官方 `*_consistency_preprocess_dump.md` 支持范围(verl + fsdp/megatron + 官方名单模型,名单以官方文档为准)?范围内 → 采集姿势跟官方;范围外 → 本 skill 通用方法。能否用官方 `--consistent_check` 自动比对由分析侧判断。

> 改任何文件前先 `grep` 确认符号存在——适配文档可能过时或错误,真实排查遇过 `get_prompts_only_batch`、`dispatch_logger.py`、`_ensure_debugger` 不存在需新增、`get_data_iterator` 签名与文档不同;不存在就按"新增"处理并记入改动清单。
> 引擎起不来先查自己改动:确认子进程实际拿到的可见卡(如 `ASCEND_RT_VISIBLE_DEVICES`)→ 沿启动链定位 → 回滚做原生对照,别先断言框架"不支持分卡"。机制与教训见 `vime.md`。

## 4 输入对齐

让训练/推理 prefill 数学等价、逐卡逐样本可对应。框架无关,按阶段 3 现场确认落实。

1. **`PROMPTS_ONLY=1`** 裁掉 response,只留 prompt。
2. **batch 约束**:`mini_batch_num=1`、`gac=1`、`balance_batch=False`(同一请求两侧落在同一卡)。
3. **去 padding(必须,否则 shape 对不上)**:`use_remove_padding=True`(verl);slime 用 `--data-pad-size-multiplier 1`,TP>1 时两侧按同一 pad_size。训练侧按真实 prompt token 数截取输入与 logprob(`log_probs[:, :-1]` 前先切),保证两侧 dump 输入 shape=真实 prompt 长度、逐 token 1:1。采完验:两侧同一边界输入 shape 相等且=真实长度;[256,...] vs [190,...] 即 padding 没去净,回修重采(P18)。
4. **eager 与确定性**:`enforce_eager=True`、`TORCHDYNAMO_DISABLE=1`、SGLang `--sglang-disable-cuda-graph`(msprobe 只在 eager 下能钩住算子);`seed_all()` 固定种子,关 shuffle/随机采样。
5. **并行一致**:TP/PP/CP/DP 两侧一致,dump 才能按 rank 一一对应。
6. **排除干扰**:关验证/预热前向(verl `val_before_train`),避免其 generate 污染 dump。

> **裁剪副作用**:裁剪副本**只喂 log_prob 前向**,别替换整个 rollout_data(整体替换会因 response_lengths=0 split 崩);`loss_masks` 裁 `m[:0]` 靠 pad 展开;adv/correction 让 `rollout_corr_helper` 在 PROMPTS_ONLY=1 时直接 `return batch, {}` 早退,并同步改 loss/metrics 兜底,**训练照常走完**(P1-P3、P6b)。

## 5 参数对齐校验(dump 前前置)

目的:接入采集、跑 dump 前,先校验训练侧与推理侧模型的**配置参数、参数结构、权重**是否对齐;不一致则作为**参数不一致点**输出、打桩对齐后重建模型,再进阶段 6 采 dump。动机案例:mHC 配置不一致——训练侧 Megatron-Bridge 转换字段名写错(`enable_hyper_connections` 应为 `enable_mhc_connections`),`config.json` 的 mHC 开关/超参训练侧全落空,训练跑无 mHC 的退化模型而推理恒启用,两侧模型结构都不同,dump 天然对不上。这类**模型层不一致必须 dump 前解决**,不能等 tensor 比对才发现。**开关对齐 ≠ 参数对齐**:结构开关对齐后还要核该模块的 ② 配置参数(须是模块实际消费的字段,不是 bridge 写了就算)与 ③ 派生形状 vs checkpoint 权重(实例化模块核对);三要素清单见 `references/param-alignment.md` §1.5。完整手册见 `references/param-alignment.md`。

1. **静态 config 比对**(必做):读训练侧**实际生效** config(provider/bridge 转换结果,不能只读原始 config.json——mHC 案例原始 config 有 `hc_mult` 但 bridge 没接住)+ 推理侧 `hf_config`,逐项比对。训练侧实际生效 config 获取:**已落盘** bridge/provider config 优先;没有则用 `scripts/param_align_provider_dump.py` 从训练侧实际 bridge 实例化 provider dump(`python3 scripts/param_align_provider_dump.py <hf模型目录> <out.json> --bridge <训练侧bridge类名>`,默认 `DeepSeekV4FlashBridge`,内部含 nvrx 环境适配;bridge 源码路径与默认容器布局 `/workspace/Megatron-Bridge/src`、`/workspace/MindSpeed-Bridge` 不符时,追加 `--bridge-src <源码根目录>` 指定)。通用比对脚本 `scripts/param_align_check.py`(**不绑定任何模型**):`--train-cfg <训练侧实际生效config> --infer-cfg <推理侧config.json> [--mapping <字段映射json>]`——传映射按语义逐项比(两侧字段名可不同,支持嵌套点分路径与结构断言 `infer_expected`,示例 `scripts/param_alignment_mapping.example.json` 复制改写);不传映射自动同名键比对兜底。结构开关(如 `enable_mhc_connections`)是最高优先级——一侧有模块另一侧没有,比数值不一致更严重。
2. **不一致 → 参数不一致点**:产出 `param_alignment_report.json`(逐项 `EXACT/DIFF/MISSING_SIDE` + `mismatch_points`),编号 P1/P2...
3. **打桩对齐**:有 DIFF 时产出 `param_alignment_instruction.json`(模板 `references/templates/param_alignment_instruction.example.json`):`align_side`/`method`(`config/code`|`weight`)/`target`/`action`/`verify`。执行对齐(修 config/代码开关/统一权重)→ **重建模型** → 重跑校验,不一致点变 EXACT 才进采集。**msprobe `load` 只覆盖模块输入 tensor,不覆盖参数(`.parameters.weight.pt` 不参与),参数对齐不能靠 load**,走 config/code/weight。
4. **动态结构/权重比对**(模型构建后、采集前):dump 两侧 `named_parameters()` 清单比结构(`train_only_keys`/`infer_only_keys`)+ 对应参数 `torch.equal` 比权重;不一致回第 3 步对齐。可复用 rl-consistency-dump-analysis 的语义映射方法论(`references/semantic-mapping.md`),本阶段只做参数/结构/权重层。
5. **登记**:对齐改动与阶段 6 接入改动同册登记,阶段 9 统一恢复;对齐结果(哪些参数不一致已对齐/未对齐)写进 task_card 交接分析侧。

> 别把参数对齐当输入对齐:阶段 4 对齐输入数据(数据层),本阶段对齐模型本身(参数层);两层都 EXACT 才能采可比 dump。

## 6 接入采集

在训练/推理两侧包住目标前向,用 msprobe PrecisionDebugger 采集。**每一步改动都记录**(阶段 9 恢复依据)。接入会改目标代码,预期内。

**改代码前先读 reference(硬性)**:先按 Reference 表"何时读"列,读完本阶段对应 reference(`verl.md`/`vime.md`/`msprobe-config.md`/`param-alignment.md`/`pitfalls.md` 等),再对照现场 grep 确认符号;未读对应 reference 不得开始接入/改代码。

> 两个真实翻车点,先读再改:① 补丁脚本定界符避开目标代码里的引号(用 `'''`),应用前 `py_compile` 自检(P11);② 推理引擎返回字段别照抄文档——vLLM-ascend 的 `prompt_logprobs` 是 **top-k 嵌套 dict**,解析前先打 `[DUMP_DIAG]` 确认结构(P12)。

1. **训练侧接入(官方做法:直接改源码,不写 hook 文件、不走框架预留 hook 入口)**:worker `__init__`/训练类构造处,`DUMP_ON=1` 时 `from msprobe.pytorch import PrecisionDebugger, seed_all; seed_all(mode=True)` 并实例化 debugger(`task='tensor', level='L0', step=[0], dump_path=...`);`DUMP_ON=0` 为 `None`(env-gated)。只 actor 引擎建 debugger,ref 引擎(`forward_only`)跳过。包主前向:`debugger.start(model)` → 前向 → `stop()` → `step()`;FSDP 包 micro_batch 循环体,Megatron 插 forward 内部。同时把 micro_batch 的 request_id 写入训练日志(verl 系 `update_actor_log.jsonl`)。
2. **PROMPTS_ONLY 裁剪 + 改 loss/metrics 兜底(官方做法,训练照常走完)**:裁剪后训练不能崩——改三处:裁剪(`compute_log_prob` 截 input_ids/attention_mask/position_ids 的 response 段并 pop response 相关字段)、改 loss(`response_length==0` 用 prompt 段切 logits、`response_mask=None` 用全 True 兜底)、改 metrics(缺字段主动返回 0 值不崩)。**不要在采集开关下跳过训练**,正确做法是让裁剪后的 loss 也能算(P6b)。
3. **prefill logprob_diff(采集侧数值指标,训练循环内直接算、打印训练日志)**:找两侧 logprob 碰头处(actor 重算的 log_probs 与 rollout 带回的推理侧 prompt 段 logprob 同时可见),逐 token `|exp(lp_train)−exp(lp_infer)|` + mean/max/std,带统一关键词打印(如 `prefill_logprob_diff_mean`),env-gated。**拿不到推理侧 prompt 段 logprob 就算不了,别拿 response 段 `rollout_log_probs` 顶替**。通用步骤见 `references/prefill-logprob-diff.md`。
4. **response 段 E2E 指标口径**:`rollout_probs_diff`/`train_rollout_logprob_abs_diff` 锚定 **response 段**,PROMPTS_ONLY 裁剪下天然算不出——日志 valid=0/mean=0 是"没算"不是"一致";要实测须补跑 `PROMPTS_ONLY=0` 标准训练 step,没抓到标 `evidence_missing`。详见 `references/probs-diff-monitoring.md`。
5. **推理侧**:先确认引擎原生 msprobe 集成——vLLM-ascend/SGLang ≥0.5.11 都原生集成,传 config 即启用(vLLM-ascend `--vllm-enforce-eager` + `--vllm-additional-config '{"dump_config_path":...}'`;SGLang `--sglang-msprobe-dump-config`);需 request_id 调度日志才在 model_runner 注入 DispatchLogger(verl 系);引擎无原生集成 → 手写推理侧接入(`new-framework.md` §2)。
6. **request_id 贯穿**:确认 request_id 能从推理引擎一路流到训练 micro_batch 日志。
7. **msprobe 配置**:两侧各一份 config(`msprobe-config.md` 模板),`task=statistics`(轻量)或 `tensor`,`level=L0/mix`,dump_path 分开。要算 prefill logprob_diff,采集级别须覆盖 final 输出边界(logits/output_layer/lm_head)。
8. **记录改动清单**:`git diff` 快照或备份,登记每处改动(文件/位置/内容/恢复方式)。

## 7 采集验证

1. 基线命令追加采集开关(`DUMP_ON=1`、`PROMPTS_ONLY=1`,推理侧 `dump_config_path`;开关名按阶段 3 现场确认)跑一次。
2. **验产物**:两侧 dump 路径出现 `step_N/rank_M/dump.json`(或等价),非空。
3. **验关联**:推理调度日志挑 phase=prefill 单请求 step,拿 request_id 去训练日志找对应 step/rank;vllm ≥0.14 会给 request_id 追加 8 位 hex 后缀,匹配前剥掉。
4. **验可比**:两侧样本 token 序列一致(长度/内容),输入 shape 相等且=真实 prompt 长度;[256,...] vs [190,...] → 回阶段 4 修掉重采,别硬比对(P18)。
5. **记录**:step/rank、两侧 dump 路径、关联关系,写 task_card。
6. **验指标产出**:训练日志出现 `prefill_logprob_diff_mean` 且非空、数值合理(不能光看没报错);其"是否一致"的解读归分析侧。E2E 0 值=没算。逐边界比对/首差异定位不在本 skill。
7. **交接 round 0(配合使用时)**:把 dump 路径、step/rank、request_id 关联方式、采集配置、task_card 交接给 rl-consistency-dump-analysis,等它返回首差异结论与打桩指令。**先别恢复代码**——打桩轮次还要用同一套接入。**单独使用时**:阶段 7 验证通过即完成,跳过阶段 8,直接阶段 9 恢复代码。

> **采集踩坑**:训练侧有效 dump 在 **step0**(compute_log_prob 主前向),step1 未 flush 为空;只有 `can_reuse_log_probs_in_loss=False` 时主前向才执行,需 `--get-mismatch-metrics` 强制。docker exec 清理用 `[r]ay` 正则防自匹配、长任务用 `-d` 分离;每次采集前确认目标卡 HBM 回基线(残留 EngineCore 会让二次采集 OOM)。
> **启动失败先对坑,别从头 debug**:① 基于基线重建的脚本逐段 diff 基线,漏 `--rm-type` 报 `Rule-based RM type is not specified`(P13);② 含空格 JSON 过 `ray job submit` 被重新 shell 解析截断,改无空格 compact 形式(P14);③ 容器内慢命令用长 timeout + `-d` + `tr -d '\n\r'`(P15)。详见 `references/pitfalls.md`。

## 8 打桩二次采集(数据覆盖加载,与分析侧循环)

拿到 analysis 侧差异结论后,若下游还有待确认独立差异,按它的**打桩指令**做隔离再采,迭代找出**所有独立差异点**。原理:用先前采集的模块级输入 tensor 覆盖目标模型该模块实际输入(msprobe `load`,forward_pre_hook 注入),把差异边界强行对齐、隔离前序误差,再 dump 下游——覆盖后若仍有非舍入级差异,就是下一个独立差异点,继续打桩,直到下游全 EXACT/只剩余舍入级。完整执行手册见 `references/stubbing.md`。

> **前提**:msprobe 需为含 `module_load/tensor_loader.py` 的版本(26.1 正式版没有,需上游 master 源码安装;验证 `python3 -c "import msprobe, importlib.util; print(importlib.util.find_spec('msprobe.pytorch.dump.module_load.tensor_loader'))"`)。

1. **接打桩指令**:analysis 侧输出 `stub_instruction.json`(round N)= `target_side`(被覆盖侧 train/infer)/`source_side`(数据侧)/`isolate_boundary`(边界语义名)/`modules[]`(目标条目名 + 源 .pt + 变换规则)/`dump_after_load`。本阶段唯一输入。
2. **查源 dump**:必须 `level=L0` 或 `mix`(load 只读模块级 tensor);step/rank 自动对齐。
3. **改造源数据(训推算子名不同,必须做)**:load 要求源与目标"模块结构相同、命名一致";训推两侧天然不同(如训练独立 Q 投影 ↔ 推理融合 QKV 输出的 q 段)。按变换规则把源 `.pt` 重命名为目标条目名、对齐 shape/dtype(切片/reshape),放新目录(如 `stub_source_roundN/`),`load.path` 指向它。
4. **写 load 配置**(目标侧 config 顶层加 `load` 段):
   ```json
   {
       "task": "tensor",
       "dump_path": "/<本轮 dump 根目录>",
       "level": "L0",
       "rank": [], "step": [],
       "load": {
           "path": "/<改造后源dump根>",
           "modules": ["Module.<目标模块路径>.<类名>.forward.0"],
           "step": [], "rank": [],
           "dump_after_load": true
       }
   }
   ```
   条目名=`Module.{dotted_path}.{ClassName}.forward.{N}`(取 `.input.{i}.pt` 之前的部分),以 `Module.` 开头、`.forward.{N}` 结尾,能被目标模型 `named_modules()` 命中。
5. **重跑采集**:沿用阶段 6 接入不变,只改 config(覆盖哪侧改哪侧并重跑);验证可 `dump_after_load=false` 只覆盖不 dump,观察下游指标恢复。注意 load 覆盖后,被覆盖模块及**直接上游**的 backward 数据在 dump 中可能缺失——只影响 dump 采集,不影响实际反向结果。
6. **验 load 生效**:覆盖边界两侧输入 cosine=1.0;`stop()` 无"未命中条目"warning;`load.path` 缺失/modules 空/条目格式非法启动报错;shape/dtype 不匹配只 warning 仍替换,forward 可能报错——改造源数据务必对齐。
7. **交接下一轮**:新 dump 路径(dump_path)+ round 号 + 目标侧 + 被覆盖条目清单,交给 analysis 侧比对下游。

> **打桩方向**:要隔离/验证训练侧前序差异 → 用推理侧 dump 覆盖训练侧模块输入;反之亦然。选改动成本低、语义映射最清晰的一侧。目的都是让两侧从覆盖边界起输入一致,判断下游是否还有独立差异。

## 9 恢复代码

全部轮次结束(analysis 给出整体结论或用户叫停)后恢复,无论结果如何:

1. 按改动清单逐一还原(反向 patch / `git checkout` / 恢复备份)。
2. 验证:`git diff` 干净或与备份完全一致。
3. 确认无残留:`DUMP_ON`、`PROMPTS_ONLY`、`dump_config_path`、debugger 调用、`load` 配置段/改造后源 dump 目录。

恢复是采集的组成部分,跳过等于没做完。**打桩轮次之间不要恢复**(7→8→7 循环共用同一套接入)。

## 职责边界

本 skill 只负责采集、参数对齐与打桩:产"两侧可比的 prefill dump"、dump 前参数对齐校验(阶段 5)、按 analysis 打桩指令 load 二次采集产"打桩后 dump";**差异分析、首差异定位、输出打桩指令、整体结论由 `rl-consistency-dump-analysis` 承担**。每轮采集后把 dump 路径与 task_card(含参数对齐结果)交接给它,等它的打桩指令;全部轮次结束后恢复代码。本 skill 不承担任何分析/报告职责。

## 纪律

- **现场验证优先于 reference**:改动以现场 grep/数据流为准,references 仅线索;reference 有错以现场为准并采完修正。
- **证据决定采集完成**:没有 `dump.json`、关联不上,不算采集完成。
- **只记事实**:未知标 `unknown`/`evidence_missing`,不填默认值。
- **不泄露**:原始 tensor/dump/prompt/凭据不进采集产物与交接文档。
- **最小 case 不是目标**:最小化是为了对齐,不代表最终范围。

## Reference 文件

| 文件 | 内容 | 何时读 |
|---|---|---|
| `references/verl.md` | verl 线索卡(SPMD ≤0.6 / Async v0.7–0.8 / V1 v0.9):先识别版本架构再看对应节 | 识别到 verl |
| `references/slime.md` | slime v0.2.2(Megatron + SGLang)线索卡:训推分卡、padding 约束、SGLang 接入 | 识别到 slime |
| `references/vime.md` | vime(Megatron-Bridge + vLLM-ascend)适配:设备分配(base_gpu_id=ray 物理卡号,勿二次映射)、引擎起不来排查、训练侧接入、指标碰头点 | 识别到 vime |
| `references/msprobe-config.md` | PrecisionDebugger 接口 + config 模板 + dump.json 字段 | 配置/验证时 |
| `references/param-alignment.md` | 参数对齐校验(dump 前前置):三层校验(config 超参/结构/权重)、字段映射表、报告与对齐指令、执行流程、与 tensor 打桩的区别 | 阶段 5 必读 |
| `references/stubbing.md` | 打桩二次采集(msprobe `load`)执行手册:官方要点、训推命名改造、config 写法、验证与坑 | 阶段 8 前必读 |
| `references/templates/stub_instruction.example.json` | 打桩指令示例模板(算子名/路径/维度为占位符,以现场语义映射为准) | 阶段 8 接指令时对照 |
| `references/prefill-logprob-diff.md` | prefill logprob_diff 通用步骤:训练循环内 `log_softmax` → 按 token 对齐(去 padding/末位)→ 逐 token 概率差 + mean/max/std → 打印训练日志 | 阶段 6 接入、阶段 7 验证 |
| `references/probs-diff-monitoring.md` | response 段 E2E 指标(`rollout_probs_diff`/`train_rollout_logprob_abs_diff`):含义、verl 原生开关、通用落地;裁剪下 0 值=没算 | 阶段 6 接入、阶段 7 验证 E2E |
| `references/new-framework.md` | 新框架/未知框架通用方法论:现场识别清单、三件事找法、reference 自我纠错 | 各框架阶段 3(含新框架/文档过时) |
| `references/pitfalls.md` | 实战失败教训(PROMPTS_ONLY 副作用、msprobe step 语义、docker/残留进程、补丁嵌套引号、prompt_logprobs 嵌套、漏 rm-type、ray job 拆 JSON、TE/F.silu 接入坑等) | 流程前通读一次;阶段 6/7/8 遇症状对照 |
| `references/task_card.json` | 目标/授权/环境记录模板 | 阶段 0 |
