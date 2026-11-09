---
name: rl-consistency-dump-analysis
description: 通用训推一致性(prefill)的 dump 数据分析 skill。分析**已采集**的训推两侧 dump:探查 dump 结构 → 语义边界映射(不要求算子同名)→ 逐边界精确门禁比对 → 定位首差异 → ULP 量化区分舍入/逻辑级 → 追放大链 → 差异点隔离验证(打桩迭代:输出打桩指令给采集侧,等它用 msprobe `load` 覆盖边界再采后比下游,枚举所有独立差异点)→ 排除根因假设 → 输出整体结论 → 产出分析报告。与具体框架/硬件解耦,适用 msprobe PrecisionDebugger 或等价工具采的任何 dump。触发词:训推一致性分析、dump 数据分析、dump 已采好帮我分析、首差异定位、所有差异点、差异点隔离、打桩指令、对齐报告、训推不一致结论、比较两侧 tensor。采集与打桩再采由 rl-prefill-consistency-dump 负责,本 skill 只消费已落盘 dump、输出打桩指令驱动再采。
---

# 训推一致性 Dump 数据分析

拿到两侧已采集的 dump(training prefill forward vs inference prefill forward,同一步/同一请求),离线比对训推不一致的首差异在哪、根因、放大到多大,并**通过打桩迭代枚举所有独立差异点**,产出机器可读结论与分析报告。**核心约束:按语义边界比对,不按算子名**——两侧算子命名差异极大(训练独立 Q 投影 ↔ 推理融合 QKV 输出的 q 段是常态),同名比对等于没比。本 skill 只读 dump、不改代码;**需要再采时输出"打桩指令"交给采集侧 `rl-prefill-consistency-dump` 执行(msprobe `load` 覆盖边界再采),拿下一轮 dump 继续比下游,直到枚举出所有独立差异点**。

## 做什么 / 不做什么

- **做**:读 dump、探查算子轨迹、语义边界映射、逐边界精确比对、定位首差异、ULP 量化、追放大链、输出打桩指令驱动隔离、枚举独立差异点、排除假设、输出整体结论、产出分析报告。
- **不做**:采集/打桩重跑 dump(那是 `rl-prefill-consistency-dump`)、修复差异并验证、接入训练循环算指标。
- **纪律**:dump 只读;原始 tensor/prompt/凭据不进报告与仓库,只留聚合数值与边界统计;证据不足标 `evidence_missing`,绝不编数值。**可"要求再采一轮"(输出打桩指令),但自己不碰采集/代码。**

## 执行顺序

```
0 锁定范围 → 1 探查 dump 结构 → 2 语义边界映射 → 3 逐边界精确比对
  → 4 首差异定位与定性 → 5 差异点隔离验证(打桩迭代)
  → 6 排除假设 → 7 所有独立差异点整体结论 → 8 产出报告
```

按序走,1 和 2 可合并(探查时顺手记两侧算子命名规律)。阶段 5 是与 `rl-prefill-consistency-dump` 的**循环接缝**:本 skill 输出打桩指令 → 采集侧 load 覆盖再采 → 本 skill 比下游 → 有独立差异就再输出下一轮指令,直到下游全 EXACT/只剩余舍入级。首轮可没有阶段 5(下游全 EXACT 直接到结论)。

**单独使用出口(只分析已采 dump,不触发再采)**:本 skill 可单独使用,也可配合采集侧使用。
- **单独使用**:只消费你提供的两侧 dump,阶段 0 目标选**"仅首差异"**,走 1→4→6→7→8;未做打桩隔离的独立差异如实标 `evidence_missing`,不宣称枚举完。
- **配合使用**:阶段 4 定性后若需确认"下游差异是首差异放大,还是存在第二个独立差异",进阶段 5 输出打桩指令交 `rl-prefill-consistency-dump` 再采,循环枚举所有独立差异点后再到结论。
阶段 5 打桩迭代**仅在配合使用时进入**;单独使用到结论前把"未隔离验证"如实标注即可。

## 0 锁定范围

写清并记入 `references/task_card.json`:

- 两侧 dump 路径(training/inference 各 `dump_tensor_data` 目录)
- 模型 / 训练后端 / 推理引擎及版本 / 硬件
- 采集配置:step、rank、request_id 关联方式(若能拿到)、prompt token 长度、并行切分(TP/PP/EP)
- 目标:仅首差异?还是完整报告?(默认完整;**单独使用(不再采)建议选"仅首差异"**;完整枚举所有独立差异点需配合采集侧打桩再采)
- 已知陷阱:训练侧 tensor 常带前导/中间 1 维 batch 维、两侧是否有 padding、推理侧是否逐 position 而非全量

只记事实,未知标 `unknown`,不编默认值。

## 1 探查 dump 结构

用 `scripts/probe_dump.py`(通用探查,不绑定架构)摸清两侧 dump:

1. **读元数据**:两侧 `dump.json`(task/level/framework)、`construct.json`(算子轨迹,key 顺序即执行顺序)、`stack.json`。
2. **列算子轨迹**:两侧 `construct.json` 的 key 列表——每条是 `.` 分隔的算子路径(常为 `前缀.模块路径.算子类名.forward.<序号>`);对比两侧前缀规律、算子类名差异、融合算子。
3. **列 tensor 文件**:`dump_tensor_data/*.pt` 文件名 = 算子路径 + `forward.{idx}.{output|input}.{n}.pt` / `.parameters.weight.pt`;统计每侧文件数与 input/output/weight 分布。
4. **抽查 shape/dtype**:任挑几个同语义算子两侧 tensor,记差异规律(前导 batch 维、中间广播维、逐 position vs 全量)。

探查结论记入分析记录:两侧命名规律(前缀替换规则、融合算子清单)、需切片对齐的边界候选。字段与命名规则见 `references/dump-structure.md`。

## 2 语义边界映射

核心方法论:**按语义对齐,不按名字**。语义边界 = 数学上同一位置的张量(如「GEMM 前输入归一化输出」「Q 下投影 GEMM 输出」「attention 输出」「残差加和输出」「最终 logits」)。完整方法见 `references/semantic-mapping.md`。

1. **划语义边界清单**,沿数据流:embedding → 各层输入归一化 → Q/K/V 下投影 → (norm →) 上投影 → RoPE → attention → o_proj → 残差 → MLP(fc1/act/fc2 或 MoE)→ 最终归一化 → logits/lm_head。
2. **逐边界对齐两侧算子**:训练侧某算子输出数学上等于推理侧哪算子的输入/输出?靠 construct 轨迹顺序 + 数据流 + shape 校验。融合算子用**切片/拆分**对齐(如推理侧融合 QKV 输出 `[seq, q_kv总维]` 按权重行序切 q 段与 kv 段);切片前先做相关性扫描确认布局。
3. **权重边界单独对齐**:两侧权重(embedding/qkv/o_proj/lm_head)按语义切好对齐,用于阶段 6 排除权重不同步。
4. **映射不了的标 `evidence_missing`,不猜**:shape 对不上、语义找不到、结构差异过大(MoE 内部路由/permute)的,记下但不硬比。

产出:**语义映射表**(边界名/训练侧算子 glob/推理侧算子 glob/变换规则/映射依据)= `scripts/analyze_dump_boundaries.py` 的 `--mapping` 输入。**本 skill 不带预设映射**,每个新 dump 现场建映射。

## 3 逐边界精确比对

用 `scripts/analyze_dump_boundaries.py --train <训练侧 dump_tensor_data> --infer <推理侧 dump_tensor_data>` 对每个边界跑精确门禁(`same_shape/same_dtype/no_nonfinite/torch.equal/max_abs_diff=0`),沿 `execution_order` 输出每条边界的证据,直到第一个不等的边界。

```bash
python3 scripts/analyze_dump_boundaries.py \
  --train <train>/step0/rank0/dump_tensor_data \
  --infer <infer>/step0/rank0/dump_tensor_data \
  --mapping semantic_mapping.json \
  --output boundary_results.json
```

`--mapping` 传自建语义映射 JSON(list of BOUNDARY,格式见脚本头注释),**新架构直接写 JSON,不必改脚本**。比对只读 dump。

结果分类:每边界 ∈ {`EXACT`, `NOT_EXACT`, `SHAPE_DIFF`, `DTYPE_DIFF`, `NONFINITE`, `MISSING`}(脚本输出)+ 语义层映射不上的标 `evidence_missing`。分类含义与结果必读项见 `references/analysis.md` §1。

## 4 首差异定位与定性

首差异 = **沿语义执行顺序第一个非 EXACT 的边界**(脚本已按 execution_order 排序并给出 `first_divergence`)。

关键纪律:**首差异必须是最早的那一个**——下游所有非零差异都是它放大的结果。若某边界是首个 NEQ 但更靠前的边界被跳过/漏比,先补全前序边界再定论;首差异的**所有前序边界必须全部 EXACT**。

首差异往往是极小值(如融合 vs 独立 GEMM 的 ±1~2 ULP),必须量化定性**舍入级**还是**逻辑级**:

1. **ULP 量化**:对首差异边界算 `diff / ULP(value)` 倍数分布。典型指纹:
   - **舍入级**(融合/独立 GEMM、累加顺序差):差异极稀疏(<1‰)、倍数集中于 ±1~2 ULP、position 稀疏分布 → 属预期浮点差异,非逻辑错误。
   - **逻辑级**(配置/权重/内核错误):差异密集或倍数远超 ULP、形态有规律(整行/整列/整 position)。
2. **放大链**:记录首差异量级 → 沿下游逐层 max_abs 变化(首差异常 1e-4~1e-2,经 norm/attention/MLP 放大到 1e0~1e2),证明"首差异小但被逐层放大到最终指标非零"。
3. **定性决定是否进打桩**:首差异逻辑级 → 下游差异可能只是放大、也可能有第二独立差异,进阶段 5;首差异舍入级 → 下游也只剩舍入级可不再打桩(记已知 gap),下游仍有逻辑级差异则仍需打桩隔离后再确认。

判定指纹表与 ULP 计算见 `references/analysis.md` §3。

## 5 差异点隔离验证(打桩迭代,输出打桩指令;仅配合使用时进入)

要回答"下游非零差异是首差异的放大,还是存在第二个独立差异?",用**打桩(数据覆盖加载)**把已确认差异边界强行对齐后比下游。完整方法论、打桩指令 schema 与判定见 `references/stubbing-loop.md`。

> 进打桩前先做**低成本排除**(权重不同步 / padding 未去净 / 两侧配置不等价——见阶段 6 清单),避免把根因差异误当独立差异去打桩。输出打桩指令的前提:采集侧已用 `rl-prefill-consistency-dump` 完成接入(0→7)且未恢复代码;若 dump 非该 skill 采集,先让采集侧补做接入再进阶段 8。

1. **判定是否需下一轮**:
   - 下游全 EXACT → 已隔离差异点是唯一独立差异,无下一轮。
   - 下游存在逻辑级差异(非 ULP 指纹)→ 可能为下一独立差异,进打桩。
   - 下游仅剩余舍入级 → 记入舍入级差异点清单,可结束。
   - 下游存在 evidence_missing 未比边界 → 先补映射或标注,不因"没比"误判"无差异"。
2. **输出打桩指令** `stub_instruction.json`(round N,交 `rl-prefill-consistency-dump`;模板见 `references/templates/stub_instruction.example.json`):
   - `source_side`/`target_side`:覆盖方向(要隔离哪侧前序误差,就用另一侧数据覆盖它)
   - `isolate_boundary`:本轮隔离边界(语义名 + 两侧算子)
   - `modules[]`:目标条目名 + 源 `.pt` 文件 + **变换规则**(重命名/切片/reshape/dtype,来自阶段 2 语义映射)
   - `dump_after_load`:覆盖后继续 dump 下游
   - `expected`:覆盖后该边界两侧输入应一致;若下游仍逻辑级差异 → 确认下一独立差异
3. **下一轮分析**(采集侧返回 round N+1 dump 后):先验证**隔离边界本身**两侧 bitwise 一致(load 生效证据;不一致说明打桩没生效,先别比下游),再比对隔离边界之后的下游:
   - 全 EXACT → 确认此前的差异都是已隔离差异点的放大,结束。
   - 仍逻辑级差异 → 确认第 N+1 个独立差异点,记录边界/定性/证据 → 输出下一轮打桩指令。
   - 残留舍入级 → 记入舍入级差异点清单。
4. **停止条件**:下游全 EXACT / 只剩舍入级 / 剩余边界 evidence_missing(标注后停止)。枚举完进阶段 7。

## 6 排除假设

把可能根因逐一列出并排除,每条记录排除依据(通用清单按序):

1. **权重不同步**:权重边界(embedding/qkv/o_proj/lm_head)是否 `torch.equal`?是 → 排除。
2. **输入未对齐/padding**:首差异前序输入边界是否 EXACT?训练侧是否有 padding(带 batch 维与推理侧 shape 不一致)?是未去净 → 修正映射或标注,不回退。
3. **配置差异**:两侧注意力/MLP 结构(独立 vs 融合、norm 融合位置、MoE 路由)是否语义等价?结构不同但数学等价 → 不算根因。
4. **并行切分不一致**:两侧 TP/PP/EP 一致?是 → 排除。
5. **fused 内核/累加顺序**:首差异形态符合 ULP 指纹 → 定性浮点舍入,非错误。
6. **随机性**:单样本确定性配置 + 前序 EXACT → 排除。

排除假设的证据与首差异证据同等重要——结论必须写清「为什么不是权重/配置/输入问题」。详见 `references/analysis.md` §5。

## 7 所有独立差异点整体结论

1. **边界级分类**:每边界 `EXACT`/`NOT_EXACT`/`evidence_missing`(SHAPE_DIFF/DTYPE_DIFF/NONFINITE/MISSING 归入 NOT_EXACT 或不比)。
2. **差异点清单(核心)**:枚举**所有独立差异点**(经阶段 5 打桩隔离验证),每个记:序号 + 隔离轮次、边界位置(语义名 + 两侧算子)、定性(舍入级已知 gap / 逻辑级)、放大链、根因假设 + 排除依据、是否可解。附舍入级差异点清单、evidence_missing 清单。
3. **整体级四分类**(映射到报告):全 EXACT → `exact`;存在逻辑级独立差异 → `nonzero`;未执行 → `not_run`;证据不足 → `evidence_missing`。全部差异点仅舍入级 → 训推一致(允许浮点差异,须写明每个舍入级差异点)。
4. **结论口径**:训推不一致结论必须指向「**全部独立差异点(位置/定性/根因/证据链/是否可解)**」,不能只报首差异或最终指标非零。
5. **数值指标口径**(如有 prefill logprob diff / E2E,只引用训练日志实测):
   - prefill 段 logprob diff(训练循环直接算,日志实测 mean/max/std):`mean≈0` 是训推一致,非 0 结合逐边界比对定位;日志没有就 `evidence_missing`,不编。口径见 `references/prefill-logprob-diff.md`。
   - response 段 E2E(`rollout_probs_diff`)在 prompt-only 裁剪下无实测(valid=0 是「没算」不是「一致」);要数值须跑含 response 的标准训练实测,否则 `evidence_missing`。
   - 两指标不同层面,不互相冒充。

## 8 产出报告

产出两类:**分析结论记录**(每次必做)与**分析报告**(评估/交付必做)。

### 8.1 分析结论记录(必须)

- 两侧 dump 路径 / step / rank / request_id 关联(含每轮 round N 的 dump 路径)
- 语义映射表(或映射文件路径)
- 比对汇总:边界总数、EXACT/NOT_EXACT/evidence_missing 计数
- 打桩迭代记录:每轮打桩指令 + 采集侧返回的下游结果
- **所有独立差异点清单**(每个:序号、隔离轮次、边界位置、定性、放大链、根因假设、是否可解)
- 舍入级差异点清单(已知 gap)、evidence_missing 清单
- 排除假设清单、整体结论 + 未解决问题

### 8.2 分析报告(建议;评估/交付必做)

从单一事实源 `report_source.json` 用 `scripts/generate_alignment_report.py` 渲染 MD/HTML 两种格式:

```bash
# 首次: 从内置中文模板初始化 report_source.json(可跳过,直接手写也行)
python3 scripts/generate_alignment_report.py --init-source report_source.json
# 编辑 report_source.json 填入本案例事实后,生成 MD/HTML 两种格式:
python3 scripts/generate_alignment_report.py --input report_source.json --output-dir reports/
# 产出: reports/alignment_report.md / alignment_report.html
```

**report_source.json 要点**(完整 schema 见 `references/report-source.md` 与 `references/templates/alignment_report_source.zh-CN.json`):

- `results`:`native/conditional_prefill_status` 与 `e2e_metric_status` 取 `exact|nonzero|not_run|evidence_missing`;**nonzero/zero 必须带数值 `e2e_metric_value`;没抓到实测就 `evidence_missing` + `value=null`**。
- `differences[]`:`status ∈ {excluded_known_gap, pending, resolved}`;**每个独立差异点一条**;逻辑级独立差异在整体 nonzero 时标 `excluded_known_gap`,不伪装 resolved;舍入级可在 differences 或 limitations 注明「浮点级,非错误」。
- `verification_axes`:`tensor_prefill` 必须与 `native_prefill_status` 一致;`weight_sync` 记 `exact`(权重 bitwise 一致证据)。
- `limitations[]`:如实记未覆盖边界、E2E 数值缺失、诊断模型范围。
- `provenance`/`comparison`/`artifacts`:版本、prompt 对齐口径(长度/切片/因果位移)、产物清单。

> **环境提示(先读)**:三个脚本都不能在裸宿主 python3 上直接跑——`probe_dump.py`/`analyze_dump_boundaries.py` 依赖 torch;`generate_alignment_report.py` 需要 Python 3.11+(f-string 表达式已用单引号写法,不再依赖 3.12 的 PEP 701)。宿主默认 python3 若是 3.9 且无 torch,就在容器(有 torch + 3.11,常见的 3.10/3.11 torch 容器即可)里执行,产物拷回。
> **报告生成器的模板定位**:`generate_alignment_report.py` 从"脚本所在目录上一级 + `references/templates/alignment_report_source.zh-CN.json`"读模板,**单独拷一个 .py 出来跑会报 `cannot read report source`**。容器内跑保持相对结构(把整个 skill 目录拷进去,SKILL.md 里 `scripts/...` 相对命令即可原位执行);`--init-source` 是标准起点。

## 纪律

- **架构无关是设计约束**:主体方法论不绑定任何框架/模型/硬件;不带预设映射,一切以现场 dump 为准。
- **现场验证优先于 reference**:映射/比对以两侧 dump 实际算子名、shape、dtype 为准;reference 只是线索,冲突以现场为准并修正。
- **只记事实**:未知标 `unknown`/`evidence_missing`,不填默认值、不编数值。
- **不泄露**:原始 tensor/dump/prompt/凭据不进报告与仓库。
- **只读**:不改 dump、不写回采集侧代码。
- **可要求再采,不自采**:需要隔离验证时输出打桩指令给 `rl-prefill-consistency-dump`,等它 load 覆盖再采后比下游;本 skill 不碰采集/代码,不在"没比到"的边界上假装"无差异"。

## Reference 文件

| 文件 | 内容 | 何时读 |
|---|---|---|
| `references/dump-structure.md` | msprobe/等价 dump 通用结构:dump.json/construct.json/stack.json/dump_tensor_data 字段与命名、两侧 shape/dtype 差异规律 | 阶段 1 探查前 |
| `references/semantic-mapping.md` | **语义边界映射方法论(架构无关)**:边界清单、融合算子切片对齐、权重边界、布局相关性扫描、evidence_missing 判定 | 阶段 2 |
| `references/analysis.md` | 逐边界精确门禁、首差异定位、ULP 量化、放大链、排除假设、分类结论的详细方法与判定指纹 | 阶段 3-4、6 |
| `references/stubbing-loop.md` | **差异点隔离验证(打桩迭代)方法论 + 打桩指令 JSON schema**:独立差异 vs 放大差异判定、指令格式、迭代停止条件、实战沉淀的判定方法论 | 阶段 5 |
| `references/prefill-logprob-diff.md` | prefill logprob diff(训练循环直接算的数值指标)的通用计算口径与解读陷阱 | 阶段 7 填数值指标时 |
| `references/pitfalls.md` | 分析侧实战失败教训(A 系列):shift-by-one 位移陷阱、lm_head 按 shape 找权重、logits 定性三件套等 | 阶段 1-5 遇症状对照 |
| `references/report-source.md` | `report_source.json` 字段说明与填法(结合模板) | 阶段 8 填报告源时 |
| `references/task_card.json` | 目标/范围记录模板 | 阶段 0 |
| `references/templates/stub_instruction.example.json` | 打桩指令 `stub_instruction.json` 示例模板(融合 vs 独立算子场景;算子名/路径/维度为占位符,以现场映射为准) | 阶段 5 输出打桩指令时 |
| `references/templates/alignment_report_source.zh-CN.json` | `report_source.json` 中文模板与字段 schema | 阶段 8 初始化/填报告源时 |
| `scripts/probe_dump.py` | 通用 dump 探查脚本:读元数据、列算子轨迹、列 tensor 文件、抽查 shape/dtype | 阶段 1 |
| `scripts/analyze_dump_boundaries.py` | 逐边界精确门禁比对脚本(`--mapping` 自建语义映射),输出 JSON + 首差异 | 阶段 3 |
| `scripts/generate_alignment_report.py` | 分析报告生成(MD/HTML),来自 true-on-policy-align | 阶段 8 |
