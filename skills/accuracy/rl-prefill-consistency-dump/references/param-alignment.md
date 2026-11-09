# 参数对齐校验(dump 前前置步骤)

配合 SKILL 阶段 5。**参数对齐校验** = 在接入采集、跑 dump 之前,先校验训练侧与推理侧模型的**配置参数、参数结构、权重**是否对齐;不一致就作为**参数不一致点**输出,打桩对齐两侧参数后,再接入采集跑 dump。它与 dump 后的 tensor 打桩(`references/stubbing.md`,msprobe `load` 覆盖模块输入)是两个不同层面:**本阶段保证"两侧模型本身一致",tensor 打桩保证"两侧输入数据一致"**。

## 0 为什么需要(动机案例:mHC 配置不一致)

真实案例:训练侧通过 Megatron-Bridge 把 `config.json` 转成 Megatron provider 时,字段名写错(`provider.enable_hyper_connections = True` 应为 `provider.enable_mhc_connections = True`;`provider.num_residual_streams` 应为 `provider.mhc_num_residual_streams`),导致 `config.json` 里的 mHC 配置(开关、`hc_mult`)在训练侧**全部落空**:训练侧 `enable_mhc_connections=False`,mHC 模块根本没被注入,训练跑的是无 Hyper-Connections 的退化模型;而推理侧 vllm 直接读 `config.hc_mult`,**无条件启用 mHC**。

后果:两侧**模型结构都不一样**(训练侧无 `hc_head_*`/`hc_attn_*` 参数,推理侧有),dump 出来的 tensor 天然对不上——这种问题等到 tensor 比对才发现,会白耗一轮分析。**参数对齐校验把它提前到 dump 之前**:两侧配置/结构/权重不一致,先对齐,再采集。

> 本质:**这是"模型层不一致",不是"输入数据不一致"**。模型层不一致必须 dump 前解决;输入层不一致由阶段 4 输入对齐 + dump 后 tensor 打桩处理。

## 1 校验内容:三层

| 层 | 校验什么 | 时机 | 发现什么 |
|---|---|---|---|
| ① config 超参比对 | 两侧配置参数逐项比对(数值) | 静态:采集前读两侧 config/checkpoint 元数据 | 超参数值不一致(如 `hc_mult` 训练侧没接住、`sliding_window` 两侧不同) |
| ② 参数结构比对 | 两侧模型实例化后的参数 key 清单对齐 | 动态:模型构建后、跑前向采集前 | 结构不一致(一侧有模块另一侧没有,如 mHC `hc_head_*` 训练侧缺失) |
| ③ 权重比对 | 两侧对应参数数值 `torch.equal`/max_abs_diff | 动态:模型构建后 | 权重不同步(两侧加载的权重数值不一致) |

**①必须做**;②③在同一侧模型构建后顺手做(见 §5.2),避免额外一轮构建。

## 1.5 对齐检查三要素(通用):开关 / 参数 / 形状

**开关对齐 ≠ 参数对齐**。开关只是"注入模块的门";结构开关打开后,顺带核三件事(通用套路,不绑定模型):

- **① 开关**:结构的启用开关两侧一致;推理侧无开关字段(结构上恒启用)时,训练侧必须为 True。
- **② 参数**:开关驱动的模块所用配置参数(流数/迭代次数/eps/topk/heads/窗口/压缩率…)两侧一致。字段映射的 train 字段必须是**模块实际消费的字段**(grep 模块 builder 里 `config.<字段>` 读取点确认),不是 bridge 写了就算——bridge 可能写死字段、字段名写错、或未接线而用 dataclass 默认值(与参考一致可能是巧合)。
- **③ 形状**:模块由参数派生的权重形状与参考 checkpoint 权重逐位对上——这是"结构确实按参考建、权重能装"的最终裁判。config 层只证配置值,构造层才证构建结果;涉及结构开关时实例化模块,打印推导属性/派生形状核对。

上报时区分「config 层已核」(①②)与「构造层已核」(③),别把前者当结构对齐结论。

## 2 字段映射表(训练侧 vs 推理侧)

训练侧字段名(provider/Megatron config)与推理侧字段名(hf_config)往往**不同名**——这正是 mHC bridge bug 的形态。比对前先建字段映射,别按同名比。以 DeepSeek-V4-Flash-Vision 实测为例:

| 语义 | 训练侧字段(provider/Megatron) | 推理侧字段(hf_config) | 判定 |
|---|---|---|---|
| Hyper-Connections 开关 | `enable_mhc_connections` | 推理侧无独立开关(结构上恒启用)→ 训练侧必须为 `True` | 结构开关:训练侧 False + 推理侧启用 = 结构不一致 |
| Hyper-Connections 流数 | `mhc_num_residual_streams` | `hc_mult` | 数值须一致 |
| Sinkhorn 迭代 | `mhc_sinkhorn_iterations` | `hc_sinkhorn_iters` | 数值须一致 |
| HC eps | 训练侧若代码硬编码/无字段 | `hc_eps` | 训练侧无对应字段时标记 `MISSING_SIDE` + 查代码是否硬编码同值 |
| CSA 窗口 | `csa_window_size` | `sliding_window` | 数值须一致 |
| CSA 压缩率 | `csa_compress_ratios` | `compress_ratios` | 数值须一致 |
| DSA indexer heads | `dsa_indexer_n_heads` | `index_n_heads` | 数值须一致 |
| DSA indexer head_dim | `dsa_indexer_head_dim` | `index_head_dim` | 数值须一致 |
| DSA indexer topk | `dsa_indexer_topk` | `index_topk` | 数值须一致 |
| MTP 层数 | `mtp_num_layers` | `num_nextn_predict_layers` | 数值须一致 |
| Hash 层数 | `moe_n_hash_layers` | `num_hash_layers` | 数值须一致 |
| 隐藏维度 | `hidden_size` | `hidden_size` | 数值须一致 |
| 层数 | `num_layers` | `num_hidden_layers` | 数值须一致 |
| 注意力头 | `num_attention_heads` | `num_attention_heads` | 数值须一致 |
| 词表 | `actual_vocab_size` | `vocab_size` | 数值须一致 |

> **字段映射表按现场模型实测建,本表只是 DSV4-Flash-Vision 的起点**。新模型先列两侧 config 全部 key,按语义对齐,再比对。现场验证优先于本表(与 SKILL 阶段 3 同一纪律)。
> **train 字段必须选「模块实际消费的字段」**(grep 模块 builder 的读取点),不是 bridge 写了就算(§1.5 ②)。

## 3 校验产物 `param_alignment_report.json`

逐项比对结果 + 参数不一致点清单。格式:

```json
{
  "model": "DeepSeek-V4-Flash-Vision (4-layer)",
  "checked_at": "<时间/现场>",
  "items": [
    {
      "id": "P1",
      "semantic": "mHC enable 开关",
      "train": {"field": "enable_mhc_connections", "value": false, "source": "Megatron provider (bridge 输出)"},
      "infer": {"field": "<无独立开关, 结构上恒启用>", "value": true},
      "status": "DIFF",
      "verdict": "结构不一致: 训练侧 mHC 未启用, 推理侧启用"
    }
  ],
  "mismatch_points": ["P1"],
  "weights": {"status": "EXACT", "checked_keys": 3450, "diff_keys": 0},
  "structure": {"status": "DIFF", "train_only_keys": [], "infer_only_keys": ["decoder.hc_head_fn", "decoder.hc_head_base", "decoder.hc_head_scale", "layers.*.hc_attn_*", "layers.*.hc_ffn_*"]},
  "summary": "参数对齐: 1 个不一致点 (P1 结构开关), 权重 EXACT; 需对齐后采 dump"
}
```

- `status ∈ {EXACT, DIFF, MISSING_SIDE, N/A}`;结构层 `train_only_keys`/`infer_only_keys` 记录只在一侧存在的参数 key(模式通配)。
- **结论**:全部 EXACT → 可采;存在 DIFF → 出对齐指令对齐后采。

## 4 参数对齐指令 `param_alignment_instruction.json`

存在不一致点时输出(模板 `references/templates/param_alignment_instruction.example.json`)。格式:

```json
{
  "round": 0,
  "mismatch_points": ["P1"],
  "alignment": [
    {
      "point_id": "P1",
      "align_side": "train",
      "semantic": "mHC enable 开关",
      "method": "config/code",
      "target": {"field": "enable_mhc_connections", "value": true},
      "action": "修 Megatron-Bridge deepseek_v4_bridge.py L463: enable_hyper_connections → enable_mhc_connections (上游字段名笔误); L465 num_residual_streams → mhc_num_residual_streams; 重建模型",
      "verify": "重跑参数对齐校验, P1 变 EXACT"
    }
  ],
  "then": "对齐 → 重建模型 → 重跑校验(阶段 5)→ 接入采集(阶段 6)→ 跑 dump"
}
```

**对齐方式**(重要约束):

| method | 做法 | 适用 |
|---|---|---|
| `config/code` | 改一侧 config 参数或代码开关,重建模型(默认,主要方式) | 结构/开关不一致(mHC)、超参数值不一致 |
| `weight` | 加载时用对齐后的权重(从另一侧导出/转换后加载,或统一权重来源) | 权重不同步 |
| ~~msprobe `load`~~ | **不可用于参数对齐**:`load` 只覆盖模块输入 tensor,`.parameters.weight.pt` 不参与覆盖(见 `stubbing.md` §1) | — |

> **对齐方向**:以"权威侧"为准。哪一侧更接近官方/config 定义就对齐到哪一侧;两侧都偏离时先修到 config 定义值再比。mHC 案例:推理侧直接读 config(权威),把训练侧对齐到推理侧(启用 mHC)。

## 5 执行流程

```
阶段 3 识别现场 → 阶段 4 输入对齐
  → 5.1 静态 config 比对(读两侧 config,按字段映射表比对)→ param_alignment_report.json
  → 5.2 若 DIFF: 出 param_alignment_instruction.json → 执行对齐(改 config/code/weight)
      → 重建模型 → 重跑校验, 不一致点变 EXACT
  → 5.3 动态结构/权重比对(模型构建后、采集前, 见 §5.2)
  → 全部 EXACT → 进阶段 6 接入采集
```

### 5.1 静态 config 比对(先做,零成本)

- 读训练侧 provider/Megatron config(从 bridge 构造逻辑或已落盘 config 取)+ 推理侧 `hf_config`(`config.json` 或 vllm 加载后的 `model_config.hf_config`)。
- **训练侧实际生效 config 获取**:① 已落盘 bridge/provider config 直接读;② 没有则用 `scripts/param_align_provider_dump.py <hf模型目录> <out.json> [--bridge <训练侧bridge类名>]`——实例化训练侧实际 bridge 的 provider 并 dump 全部可序列化属性(默认 `DeepSeekV4FlashBridge`;内含 Megatron-LM nvrx 版本探测的环境适配,134 实测可跑)。dump 出的属性名即映射 `train` 字段取值来源。
- 通用脚本 `scripts/param_align_check.py`(**不绑定任何模型**,纯 Python 无 torch,可在任意容器/宿主跑):`--train-cfg <训练侧实际生效config> --infer-cfg <推理侧config.json> [--mapping <字段映射json>] [--out <报告路径>]`。
  - **传映射**:按 §2 字段映射逐项语义比对(两侧字段名可不同、嵌套点分路径、结构断言 `infer_expected`)。映射写成外部 json,示例 `scripts/param_alignment_mapping.example.json` 复制改写(脚本不内建任何模型映射)。
  - **不传映射**:自动同名键比对兜底——枚举两侧顶层键并集,同名键直接比,单侧缺失标 `MISSING_SIDE`,嵌套 dict 值标 `SKIP`;适合快速筛查与建映射前的摸底。
- **训练侧字段必须从实际转换结果取**(如 provider 实例),不能只读 `config.json` 原始值——mHC 案例就是 `config.json` 有 `hc_mult` 但训练侧 bridge 没接住,只比 `config.json` 会漏。
- 产出 `param_alignment_report.json` 的 `items`(status ∈ `EXACT/DIFF/MISSING_SIDE/SKIP/N/A`)+ `mismatch_points`。
- 有 DIFF/MISSING_SIDE → §5.2;全 EXACT → §5.3。

### 5.2 打桩对齐 + 验证

1. 按 `param_alignment_instruction.json` 的 `alignment[]` 执行(改 config/code/weight)。
2. **重建模型**(对齐涉及结构/权重,必须重建,不能热改)。
3. 重跑 §5.1,确认对应 `point_id` 从 `DIFF` 变 `EXACT`;**未变 EXACT 不得进采集**。
4. **构造层验证(§1.5 ③,对齐涉及结构开关时必须做)**:用训练侧真实 bridge 生成 provider(含对齐后的 override)→ 实例化被开关驱动的模块 → 打印推导属性与派生参数形状 → 与 checkpoint 对应权重形状逐位比对。这是「开关打开后模块确实按参考结构建、权重能装」的主动证明,不能停在 config 层。
5. 记录改动到改动清单(阶段 9 恢复依据,与阶段 6 接入改动同册登记)。

### 5.3 动态结构/权重比对(模型构建后,采集前)

模型构建完成、跑前向采集之前,做一次**结构 + 权重**比对(msprobe 或直接 torch,在模型 forward 入口之前挂一次):

1. **结构比对**:dump 两侧模型 `named_parameters()`/权重 key 清单,按语义映射对齐,列 `train_only_keys` / `infer_only_keys`。有只在一侧存在的参数 → 结构不一致(如 mHC `hc_head_*` 训练侧缺失),回 §5.2 对齐。
2. **权重比对**:两侧对应参数 `torch.equal`(或 `max_abs_diff=0`)。不等 → 权重不同步,回 §5.2 `weight` 方式对齐。
3. 两者 EXACT → 记录到报告,进阶段 6。

> 动态比对放**接入采集之后、跑前向之前**:接入采集已让模型可用,直接挂一次参数比对即可,不需要额外构建。若模型构建耗时,可与采集接入共用一次构建。

## 6 与 tensor 打桩的区别(别混)

| | 参数对齐校验(本阶段) | tensor 打桩(`stubbing.md`) |
|---|---|---|
| 时机 | **dump 前** | dump 后(首差异定位后) |
| 对齐对象 | 模型**配置/结构/权重** | 模块**输入 tensor** |
| 手段 | 改 config/code/weight,重建模型 | msprobe `load` 覆盖输入 |
| 目标 | 两侧模型本身一致 | 隔离前序误差,枚举独立差异点 |
| 触发 | 采集侧固定前置步骤 | 分析侧打桩指令 |

两层互补:**先保证模型一致,再采 dump;采完若 tensor 仍有差异,再打桩隔离**。参数对齐不解决 tensor 级差异,反之亦然。

## 7 验证与坑

- **别只比 `config.json`**:训练侧实际生效值要从 bridge/provider 转换结果取(字段名可能接错,如 mHC)。
- **结构开关是最高优先级不一致**:`enable_mhc_connections` 这类"一侧有模块另一侧没有"的差异,比数值不一致更严重,必须先对齐。
- **开关对齐 ≠ 参数对齐(§1.5)**:开关只是门;开开关后还要核 ② 结构参数(且须是模块实际消费的字段)与 ③ 派生形状 vs checkpoint 权重,区分「config 层已核」与「构造层已核」。
- **对齐必须重建模型**:改 config/代码后热改参数不算数,结构变化必须重建再校验。
- **`load` 覆盖不了参数**:别试图用 msprobe `load` 对齐权重,它只覆盖模块输入(`stubbing.md` §1 明确).
- **对齐改动要登记**:与阶段 6 接入改动同册,阶段 9 统一恢复(本阶段改动属于诊断前置,采完一并恢复)。
- **权重来源统一**:两侧若从同一权重文件加载,权重比对通常 EXACT;若一侧用了转换后权重,必须比对。
- **参数不一致点交接给分析侧**:对齐结果记录进 `task_card.json`,让分析侧知道"dump 是在参数对齐后采的、哪些参数不一致已对齐、哪些未对齐未处理"。不改变分析 skill 职责,只补充交接信息。

## 8 参考资料

- mHC 案例证据链:Megatron-Bridge `deepseek_v4_bridge.py` 字段名笔误 → 训练侧 `enable_mhc_connections` 落空;vllm 直接读 `config.hc_mult` 无条件启用。
- 权重/结构比对复用 rl-consistency-dump-analysis 的语义映射方法论(`references/semantic-mapping.md`),本阶段只做参数/结构/权重层,不做 tensor 数据比对。
