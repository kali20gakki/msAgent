# 打桩二次采集(数据覆盖加载,msprobe `load`)执行手册

配合 SKILL 阶段 8。**打桩** = 用先前采集的模块级(L0/mix)输入 tensor,在目标模型模块前向时**覆盖其实际输入**(msprobe `load`),把训推差异边界"强行对齐",隔离前序累计误差,再继续 dump 下游,迭代找出所有独立差异点。官方功能名**数据覆盖加载**,官方文档 `docs/zh/user_guide/dump/pytorch_data_load_instruct.md`(上游 master 源码内有;容器 26.1 正式版无此功能,需源码安装)。

## 1 前提

- msprobe 版本带 `module_load/tensor_loader.py`(验证:`python3 -c "import msprobe, importlib.util; print(importlib.util.find_spec('msprobe.pytorch.dump.module_load.tensor_loader'))"`)。26.1 正式版没有,需上游 master 源码编译安装。
- 必须有先前 `level=L0` 或 `mix`、`task=tensor` 采集的源 dump(`dump_tensor_data/*.pt`)。
- `load` 只覆盖**输入 tensor**(位置参数 args 的 `.input.{i}.pt`、关键字参数 kwargs 的 `.kwargs.{key}.pt`);`.parameters.weight.pt`、`.backward.*` 不参与覆盖。
- **本阶段输入是分析侧的打桩指令** `stub_instruction.json`(示例模板 `references/templates/stub_instruction.example.json`,算子名/路径仅为占位符);字段含义见 rl-consistency-dump-analysis 的 `references/stubbing-loop.md` §2。

## 2 官方机制与 config

工具在模块 `forward_pre_hook` 里按 `full_forward_name` 精确匹配 `load.modules`,匹配成功就把源 dump 的 tensor 替换进模块 forward 输入。config 顶层 `load` 段:

```json
{
    "task": "tensor",
    "dump_path": "/home/round1_dump_b",        // 本轮 dump 产物根(覆盖后数据落这里)
    "level": "L0",
    "rank": [], "step": [],
    "load": {
        "path": "/home/stub_source_round1",     // 改造后源 dump 根目录(step{N} 的上一层)
        "modules": ["Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"],
        "step": [], "rank": [],                 // 空 = 每个 step/rank 都 load,按源对应 step/rank 自动对齐
        "dump_after_load": true                 // true=覆盖后继续 dump 下游;false=只覆盖不 dump
    }
}
```

参数:

| 参数 | 必选 | 说明 |
|---|---|---|
| `path` | 必选 | 源 dump 根目录(即 `step{N}/` 上一层),工具自动在 `{path}/step{N}/rank{M}/dump_tensor_data/` 找 `.pt`。单卡场景源是 `proc{pid}` 目录,工具自动查找 |
| `modules` | 必选 | 覆盖条目名 list[str],格式 `Module.{dotted_path}.{ClassName}.forward.{N}`(`forward.{N}` 是模块前向调用序号,从 0 起,**不可省略**——梯度累积下同一模块多次调用要指定第几次) |
| `step` / `rank` | 可选 | 控制哪些 step/rank 执行 load(空 = 全部),加载始终从源对应 step/rank 自动对齐 |
| `dump_after_load` | 可选 | 覆盖后是否继续 dump,默认 `false` |

## 3 训推场景:源数据改造(必须做,否则匹配不上)

**两个路径别混**:打桩指令里的 `source_dump` 是**改造前**读源 `.pt` 的目录(常是源侧 `.../step0/rank0/dump_tensor_data`);`load.path` 是**改造后**新目录的**根**(`step{N}` 的上一层)。改造 = 从 `source_dump` 读 `.pt` → 重命名/切分/对齐 → 按 `step{N}/rank{M}/dump_tensor_data/` 结构写入 `load.path` 下。

**约束**:`load` 要求源 dump 与目标模型"模块结构相同、模块命名一致(`named_modules()` 点分路径对应)"。训推两侧算子名/结构**天然不同**——不能把训练侧 dump 直接塞给推理侧 `load`。

| 场景 | 训练侧算子 | 推理侧算子 |
|---|---|---|
| 例 | `<训练侧模块路径>.<训练侧算子类名>` | `<推理侧模块路径>.<推理侧算子类名>` |
| 影响 | 条目名不同 | 融合 vs 独立 GEMM,shape/切分不同 |

所以打桩前必须**改造源 dump 数据**:

1. **重命名 `.pt` 文件**为目标的条目名格式:`Module.<目标模块点分路径>.<类名>.forward.<N>.input.<i>.pt`(或 `.kwargs.<key>.pt`)。
2. **对齐 shape/dtype**:融合算子输出 → 按权重行序切片拆成独立投影的 q/kv/v 段;需要 squeeze/reshape 的做变换。切片前先做相关性扫描确认布局(见 rl-consistency-dump-analysis 的 `references/semantic-mapping.md` §布局确认)。
3. 改造产物放**新目录**(别污染源 dump),`load.path` 指向它。**新目录必须保持与源相同的层级结构** `step{N}/rank{M}/dump_tensor_data/`——load 按 `{load.path}/step{N}/rank{M}/dump_tensor_data/{目标条目名}.input.{i}.pt` 精确找文件。变换规则来自 analysis 侧打桩指令的 `modules[].transform`。
4. 检查目标模块 forward 的**参数顺序**:工具按位置 index 加载——A 模型 `forward(x, mask)` 与 B 模型 `forward(mask, x)` 会被按位置错配(shape 碰巧对上但语义错误),工具无法自动检测,须保证两侧 forward 签名一致。

## 4 流程(与 SKILL 阶段 8 对应)

```
接 stub_instruction.json(analysis 侧)
  → 检查源 dump 是 L0/mix
  → 改造源 .pt(重命名/切分/对齐 dtype)→ 新目录
  → 目标侧 config 加 load 段
  → 重跑采集(沿用采集开关/接入)
  → 验证 load 生效(覆盖边界 cosine=1.0)
  → 交接 round N+1 dump 给 analysis 侧
```

打桩方向:**要隔离/验证训练侧前序差异 → 用推理侧 dump 覆盖训练侧模块输入**(训练侧打桩);反之同理。选改动成本低、映射清晰的一侧。目标都是"让两侧从覆盖边界起输入一致,判断下游是否还有独立差异"。

## 5 验证与坑

**验证 load 生效**:
- 覆盖边界:目标侧 dump 的该模块 input 与源 dump 对应 tensor `torch.equal` / cosine=1.0(官方示例用 `msprobe compare` 比对确认)。
- 无"未命中条目"warning(`debugger.stop()` 时提示条目从未命中 = `forward.{N}` 序号不对)。
- 日志出现 `[load] module validation: X/Y modules valid in model`。

**异常处理表**(官方):

| 异常 | 处理 |
|---|---|
| `load.path` 不存在/非目录 | 启动报错(`MsprobeException`) |
| `load.modules` 空 / 非 list / 条目格式不合法(不以 `Module.` 开头、不以 `.forward.{N}` 结尾、含 `.backward.`、重复) | 启动报错 |
| 配了 `modules` 没配 `path` | 启动报错 |
| 模块名在目标模型不存在 | warning,训练继续;条目匹配不上 = 覆盖无效,会误导结论 |
| 条目配了但 `forward.{N}` 序号从未命中 | `debugger.stop()` 时 warning,训练继续 |
| 源 `.pt` 缺失 | warning,该参数保持原值,训练继续 |
| shape/dtype 不匹配 | warning,仍执行替换,forward 可能 `RuntimeError`——改造源数据务必对齐 |

**特别注意**:
- 覆盖会替换模块输入 tensor(经 `torch.load`),被覆盖模块及其**直接上游**的 backward 数据(反向输入输出、参数梯度)在 dump 中可能缺失——**只影响 dump 采集,不影响模型实际反向结果**。训推一致主要看 forward,一般无碍。
- load 与 dump 的 `level` 解耦:load 强制挂模块 hook,`level=L1` 也能覆盖;`dump_after_load=true` 时 dump level 任意。
- 一个边界**隔离一次**:首差异边界覆盖后,下游若恢复 EXACT 说明下游差异都是它放大的;若仍非舍入级差异 → 第二个独立差异点,下一轮打桩覆盖第二个边界。**不要一次覆盖多个未确认边界**,否则分不清哪个是独立差异(多个独立差异需要逐个隔离才能编号;只有要"验证整体下游是否还有任何独立差异"时可一次性覆盖多个已知差异边界)。
- 源 dump 的 step/rank 需与目标运行对齐;训练侧有效 dump 在 step0(P4)。

## 6 参考资料

- 官方:`/home/z00938604/msprobe/docs/zh/user_guide/dump/pytorch_data_load_instruct.md`(宿主源码)、上游 `python/msprobe/pytorch/dump/module_load/tensor_loader.py`。
- 分析侧打桩指令 schema 与独立差异判定:rl-consistency-dump-analysis 的 `references/stubbing-loop.md`。
