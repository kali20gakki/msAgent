# dump 数据结构(通用,msprobe PrecisionDebugger / 等价工具)

训推两侧 dump 由采集工具(msprobe PrecisionDebugger 或等价)落盘,**布局通用、与具体框架无关**。探查脚本见 `scripts/probe_dump.py`。下面是通用的字段与命名规则。

## 目录布局

```text
{step}/rank{M}/
├── dump.json             # 采集元数据
├── construct.json        # 算子轨迹(dict,key 顺序即执行顺序)
├── stack.json            # 调用栈信息
└── dump_tensor_data/     # 实际 tensor 文件(*.pt)
    ├── <op>.forward.input.0.pt
    ├── <op>.forward.output.0.pt
    ├── <op>.forward.0.output.0.pt
    └── <op>.forward.0.parameters.weight.pt
```

两侧各一份(train / infer),`rank{M}` 对应并行卡位。

## dump.json(元数据)

```json
{
  "task": "tensor",            // statistics | tensor 等
  "level": "mix",              // L0 / mix 等,决定采集深度(要采 logits 必须覆盖 output_layer/lm_head 边界)
  "framework": "pytorch",
  "dump_data_dir": "<dump_tensor_data 相对路径>",
  "data": { ... }              // 按 step 索引的算子路径清单(可作执行顺序交叉验证)
}
```

**读法**:先读 task/level/framework 确认采集范围;`level` 是否覆盖到 final 输出边界直接决定能否算 logits 级指标。

## construct.json(算子轨迹)

- 根对象为 dict,**key = 算子完整路径,顺序 = 执行顺序**——这就是「语义执行顺序」的事实来源。
- key 形态:一串以 `.` 分隔的算子路径,常为 `前缀.模块路径.算子类名.forward.<序号>`。训练侧与推理侧的路径前缀、模块层级、算子类名、融合方式都可能不同。
- 前缀规律(`probe_dump.py` 会自动统计):首 token 反映算子来源——`Module` 带完整模块路径,`Functional`/`Torch`/`Tensor` 是底层算子,其余前缀多为框架/NPU 专属算子(如融合类 `NPU`/`_C_ascend`/`Triton`)。两侧前缀一致与不一致都是常态,不一致正是需要语义映射的原因。
- 用法:两侧 key 列表就是比对地图;同名 key(两侧都有)是**可直接精确比对的候选边界**,仅一侧出现的 key 是**结构差异来源**(融合算子的常见出处)。

## dump_tensor_data(实际 tensor)

文件名 = 算子路径 + `forward[.序号].{kind}.{n}.pt`。两种形态(解析规则见 `probe_dump.py`):

| 形态 | 文件名 | 说明 |
|---|---|---|
| 无序号 | `Functional.dropout.0.forward.input.0.pt` | 算子路径到 `.forward` 为止 |
| 有序号 | `...VocabParallelEmbedding.forward.0.output.0.pt` | 路径含 `.forward.<idx>`,同算子多次调用按 idx 区分 |

kind ∈ `input.<n>`(第 n 个输入)/ `output.<n>`(第 n 个输出)/ `parameters.weight`(权重)。**input/output 可多份**;shape/dtype 在文件内。

## 两侧 shape/dtype 差异规律(探查时重点核对)

- **训练侧常带前导/中间 1 维 batch 维**:如 `[1, seq, hidden]`、`[seq, 1, hidden]`、`[seq, 1, 1, hidden]`;推理侧常直接 `[seq, hidden]`。比对前按实际 squeeze 对齐。
- **推理侧可能逐 position 而非全量**:logits 若按 position 逐个 dump(每个输出 `[1, vocab]`,多个 position 文件),需 concat 成 `[seq, vocab]` 再比。
- **融合算子输出是多段拼接**:如融合 QKV 下投影输出 `[seq, q_dim+kv_dim]` = q 段 + kv 段;融合 GEMM(gate+up)输出 `[seq, 2*hidden]`。比对前按权重行序切片。

## 探查动作清单(阶段 1)

1. 读两侧 `dump.json`(task/level/framework)。
2. 读两侧 `construct.json` 轨迹,记录 key 数量、前缀规律、**同名 key**、**仅一侧出现的 key**(尤其含 `fused/merge/moe/grouped/add_rms/swiglu/flash` 关键词的——融合算子切片对齐的高频来源)。
3. 列 `dump_tensor_data` 文件数、kind 分布(input/output/weight 各多少)。
4. 抽样同名算子两侧 shape/dtype,记下 squeeze/切片需要(前导 batch 维、中间广播维、逐 position 拼接)。
