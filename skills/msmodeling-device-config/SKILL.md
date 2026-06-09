---
name: msmodeling-device-config
description: 在需要根据自然语言规格为未支持硬件新增或更新 DeviceProfile 设备画像条目时使用
version: 0.4.0
source: local-session-analysis
---

# 设备画像自然语言导入器

引导用户把自然语言硬件描述转换为可运行、可校准、可复现的 `DeviceProfile` 设备画像。

## 适用场景

- 用户想要添加尚未支持的加速器、GPU、NPU、板卡、die、chiplet 或 SKU。
- 用户提到类似 `ATLAS_800_A3_752T_128G_DIE` 的 profile，并想添加相似条目。
- 用户以文字、表格、截图、数据手册或零散笔记的形式提供规格。
- 用户只有部分规格，但希望先得到一个可运行的近似 profile，后续再校准。

## 默认策略

采用两阶段导入：

1. 基于可靠事实构建最小可运行 profile。
2. 将用户确认的估值、助手推断的字段和兜底默认值全部标记为 `needs calibration`，如果用户希望提高准确性，再继续收集校准数据。

每轮对话都维护一张内部事实表：

- `confirmed`：用户明确提供或确认的事实。
- `ambiguous`：单位、粒度、范围或指标含义不明确的事实。
- `missing`：写出可运行 profile 仍缺少的必要事实。
- `needs calibration`：为先跑通而暂用的默认值、估值、假设或省略项。

默认写入目标：`tensor_cast/device.py`。生成的 `DeviceProfile` 应直接添加到该文件中，作为可被 `--device <PROFILE_NAME>` 直接引用的内置 profile。

不要默认新建 `tensor_cast/device_profiles/<device_slug>.py`。只有当用户明确要求临时 profile、自定义 profile 文件或隔离实验时，才写入 `tensor_cast/device_profiles/`。

## 当前设备画像代码结构

编辑前先阅读 `tensor_cast/device.py`，确认现有厂商类、profile 命名风格、互联常量和 `DeviceProfile` 注册方式。只有在用户明确要求写入 `tensor_cast/device_profiles/` 时，才需要额外阅读 `tensor_cast/device_profiles/README.md` 和 `tensor_cast/device_profiles/__init__.py`。

重要类型：

- `DeviceProfile(name, vendor, comm_grid, mma_ops, gp_ops, compute_efficiency, memory_size_bytes, memory_bandwidth_bytes_ps, memory_efficiency, static_cost)`
- `CommGrid(grid=torch.arange(...).reshape(...), topologies={start_dim: InterconnectTopology(...)})`
- `InterconnectTopology(bandwidth_bytes_ps, latency_s, comm_efficiency, type=InterconnectType.CLOS/FULL_MESH)`
- `StaticCost(mma_op_cost_s, gp_op_cost_s, comm_op_cost_s)`

`DeviceProfile.__post_init__` 会把每个 profile 注册到 `DeviceProfile.all_device_profiles`，因此 `name` 必须唯一。默认把新增 profile 写入 `tensor_cast/device.py`；虽然 `tensor_cast/device_profiles/__init__.py` 会自动导入该目录下的每个 `.py` 文件，但这只作为用户明确要求自定义文件时的备用路径。

`CommGrid.topologies` 的 key 是 `start_dim`，不是随意的层级编号。通常 `0` 表示覆盖整个 grid 的最慢互联层，`1` 表示从第 1 维开始的更快子拓扑，最后一个维度表示最快的内层互联。当前实现要求 `grid.ndim == len(topologies)`，并且每个 grid 维度至少为 2。

## 工作流程

默认使用“新手引导模式”，除非用户主动一次性给出完整规格或要求高级模式。

1. 首轮只问最容易回答的 2 到 3 件事：硬件名称/厂商、资料来源、希望按单卡还是单 die 建 profile。用户不知道时允许回答“不知道”。
2. 先解释，再追问。每个专有概念都用一句中文解释，并给出用户可直接照抄的回答示例。
3. 不要在首轮暴露 `mma_ops`、`gp_ops`、`CommGrid` 等内部字段名；等用户理解对应含义后，再说明它们会写入哪个字段。
4. 解析出候选事实，并用简短清单复述：已确认、还不确定、可以先默认的项。
5. 每轮最多问 2 到 3 个问题。每个问题都提供“我不知道 / 先跳过 / 用默认值”的出口。
6. 如果用户无法提供某项事实，给出明确选项：省略、使用用户确认的估值、或使用兜底默认值。
7. 写文件前再次确认所有会被写入 `tensor_cast/device.py` 的估值、假设和插入位置。
8. 写入后运行导入/注册检查，并在最终回复中列出 `needs calibration` 和可执行命令行示例。

不要把“看起来合理”的硬件常识直接写成事实。只要不是用户明确给出的值或项目已有代码中的确定值，就必须标为待校准。

## 新手引导模式

首轮提示应像下面这样，不要直接抛出完整参数表：

```text
我们先不用填复杂参数，我会一步步问。
第一步只需要你告诉我三件事：
1. 这是什么硬件？比如厂商、型号或截图里的名称。
2. 你手头有什么资料？比如官网规格、截图、表格、口头描述。
3. 你希望先按“单卡”建 profile，还是硬件里有多个 die/chiplet 需要拆开？如果不确定，直接说“不确定”。

你可以直接粘贴原始描述或截图里的文字，我会帮你翻译成设备画像需要的字段。
```

后续按阶段推进：

1. 基本身份：厂商、型号、profile 名称。
2. 建模粒度：单卡、单 die、单 chiplet 或其他调度单元。
3. 内存信息：显存容量和内存带宽。
4. 算力信息：先问用户资料里有哪些 FP32、FP16、BF16、INT8、FP8 等峰值，再映射到 `mma_ops`。
5. 通用算力：解释 `gp_ops`，优先找官方 vector/general 指标；没有时提供“留空”或“临时估值”选项。
6. 通信拓扑：从“几张卡、卡之间怎么连”开始问，再转换为 `CommGrid`。
7. 最终确认：展示将写入的字段、默认值和 `needs calibration`。

如果用户表现出不熟悉硬件术语，应优先给例子，而不是增加术语密度。

## 术语速查

向新手解释时使用这些说法：

- `DeviceProfile`：一张“硬件能力卡片”，记录某种硬件的算力、显存、带宽和互联。
- profile 粒度：这张能力卡片代表的最小硬件单元。常见选择是“单卡”或“单 die”。
- die / chiplet：一张卡里面可能有多个计算芯片小块；如果软件能分别调度它们，可能要按 die 建 profile。
- dtype：数据类型，比如 FP32、FP16、BF16、INT8、FP8。
- `mma_ops`：矩阵/张量计算峰值，主要影响矩阵乘、Linear、Attention 这类大算子。资料里常写成 FP16/BF16 TFLOPS、TOPS、Tensor Core、AI Core 算力。
- `gp_ops`：通用/vector 计算峰值，主要影响 softmax、norm、激活函数、逐元素计算等非矩阵算子。很多资料不会单独给，缺失时可以先留空或用待校准估值。
- 内存带宽：HBM/显存每秒能读写多少数据，常见单位 GB/s 或 TB/s。
- 互联带宽：多卡或多 die 之间通信的带宽，必须确认是单向还是双向。
- `needs calibration`：为了先跑通而暂时使用的默认值、估值或不确定项，后续需要用实测或官方资料校准。

## 需要收集的必要事实

以下是助手内部检查清单，不要原样丢给用户填写。对新手应拆成多轮问题，并配合上面的术语解释：

1. `profile_name`：稳定的大写名称，例如 `ATLAS_800_A3_752T_128G_DIE`。
2. `vendor`：硬件厂商。
3. `sku`：硬件型号或规格名，用于生成清晰的 profile 名称。
4. `granularity`：该 profile 代表单卡、单 die、单 chiplet，还是其他调度单元。
5. `source_scope`：用户给出的显存、带宽和算力是每个 profile 单元、整卡、整机、整节点还是整集群的值。
6. `memory_size`：每个 profile 单元的 HBM/设备内存容量。
7. `memory_bandwidth`：每个 profile 单元的内存带宽。
8. `mma_ops`：按 dtype 区分的 tensor/matrix 峰值吞吐。
9. `gp_ops`：按 dtype 区分的 general/vector 峰值吞吐。
10. `comm_grid`：拓扑形状，以及每一层互联的带宽、延迟、效率和类型。

如果缺少某项事实，提出有针对性的追问。如果用户无法提供，则只能在用户接受后使用兜底默认值或估值，并标记为 `needs calibration`。

## 首轮用户提示

首轮目标是降低门槛，不是收齐参数。如果用户使用中文，则用中文询问：

```text
我们先不用填复杂参数，我会一步步带你配。
你先给我任意一种信息即可：硬件型号、官网规格截图里的文字、表格、或你知道的口头描述。

为了开始，我只问 3 个小问题：
1. 这是什么硬件？例如厂商和型号。
2. 你想先按一整张卡建 profile，还是这张卡里有多个 die/chiplet 需要拆开？不知道也没关系。
3. 资料里有没有写“显存容量”“显存带宽”或“FP16/BF16/INT8 算力”？有的话原样贴出来即可。
```

然后解析用户提供的信息，只追问缺失或存在歧义的必要事实。后续追问要解释为什么需要该信息，以及用户可以如何回答。

## 追问优先级

写文件前必须优先确认：

1. 所有容量、带宽和峰值是否都是“每个 profile 单元”的值，而不是整卡、整机或整节点总量。
2. profile 粒度是单卡、单 die、单 chiplet 还是其他调度单元。
3. `mma_ops` 与 `gp_ops` 是否来自不同官方指标；不得默认二者相同。
4. 公开互联带宽是单向还是双向。
5. 互联类型更接近 `InterconnectType.FULL_MESH` 还是 `InterconnectType.CLOS`。
6. 所有估值是否获得用户确认。

如果用户只想快速跑通，优先追问会导致代码错误或数量级错误的问题；其余问题可以进入 `needs calibration`。

## 解析规则

显式规范化单位：

- `T`、`TFLOPS`、`TOPS` -> `* 1e12`
- `P`、`PFLOPS`、`POPS` -> `* 1e15`
- `GB/s` 十进制带宽 -> `* 1e9`
- `TB/s` 十进制带宽 -> `* 1e12`
- `GiB` -> `* (1024**3)`
- `TiB` -> `* (1024**4)`
- GB 级设备内存容量规格默认按 GiB 处理，除非用户明确说明使用十进制 GB；该歧义需要列入 `needs calibration`
- `us` / `microsecond` 延迟 -> `* 1e-6`
- `ns` / `nanosecond` 延迟 -> `* 1e-9`

映射 dtype 名称：

- `FP32`、`float32` -> `torch.float32`
- `FP16`、`half`、`float16` -> `torch.half`
- `BF16`、`bfloat16` -> `torch.bfloat16`
- `INT8` -> `torch.int8`
- `FP8` -> `DTYPE_FP8`
- `FP4`、`MXFP4` -> 仅当用户确认这对应当前 FP4 建模路径时，映射为 `DTYPE_FP4`

不要编造 dtype 吞吐。如果某种关系看起来合理，应请用户确认，例如：`INT8 峰值是否等于 FP16 的 2 倍？`

如果用户给出的是“训练/推理峰值”“稀疏/稠密峰值”“Boost/Typical 峰值”或带条件的峰值，保留条件描述，并确认写入 profile 的值应该使用哪一个。

## `mma_ops` 和 `gp_ops` 输入引导

`mma_ops` 和 `gp_ops` 直接让用户填字段很容易卡住。改为以下流程：

### 第一步：从资料提取峰值数字

直接问用户：“你的资料里有没有写 FP16/BF16/INT8 这类算力？直接贴过来就行，我来帮你对应到设备画像字段。”

支持的关键词映射（用户不需要知道这些术语）：

| 用户资料里可能的写法 | 对应设备画像字段 |
|---|---|
| FP16 TFLOPS、FP16 算力、Tensor Core 算力 | `mma_ops[torch.half]` |
| BF16 TFLOPS、BF16 算力、AI Core 算力 | `mma_ops[torch.bfloat16]` |
| INT8 TOPS、INT8 算力 | `mma_ops[torch.int8]` |
| FP32 矩阵/张量 TFLOPS | `mma_ops[torch.float32]` |
| FP8 算力、MXFP8 | `mma_ops[DTYPE_FP8]` |

用户只需贴文字，例如：`BF16 500 TFLOPS`、`FP16 800T`，不必写成 Python 字典格式。

### 第二步：解释 `gp_ops`（通用算力）

如果资料里只有 `mma_ops` 相关的峰值，问用户：“资料里有没有单独给 vector/通用计算的峰值？比如 FP32 向量算力。很多硬件不会单独列这一项。”

如果用户有：按官方值填写 `gp_ops`，例如 FP32 向量算力对应 `gp_ops[torch.float32]`。
如果用户没有，提供以下选项，不要让用户空着不知所措：

- **留空**：`gp_ops={}` — 影响 softmax、norm、激活等逐元素算子的估算准确性，适合只想跑通整体流程的场景。
- **临时估值**：按 `mma_ops` 的某个比例估算，例如 `gp_ops[BF16] ≈ mma_ops[BF16] / 10`。必须明确标注 `needs calibration`，并说明这是临时假设。
- **留空但加入校准清单**：明确告诉用户缺少 `gp_ops` 会导致哪些算子无法被正确估算，建议后续补充。

### 第三步：dtype 倍率关系

如果用户只给了 BF16 峰值，问：`INT8 峰值是否等于 BF16 的 2 倍？FP8 是否等于 BF16 的 2 倍？`
如果用户无法确认，分别用“待确认”和“待确认（2x BF16）”标注，列入 `needs calibration`。

### 第四步：写入格式

将用户提供的峰值归一化后，用以下格式写入代码（不要暴露给用户看内部格式）：

```python
mma_ops={
    torch.float32: <FP32 TFLOPS> * 1e12,
    torch.bfloat16: <BF16 TFLOPS> * 1e12,
    torch.half: <FP16 TFLOPS> * 1e12,
    torch.int8: <INT8 TOPS> * 1e12,
    DTYPE_FP8: <FP8 TOPS> * 1e12,
}
gp_ops={
    torch.float32: <FP32 向量 TFLOPS> * 1e12,
    torch.bfloat16: <BF16 向量 TFLOPS> * 1e12,
    torch.half: <FP16 向量 TFLOPS> * 1e12,
}
```

## 兜底默认值

仅当用户无法提供校准值且接受先跑通时使用：

- `compute_efficiency=0.7`
- `memory_efficiency=0.6`
- `comm_efficiency=0.7`
- `StaticCost(mma_op_cost_s=5 * 1e-6, gp_op_cost_s=2 * 1e-6, comm_op_cost_s=10 * 1e-6)`

所有兜底值都必须在最终回复的 `needs calibration` 下列出。

不要对以下核心硬件事实静默使用兜底：`granularity`、`source_scope`、每 profile 单元的显存、每 profile 单元的内存带宽、已写入 dtype 的吞吐、互联带宽方向。缺失这些事实时，应追问或明确得到用户同意后才使用估值。

## 多 die / 多 chiplet 规则

当用户描述“整卡包含多个 die/chiplet”或提供的是整卡总规格时，先追问：

1. profile 粒度希望选整卡、单 die、单 chiplet，还是调度可见的其他单元？
2. 如果选单 die 或单 chiplet，整卡显存、带宽和算力是否可以按数量均分？
3. die 内、卡内、节点内、节点间分别有多少单元、带宽、延迟、效率和互联类型？
4. 软件调度是否能独立选择 die/chiplet；如果不能，整卡 profile 可能更合适。

不要静默地把整卡规格除以 die 数。只有用户确认“可以均分”或数据手册明确给出 per-die 规格时，才能写入 per-die profile。

## 通信拓扑访谈

先用硬件术语询问：

1. grid 应覆盖多少个 profile 单元？
2. 从最慢到最快有哪些拓扑层级？
3. 每一层的组大小、单向带宽、延迟、效率和类型分别是什么？
4. 公开带宽是单向还是双向？
5. 如果只知道一部分层级，是否先只建模已知范围？

将其转换为 `CommGrid`，外层维度表示更慢的层级，内层维度表示更快的层级：

```python
CommGrid(
    grid=torch.arange(<total_units>).reshape(<slow>, <middle>, <fast>),
    topologies={
        0: InterconnectTopology(...),
        1: InterconnectTopology(...),
        2: InterconnectTopology(..., type=InterconnectType.FULL_MESH),
    },
)
```

`grid.ndim` 必须等于 `len(topologies)`，并且每个 grid 维度都至少为 2。不要为了满足该约束而凭空增加不存在的拓扑层；未知层级应暂不建模，或让用户确认一个用于模拟的最小拓扑。

对于没有真实多设备拓扑的单卡或单 die profile，询问用户希望模拟的最小拓扑。不要创建单元素 grid，因为 `CommGrid` 会拒绝小于 2 的维度。

## 写入前检查清单

在编辑 `tensor_cast/device.py` 前，确认：

- `profile_name` 唯一，并且不会与 `DeviceProfile.all_device_profiles` 中已有名称冲突。
- 插入位置清晰：优先放入现有厂商类；没有合适类时，在文件末尾新增一个小型厂商/SKU 类。
- 不新建 `tensor_cast/device_profiles/<device_slug>.py`，除非用户明确要求。
- 所有写入的数值都能追溯到用户输入、用户确认的估值或明确的兜底默认值。
- `DTYPE_FP4` / `DTYPE_FP8` 只在实际用到时导入。
- `CommGrid` 的每个维度都至少为 2，且 `topologies` 数量与 `grid.ndim` 一致。
- 已准备好最终命令行示例，并确认 `--device` 后面的值等于 `DeviceProfile.name`。

## 代码生成模式

将生成的 `DeviceProfile` 直接添加到 `tensor_cast/device.py` 中现有厂商类的末尾（例如 `ATLAS_800` 类内），或在文件末尾新建小类来容纳。

```python
# 放在 tensor_cast/device.py 中现有类末尾，或文件末尾新建类
class <CUSTOM_CLASS_NAME>:
    _STATIC_COST = StaticCost(...)

    _INTERCONNECT = CommGrid(...)

    <PROFILE_NAME> = DeviceProfile(
        name="<PROFILE_NAME>",
        vendor="<VENDOR>",
        mma_ops={...},
        gp_ops={...},
        memory_size_bytes=<...>,
        memory_bandwidth_bytes_ps=<...>,
        compute_efficiency=<...>,
        memory_efficiency=<...>,
        comm_grid=_INTERCONNECT,
        static_cost=_STATIC_COST,
    )
```

插入时注意：

- `profile_name` 已在 `DeviceProfile.all_device_profiles` 中注册为唯一，不要覆盖已有名称。
- 如果在现有类（如 `ATLAS_800`）下添加，注意类已有的 `_STATIC_COST` 和互联常量是否可复用。
- 如果新建类，命名风格与现有类保持一致，例如 `<VENDOR>_<SKU>`。
- 仅在实际使用时导入 `DTYPE_FP4` 或 `DTYPE_FP8`。

## 验证

运行导入/注册检查，确认 `tensor_cast/device.py` 中新增的 profile 已在模块导入时注册：

```bash
python - <<'PY'
from tensor_cast.device import DeviceProfile
print(DeviceProfile.all_device_profiles["<PROFILE_NAME>"])
PY
```

如果本地环境使用特定 Python 可执行文件，则遵循项目惯例。

验证后还要生成一条可执行命令，确保命令中的 `--device` 值就是新注册的 `DeviceProfile.name`：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device <PROFILE_NAME>
```

如果导入失败，先修复 `tensor_cast/device.py` 中新增 profile 的问题；不要通过跳过注册或修改全局注册逻辑来掩盖问题。

## 命令行生成规则

最终必须给出一条可直接复制执行的命令。不要保留 `TEST_DEVICE`、`TESE_DEVICE` 或 `<PROFILE_NAME>` 这类占位符；必须替换成真实的 `DeviceProfile.name`。

默认命令模板：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device <PROFILE_NAME>
```

生成时遵循：

- 如果用户指定了模型、query 数量、query 长度或其他参数，使用用户提供的值。
- 如果用户没有指定，使用上面的默认模板。
- `--device` 后必须是刚注册成功的 profile 名称，例如 `--device FOO_X200_800T_96G`。
- 如果验证失败，不要给出“已可运行”的命令，只给出修复建议。

## 示例模拟：新手引导流程

用户输入：

```text
我们有个 Foo X200 单卡，显存 96GB，带宽 3.2TB/s。BF16 是 800T，FP16 是 800T，INT8 是 1600T，FP32 是 100T。8 卡机器，卡间 400GB/s。
```

期望的新手引导流程：

**第一轮（助手）：** 先复述已理解的信息，并提出下一批问题：

```text
好的，我理解了：
- 厂商/型号：Foo X200，粒度先按单卡。
- 显存：96GB，带宽 3.2TB/s。

接下来还需要几件事：

1. 【profile 名称】你想给这张卡起什么名字？例如 FOO_X200_800T_96G。我先按这个生成。

2. 【算力确认】资料里 BF16、FP16、INT8、FP32 的峰值都给了，很完整。你的资料里有没有单独给 vector（通用计算）算力？很多硬件不会单独列这一项。
   - 如果有，直接告诉我数字。
   - 如果没有，可以选：
     a) 先留空，后续我来校准。
     b) 先用 BF16 的 1/10 作为临时估值（我会标成待校准）。

3. 【卡间互联】8 卡之间 400GB/s 是单向还是双向？互联方式是全互联还是 CLOS？不知道的话可以直接说“我不确定，用默认的就行”。
```

**第二轮（用户）：** 用户只回复第 2 和第 3 项：`BF16 没有单独的 vector 算力，用留空吧。带宽我不确定单向还是双向`。

**第三轮（助手）：** 展示将写入 `tensor_cast/device.py` 的 profile、插入位置和 `needs calibration`，让用户确认后再写入；写入并验证成功后输出带真实 `--device FOO_X200_800T_96G` 的命令行。

## 示例模拟：信息较完整

用户输入：

```text
我们有一个 Foo X200 单卡，感觉可以先按单卡建 profile。显存 96GB，带宽 3.2TB/s。BF16/FP16 是 800T，FP32 是 100T，INT8 是 1600T。8 卡机器，卡间 400GB/s，延迟先按 0.3us。节点间还不确定。
```

期望的助手行为：

1. 提取直接事实：厂商/SKU 候选为 `Foo X200`，粒度可能是单卡，内存为 96 GiB，内存带宽为 3.2 TB/s，`mma_ops` 包含 FP32/BF16/FP16/INT8，以及一个 8 卡拓扑候选。
2. 写入前提出有针对性的追问：
   - `profile_name` 是否使用 `FOO_X200_800T_96G`？
   - `vendor` 是否写成 `FOO`？
   - 96GB 是否按 96 GiB 写入？
   - `gp_ops` 是否有官方 vector/general compute 峰值？如果没有，是否先用 FP32 10T、BF16/FP16 20T 作为待校准估值？
   - 卡间 400GB/s 是单向还是双向？互联类型更接近 FULL_MESH 还是 CLOS？
3. 如果用户无法提供 `gp_ops`，不要静默编造。可以在用户接受的情况下只保留已知 dtype，或使用清晰说明的临时值。
4. 如果节点级拓扑未知，只创建用户已描述的 8 卡拓扑，因为 `CommGrid` 可以先建模已支持的范围。

## 示例模拟：多 die 且规格范围不明

用户输入：

```text
Bar Z9 一张卡 2 个 die，总算力 BF16 1200T，显存 192GB，卡内互联 900GB/s。我们调度时看起来像两个设备。
```

期望的助手行为：

1. 不要直接把 1200T、192GB、900GB/s 写入单 die profile。
2. 先追问：profile 是否按单 die 建？如果按单 die，算力和显存是否可均分为每 die 600T、96GB？卡内 900GB/s 是 die 间单向还是双向？
3. 如果用户确认软件调度看到两个 die，且整卡规格可均分，再生成类似 `BAR_Z9_600T_96G_DIE` 的 per-die profile。
4. 如果用户不能确认均分，则建议先建整卡 profile，或把 per-die 数值列为 `needs calibration`。

## 示例模拟：只想快速跑通

用户输入：

```text
只有截图：Baz NPU，FP16 500T，64G 显存。互联和 gp_ops 都没有，先让我能跑模型估算。
```

期望的助手行为：

1. 追问最少阻塞项：profile 粒度、内存带宽是否知道、是否接受用默认效率和 static cost。
2. 明确说明 `gp_ops`、内存带宽和互联拓扑缺失会影响估算准确性。
3. 如果用户同意快速跑通，使用用户确认的临时值或项目兜底默认值，并把每一项列入 `needs calibration`。
4. 不要声称该 profile 已具备性能准确性，只能说明它可导入、可用于粗略估算。

## 最终回复格式

简洁报告：

- 修改的文件（通常是 `tensor_cast/device.py`）和插入位置。
- 已注册的 `DeviceProfile.name`，如果验证失败则不要声称已注册。
- 直接使用的用户提供事实。
- 用户确认的估值。
- `needs calibration`：所有兜底值、假设、遗漏 dtype、近似带宽/延迟或单位歧义。
- 验证命令和结果。
- **可执行命令行示例**：把新 profile 名称代入 `--device` 参数，最终回复必须使用真实名称，例如：

```bash
python -m cli.inference.text_generate Qwen/Qwen3-32B --num-queries 2 --query-length 3500 --device FOO_X200_800T_96G
```

如果项目还有其他常用命令（如 `tensor_cast` 内省、`tensor_cast simulate` 等），也可以一并给出。

- 下一条最重要的校准问题，如有。

## 安全规则

- 新增前检查 `DeviceProfile.all_device_profiles` 中是否已有重复名称。
- 不要覆盖、改名或删除已有内置 profile；只追加新增 profile，除非用户明确要求修改已有项。
- 不要为用户未提供或未确认的 dtype 添加吞吐。
- 不要静默地把整卡规格转换为按 die 规格；需要询问粒度和是否可均分。
- 未经确认，不要把双向带宽当作单向带宽。
- 不要为了满足 `CommGrid` 约束而凭空创造拓扑层级或设备数量。
- 不要默认新建 `tensor_cast/device_profiles/` 文件；本 skill 默认追加到 `tensor_cast/device.py`。

## 完成标准

- profile 可以成功导入。
- `DeviceProfile.all_device_profiles` 包含新的 profile 名称。
- 用户可以把该 profile 名称传给设备选择逻辑。
- 最终回复包含已替换真实 profile 名称的可执行命令行示例。
- 所有默认值、估值、推断值和不确定性都对用户可见。
