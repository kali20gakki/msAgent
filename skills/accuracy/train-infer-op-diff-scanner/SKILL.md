---
name: train-infer-op-diff-scanner
description: RL 强化学习训练与推理（rollout）路径的算子差异性扫描。当用户提供训练启动脚本路径，要求对比 Megatron 训练路径与 vLLM 推理路径的算子差异（尤其是融合算子 vs 单算子不一致）时使用此 skill。典型触发词包括"训推算子扫描"、"训推差异性"、"算子差异报告"、"融合算子对比"、"train vs infer op diff"。注意：本 skill 通过运行完整 RL 训练脚本（集成 profiling）来采集真实运行时算子（非静态源码扫描，非独立分离运行）。当训练、推理使用不同脚本时禁止触发。
---

# Train-Infer Operator Diff Scanner（RL 训推算子一致性扫描 ）

面向 Ascend NPU 的 RL 训推算子差异性**运行时扫描**技能。

## 核心原则

1. **必须运行完整 RL 脚本**：算子采集必须通过运行完整 RL 训练脚本（集成 verl profiler）完成。**禁止**编写独立脚本分离运行 Megatron 训练和 vLLM 推理来分别采集算子数据
2. **运行时采集（非静态扫描）**：所有算子信息必须来自真实 NPU 运行时 profiling，静态源码扫描结果不作为最终结论
3. **profiler 集成到 RL 脚本**：通过 `global_profiler` + 各组件 profiler 配置在 RL 脚本中开启 profiling（训练路径用 e2e profiler，推理路径用 `discrete=True` 分离采集）
4. **代码栈贴入报告**：每个差异算子需附对应的代码调用栈（从 profiler DB 的 MSTX/CPU callstack 提取或从源码路径标注）

## 触发条件

- 对比 RL 训练（Megatron）和推理（vLLM）的算子差异
- 扫描训练和推理路径中的融合算子差异
- 生成训推算子一致性报告
- 用户提供了 RL 训练启动脚本路径

## 不适用场景

- 运行时 dump 数据的逐值对比（应使用 `rl-consistency-dump-analysis` skill）
- 纯性能 profiling 分析（应使用 msprof-analyze-cli skill）
- 无法执行脚本或 NPU 不可用的环境

---

## 工作流程（5 阶段）

### 阶段 1：环境与配置解析（预计 2-3 分钟）

#### 1.1 解析训练脚本

**必须先读取用户提供的 RL 脚本**，提取以下配置项：

| 解析项 | 脚本字段 | 示例 |
|--------|---------|-----|
| 训练后端 | `actor_rollout_ref.actor.strategy` | `megatron` |
| 推理引擎 | `actor_rollout_ref.rollout.name` | `vllm` |
| 模型路径 | `MODEL_PATH` 或 `actor_rollout_ref.model.path` | `/workspace/models/Qwen3-0.6B` |
| 模型 ID | `MODEL_ID` | `Qwen/Qwen3-0.6B` |
| TP/PP/EP | `tensor_model_parallel_size` 等 | `1` |
| max_prompt_length | `data.max_prompt_length` | `512` |
| max_response_length | `data.max_response_length` | `1` |
| train_batch_size | `data.train_batch_size` | `4` |
| total_training_steps | `trainer.total_training_steps` | `1` |
| NPU 设备 | `ASCEND_RT_VISIBLE_DEVICES` | `1` |
| rollout 配置 | `enforce_eager`, `calculate_log_probs`, `n` | `True`, `True`, `2` |

#### 1.2 环境检查（必须全部通过，失败则告知用户并停止）

```bash
python -c "import torch; import torch_npu; print(torch.npu.device_count(), 'NPUs available')"
which msprof
pip show megatron-core vllm vllm-ascend mbridge 2>/dev/null | grep -E "^(Name|Version):"
ls -la ${MODEL_PATH}/config.json
```

#### 1.3 备份原始脚本

```bash
cp <script_path> <script_path>.bak_$(date +%Y%m%d_%H%M)
```

---

### 阶段 2：集成 Profiler 到 RL 脚本（预计 2-3 分钟）

**目的**：修改 RL 训练脚本，嵌入 profiling 配置，使单次运行即可同时采集训练和推理路径的 NPU 算子。

#### 2.1 检查脚本是否已含 profiler 配置

搜索脚本中是否存在 `global_profiler` 或 `PROFILER` 相关配置块。若已存在则跳过 2.2。

#### 2.2 在脚本中添加 Profiler 配置块

在脚本中 `launch` 命令之前插入以下配置,（参考 `references/run_qwen3_0_6b_megatron_vllm_ascend.sh` 的实际实现）：

```bash
PROFILER_OUTPUT_DIR=${PROFILER_OUTPUT_DIR:-<脚本目录>/profiler_output}
rm -rf ${PROFILER_OUTPUT_DIR} 2>/dev/null || true
mkdir -p ${PROFILER_OUTPUT_DIR}

# 关键：训练路径用 e2e profiler，推理路径用 discrete=True 分离采集
PROFILER=(
    global_profiler.tool=npu
    global_profiler.steps=[1]
    global_profiler.save_path=${PROFILER_OUTPUT_DIR}
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.tool_config.npu.contents="['npu','cpu','shapes']"
    actor_rollout_ref.actor.profiler.tool_config.npu.level=level1
    actor_rollout_ref.ref.profiler.enable=True
    actor_rollout_ref.ref.profiler.tool_config.npu.contents="['npu','cpu','shapes']"
    actor_rollout_ref.ref.profiler.tool_config.npu.level=level1
    actor_rollout_ref.rollout.profiler.enable=True
    actor_rollout_ref.rollout.profiler.tool=npu
    actor_rollout_ref.rollout.profiler.tool_config.npu.contents="['npu','cpu','shapes']"
    actor_rollout_ref.rollout.profiler.tool_config.npu.level=level1
    actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=True
)
```

在 launch 命令参数末尾追加 `"${PROFILER[@]}"`。

#### 2.3 配置说明

| 配置项 | 说明 |
|--------|------|
| `global_profiler.tool=npu` | 全局开启 NPU profiling |
| `global_profiler.steps=[1]` | 仅采集 step 1（节省时间） |
| `actor.profiler.enable=True` | 训练 actor 路径采集 |
| `ref.profiler.enable=True` | 训练 ref 路径采集 |
| `rollout.profiler.enable=True` | 推理 rollout 路径采集 |
| `rollout.profiler.tool_config.npu.discrete=True` | **关键**：触发 vLLM 侧独立 `torch_npu.profiler` 采集，产出 raw `*_ascend_pt` 目录（**非 DB**），须事后 `analyse()` 才落 DB（见阶段 3.3） |
| `level=level1` | 采集 CANN 算子级别数据 |

#### 2.4 输出路径约定（训练/推理采集后端不同，产物形态不同）

运行完成后，profiler 输出目录结构如下。**关键：训练与推理两条路径的 profiler 后端不同，产物形态也不同**：

```
profiler_output/
├── e2e/                                              # 训练路径（Megatron actor+ref）
│   └── <hostname>_<pid>_<ts>_ascend_pt/
│       └── ASCEND_PROFILER_OUTPUT/
│           └── ascend_pytorch_profiler_0.db          # ← 训练算子（msprof，已自动解析）
└── agent_loop_rollout_replica_0/                     # 推理路径（vLLM rollout）
    └── <hostname>_<pid>_<ts>_ascend_pt/              # ← vllm-ascend torch_npu profiler 原始目录
        └── （需执行 analyse() 离线解析后才生成 ASCEND_PROFILER_OUTPUT/*.db）
```

**机制差异（决定后续流程）**：

| 路径 | profiler 后端 | export_type | 是否自动解析 | 产物 |
|------|--------------|-------------|:---:|------|
| 训练（e2e） | CANN msprof（verl） | Db | ✅ 自动 | 直接生成 `ascend_pytorch_profiler_0.db` |
| 推理（discrete=True） | `torch_npu.profiler`（vllm-ascend） | Text | ❌ 需手动 | 只落 `*_ascend_pt` 目录，须 `analyse()` 后才有 DB |

> 推理侧的 `discrete=True` 由 vLLM 引擎触发 profiler，走 `vllm_ascend/worker/worker.py` 的 `_init_profiler()`（`export_type=ExportType.Text`），不会像训练侧那样自动解析出 DB。

---

### 阶段 3：运行 RL 脚本采集算子数据（预计 5-15 分钟）

#### 3.1 运行完整 RL 训练脚本

```bash
cd <脚本所在目录>
bash <脚本名> 2>&1 | tee profiler_output/run.log
```

**注意**：必须等待脚本完整执行完毕。此脚本同时运行 Megatron 训练和 vLLM rollout，运行时间取决于模型大小（0.6B 模型约 5-10 分钟）。

#### 3.2 验证采集产物

**训练路径**（已自动解析为 DB）：
```bash
find profiler_output/e2e -name "ascend_pytorch_profiler_0.db" -type f
```
确认存在 `profiler_output/e2e/*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db`。

**推理路径**（原始目录，尚未解析）：
```bash
find profiler_output/agent_loop_rollout_replica_0 -maxdepth 2 -type d -name "*_ascend_pt"
```
确认存在 `profiler_output/agent_loop_rollout_replica_0/<hostname>_<pid>_<ts>_ascend_pt/` 目录（这是 vllm-ascend `torch_npu.profiler` 的原始输出，**不是 DB**，目录内含 `FRAMEWORK/torch.op_range` 与 `PROF_*/host/data`）。

#### 3.3 对推理侧 ascend_pt 目录执行离线解析（关键步骤，不可跳过）

推理侧 profiler 走 vLLM-Ascend 的 `torch_npu.profiler`（`export_type=Text`），**默认不自动解析**。必须手动调用 `analyse()` 生成 DB，否则推理侧拿不到 device 算子：

```bash
INFER_PT_DIR=$(find profiler_output/agent_loop_rollout_replica_0 -maxdepth 2 -type d -name "*_ascend_pt" | head -1)
python3 -c "
from torch_npu.profiler.profiler import analyse
analyse('${INFER_PT_DIR}')
"
```

解析完成后，推理侧会生成 `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db`（含 `COMPUTE_TASK_INFO` 表 = device 侧 AI Core 真实执行算子），同时 text 模式还会额外产出 `api_statistic.csv` / `operator_details.csv` / `trace_view.json`（host 侧备选）。验证 DB 已生成：

```bash
find profiler_output/agent_loop_rollout_replica_0 -name "ascend_pytorch_profiler_0.db" -type f
```

> **机制说明（关键）**：`analyse()` 不带 `export_type` 时，会从 `profiler_info_*.json` 读取采集时的 `_export_type`。由于 vllm-ascend 采集时是 `["text"]`，本应只走 text 解析；但 **CANN 8.5.0 的 msprof 支持 `text(which will also export the database)` 默认行为**（`torch_npu` 内部 `CannPackageManager.is_support_default_export_db()` 检测），因此会自动补上 db parser，从而同时产出 DB 与 CSV。**若某 CANN 版本不支持该默认行为**（`msprof --help` 输出无 `text(which will also export the database)`），则须显式传 `export_type=['db']`：
>
> ```bash
> python3 -c "
> from torch_npu.profiler.profiler import analyse
> analyse('${INFER_PT_DIR}', export_type=['db'])
> "
> ```
>
> **告警说明**：analyse 过程中出现的 `Failed to get acl to npu flow events`、`Failed to get task data from db`、`no such table: TASK` 等告警是 `export_type=Text` 模式的正常现象（推理侧 msprof 格式的 device 二进制 `device_*/data/` 为空），**不影响**最终 `COMPUTE_TASK_INFO` 中 device 算子的解析——真实算子来自 host 侧 CANN api_event 记录（`CANN_API` 表），analyse 会将其还原为 device 侧算子。

#### 3.4 验证推理侧 device 算子已落地（证据闭环）

解析完成后，必须用 SQL 确认 `COMPUTE_TASK_INFO` 表非空（该表 = device 侧 AI Core 真实执行算子），**不能只确认 DB 文件存在**：

```bash
python3 -c "
import sqlite3, glob
dbs = glob.glob('profiler_output/agent_loop_rollout_replica_0/*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db')
assert dbs, '推理侧 DB 缺失'
c = sqlite3.connect(dbs[0])
n = c.execute('SELECT COUNT(*) FROM COMPUTE_TASK_INFO').fetchone()[0]
print('COMPUTE_TASK_INFO rows =', n)
"
```

- ✅ `n > 0`：推理侧 device 算子已落地，进入阶段 4。
- ❌ DB 缺失或 `n == 0`：按以下顺序兜底：

| 兜底 | 触发条件 | 操作 |
|------|---------|------|
| 1. 显式导出 DB | `analyse()` 后无 `ascend_pytorch_profiler_0.db`（CANN 不支持默认 `text` 同时导 db） | `analyse('${INFER_PT_DIR}', export_type=['db'])` 后重新验证 |
| 2. host 侧 CSV 备选 | DB 存在但 `COMPUTE_TASK_INFO` 为空 | 改用 `ASCEND_PROFILER_OUTPUT/api_statistic.csv`（或 `operator_details.csv`）中的 CANN 算子名统计，报告中标注「host 侧备选，非 device 侧」 |
| 3. 终止扫描 | 1、2 兜底操作失败 | 跳过阶段 4、5，直接输出报告，记录已进行的操作与结果 |


### 阶段 4：算子提取与对比分析（预计 3-5 分钟）

#### 4.1 从 Profiler DB 提取算子列表

使用 `ascend_pytorch_profiler_db_explorer` skill 或直接 SQL 查询 DB。

**重要**：`COMPUTE_TASK_INFO` 表中算子名字段是 `name`（INTEGER 外键），真实算子名字符串在 `STRING_IDS` 字典表中，必须 JOIN 查询，不能直接查 `task_name`（该列不存在）。

**训练路径算子提取**（从 `profiler_output/e2e/*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db`）：

```sql
-- 提取训练路径所有唯一 CANN 算子名及调用次数（device 侧真实执行算子）
SELECT s.value AS task_name, COUNT(*) AS cnt
FROM COMPUTE_TASK_INFO t
JOIN STRING_IDS s ON t.name = s.id
GROUP BY s.value
ORDER BY cnt DESC;
```

**推理路径算子提取**（从 `profiler_output/agent_loop_rollout_replica_0/*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db`，即 3.3 解析后产物）：

```sql
-- 同上，提取推理路径所有唯一 CANN 算子（device 侧真实执行算子）
SELECT s.value AS task_name, COUNT(*) AS cnt
FROM COMPUTE_TASK_INFO t
JOIN STRING_IDS s ON t.name = s.id
GROUP BY s.value
ORDER BY cnt DESC;
```

将结果分别保存为：
- `profiler_output/megatron_runtime_ops.json`
- `profiler_output/vllm_runtime_ops.json`

#### 4.2 构建算子全量对比表

按 Transformer 层结构逐模块对比，生成标准对比表：

| 维度 | 训练路径 (Megatron) | 推理路径 (vLLM) | 差异类型 | 差异等级 |
|------|---------------------|-----------------|----------|:--------:|
| 注意力 | 列出运行时算子（如 `FlashAttentionScore`） | 列出运行时算子（如 `FusedInferAttentionScore`） | 算子不同 / 融合vs非融合 | 🔴 |
| 残差+归一化 | `Add` + `RmsNorm` | `AddRmsNormBias` | 融合程度不同 | 🟠 |
| 激活函数 | `Swish`（独立 SiLU） | `SwiGlu`（SiLU+Gate 融合） | 融合程度不同 | 🟠 |
| RoPE | `Sin` + `Cos`（独立） | `_triton_rope`（融合） | 后端实现不同 | 🟠 |
| MatMul（线性层） | `MatMulV2` | `MatMulV2` | 相同 | 🟢 |
| KV Cache | 无 | `ReshapeAndCacheNdKernel` | 推理独有 | 🟡 |
| 反向传播 | `RmsNormGrad`, `FlashAttentionScoreGrad` 等 | 无 | 训练独有 | 🟡 |

#### 4.3 差异等级判定规则

| 标记 | 颜色 | 判定条件 | 精度风险 |
|:----:|:----:|---------|:--------:|
| 🔴 | 红 | 一方使用融合 kernel，另一方使用多个独立 kernel（如 Attention：分离 MatMul+Softmax+MatMul vs 单 kernel FusedInferAttentionScore） | 高 |
| 🟠 | 橙 | 同类操作但融合程度不同（如 Add+RMSNorm vs AddRmsNormBias），或不同后端实现同一数学运算（如 NPU RoPE vs Triton RoPE） | 中 |
| 🟡 | 黄 | 实现方式不同但数学语义等价，或仅单侧存在且另一侧有等价替代 | 低 |
| 🟢 | 绿 | 相同基本算子，或训练/推理单侧独占且无等价替代 | 极低 |

#### 4.4 提取代码调用栈（关键）

对每个 🔴 和 🟠 差异算子，从以下源码路径标注代码调用栈：

| 算子类别 | 训练路径关键源码 | 推理路径关键源码 |
|---------|----------------|----------------|
| Attention | `megatron/core/transformer/dot_product_attention.py` → `FlashAttentionScore` | `vllm_ascend/attention/` → `FusedInferAttentionScore` |
| RMSNorm | `megatron/core/transformer/torch_norm.py` | `vllm_ascend/layers/layernorm.py` → `AscendRMSNorm` |
| SwiGLU | `megatron/core/transformer/mlp.py` | `vllm_ascend/layers/activation.py` → `AscendSiluAndMul` |
| RoPE | `megatron/core/transformer/attention.py` → `Sin`/`Cos` | `vllm_ascend/layers/rotary_embedding.py` → `_triton_rope` |

**提取方式**（优先使用 MSTX 标记）：
1. 查询 DB 中 `MSTX` 表获取算子与 Python 调用栈的对应关系
2. 若 MSTX 不可用，则标注算子对应的源码文件路径和关键函数名
3. 在报告中以 call stack 形式展示：`上层调用路径 → 算子类/函数 → CANN 算子名`

---

### 阶段 5：生成最终产物（预计 3-5 分钟）

必须生成以下 **3 个文件**：

#### 产物 1：完整报告 (Markdown) → `<工作目录>/train_infer_op_diff_report.md`

**必须包含**以下章节（参考 `references/report_template.md`）：

1. **标题 + 元信息**（模型名、配置、脚本路径、采集方式、生成时间）
2. **算子证据来源**（表格：路径、数据源、DB 文件、算子数）
3. **训练路径算子全量列表**（运行时采集的 CANN 算子 TOP30+，含调用次数和用途分类）
4. **推理路径算子全量列表**（同上）
5. **核心差异：训练 vs 推理算子对比表**（含差异类型和 🔴🟠🟡🟢 等级标记）
6. **代码调用栈**（每个 🔴🟠 差异算子附源码路径和调用链）
7. **结论与建议**（P0-P3 优先级，含具体操作）

格式要求：
- 使用 HTML 颜色标记差异等级：`<span style="color:red">🔴 高差异</span>`、`<span style="color:orange">🟠 中差异</span>`、`<span style="color:gold">🟡 低差异</span>`、`<span style="color:green">🟢 无差异</span>`
- 代码调用栈使用折叠块 `<details><summary>调用栈详情</summary>...</details>`

#### 产物 2：算子对比表 (CSV UTF-8 BOM) → `<工作目录>/operator_diff_table.csv`

```csv
操作,训练算子,推理算子,训练调用次数,推理调用次数,差异类型,差异等级,训练代码路径,推理代码路径
注意力,FlashAttentionScore,FusedInferAttentionScore,672,224,算子不同,🔴,megatron/core/transformer/dot_product_attention.py,vllm_ascend/attention/ascend_attention.py
残差+归一化,Add(6484)+RmsNorm(2712),AddRmsNormBias(448),6484+2712,448,融合vs非融合,🟠,megatron/core/transformer/transformer_layer.py,vllm_ascend/layers/layernorm.py
```

**保存格式**：文件开头写入 UTF-8 BOM `\xef\xbb\xbf`


---

## 执行顺序约束

```
阶段 1 (配置解析 + 备份脚本)
  └─→ 阶段 2 (集成 Profiler 到脚本)
        └─→ 阶段 3 (运行 RL 脚本采集算子)  ← 最耗时
              └─→ 阶段 4 (提取算子 + 对比分析)
                    └─→ 阶段 5 (生成报告)
```

- 各阶段严格串行，不可跳过
- 阶段 3 运行期间需等待脚本执行完成
- 每个阶段完成后立即整理中间结果，避免丢失

---

## 时间控制

| 阶段   | 预计耗时 | 说明 |
|------|---------|------|
| 阶段 1 | 2-3 min | 纯文本解析 + 环境检查 |
| 阶段 2 | 2-3 min | 修改脚本（文本操作） |
| 阶段 3 | 5-15 min | 运行 RL 脚本（取决于模型大小） |
| 阶段 4 | 3-5 min | SQL 查询 + 对比分析 |
| 阶段 5 | 3-5 min | 生成报告和 CSV |

---

## 重要约束

1. **绝对禁止独立分离运行**：不允许编写独立脚本分别运行 Megatron 训练和 vLLM 推理来采集算子数据
2. **绝对禁止跳过运行时采集**：不允许仅凭源码静态扫描生成最终对比表
3. **备份优先**：修改脚本前必须先备份
4. **证据闭环**：每个结论必须附 profiler DB 提取的运行时算子证据

## 注意事项

- 脚本运行时间可能较长（0.6B 模型约 5-10 分钟），需在开始前告知用户预计等待时间
- 如果 profiler DB 表结构不同（如 `ge_task_merge` vs `ge_summary`），需先查询 schema
- **推理侧必须先 `analyse()` 离线解析**：推理 profiler 产物是 `*_ascend_pt` 目录（vllm-ascend `torch_npu.profiler`，`export_type=Text`），不解析就没有 DB。`device_*/data` 为空（无 `ffts_profile.data`/`stars_soc.data`）是该模式的正常现象，**不影响** device 算子提取——`analyse()` 会从 host 侧 `CANN_API`（api_event 记录）还原出 `COMPUTE_TASK_INFO`（device 侧算子）；仅在 `COMPUTE_TASK_INFO` 仍为空时才需要走 3.4 的兜底（`export_type=['db']` 或 `api_statistic.csv`）
- CSV 文件必须使用 UTF-8 with BOM 编码（`\xef\xbb\xbf` 文件头）

## References

- `references/report_template.md` — 报告模板
- `references/run_qwen3_0_6b_megatron_vllm_ascend.sh` — Qwen3-0.6B GRPO 训练脚本
