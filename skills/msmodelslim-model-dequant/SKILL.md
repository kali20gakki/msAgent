---
name: msmodelslim-model-dequant
description: 为 msModelSlim 适配流程注入反量化能力。先识别模型权重是否可反量化，再实现反量化脚本并接入 model_adapter。当前仅覆盖 FP8 的 per-block 与 per-channel 两类；若格式不确定或无公开反量化规则，要求用户提供反量化脚本或浮点权重。
---

# msModelSlim 反量化接入 Skill

本 Skill 用于在模型适配前处理 FP8 量化权重，将可识别权重恢复为 bf16/fp16 再进入后续适配与量化流程。

## 适用范围

- 支持：
  - FP8 per-block（块级 scale）
  - FP8 per-channel（通道级 scale）
- 不支持（默认阻塞）：
  - 无法确认反量化规则的自定义量化格式
  - 缺少必要 scale/shape 元数据的量化权重
  - 仅有推理框架专用格式且无训练侧可逆映射说明
  - 非 FP8（例如 INT4/INT8/FP4）在本 Skill 中默认不处理

## 必需输入

- 模型目录（至少含 `config.json`、`model.safetensors.index.json`）
- 目标模型的 `model_adapter.py`

## 硬门禁：先识别“可接受反量化权重”

在编写任何脚本前，必须先判定是否属于可接受类型：

1. 读取 `config.json`，提取可用量化配置字段（如量化类型、group/block 相关参数）
2. 读取 `model.safetensors.index.json` 的 `weight_map`
3. 按键名模式识别：
   - **FP8 per-block 可接受**：存在与权重匹配的块级 scale（常见键后缀如 `xxx.weight_scale_inv` / `xxx.scale`），且可推导块大小 `block_size`
   - **FP8 per-channel 可接受**：存在与权重某一维匹配的通道级 scale（常见形态如 `[out_features]`、`[in_features]` 或可广播到对应通道维）
4. 用 scale 形状对权重形状做一致性检查：
   - per-block：`M` 与 `N` 可按 `block_size` 分块并与 scale 网格对应
   - per-channel：scale 的长度或广播维度与指定 channel axis 一致
5. 若未命中上述模式，标记为 **未知/不接受格式** 并停止实现

> 必须输出判定依据（命中的键模式、样例键名、对应文件）。

## 工作流

### 1) 识别与分流

- `未量化`：直接跳过反量化步骤，进入后续适配
- `可接受量化（FP8 per-block/per-channel）`：进入脚本实现与适配器接入
- `未知/不接受量化`：阻塞并向用户提出材料请求（见“阻塞话术”）

### 2) 实现反量化脚本（`convert_*_to_bf16.py`）

按 FP8 模式创建脚本，最小要求：

- 提供 `convert_*` 主转换函数（必须带可见进度条与异常提示）
- 脚本只需覆盖当前模型确认的量化模式（可仅实现 per-block）
- 将结果统一转换到 `torch.bfloat16`（或项目默认浮点 dtype）
- 参数回写必须 dtype-safe（强约束）：
  - 若原始权重存储 dtype 为 FP8，禁止直接将反量化得到的 BF16/F16 张量回写到该 FP8 存储；否则会发生静默类型回退
  - 必须先将“原参数存储”提升到目标浮点 dtype（如 BF16），再执行赋值/拷贝
  - 允许不同代码风格实现（原地改写或整体替换），但必须保证 dtype 与 device 一致
  - 回写后必须有显式校验（断言或日志）确认最终参数 dtype 已是目标浮点类型，而非 FP8
- 对 index 与 safetensors 不匹配场景给出 warning 并安全跳过
- 进度条要求：
  - 必须使用 `tqdm`（或等价方案）显示转换进度，禁止仅打印日志让用户等待
  - `total` 必须可追踪（例如按待转换参数个数/层数），并实时更新 `desc`（如当前层名）
  - 长耗时阶段（扫描权重、逐层反量化、保存结果）至少覆盖主耗时环节中的一个；优先覆盖逐层反量化主循环

建议函数骨架：

```python
@lru_cache(maxsize=1)
def get_weight_map(model_path: str) -> Dict[str, str]:
    ...

@torch.no_grad()
def convert_module_xxx_to_bf16(name: str, module: nn.Module, model_path: str, weight_map: Dict[str, str]):
    ...
```

per-block 示例：

```python
def decode_fp8_per_block(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    m, n = weight.shape
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)[:m, :n]
    return (weight.float() * scale_expanded.float()).to(torch.bfloat16)
```

参数回写（通用伪代码）：

```python
dequant_weight = dequantize(fp8_weight, scale_info)   # dequant_weight: BF16/F16
if original_param_storage_is_fp8:
    promote_original_param_storage_to_target_float_dtype()
assign_or_copy_with_device_dtype_alignment(dequant_weight)
verify_param_dtype_is_target_float_dtype_not_fp8()
```

可选工具脚本（用于判定权重/scale 形状）：

- `scripts/inspect_safetensors.py`
- 用法：
  - `python scripts/inspect_safetensors.py -m /path/to/model`
  - `python scripts/inspect_safetensors.py -m /path/to/model -p "model.layers.*.mlp*.weight*"`

### 3) 接入 `model_adapter.py`

在适配器中注入反量化调用，要求：

- 在调用 transformers 加载模型前，先检查模型目录 `config.json` 是否包含量化字段（如 `quantization_config`、`quant_method`、`fmt`、`weight_block_size`、`modules_to_not_convert` 等）
- 若检测到量化字段，必须主动删除这些字段后再进行加载，避免 transformers 按量化方式装载权重
- 删除原因必须在日志中说明：昇腾当前不支持 FP8 量化权重直载，若不清理配置会导致错误加载路径
- 可接受实现方式：
  - 直接修改模型目录下 `config.json` 后加载；或
  - 生成去量化字段的临时 `config` 并确保后续加载明确使用该配置
- 无论采用哪种方式，都必须保证“最终传给 transformers 的 config 不包含量化相关字段”
- 在模型权重可用后、量化流程前调用 `convert_*_to_bf16`
- 仅对命中的量化子模块执行，避免全模型无谓遍历
- 转换失败不应静默吞掉；至少记录 warning 与原因

接入示例：

```python
from .convert_fp8_to_bf16 import convert_module_fp8_to_bf16

def init_model(...):
    model = ...
    convert_module_fp8_to_bf16(name="", module=model, model_path=model_path, weight_map=...)
    return model
```

### 4) 最小验证

- 校验可读性：脚本能读取 index 与目标 safetensors
- 校验转换：目标层权重 dtype 由 FP8 变为 bf16/fp16
- 校验接入：adapter 初始化路径已触发自动转换函数
- 校验回退：非量化权重或不匹配键时不报致命错误
- 校验模式：输出明确 `per-block` 或 `per-channel` 判定结果
- 校验赋值安全：确认未出现“BF16 copy_ 到 FP8 存储后仍为 FP8”的回退问题

## 阻塞话术（必须原样遵循语义）

当量化格式不确定或缺反量化规则时，必须明确告知：

- 当前权重格式无法确认可逆反量化规则，不能安全实现通用反量化。
- 请用户二选一提供：
  1. 官方/已验证的反量化脚本（含权重键映射与scale定义）
  2. 对应浮点权重（bf16/fp16/fp32）
- 在补齐材料前，不进入模型适配代码改写阶段。

## 输出模板

Agent 每次执行本 Skill 后，输出以下结构：

```markdown
## 反量化判定结果
- 量化状态：未量化 | FP8 per-block(可接受) | FP8 per-channel(可接受) | 未知(阻塞)
- 判定依据：{keys/file evidence}

## 反量化实现动作
- 新增/复用脚本：{path}
- 核心函数：{functions}
- 适配器接入点：{adapter_path + function}

## 验证结果
- 读取校验：通过/失败
- dtype 校验：通过/失败
- 适配器触发校验：通过/失败

## 后续动作
- 若阻塞：请求用户提供反量化脚本或浮点权重
- 若通过：进入模型适配主流程
```
