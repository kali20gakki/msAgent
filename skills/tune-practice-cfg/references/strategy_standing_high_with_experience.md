# standing_high_with_experience 调优策略

在 standing_high 基础上引入**专家经验**，实现结构感知的量化配置。根据模型结构类型自动匹配预定义的 QConfig 模板和离群值抑制策略。

最大迭代轮次由入参 `max_iterations` 指定。

---

## 工作流程

```
1. 用户指定 quant_type（w8a8/w4a8）+ structure_configs
2. 读取专家经验配置
3. 根据 quant_type 选择离群值抑制策略和结构映射
4. 根据 structure_configs 为不同结构类型分配 QConfig
5. 构建完整的调优配置
6. 根据模型能力过滤不支持的离群值抑制策略
7. 委托 standing_high 策略执行实际调优
```

---

## 专家经验配置

### 支持的量化类型

- `w8a8`：8-bit 权重 + 8-bit 激活
- `w4a8`：4-bit 权重 + 8-bit 激活

### 预定义 QConfig 模板

| 模板名 | act.scope | act.dtype | weight.scope | weight.dtype | weight.method |
|--------|-----------|-----------|-------------|-------------|---------------|
| `w8a8_default` | per_tensor | int8 | per_channel | int8 | minmax |
| `w8a8_dynamic` | per_token | int8 | per_channel | int8 | minmax |
| `w4a8_dynamic` | per_token | int8 | per_channel | int4 | ssz |

### 预定义离群值抑制策略

| 策略名 | type | 说明 |
|--------|------|------|
| `flex_smooth_quant_default` | flex_smooth_quant | 标准 SmoothQuant |
| `iter_smooth_default` | iter_smooth | 迭代 SmoothQuant |
| `flex_awq_ssz_default` | flex_awq_ssz | AWQ + SSZ 组合（step=10） |
| `quarot_default` | quarot | Quantization-Aware RoT |

### 结构-量化映射（w8a8）

| 模型结构 | QConfig 模板 | 说明 |
|----------|-------------|------|
| MHA | w8a8_default | 多头注意力 |
| GQA | w8a8_default | 分组查询注意力 |
| MLA | w8a8_default | 多头潜在注意力 |
| FFN | w8a8_dynamic | 前馈网络 |
| MoE | w8a8_dynamic | 混合专家 |
| DSA | bf16（不量化） | 动态稀疏注意力 |
| SWA | bf16（不量化） | 滑动窗口注意力 |
| GatedDeltaNet | bf16（不量化） | 门控 Delta 网络 |

### 结构-量化映射（w4a8）

| 模型结构 | QConfig 模板 |
|----------|-------------|
| MHA / GQA / MLA | w8a8_default |
| MoE | w4a8_dynamic |
| DSA / SWA / GatedDeltaNet | bf16（不量化） |

### 各量化类型的离群值抑制策略

- w8a8: `flex_smooth_quant_default`, `iter_smooth_default`
- w4a8: `flex_awq_ssz_default`

---

## 关键配置项

| 配置项 | 说明 |
|--------|------|
| **quant_type** | 量化类型：w8a8 / w4a8 |
| **structure_configs** | 模型各结构的配置（include/exclude + 结构类型） |

---

## 调优搜索空间

| 配置项 | 说明 |
|--------|------|
| **离群值抑制策略候选** | 由专家经验自动生成 |
| **允许的数据集列表** | 由专家经验自动生成 |
| **最大回退层数** | 由专家经验自动生成 |

控制摸高算法的搜索边界。

---

## 使用该策略的最小输入

| 必要输入 | 产出 |
|----------|------|
| 模型、quant_type（w8a8/w4a8）、structure_configs | 多轮 Practice YAML |
