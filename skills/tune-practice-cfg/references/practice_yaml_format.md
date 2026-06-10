# 量化配置格式（Practice YAML）

## 整体结构

```yaml
apiversion: modelslim_v1          # API 版本：modelslim_v0 | modelslim_v1 | multimodal_vlm_modelslim_v1
metadata:
  config_id: "unique-config-name" # 配置唯一标识
  score: 100.0                    # 排序分数（越高越优先）
  verified_model_types: []        # 已验证适用的模型类型列表
  label:                          # 过滤标签（必填 dict，禁止写成字符串）
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: false
  verified_tags: {}               # {model_type: [[tag1, tag2], [tag3]]}
spec:
  process: []                     # 量化处理器列表（见下文）
  dataset: "mix_calib.jsonl"          # 校准数据集（无路径时用短名，在 lab_calib 解析）
  save: []                        # 保存配置
```

- **metadata**：仅上述键；勿塞 `quantization` 等与顶层无关项。
- **spec**：仅 V1 允许的块；勿混 GPTQ/C4 等无关段名。
- **process 顺序**：一般先离群/旋转等前处理，再 `linear_quant` / `autoround_quant` 等。

## Process 处理器类型

`spec.process` 是一个有序列表，每个元素定义一个处理步骤。

### linear_quant — 线性层量化

```yaml
- type: "linear_quant"
  qconfig:
    act:
      scope: "per_token"
      dtype: "int8"
      symmetric: true
      method: "minmax"
      ext: {}                    # 扩展参数，如 ssz 的 step: 10
    weight:
      scope: "per_channel"
      dtype: "int8"
      symmetric: true
      method: "minmax"
      ext: {}
  include: ["*"]                 # 包含的层模式（fnmatch 匹配）
  exclude: []                    # 排除的层模式（fnmatch 匹配）
```

### flex_smooth_quant — 离群值抑制（SmoothQuant）

```yaml
- type: "flex_smooth_quant"
  include: ["*"]
```

### iter_smooth — 迭代式 SmoothQuant

```yaml
- type: "iter_smooth"
  include: ["*"]
```

### flex_awq_ssz — AWQ + SSZ 组合

```yaml
- type: "flex_awq_ssz"
  qconfig:
    act:
      scope: "per_token"
      dtype: "int8"
      symmetric: true
      method: "minmax"
    weight:
      scope: "per_channel"
      dtype: "int4"
      symmetric: true
      method: "ssz"
      ext:
        step: 10
  enable_subgraph_type:
    - "norm-linear"
    - "linear-linear"
    - "ov"
    - "up-down"
```

### quarot — Quantization-Aware RoT

```yaml
- type: "quarot"
```

## QConfig 字段取值

| 字段 | 有效值 | 说明 |
|------|--------|------|
| **dtype** | `int8`, `int4`, `float`, `mxfp8`, `mxfp4`, `fp8_e4m3` | 量化数据类型 |
| **scope** | `per_tensor`, `per_channel`, `per_group`, `per_block`, `per_token`, `pd_mix`, `per_head` | 量化粒度 |
| **symmetric** | `true`, `false` | 是否对称量化 |
| **method** | `minmax`, `mse`, `ssz`, `awq`, `quarot`, `none` | 校准算法 |
| **ext** | 对象，如 `{step: 10}` | 算法扩展参数 |

> `dtype: "float"` 表示保持浮点不量化（用于回退层）。`method: "none"` 表示不使用校准。

## Save 保存配置

```yaml
save:
  - type: "ascendv1_saver"
    part_file_size: 4            # 分片大小（GB）
```

- save的配置默认优先按照以上示例填写

## 完整示例（W8A8 默认配置）

```yaml
apiversion: modelslim_v1
metadata:
  config_id: default-w8a8
  score: 50
  verified_model_types: []
  label:
    w_bit: 8
    a_bit: 8
    is_sparse: false
    kv_cache: false
spec:
  process:
    - type: "iter_smooth"
      include: ["*"]
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"
      include: ["*"]
  dataset: "mix_calib.jsonl"
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

## 常见错误

- `metadata.label` 写成字符串而非 dict
- `type` 与字段不匹配（如 `flex_awq_ssz` 缺少 `qconfig`）
- `dataset` 虚构不存在的文件名，未用 `lab_calib` 短名
- `save` 字段的 `type` 不为 `"ascendv1_saver"`
