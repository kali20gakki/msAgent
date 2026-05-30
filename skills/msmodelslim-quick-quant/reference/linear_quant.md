# linear_quant 参数参考（简洁版）

## 处理器级参数

| 参数 | 类型 | 说明 | 常用值 |
|---|---|---|---|
| `type` | string | 处理器类型，固定为线性量化 | `"linear_quant"` |
| `qconfig` | object | 量化核心配置，包含 `act` 和 `weight` | `{ act: ..., weight: ... }` |
| `include` | array[string] | 包含的层名模式（支持通配符） | `["*"]`, `["*self_attn*"]` |
| `exclude` | array[string] | 排除的层名模式，优先级高于 `include` | `["*down_proj*"]` |

## qconfig.act（激活量化）

| 参数 | 类型 | 可选值 | 默认值 | 说明 |
|---|---|---|---|---|
| `scope` | string | `"per_tensor"`, `"per_token"`, `"pd_mix"` | `"per_tensor"` | 激活量化粒度/模式 |
| `dtype` | string | `"int8"`, `"int4"`, `"float"` | `"int8"` | 激活量化数据类型 |
| `symmetric` | bool | `true`, `false` | `false` | 是否对称量化 |
| `method` | string | `"minmax"`, `"histogram"` | `"minmax"` | 激活量化算法 |

## qconfig.weight（权重量化）

| 参数 | 类型 | 可选值 | 默认值 | 说明 |
|---|---|---|---|---|
| `scope` | string | `"per_tensor"`, `"per_channel"`, `"per_group"` | `"per_channel"` | 权重量化粒度 |
| `dtype` | string | `"int8"`, `"int4"` | `"int8"` | 权重量化数据类型 |
| `symmetric` | bool | `true`, `false` | `true` | 是否对称量化 |
| `method` | string | `"minmax"`, `"ssz"`, `"gptq"` | `"minmax"` | 权重量化算法 |

## 最小可用建议

- 优先从 `act.scope: "per_tensor"` + `weight.scope: "per_channel"` 开始。
- 优先使用 `method: "minmax"` 作为基础配置。
- 先用 `include: ["*"]` 验证流程，再按需增加 `exclude`。
- 对 MoE 模型，路由器 `gate` 模块一般不量化，建议 `exclude: ["*.gate"]`。
