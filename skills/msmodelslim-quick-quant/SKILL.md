---
name: msmodelslim-quick-quant
description: 提供 msModelSlim 的通用快速量化指引，包含安装、最简 YAML 配置与基础执行校验。适用于用户询问 msmodelslim 安装、快速量化、配置 yaml、linear_quant 或 minmax 基础参数时。
---

# msModelSlim Quick Quant Skill

## 适用场景

- 需要安装 `msmodelslim`
- 需要用 YAML 启动基础量化流程
- 仅需基础量化器配置（不涉及复杂策略）

## 执行规则

- 禁止使用绝对路径或仓库私有路径。
- 统一使用占位符：`${MODEL_PATH}`、`${SAVE_PATH}`、`${MODEL_TYPE}`、`${CONFIG_PATH}`。
- YAML 仅介绍 `linear_quant` 基础配置。
- 复杂配置（多阶段流程、MoE 混合策略、复杂 outlier 组合）不在本 Skill 覆盖范围。
- MoE 结构中路由器 `gate` 模块一般不量化，默认应排除。
- 回答保持简洁，只解释必要参数与命令。

## 最小流程

### 1) 安装与校验

- 检查 Python 版本（>=3.8）与依赖环境（如 CANN）。
- 按安装文档执行在线/离线/源码安装。
- 安装后验证：
  - `msmodelslim quant --help`
  - `python -c "import msmodelslim"`

### 2) 生成最简 YAML

```yaml
apiversion: "modelslim_v1"
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: false
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"
      include: ["*"]
      exclude: ["*.gate"]
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

`linear_quant` 参数说明见 [reference/linear_quant.md](reference/linear_quant.md)。

### 3) 执行量化

```bash
msmodelslim quant \
  --model_path ${MODEL_PATH} \
  --save_path ${SAVE_PATH} \
  --device npu \
  --model_type ${MODEL_TYPE} \
  --config_path ${CONFIG_PATH} \
  --trust_remote_code True
```

### 4) 最小验证

- 命令执行无报错退出。
- `save_path` 下生成量化结果文件。
- 日志中可看到 `linear_quant` 处理过程。
