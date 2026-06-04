# Quantizer 模型量化

`Quantizer` 是面向 msModelSlim 模型分析与适配场景的 Agent，负责协助用户完成新模型接入量化/压缩流程前的可行性评估、实现来源与结构性风险排查，并在分析通过后按约定完成模型适配器（Model Adapter）的开发与验证。

## Agent 定位

- 面向大语言模型（LLM）的 msModelSlim 适配场景
- 聚焦接入前可行性分析、模型结构解析、适配器代码生成及关键验证
- 适合处理基础 Transformers 模型适配，以及应对 MoE packed 权重拆解、超大模型逐层加载等复杂模型接入需求

## 核心能力

- 评估模型实现来源、结构特征及量化接入的可行性风险
- 为 Decoder-only LLM 创建基于 Transformers 的适配器
- 为超大模型提供逐层加载（懒加载）等解决方案以规避内存瓶颈
- 严格遵循门禁规则与多步验证流程，确保结论由实际证据（配置、日志、命令输出）支撑

## 环境准备

- 请准备推理运行环境，推荐使用 vllm-ascend 镜像，使用 Docker 安装 vllm-ascend 指导：[vllm-ascend安装](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/installation.html#set-up-using-docker)，推荐在容器内安装 msagent 并使用 Quantizer
- 请根据模型安装合适的 transformers 版本

## 推荐使用方式

- 开始前请先下载 [msModelSlim 代码仓](https://gitcode.com/Ascend/msmodelslim)
- 模型权重请用户自行准备并下载到本地可访问路径，再将模型名称或路径提供给 Quantizer
- 在开始适配前，提供模型名称或路径，让 Quantizer 先进行模型分析和风险评估
- Agent生成模型适配器后，请重新源码安装msModelSlim，执行install.sh安装脚本，具体操作可参考[安装指导](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/getting_started/install_guide.md#23-%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85)，安装完成后使用一键量化生成量化权重，具体操作可参考[一键量化使用指导](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/feature_guide/quick_quantization_v1/usage.md)
- 若量化过程出现报错，可以将报错信息复制给agent，由agent定位并解决问题
- 推荐使用以下量化配置生成w8a8量化权重

```yaml
apiversion: modelslim_v1

default_w8a8_dynamic: &default_w8a8_dynamic
  act:
    scope: "per_token"
    dtype: "int8"
    symmetric: True
    method: "minmax"
  weight:
    scope: "per_channel"
    dtype: "int8"
    symmetric: True
    method: "minmax"

spec:
  process:
    - type: "linear_quant"
      qconfig: *default_w8a8_dynamic
      include:
        - "*"
      exclude:
        - "*.gate"
        - "*mlp.down_proj"
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

## 使用注意
- 当前生成的模型适配器仅支持LLM的w8a8、LLM结构中的MOE的w4a8量化等线性层量化，暂不支持离群值抑制系列算法和FA3量化等复杂算法的适配。
- Agent在模型分析过程若发现较难适配的风险点，会中断开发适配器，提前告知风险，需要用户确认风险，并告知继续开发适配器后，才会继续开发模型适配器。

## 典型使用场景

| 场景 | 示例提示词 | 效果展示|
|---|---|---|
| 基础模型适配 | ` {模型路径} 是该模型}的权重，请完成对该模型的适配风险分析，并完成{msModelSlim代码路径}项目对于该模型的适配器开发和验证工作。` | <img src="../figures/modelslim_adapt.jpg" alt="MsModelslim 适配示例" width="800"> |
