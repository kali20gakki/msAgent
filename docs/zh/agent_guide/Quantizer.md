# Quantizer 模型量化调优

`Quantizer` 是面向 msModelSlim 模型量化场景的 Agent，负责通过**端到端调优流程**完成量化精度调优：用户以自然语言描述需求，Agent 编排环境检查、模型准备、配置搜索、量化执行与精度评测，在指定精度约束下搜索满足要求的量化配置并交付量化权重。

## Agent 定位

- 面向大语言模型（LLM）的 msModelSlim **量化精度调优**
- 用户无需手写完整 YAML，以自然语言描述模型路径、量化方案、设备与精度目标即可启动全流程
- 若目标模型尚未被 msModelSlim 支持，流程会在**模型准备**阶段触发可行性分析与适配器开发，无需单独发起适配任务
- 适合处理基础 Transformers 模型适配，以及 MoE packed 权重拆解、超大模型逐层加载等复杂模型的接入与调优

## 核心能力

- **模型适配**：评估模型接入可行性，完成 msModelSlim 适配器开发与验证；若模型尚未注册，在调优流程的模型准备阶段触发
- **精度调优**：根据用户指定的精度约束，生成量化与评测配置，执行量化与 AISBench 评测，迭代搜索满足要求的量化方案

## 前置准备

- 请准备推理运行环境，推荐使用 vllm-ascend 镜像，使用 Docker 安装 vllm-ascend 指导：[vllm-ascend安装](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/installation.html#set-up-using-docker)，推荐在容器内安装 msagent 并使用 Quantizer
- 请根据模型安装合适的 transformers 版本，特殊说明：如果 msModelSlim 模型量化与推理引擎服务化要求的 transformers 版本不一致，可以将相关信息告知 Agent，让其自行管理使用对应版本。
- 量化调优需在容器内安装 msModelSlim ；安装指导见 [msModelSlim 安装](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/getting_started/install_guide.md#23-%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85)
- 调优评测依赖 AISBench 评测服务，安装与使用说明见其 [README](https://github.com/AISBench/benchmark/blob/main/README.md)；测评所需数据集（如 gpqa、aime25 等）须自行准备，可参考 [AISBench 数据集准备指南](https://yh-ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html)

## 推荐使用方式

用自然语言描述量化调优需求即可，Agent 会提取参数、回显确认，再依次推进全流程。

**需提供的关键信息：**

| 信息 | 说明 |
|------|------|
| 模型路径 | 本地目录或 HuggingFace 仓库名 |
| 保存路径 | 量化产物与过程输出目录 |
| 量化方案 | 如 W8A8；未说明时 Agent 会提议默认方案并经你确认 |
| 设备 | NPU / CUDA / CPU 及卡号 |
| 精度需求 | 相对容差（如「精度损失不超过 2%」）或绝对目标（如「gsm8k 不低于 83%」） |
| `trust_remote_code` | 使用 HuggingFace 自定义代码模型时需确认 |

**示例提示词：**

```
帮我把 path/to/Qwen3-32B 进行 W8A8 量化，使用 NPU 0卡，gsm8k 相较于基线精度损失控制在 1% 以内
```

```
{模型路径} 是新模型，请帮我完成 W8A8 量化调优，gpqa 数据集精度损失不超过 1%
```

## 端到端调优流程

Quantizer 按以下阶段编排（各阶段经用户确认后进入下一步）：

1. **用户输入对齐**：提取并回显关键参数（模型路径、保存路径、量化方案、设备、精度需求等）
2. **环境准备**：确认 msModelSlim 可 import、Ascend 环境变量与 NPU 卡号就绪
3. **模型准备**：检查 `config.ini` 是否已注册目标模型；**未注册时进入模型适配子流程**（见下节），已注册则跳过
4. **量化配置调优**：循环执行「生成量化配置 → 量化 → 评测 → 记录历史」，直至精度达标或达到最大迭代次数
5. **结果交付**：输出满足精度要求的量化权重、评测报告与调优历史

各阶段由专用 SubAgent 分工完成：

| SubAgent | 所属阶段 | 职责 |
|--------|----------|------|
| `msmodelslim-model-analysis` | 模型准备 | 适配前分析：实现来源、结构 / MoE / 逐层加载等风险评估 |
| `msmodelslim-model-adapt` | 模型准备 | 分析通过后：适配模板、注册、`config.ini` 与四步验证 |
| `quant-tuning-evaluation-generator` | 量化配置调优 | 生成测评配置（Evaluation YAML） |
| `quant-tuning-practice-generator` | 量化配置调优 | 生成 / 调整量化配置（Practice YAML） |
| `quant-tuning-quantizer` | 量化配置调优 | 依据 Practice YAML 执行模型量化 |
| `quant-tuning-evaluator` | 量化配置调优 | 对量化模型执行 AISBench 精度评测 |

### 模型准备（模型适配子流程）

当目标 `model_type` 尚未在 msModelSlim `config.ini` 中注册时，Agent 委派分析与适配 SubAgent，完成后继续进入量化配置调优。

该子流程主要完成：

- 评估模型实现来源、结构特征及量化接入的可行性风险
- 为 Decoder-only LLM 创建基于 Transformers 的适配器
- 为超大模型提供逐层加载（懒加载）等解决方案以规避内存瓶颈
- 严格遵循门禁规则与多步验证流程，确保结论由实际证据（配置、日志、命令输出）支撑

## 使用注意

- 当前模型适配子流程主要支持 LLM 的 W8A8 等线性层量化；离群值抑制与 FA3 等复杂算法暂不支持
- 模型分析阶段若发现较难适配的风险点，会中断流程并提前告知；需你确认风险并同意继续后，才会进入适配与后续调优
- 若未提供浮点 baseline 精度，Agent 会先对浮点模型执行评测获取 baseline，再进入调优循环
