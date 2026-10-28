# Quantizer 模型量化调优

`Quantizer` 是面向 msModelSlim 模型量化场景的 Agent，负责通过**端到端调优流程**完成量化精度调优：用户以自然语言描述需求，Agent 编排环境检查、模型准备、配置搜索、量化执行与精度评测，在指定精度约束下搜索满足要求的量化配置并交付量化权重。

## 1. Agent 定位

- 面向大语言模型（LLM）、多模态理解模型（VLM）与多模态生成模型（DiT）的 msModelSlim **量化精度调优**。
- 用户无需手写完整 YAML，以自然语言描述模型路径、量化方案、设备与精度目标即可启动全流程。
- 若目标模型尚未被 msModelSlim 支持，流程会在**模型准备**阶段触发可行性分析与适配器开发，无需单独发起适配任务。
- 适合处理基础 Transformers 模型适配，以及 MoE packed 权重拆解、超大模型逐层加载、DiT 独立推理仓接入等复杂模型的接入与调优。

## 2. 核心能力

- **模型适配**：评估模型接入可行性，完成 msModelSlim 适配器开发与验证；若模型尚未注册，在调优流程的模型准备阶段触发。
- **离群值抑制适配**：基础模型适配验证通过后，在离群值抑制阶段先于源码完成所选算法的接口及映射、安装与前置检查；用户未指定时默认独立验证 `quarot`、`flex_smooth_quant`、`flex_awq_ssz`、`iter_smooth` 4 项，使用已安装 msModelSlim API 提取逐算法 DOT 图，并从同一原始浮点模型分别独立完成 logits 门禁和人类可读汇总报告。失败项留痕并撤回其本次新增的专用适配代码；调优阶段只检查已安装 Adapter 的对应接口。
- **精度调优**：根据用户指定的精度约束，生成量化与评测配置，执行量化与 AISBench 评测，迭代搜索满足要求的量化方案。
- **多模态支持**：Agent 会自动识别模型类型（大语言模型 / 多模态理解模型 / 多模态生成模型），并路由到对应的分析、量化与评测子流程。

## 3. 前置准备

- 请准备推理运行环境，推荐使用 vLLM-Ascend 镜像，使用 Docker 安装 vLLM-Ascend 指导：[vLLM-Ascend安装](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/installation.html#set-up-using-docker)，推荐在容器内安装 msagent 并使用 Quantizer
- 请根据模型安装合适的 transformers 版本，特殊说明：如果 msModelSlim 模型量化与推理引擎服务化要求的 transformers 版本不一致，可以将相关信息告知 Agent，让其自行管理使用对应版本。
- 量化调优需在容器内安装 msModelSlim；安装指导见 [msModelSlim 安装](https://gitcode.com/Ascend/msmodelslim/blob/26.2.0/docs/zh/install_guide/install_guide.md)
- 调优评测依赖 AISBench 评测服务，安装与使用说明见其 [README](https://github.com/AISBench/benchmark/blob/main/README.md)；评测所需数据集（如 gpqa、aime25 等）须自行准备，可参考 [AISBench 数据集准备指南](https://yh-ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html)

## 4. 推荐使用方式

用自然语言描述量化调优需求即可，Agent 会提取参数、回显确认，再依次推进全流程。

**需提供的关键信息：**

| 信息 | 说明 |
|------|------|
| 模型路径 | 本地目录或 HuggingFace 仓库名。 |
| 保存路径 | 量化产物与过程输出目录。 |
| 量化方案 | 如 W8A8；未说明时 Agent 会提议默认方案并经你确认。 |
| 设备 | NPU / CUDA / CPU 及卡号。 |
| 精度需求 | 相对容差（如「精度损失不超过 2%」）或绝对目标（如「gsm8k 不低于 83%」）。 |
| 推理仓路径 | 仅多模态生成模型（DiT）需要。 |
| `trust_remote_code` | 使用 HuggingFace 自定义代码模型时需确认。 |

**示例提示词：**

```text
帮我把 path/to/Qwen3-32B 进行 W8A8 量化，使用 NPU 0卡，gsm8k 相较于基线精度损失控制在 1% 以内
```

```text
{模型路径} 是新模型，请帮我完成 W8A8 量化调优，gpqa 数据集精度损失不超过 1%
```

```text
帮我对 path/to/Wan2.2-T2V-A14B 进行 W8A8 量化，推理仓在 path/to/inference_repo，使用 NPU 0卡，VBench 评分相较 FP baseline 损失不超过 1%
```

## 5. 端到端调优流程

Quantizer 按以下阶段编排（各阶段经用户确认后进入下一步）：

1. **用户输入对齐**：提取并回显关键参数（模型路径、保存路径、量化方案、设备、精度需求等）。
2. **环境准备**：确认 msModelSlim 可 import、Ascend 环境变量与 NPU 卡号就绪。
3. **模型准备**：检查 `config.ini` 是否已注册目标模型并验证基础适配；通过后独立完成离群值抑制适配。
4. **量化配置调优**：循环执行「生成量化配置 → 量化 → 评测 → 记录历史」，直至精度达标或达到最大迭代次数。
5. **结果交付**：输出满足精度要求的量化权重、评测报告与调优历史。

各阶段由专用 SubAgent 分工完成：

| SubAgent | 所属阶段 | 职责 |
|--------|----------|------|
| `msmodelslim-model-analysis` | 模型准备 | 适配前分析：实现来源、结构 / MoE / 逐层加载等风险评估，并识别模型类型（LLM / 多模态理解 / 多模态生成） |
| `msmodelslim-model-adapt` | 模型准备 | 分析通过后：适配模板、注册、`config.ini` 与四步验证；DiT 模型由其多模态生成扩展节承接适配 |
| `msmodelslim-anti-outlier-adapt` | 模型准备 | 基础适配验证通过后：执行逐算法离群值抑制与最终 logits 门禁 |
| `quant-tuning-evaluation-generator` | 量化配置调优 | 生成评测配置（Evaluation YAML） |
| `quant-tuning-practice-generator` | 量化配置调优 | 生成 / 调整量化配置（Practice YAML） |
| `quant-tuning-quantizer` | 量化配置调优 | 依据 Practice YAML 执行模型量化 |
| `quant-tuning-evaluator` | 量化配置调优 | 对量化模型执行 AISBench 精度评测 |
| `quant-tuning-quantize-dit` | DiT 量化 | 依据 Practice YAML（`apiversion: multimodal_sd_modelslim_v1`）执行 W8A8 动态 / MXFP8 量化 |
| `quant-tuning-infer-dit` | DiT 调优 | 使用 MindIE-SD 对量化权重做冒烟推理验证（可选，不评分） |
| `quant-tuning-score-dit` | DiT 调优 | 对生成的视频执行 VBench 维度评分，与 FP baseline 对比判定是否达标 |
| `quant-tuning-history-append-dit` | DiT 调优 | 记录 DiT 各轮调优历史（量化配置、评分结果与回退信息） |

## 6. 模型准备

模型准备阶段会自动识别模型类型，并按类型路由到对应的量化与评测流程：

| 模型类型 | 支持范围 | 量化方案 | 评测方式 |
|--------|----------|----------|----------|
| 大语言模型（LLM） | Decoder-only LLM | W8A8 等线性层量化方案 | AISBench 文本评测（gsm8k、gpqa 等） |
| 多模态理解模型（VLM） | 多模态理解模型的文本主干 | 与 LLM 相同（仅量化 LLM / 文本路径） | AISBench 多模态评测（textvqa、mmmu 等） |
| 多模态生成模型（DiT） | 扩散生成模型（文生图 / 文生视频） | W8A8 动态量化 / W8A8 MXFP8 | VBench 维度评分（AISBench-VBench） |

### 模型适配子流程

当目标 `model_type` 尚未在 msModelSlim `config.ini` 中注册时，Agent 先委派模型适配
SubAgent。基础适配四步验证以及所选算法的接口与映射检查通过后，再进入独立的离群值
抑制门禁流程；缺少所需接口的算法不得启动门禁。

该子流程主要完成：

- 评估模型实现来源、结构特征及量化接入的可行性风险。
- 为 Decoder-only LLM 创建基于 Transformers 的适配器。
- 为超大模型提供逐层加载（懒加载）等解决方案以规避内存瓶颈。
- 严格遵循门禁规则与多步验证流程，确保结论由实际证据（配置、日志、命令输出）支撑。

### VLM（多模态理解模型）

- 支持 VLM 的**文本主干**自动量化与调优，视觉侧（ViT）暂不在量化范围内。
- 分析与量化流程与 LLM 一致，Agent 识别为 `vlm_text` 后自动路由到 VLM 专属的分析 / 量化 / 推理子流程。
- 精度评测同样使用 AISBench 文本数据集，精度约束的表达方式与 LLM 相同。

### DiT（多模态生成模型）

- **需提供独立推理仓路径**：DiT 的实现代码（`modeling_*.py` / `pipeline/`）通常位于独立的推理仓中，与权重目录分离。Agent 在分析阶段识别出 DiT 模型后，会主动索取推理仓路径；未提供时无法继续。
- **模型适配**：分析 SubAgent 会检查模型目录是否包含 `_diffusers_version` 标记以判定 diffusers 实现，随后由适配 SubAgent 完成适配器生成与四步验证。
- **量化方案**：data-free 量化，不引入激活校准数据集，支持两种模板：
  - `w8a8_dynamic`：权重 per-channel INT8 / 激活 per-token INT8 动态量化
  - `w8a8_mxfp8`：权重 per-block / 激活 per-block MXFP8
- **精度调优回路**：以整层回退策略扩展 exclude 列表（将前 N 个 block 追加到回退配置后重新量化），再经批量推理产出视频并由 VBench 评分，与 FP baseline 对比直至精度达标、达到最大迭代次数或回退规则耗尽。
- **冒烟验证（可选）**：量化完成后可使用 MindIE-SD 进行轻量推理验证（少量 prompt，验证权重可加载、可出图 / 出视频），不计分。

## 7. 注意事项

- 离群值抑制阶段在用户未指定时默认适配 `quarot`、`flex_smooth_quant`、`flex_awq_ssz`、`iter_smooth` 4 项；用户指定时仅执行所选子集。每项使用重新加载的干净浮点模型独立校验，且不执行量化
- DiT 场景为 data-free 量化，不直接消费 `smooth` / `quarot` / `awq` 等抑制策略参数；精度调优仅通过整层回退（exclude 列表扩展）进行
- DiT 调优需准备 VBench 评测数据集与独立推理仓；FP baseline 生成视频耗时约为量化模型的 1.5-3 倍，开始执行前 Agent 会打屏预估推理时长
- 模型分析阶段若发现较难适配的风险点，会中断流程并提前告知；需你确认风险并同意继续后，才会进入适配与后续调优
- 若未提供浮点 baseline 精度，Agent 会先对浮点模型执行评测获取 baseline，再进入调优循环
