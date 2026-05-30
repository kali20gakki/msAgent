---
name: msmodelslim-model-adapt
description: 
  为 msModelSlim 创建基础 Transformers 模型适配器（Model Adapter）。
  包含创建适配器、实现必需接口与注册安装流程。
  适用：Decoder-only LLM、理解类 VLM（仅 LLM/text 部分）。
  不适用：多模态生成模型（图像/视频/语音生成）、Encoder-only、非 Transformers 架构。
---

# msModelSlim 基础模型适配 Skill

本 Skill 指导如何为新模型创建基础适配器，使其跑通 W8A8/W4A16 量化流程。

> 说明：逐层量化（按层加载/懒加载）属于高阶可选特性，不是基础适配必需项。
> 仅当 CPU 内存无法全量加载权重，或用户明确要求时，再在基础适配和四步验证（由 `msmodelslim-adapter-verification` 执行）完成后启用。

## 适用范围

- **支持**：Decoder-only LLM、理解类 VLM（只处理文本/LLM 主干）
- **不支持**：多模态生成（如 Stable Diffusion/Flux/Wan）、Encoder-only、非 Transformers

## 核心工作流

### 1. 权重来源确认（新增必做提示）
- **先询问并优先使用用户自有权重**：要求用户先提供本地模型权重路径（或已下载模型目录）。
- **仅在用户确认“没有权重”时再下载**：再协助用户执行下载流程，不要默认直接下载。
- **下载建议**：
  - 若先做结构分析，可先下载非权重文件：`modelscope download --model <org>/<model> --local_dir ./models/<name> --exclude '*.safetensors'`
  - 若进入完整量化/验证流程，需补齐可用权重文件。

### 2. 准备工作
- **分析模型**：阅读 `config.json` 与 `modeling_*.py`，确认结构与实现。
  - 详见：[模型结构分析指南](references/model_analysis.md)

### 3. 创建适配器
- **使用模板**：
  - LLM: `assets/model_adapter_template.py`
  - VLM: `assets/vlm_model_adapter_template.py`
- **实现接口**：实现 `handle_dataset`, `init_model`, `generate_model_visit`, `generate_model_forward`, `enable_kv_cache`。
- **关键原则**：
  - `visit` 与 `forward` 必须严格一致。
  - MoE 模型建议 unpack 为纯线性层。
  - 若原始模型存在需要保留的 buffer 权重，需在适配器中将其转换为 `nn.Parameter`；否则量化导出阶段通常不会保存 buffer 权重。
  - **Tokenizer pad_token 兼容性必查**：若 `tokenizer.pad_token` / `pad_token_id` 为 `None`，必须在适配器中重写 `_load_tokenizer`，将 `pad_token` 回退到 `eos_token`，避免量化过程中在 `padding=True` 时直接报错。
  - 常见报错：
    - `ValueError: Asking to pad but the tokenizer does not have a padding token.`
  - 根因链路：
    - `adapter.handle_dataset(...) -> _get_tokenized_data(...) -> tokenizer(..., padding=True, ...)`
    - 某些模型（如 MiniMax 系列）原生 tokenizer 未设置 `pad_token`。
  - 推荐修复模板：
    ```python
    def _load_tokenizer(self, trust_remote_code=False):
        """Ensure tokenizer has a pad token for quantization dataset padding."""
        tokenizer = super()._load_tokenizer(trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    ```
  - 说明：优先使用 `eos_token` 作为回退；若目标模型有更合适的专用 pad token，可按模型官方约定替换。
  - 详见：[适配器实现指南](references/implementation_guide.md)

### 4. 注册与安装
- 在 `config/config.ini` 注册模型与入口，并执行 `bash install.sh`。
- 详见：[适配器注册指南](references/registration_guide.md)

### 5. 功能性验证（独立 Skill）
- 适配器开发完成后，告知用户可自动执行功能性验证。
- 验证流程已独立为：`msmodelslim-adapter-verification`。
- 该验证 Skill 会自动按四步执行：生成测试模型 -> 全回退量化 -> 权重一致性与可加载/保存验证 -> 实际量化与描述文件规则校验。

### 6. 可选高阶特性：逐层量化
- 触发时机：
  - CPU 内存无法全量加载模型权重。
  - 用户明确要求“逐层量化/逐层加载/懒加载/按层加载”。
- 启用顺序：
  - 必须先完成基础适配与四步验证，再进入逐层量化改造。
- 实现与验证指引：
  - 详见独立 Skill：`msmodelslim-layer-wise-quantization`

## 参考资料

- [模型结构分析指南](references/model_analysis.md)
- [适配器实现指南](references/implementation_guide.md)
- [适配器注册指南](references/registration_guide.md)
- [接口检查清单](references/interface_checklist.md)
- [核心工作流](references/core_workflow.md)
