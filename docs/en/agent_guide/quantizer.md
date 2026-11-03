# Quantizer: Model Quantization Tuning

`Quantizer` is an agent for msModelSlim model quantization scenarios. It completes quantization accuracy tuning through an **end-to-end tuning process**: you describe your requirements in natural language, and the agent orchestrates environment checks, model preparation, configuration search, quantization execution, and accuracy evaluation. It searches for a quantization configuration that meets the requirements under the specified accuracy constraints and delivers the quantized weights.

## 1. Agent Positioning

- Targets **quantization accuracy tuning** of msModelSlim for large language models (LLMs), multimodal understanding models (VLMs), and multimodal generation models (DiT).
- You do not need to write a complete YAML by hand. You can start the full process by describing the model path, quantization scheme, device, and accuracy target in natural language.
- If the target model is not yet supported by msModelSlim, the process triggers feasibility analysis and adapter development during the **model preparation** phase, with no need for a separate adaptation task.
- Handles the adaptation of basic Transformers models, as well as the integration and tuning of complex models such as MoE packed weight decomposition, layer-by-layer loading of very large models, and integration of independent DiT inference repositories.

## 2. Core Capabilities

- **Model adaptation**: Evaluate the feasibility of model integration and complete the development and verification of the msModelSlim adapter. If the model is not yet registered, the adaptation is triggered in the model preparation phase of the tuning process.
- **Outlier suppression adaptation**: After the basic model adaptation verification passes, complete the interface and mapping, installation, and pre-checks of the selected algorithms in the outlier suppression phase before any source code changes. When you do not specify algorithms, the agent independently verifies the following items by default: `quarot`, `flex_smooth_quant`, `flex_awq_ssz`, and `iter_smooth`. It uses the installed msModelSlim API to extract a DOT graph for each algorithm, and independently completes the logits gate and a human-readable summary report for each from the same original floating-point model. Failed items are recorded, and the dedicated adaptation code newly added for them in this round is withdrawn. The tuning phase checks only the corresponding interfaces of the installed adapters.
- **Accuracy tuning**: According to the accuracy constraints you specify, generate quantization and evaluation configurations, run quantization and AISBench evaluation, and iteratively search for a quantization scheme that meets the requirements.
- **Multimodal support**: The agent automatically identifies the model type (LLM/multimodal understanding model/multimodal generation model) and routes to the corresponding analysis, quantization, and evaluation subprocess.

## 3. Prerequisites

- Prepare an inference runtime environment. You are advised to use the `vllm-ascend` image. For guidance on installing `vllm-ascend` with Docker, see [vllm-ascend installation](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/installation.html#set-up-using-docker). You are advised to install msagent in the container and use Quantizer.
- Install a `transformers` version appropriate for your model. Special note: if the `transformers` version required by msModelSlim model quantization differs from the version required by the inference engine for serving, you can provide the relevant information to the agent and let it manage and use the corresponding version on its own.
- Quantization tuning requires installing msModelSlim in the container. For installation guidance, see [msModelSlim installation](https://atomgit.com/Ascend/msmodelslim/blob/26.2.0/docs/en/install_guide/install_guide.md).
- Tuning evaluation depends on the AISBench evaluation service. See its [README](https://github.com/AISBench/benchmark/blob/main/README.md) for installation and usage instructions. You must prepare the datasets required for evaluation yourself (such as gpqa and aime25). You can refer to the [AISBench dataset preparation guide](https://yh-ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html).

## 4. Recommended Usage

Describe your quantization tuning requirements in natural language. The agent extracts the parameters, echoes them for confirmation, and then advances through the full process step by step.

**Key Information to Provide:**

| Information | Description |
|------|------|
| Model path | Local directory or Hugging Face repository name |
| Save path | Directory for the quantized artifacts and process output |
| Quantization scheme | For example, W8A8. If not specified, the agent proposes a default scheme and confirms it with you. |
| Device | NPU/CUDA/CPU, and the device number |
| Accuracy requirements | Relative tolerance (for example, "accuracy loss must not exceed 2%") or an absolute target (for example, "gsm8k must be at least 83%") |
| Inference repository path | Required only for multimodal generation models (DiT). |
| `trust_remote_code` | Must be confirmed when you use a Hugging Face model with custom code. |

**Example Prompts:**

```text
Quantize path/to/Qwen3-32B with W8A8 using NPU 0, and keep the gsm8k accuracy loss within 1% of the baseline.
```

```text
{Model path} is a new model. Please complete W8A8 quantization tuning for it, and keep the accuracy loss on the gpqa dataset within 1%.
```

```text
Quantize path/to/Wan2.2-T2V-A14B with W8A8. The inference repository is at path/to/inference_repo. Use NPU 0, and keep the VBench score loss within 1% of the FP baseline.
```

## 5. End-to-End Tuning Process

Quantizer orchestrates the following phases (each phase proceeds to the next only after you confirm it):

1. **User input alignment**: Extract and echo the key parameters (model path, save path, quantization scheme, device, accuracy requirements, and so on).
2. **Environment preparation**: Confirm that msModelSlim can be imported and that the Ascend environment variables and NPU device numbers are ready.
3. **Model preparation**: Check whether the target model is registered in `config.ini` and verify the basic adaptation. After verification passes, complete outlier suppression adaptation independently.
4. **Quantization configuration tuning**: Loop through "generate quantization configuration → quantize → evaluate → record history" until the accuracy target is met or the maximum number of iterations is reached.
5. **Result delivery**: Output the quantized weights that meet the accuracy requirements, the evaluation report, and the tuning history.

Each phase is handled by a dedicated subagent:

| Subagent | Phase | Responsibility |
|--------|----------|------|
| msmodelslim-model-analysis | Model preparation | Perform pre-adaptation analysis: assess the implementation source, structure/MoE/layer-by-layer loading risks, and identify the model type (LLM/multimodal understanding/multimodal generation). |
| msmodelslim-model-adapt | Model preparation | After the analysis passes: create adaptation templates, register the model, configure `config.ini`, and run the four-step verification. For DiT models, its multimodal generation extension section handles the adaptation. |
| msmodelslim-anti-outlier-adapt | Model preparation | After the basic adaptation verification passes: run per-algorithm outlier suppression and the final logits gate. |
| quant-tuning-evaluation-generator | Quantization configuration tuning | Generate the evaluation configuration (Evaluation YAML). |
| quant-tuning-practice-generator | Quantization configuration tuning | Generate or adjust the quantization configuration (Practice YAML). |
| quant-tuning-quantizer | Quantization configuration tuning | Run model quantization according to the Practice YAML. |
| quant-tuning-evaluator | Quantization configuration tuning | Run AISBench accuracy evaluation on the quantized model. |
| quant-tuning-quantize-dit | DiT quantization | Run W8A8 dynamic/MXFP8 quantization according to the Practice YAML (`apiversion: multimodal_sd_modelslim_v1`). |
| quant-tuning-infer-dit | DiT tuning | Use MindIE-SD to perform smoke inference verification on the quantized weights (optional, no scoring). |
| quant-tuning-score-dit | DiT tuning | Run VBench dimension scoring on the generated videos and compare the scores with the FP baseline to determine whether the target is met. |
| quant-tuning-history-append-dit | DiT tuning | Record the tuning history of each DiT round (quantization configuration, scoring results, and fallback information). |

## 6. Model Preparation

The model preparation phase automatically identifies the model type and routes to the corresponding quantization and evaluation process according to the type:

| Model Type | Support Scope | Quantization Scheme | Evaluation Method |
|--------|----------|----------|----------|
| Large Language Model (LLM) | Decoder-only LLM | Linear layer quantization schemes such as W8A8 | AISBench text evaluation (gsm8k, gpqa, and so on) |
| Multimodal Understanding Model (VLM) | Text backbone of multimodal understanding models | Same as LLM (quantize only the LLM/text path) | AISBench multimodal evaluation (textvqa, mmmu, and so on) |
| Multimodal Generation Model (DiT) | Diffusion generation models (text-to-image/text-to-video) | W8A8 dynamic quantization/W8A8 MXFP8 | VBench dimension scoring (AISBench-VBench) |

### Model Adaptation Subprocess

When the target `model_type` is not yet registered in the msModelSlim `config.ini`, the agent first delegates to the model adaptation
subagent. After the four-step verification of the basic adaptation and the interface and mapping checks of the selected algorithms pass, the process enters the independent outlier
suppression gate process. Algorithms that lack the required interfaces must not start the gate.

This subprocess mainly completes the following:

- Evaluate the model implementation source, structural characteristics, and feasibility risks of quantization integration.
- Create a Transformers-based adapter for Decoder-only LLMs.
- Provide solutions such as layer-by-layer loading (lazy loading) for very large models to avoid memory bottlenecks.
- Strictly follow the gate rules and the multi-step verification process to ensure that conclusions are supported by actual evidence (configurations, logs, and command output).

### Multimodal Understanding Model (VLM)

- Supports automatic quantization and tuning of the **text backbone** of VLMs. The vision side (ViT) is not currently in the quantization scope.
- The analysis and quantization process is the same as that for LLMs. After the agent identifies the model as `vlm_text`, it automatically routes to the VLM-specific analysis/quantization/inference subprocess.
- Accuracy evaluation also uses AISBench text datasets, and accuracy constraints are expressed in the same way as for LLMs.

### Multimodal Generation Model (DiT)

- **An independent inference repository path is required**: The implementation code of a DiT (`modeling_*.py`/`pipeline/`) is usually located in an independent inference repository, separate from the weight directory. After the agent identifies a DiT model during the analysis phase, it proactively requests the inference repository path. The process cannot continue if the path is not provided.
- **Model adaptation**: The analysis subagent checks whether the model directory contains the `_diffusers_version` marker to determine whether it uses the diffusers implementation. Then the adaptation subagent completes adapter generation and four-step verification.
- **Quantization scheme**: data-free quantization, which does not introduce an activation calibration dataset. Two templates are supported:
  - `w8a8_dynamic`: weight per-channel INT8/activation per-token INT8 dynamic quantization
  - `w8a8_mxfp8`: weight per-block/activation per-block MXFP8
- **Accuracy tuning loop**: Expand the exclude list with a whole-layer fallback strategy (append the first N blocks to the fallback configuration and requantize), then generate videos through batch inference and score them with VBench. Compare the scores with the FP baseline until the accuracy target is met, the maximum number of iterations is reached, or the fallback rules are exhausted.
- **Smoke verification (optional)**: After quantization, you can use MindIE-SD for lightweight inference verification (a small number of prompts to verify that the weights can be loaded and that images or videos can be generated). This step is not scored.

## 7. Usage Notes

- In the outlier suppression phase, when you do not specify algorithms, the agent adapts the following items by default: `quarot`, `flex_smooth_quant`, `flex_awq_ssz`, and `iter_smooth`. When you specify algorithms, it runs only the selected subset. Each item is verified independently with a reloaded clean floating-point model, and no quantization is performed.
- DiT scenarios use data-free quantization and do not directly consume suppression strategy parameters such as `smooth`, `quarot`, and `awq`. Accuracy tuning is performed only through whole-layer fallback (exclude list expansion).
- DiT tuning requires preparing a VBench evaluation dataset and an independent inference repository. Generating videos with the FP baseline takes about 1.5 to 3 times as long as with the quantized model. Before starting the run, the agent prints the estimated inference duration to the screen.
- If the model analysis phase finds risk points that are difficult to adapt, the process is interrupted and you are informed in advance. The process proceeds to adaptation and subsequent tuning only after you confirm the risks and agree to continue.
- If you do not provide the floating-point baseline accuracy, the agent first runs evaluation on the floating-point model to obtain the baseline, and then enters the tuning loop.
