# Text Generate Parameter Guidance

This reference summarizes high-value parameters for `python -m cli.inference.text_generate`.

## Required Or Core Parameters

- `model_id`: Hugging Face model id or local model directory.
- `--device`: target device profile; default is `TEST_DEVICE`, but real planning should specify a concrete profile.
- `--num-devices`: total device/process count; default is `1`.
- `--num-queries`: number of parallel inference queries in one batch; required.
- `--query-length`: new input sequence length in tokens; required.
- `--context-length`: existing context length in tokens; default is `0`.
- `--decode`: enables autoregressive decoding mode; omit for prefill-only simulation.

## Optimization And Quantization

- `--compile`: enables compiled path simulation; the skill adds it by default for realistic compiled-path validation, shows the assumption, and lets the user disable it before execution.
- `--compile-allow-graph-break`: allows graph breaks during compile for dynamic models.
- `--enable-sequence-parallel`: enables sequence parallel graph rewrite during compilation.
- `--quantize-linear-action`: default is `W8A8_DYNAMIC`.
- `--quantize-attention-action`: default is `DISABLED`.
- `--quantize-lmhead`: off by default because LM head quantization can affect accuracy.
- `--mxfp4-group-size`: default is `32`; relevant when linear quantization is `MXFP4`.

## Parallelism

- `--tp-size`: tensor parallel size for the whole model; default is `1`.
- `--dp-size`: data parallel size for the whole model; optional.
- `--ep-size`: expert parallel size; default is `1`.
- `--o-proj-tp-size`, `--o-proj-dp-size`: attention output projection overrides.
- `--mlp-tp-size`, `--mlp-dp-size`: MLP layer overrides.
- `--lmhead-tp-size`, `--lmhead-dp-size`: LM head overrides.
- `--moe-tp-size`, `--moe-dp-size`: expert-layer overrides.
- `--word-embedding-tp`: enables word embedding TP with `col` or `row`.
- `--enable-redundant-experts`: enables redundant experts for relevant MoE layouts.
- `--enable-shared-expert-tp`: enables vLLM-style TP for shared experts.
- `--enable-dispatch-ffn-combine`: enables dispatch/FFN/combine fusion during compilation.
- `--enable-external-shared-experts`: uses external shared experts.
- `--host-external-shared-experts`: hosts external shared experts on the current device.

## Prefix Cache And MTP

- `--prefix-cache-hit-rate`: token-level prefix cache hit-rate approximation in `[0, 1)`.
- `--num-mtp-tokens`: number of MTP tokens; use only for models with MTP capability.
- `--disable-repetition`: preserves original transformer behavior instead of exploiting repetition patterns.

## Multimodal

- `--image-batch-size`: batch size for image processing.
- `--image-height`: input image height.
- `--image-width`: input image width.

If any image option is needed, collect all three values.

## Performance Model And Debug

After the candidate command is generated, always present `--chrome-trace`, `--graph-log-url`, and `--dump-input-shapes` as optional Debug/Trace choices before final execution confirmation.

- `--performance-model analytic`: default roofline-style model.
- `--performance-model profiling`: empirical model backed by profiling CSV data.
- `--profiling-database`: required for profiling mode.
- `--export-empirical-metrics`: developer export path; requires profiling mode.
- `--chrome-trace`: writes a Chrome trace file.
- `--graph-log-url`: dumps compiled graphs when compile is enabled.
- `--dump-input-shapes`: groups table averages by input shapes.
- `--num-hidden-layers-override`: debug-only hidden layer override.
- `--remote-source`: `huggingface` by default; use `modelscope` when requested.
- `--log-level`: defaults to `error`; use `info` or `debug` for troubleshooting.

## Common Templates

Prefill-only analytic validation:

```powershell
python -m cli.inference.text_generate MODEL_ID --device DEVICE --num-devices N --num-queries B --query-length INPUT_TOKENS --context-length 0 --tp-size TP --compile
```

Decode validation:

```powershell
python -m cli.inference.text_generate MODEL_ID --device DEVICE --num-devices N --num-queries B --query-length INPUT_TOKENS --context-length CONTEXT_TOKENS --decode --tp-size TP --compile
```

Profiling validation:

```powershell
python -m cli.inference.text_generate MODEL_ID --device DEVICE --num-devices N --num-queries B --query-length INPUT_TOKENS --performance-model profiling --profiling-database PATH
```
