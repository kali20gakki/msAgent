# Throughput Optimizer Parameters

Use this file when building or validating a command for `python -m cli.inference.throughput_optimizer`.

## Core Inputs

- `model_id`: required positional model identifier.
- `--device`: target hardware profile or profiles. The CLI accepts one or more values with `nargs="+"`; pass multiple profiles as `--device A B C` to enable cross-hardware summaries.
- `--num-devices`: total device count. Required for most planning cases even though the CLI has a default. In multi-hardware runs, this same count applies to every profile.
- `--input-length`: required prompt length.
- `--output-length`: required generated length.

## Hardware Profiles

- Single hardware run: use one `--device` value.
- Cross-hardware comparison: use multiple `--device` values in one command.
- Multiple hardware profiles reuse the same model, input/output lengths, deployment mode, quantization, SLO limits, search space, and `--num-devices`.
- Use separate commands when hardware targets require different device counts, different SLOs, or different search spaces.
- Device profile names are validated by the repository. If validation fails, report the unknown profile names and the valid-profile hint from the CLI error.
- Every selected profile must have a communication grid large enough for the shared `--num-devices`.

## Modes

## Domain Deployment Mode To CLI Mapping

| Domain term | Meaning | CLI mode | Required follow-up |
|---|---|---|---|
| PD混部 | Prefill+Decode combined serving layout | aggregation | none |
| PD聚合 | Prefill+Decode combined serving layout | aggregation | none |
| 聚合部署 | Prefill+Decode combined serving layout | aggregation | none |
| PD分离: phase capability | evaluate Prefill/Decode separately | disaggregation | Prefill only, Decode only, or both |
| PD分离: ratio planning | run P and D separately, then match capacity | PD ratio optimization | P/D devices per instance |
| PD ratio | P/D instance ratio planning | PD ratio optimization | P/D devices per instance |

### Aggregation

Default mode. Do not pass `--disagg` or `--enable-optimize-prefill-decode-ratio`.

Typical use:
- one combined serving instance runs Prefill and Decode
- optimize under TTFT, TPOT, or both

### Disaggregation

Pass `--disagg`.

Interpretation:
- only `--ttft-limits`: run Prefill optimization
- only `--tpot-limits`: run Decode optimization
- both limits: run both phases separately

### PD Ratio Optimization

Pass:
- `--enable-optimize-prefill-decode-ratio`
- `--prefill-devices-per-instance`
- `--decode-devices-per-instance`

Do not combine with `--disagg`.

## Quantization

Recommended defaults to offer, not silently apply:
- `--quantize-linear-action W8A8_DYNAMIC`
- `--quantize-attention-action DISABLED`

Custom linear choices:
- `DISABLED`
- `W8A16_STATIC`
- `W8A8_STATIC`
- `W4A8_STATIC`
- `W8A16_DYNAMIC`
- `W8A8_DYNAMIC`
- `W4A8_DYNAMIC`
- `FP8`
- `MXFP4`

Custom attention choices:
- `DISABLED`
- `INT8`
- `FP8`

If linear is `MXFP4`, optionally add `--mxfp4-group-size <n>`.

## Search Dimensions

- `--tp-sizes`
- `--ep-sizes`
- `--moe-dp-sizes`

Rules:
- if all three are omitted, the CLI falls back to TP-only search
- if a search argument is present with no values, the CLI searches powers of two up to world size
- explicit values must be positive, unique enough after normalization, and must not exceed the relevant shared device count

## Performance Constraints

- `--ttft-limits`: positive float in ms
- `--tpot-limits`: positive float in ms
- `--batch-range`: `[max]` or `[min max]`
- `--max-prefill-tokens`: relevant to aggregation mode and effective input length
- `--serving-cost`: optional cost term

## Advanced Options

- `--compile`
- `--compile-allow-graph-break`
- `--jobs`
- `--reserved-memory-gb`
- `--log-level`
- `--dump-original-results`
- `--prefix-cache-hit-rate`: prefix cache hit rate in `[0, 1)`. Ask whether prefix cache is enabled before adding it.
- `--num-mtp-tokens`: number of MTP tokens. Ask whether MTP is enabled and confirm model support before adding it.
- `--mtp-acceptance-rate`: MTP acceptance-rate assumptions in `[0, 1]`. Label user-provided versus heuristic values.

## Multimodal Inputs

For VL models, ask for:
- `--image-height`
- `--image-width`

## Practical Prompting Rules

When the user is unsure:
- ask whether they want one hardware target or a cross-hardware comparison
- ask for the deployment mode first: PD混部/PD聚合, PD分离 phase capability evaluation, or PD ratio planning
- prefer a narrow first run over an exhaustive run
- recommend TP-only search for dense models
- recommend TP-first search for MoE models, then expand to EP or MOE-DP if the user wants a broader comparison
- explicitly ask whether prefix cache and MTP are enabled before the pre-execution summary
- always confirm before execution
