# Dialog Flow

## Context Reuse And Suggestions

Do not build a complete command from only a model name and planning scenario. If core inputs are missing, ask progressive questions first.

- Treat every new user request as a fresh parameter collection unless the user explicitly asks to reuse or continue a previous configuration.
- Do not fill missing values from earlier turns by default.
- When the user is unsure, provide a short recommendation and label each suggested value source:
  - CLI default
  - user-provided previous config
  - inspected local test/example
  - documented example
  - throughput_optimizer result
  - explicit heuristic
- Do not describe a configuration as regression-covered or repository-backed unless you inspected the concrete file or command output in the current turn.
- If a value is only a heuristic, keep the explanation brief and do not present it as a baseline.

## Step 1: Determine Deployment Mode

Use the following domain mapping first:

- PD混部, PD聚合, 聚合部署:
  map to aggregation mode. Explain that Prefill and Decode are modeled in one combined serving layout.
- PD分离:
  do not map immediately. Ask one clarification:
  "PD分离是想分别评估 Prefill/Decode 阶段能力，还是想指定 P/D 每实例卡数并计算实例配比？"
- PD ratio, P/D配比, 实例配比:
  map to PD ratio optimization mode.

CLI mapping:
- aggregation mode:
  no `--disagg`; no `--enable-optimize-prefill-decode-ratio`
- disaggregation mode:
  pass `--disagg`
- PD ratio optimization:
  pass `--enable-optimize-prefill-decode-ratio`
  plus `--prefill-devices-per-instance`
  plus `--decode-devices-per-instance`

If the user is unsure, explain in one sentence each:
- PD混部 / PD聚合: optimize one combined Prefill+Decode serving layout
- PD分离 phase evaluation: optimize Prefill and/or Decode as separate phases
- PD ratio: optimize Prefill and Decode separately and match their instance ratio

## Step 2: Collect Core Inputs

Ask for:
- model id
- device profile or profiles
- total device count shared by all selected profiles
- input length
- output length

Hardware prompting:
- If the user provides one hardware target, pass it as `--device <profile>`.
- If the user provides multiple hardware targets, pass them as space-separated values: `--device <profile1> <profile2> ...`.
- Explain that multiple profiles run the same workload and search settings on each hardware profile and then print cross-hardware summaries.
- Do not ask for different `--num-devices` values per profile; the current CLI uses one shared `--num-devices` for all profiles.
- If the user needs different device counts per hardware target, explain that this requires separate commands.

## Step 3: Collect SLOs

Ask what constrains the search:
- TTFT only
- TPOT only
- both TTFT and TPOT

TTFT and TPOT limit values are in milliseconds and must be positive floats.

Mode-specific follow-up:
- aggregation: one run with the provided limits
- disaggregation: explain whether the run will cover Prefill, Decode, or both
- PD ratio: recommend collecting both TTFT and TPOT

## Step 4: Search Space

Ask the smallest useful question first.

Dense model:
- ask whether TP-only search is acceptable
- if not, let the user specify explicit TP sizes

MoE model:
- ask whether to search TP only or TP plus EP
- only ask about MOE-DP if the user wants a broader MoE search

If the user does not know the values:
- use the CLI default-range behavior by passing the search flag without values

## Step 5: Quantization

Always ask:
- use recommended quantization defaults
- customize quantization

If customize:
- ask for linear quantization action
- ask for attention quantization action
- if linear is `MXFP4`, ask whether to keep group size 32

## Step 6: Prefix Cache And MTP

Ask these two questions for every throughput planning run after quantization and before other advanced options:

- whether prefix cache is enabled
- whether MTP is enabled

Prefix cache:
- If disabled, do not add `--prefix-cache-hit-rate`.
- If enabled, collect `--prefix-cache-hit-rate` and validate that it is in `[0, 1)`.
- In aggregation mode, explain that `--max-prefill-tokens` must not be smaller than the effective input length after applying prefix cache.

MTP:
- If disabled, do not add MTP parameters.
- If enabled, collect `--num-mtp-tokens` and confirm the model supports MTP.
- If enabled, collect `--mtp-acceptance-rate` assumptions. Label them as user-provided or heuristic values.
- Validate that `--num-mtp-tokens` is positive and acceptance rates are in `[0, 1]`.

## Step 7: Other Advanced Inputs

Ask only when relevant:
- multimodal image size
- batch range
- max prefill tokens
- compile toggle
- jobs

## Step 8: Pre-Execution Summary

Before running, summarize:
- deployment mode
- CLI mode and flags
- why this mapping was chosen
- hardware profiles and shared device count
- core sizing inputs
- quantization choices
- prefix cache and MTP choices
- search dimensions
- constraints
- whether the command is a single-hardware run or a cross-hardware comparison
- exact command

Then ask for explicit confirmation.

## Step 9: Post-Execution Summary

Summarize:
- best strategy
- per-hardware best strategy when multiple hardware profiles were evaluated
- cross-hardware comparison rows if present
- throughput
- TTFT
- TPOT
- concurrency
- batch size
- top candidates if available
- PD ratio fields if available
- disclaimer that results are for reference only
