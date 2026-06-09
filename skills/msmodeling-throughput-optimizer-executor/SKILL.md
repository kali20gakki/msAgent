---
name: msmodeling-throughput-optimizer-executor
description: Interactively gather parameters for `python -m cli.inference.throughput_optimizer`, generate a deployment simulation command, explain assumptions, ask for execution confirmation, then run the simulation and summarize the best parallel strategy. Use when the user wants to evaluate a model on one or more hardware profiles, compare hardware targets, compare aggregation versus disaggregation versus PD ratio optimization, choose TP/EP/MOE-DP search ranges, or obtain recommended concurrency, throughput, TTFT, and TPOT results from throughput modeling.
version: 0.1.0
source: local-session-analysis
---

# Throughput Optimizer Executor

Use this skill to turn a vague throughput-planning request into a confirmed `throughput_optimizer` run and a concise result summary.

## Workflow

1. Read the local `throughput_optimizer` implementation or docs if the current repository is available and you need to verify parameter behavior.
2. Ask only for missing inputs. Keep the dialog progressive; do not dump the full parameter list at once.
3. Build a candidate command and show a short parameter summary.
4. Highlight validation issues or assumptions before execution.
5. Ask for explicit confirmation before running the command.
6. Run the command.
7. Parse and summarize the result, then add the standard disclaimer that the output is only a simulation reference.

## Context And Suggestion Hard Rules

- If the user only provides a model and planning scenario, do not generate a full command yet. Ask for missing core inputs first.
- Do not reuse parameters from previous conversation turns unless the user explicitly says to reuse, continue, or keep the previous configuration.
- Do not claim a baseline or search plan comes from regression coverage, repository defaults, or local examples unless you inspect the concrete local file or command output in the current turn and cite that source in the summary.
- When core inputs are missing, ask progressive questions first.
- Provide suggested values only after the user says they are unsure or asks for a recommendation.
- When suggesting values, label each value source as one of: CLI default, user-provided previous config, inspected local test/example, documented example, throughput_optimizer result, or explicit heuristic.
- If a suggested value is an explicit heuristic, say it is a heuristic and explain the practical reason briefly.

## Question Flow

Follow the branching rules in `references/dialog-flow.md`.

Always collect these core inputs first:
- model id
- device profile or profiles
- total device count when relevant; the same count applies to every provided hardware profile
- input length
- output length
- deployment mode:
  - PD混部 / PD聚合 / 聚合部署 -> aggregation
  - PD分离 -> ask whether it is phase capability evaluation or P/D instance ratio planning
  - PD ratio / P/D配比 / 实例配比 -> P/D instance ratio planning
- SLO target: TTFT, TPOT, or both

Then branch:
- For PD混部, PD聚合, or 聚合部署, use aggregation mode directly.
- For PD分离, ask whether the goal is:
  1. separately evaluate Prefill and Decode phase capability, or
  2. specify P/D devices per instance and calculate the matched ratio.
- For PD分离 phase capability evaluation, use `--disagg`; ask whether to run Prefill only, Decode only, or both.
- For PD ratio / P/D instance ratio planning, use `--enable-optimize-prefill-decode-ratio`; require prefill and decode devices per instance.
- For multimodal models, ask for image height and width.
- For MoE models, ask whether to search only TP or also EP and MOE-DP.
- Ask whether prefix cache is enabled. If enabled, ask for `--prefix-cache-hit-rate`.
- Ask whether MTP is enabled. If enabled, ask for `--num-mtp-tokens`, `--mtp-acceptance-rate`, and confirm the model supports MTP.

## Defaults And Confirmation Rules

Use the parameter guidance in `references/throughput-optimizer-params.md`.

Apply these rules:
- Do not silently choose quantization settings.
- Ask whether to use the recommended quantization defaults or customize them.
- The recommended quantization defaults are:
  - `--quantize-linear-action W8A8_DYNAMIC`
  - `--quantize-attention-action DISABLED`
- If the user chooses custom quantization, ask for `quantize-linear-action` and `quantize-attention-action` explicitly.
- If `quantize-linear-action` is `MXFP4`, ask whether to keep `--mxfp4-group-size 32` or provide a custom group size.
- `--compile` should be recommended as on, but can still be changed if the user asks.
- Keep prefix cache disabled unless the user enables it. If enabled, require an explicit hit rate.
- Keep MTP disabled unless the user enables it. If enabled, require token count and acceptance rate assumptions.
- If the user does not know the parallel search space, prefer a conservative search plan:
  - dense models: TP search only
  - MoE models: TP search first, then offer EP and MOE-DP as optional expansion
- If no search dimension is selected by the user, default to TP search behavior consistent with the script.

## Validation Rules

Before execution, check these conditions and call them out clearly:
- `--device` accepts one or more hardware profiles. Multiple profiles should be passed as space-separated values, for example `--device A B C`, and the run will produce cross-hardware summaries.
- In multi-hardware runs, one shared `--num-devices` is used for every hardware profile; verify that every profile can model that device count when the repository exposes profile metadata or validation errors.
- `--enable-optimize-prefill-decode-ratio` cannot be combined with `--disagg`.
- PD ratio mode requires both `--prefill-devices-per-instance` and `--decode-devices-per-instance`.
- Any explicit TP, EP, or MOE-DP size must not exceed `--num-devices` in the relevant phase.
- `--batch-range` must be `[max]` or `[min max]` with positive integers and `min <= max`.
- `--prefix-cache-hit-rate` must be in `[0, 1)`.
- `--num-mtp-tokens` must be a positive integer when MTP is enabled.
- `--mtp-acceptance-rate` values must be in `[0, 1]` and should reflect user-provided or explicitly labeled assumptions.
- In aggregation mode, `--max-prefill-tokens` must not be smaller than the effective input length after applying prefix cache.
- If the user provides both TTFT and TPOT in disaggregation mode, explain that the tool will run both Prefill and Decode optimization separately.

## Execution Pattern

When the user confirms execution:
1. Print the exact command you will run.
2. Run it from the repository root when the repo is available.
3. Capture combined stdout and stderr.
4. If needed, use `scripts/extract_throughput_optimizer_result.py` to convert the raw output into structured JSON.
5. Return a concise summary with:
- mode
- best parallel strategy
- batch size
- concurrency
- throughput
- TTFT
- TPOT
- cross-hardware comparison rows when multiple hardware profiles are present
- top candidate rows when available
- PD ratio details when available

## Result Summary Requirements

After a successful run, always include:
- the executed command
- a short best-result summary
- a cross-hardware comparison summary when multiple hardware profiles were evaluated
- a compact top-candidates summary when the table is present
- this disclaimer, or a very close paraphrase:
  `These results come from throughput modeling and are for deployment planning reference only. Actual performance depends on runtime, topology, software stack, and real traffic patterns, so validate with real workload testing before final deployment decisions.`

## Failure Handling

If execution fails:
- show the command
- show the important error lines
- explain whether the failure is due to invalid parameters, environment issues, or model/device assumptions
- propose the smallest useful correction instead of restating the full questionnaire

## Result Explanation Handoff

If the user asks whether optimizer results are reasonable, why two hardware targets differ, how Cube/Vec/Comm/Mem explains the result, or how a best row maps to text_generate validation, use the `throughput-optimizer-explainer` skill instead of expanding this execution skill.
