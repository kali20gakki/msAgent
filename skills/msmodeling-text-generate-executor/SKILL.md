---
name: msmodeling-text-generate-executor
description: Interactively gather parameters for `python -m cli.inference.text_generate`, generate a concrete text generation simulation command, explain assumptions, ask for execution confirmation, then run the command and summarize performance metrics. Use when the user wants to validate a specific model, hardware profile, batch/query size, prefill or decode scenario, TP/DP/EP/MOE parallel strategy, profiling database, chrome trace, empirical metrics export, or a best row produced by throughput_optimizer.
version: 0.1.0
source: local-session-analysis
---

# Text Generate Executor

Use this skill to turn a text generation validation request into a confirmed `text_generate` run and concise performance summary.

## Workflow

1. Read the local `text_generate` implementation or `--help` output when the repository is available and parameter behavior needs verification.
2. Ask only for missing inputs. Keep the dialog progressive; do not dump the full parameter list at once.
3. Build a candidate command and show a short parameter summary.
4. After the candidate command is ready, always explain optional Debug/Trace switches and ask whether to add any of them.
5. Highlight validation issues or assumptions before execution.
6. Ask for explicit confirmation before running the command.
7. Run the command from the repository root when the repo is available.
8. Parse and summarize the result.

## Context And Suggestion Hard Rules

- If the user only provides a model and scenario, do not generate a full command yet. Ask for missing core inputs first.
- Do not reuse parameters from previous conversation turns unless the user explicitly says to reuse, continue, or keep the previous configuration.
- Do not claim a baseline comes from regression coverage, repository defaults, or local examples unless you inspect the concrete local file or command output in the current turn and cite that source in the summary.
- Provide suggested values only after the user says they are unsure or asks for a recommendation.
- When suggesting values, label each value source as one of: CLI default, user-provided previous config, inspected local test/example, documented example, throughput_optimizer result, or explicit heuristic.
- If a suggested value is an explicit heuristic, say it is a heuristic and explain the practical reason briefly.

## Question Flow

Follow the branching rules in `references/dialog-flow.md`.

Always collect these core inputs first:
- model id
- device profile
- total device count
- number of queries / batch size
- query length
- execution mode:
  - Prefill only -> do not add `--decode`; ask whether `--context-length` should stay `0`
  - Decode mode -> add `--decode`; require `--context-length`
  - Optimizer best-row validation -> map optimizer row fields to explicit `text_generate` parameters

Then branch:
- For Decode mode, add `--decode` and ask for `--context-length` if missing.
- For optimizer result validation, ask for TP/DP/EP/MOE-DP strategy, input/output length mapping, and whether `--num-queries` should match optimizer batch/concurrency.
- For MoE models, ask whether to set `--ep-size`, `--moe-tp-size`, `--moe-dp-size`, redundant experts, shared expert TP, or external shared experts.
- For multimodal models, ask for `--image-batch-size`, `--image-height`, and `--image-width`.
- For every text generation run, ask whether MTP is enabled. If enabled, ask for `--num-mtp-tokens` and confirm the model supports MTP.
- For Prefill-only mode, ask whether to set `--prefix-cache-hit-rate`. Do not ask for prefix cache in Decode mode.
- For profiling mode, require `--performance-model profiling` and `--profiling-database`.
- After collecting required parameters and generating the candidate command, always tell the user about optional Debug/Trace switches before execution:
  - `--chrome-trace`: export a Chrome trace file for timeline inspection.
  - `--graph-log-url`: dump compiled graph logs; requires `--compile` and may require `pydot`.
  - `--dump-input-shapes`: include input shape grouping in table averages for shape-level debugging.
- Ask whether to add any Debug/Trace switches even when the user did not initially request tracing, because users may not know these options exist.
- If the user chooses a Debug/Trace switch, update the command and summary before asking for final execution confirmation.

## Defaults And Confirmation Rules

Use the parameter guidance in `references/text-generate-params.md`.

Apply these rules:
- Use analytic performance modeling unless the user asks for profiling.
- Add `--compile` by default for realistic compiled-path simulation. Show this default explicitly and ask whether the user wants to disable it before execution.
- Do not silently choose custom quantization settings.
- The recommended quantization defaults are:
  - `--quantize-linear-action W8A8_DYNAMIC`
  - `--quantize-attention-action DISABLED`
- If the user chooses custom quantization, ask for `quantize-linear-action` and `quantize-attention-action` explicitly.
- If `quantize-linear-action` is `MXFP4`, ask whether to keep `--mxfp4-group-size 32` or provide a custom group size.
- Keep advanced layer-specific parallel overrides unset unless the user provides them or they come from an optimizer best row.
- Keep `--remote-source huggingface` unless the user asks for `modelscope`.

## Validation Rules

Before execution, check these conditions and call them out clearly:
- `--num-queries`, `--query-length`, `--num-devices`, and explicit parallel sizes must be positive integers.
- `--context-length` must be non-negative.
- `--prefix-cache-hit-rate` must be in `[0, 1)`.
- Explicit TP, DP, EP, MOE-TP, and MOE-DP sizes should be compatible with `--num-devices`.
- In optimizer best-row validation, explain that this command validates one fixed candidate and does not search.
- `--performance-model profiling` requires `--profiling-database`.
- `--export-empirical-metrics` requires `--performance-model profiling`.
- If any image option is supplied, verify the multimodal image batch size, height, and width are all present.
- `--num-mtp-tokens` should only be used for models with MTP capability.

## Execution Pattern

When the user confirms execution:
1. Print the exact command you will run.
2. Run it from the repository root when the repo is available.
3. Capture combined stdout and stderr.
4. Extract key metrics and table rows from the output when available.
5. Return a concise summary with:
- executed command
- model and device
- mode: Prefill only or Decode
- batch/query/context settings
- parallel strategy
- performance model: analytic or profiling
- latency, throughput, memory, and breakdown metrics when present
- trace, graph log, or exported empirical metrics path when requested

## Result Summary Requirements

After a successful run, always include:
- the executed command
- a short scenario summary
- the key performance metrics available in stdout
- a note when the run was based on a throughput_optimizer best row
- this disclaimer, or a very close paraphrase:
  `These results come from text_generate simulation and are for validation/planning reference only. Actual performance depends on runtime, topology, software stack, model implementation, and real traffic patterns, so validate with real workload testing before final deployment decisions.`

## Failure Handling

If execution fails:
- show the command
- show the important error lines
- explain whether the failure is due to invalid parameters, environment issues, model loading, dependency setup, device profile constraints, or profiling database assumptions
- propose the smallest useful correction instead of restating the full questionnaire

## Handoff

If the user wants to search for the best deployment strategy, use the `msmodeling-throughput-optimizer-executor` skill.
If the user asks whether optimizer results are reasonable, why hardware results differ, or how a best row maps to text_generate validation, use the `throughput-optimizer-explainer` skill.
