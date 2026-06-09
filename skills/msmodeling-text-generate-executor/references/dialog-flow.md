# Text Generate Dialog Flow

Use progressive questioning. Ask at most three short questions at a time and prefer defaults only when they are safe and visible.

## Core Flow

Do not build a complete command from only a model name and scenario. If core inputs are missing, ask progressive questions first.

1. Ask for missing core inputs:
   - model id
   - device profile
   - total device count
   - number of queries / batch size
   - query length
   - Prefill only, Decode, or optimizer best-row validation
2. If the user says Prefill only:
   - do not add `--decode`
   - ask whether `--context-length` stays `0` if the user has not provided it
3. If the user says Decode:
   - add `--decode`
   - require `--context-length`
4. If the user provides a throughput_optimizer row:
   - map model, device, input length, output/decode length, batch/concurrency, TP, DP, EP, MOE-TP, and MOE-DP when present
   - state that `text_generate` validates a fixed candidate rather than performing search
5. Ask whether MTP is enabled for this run, regardless of model family.
   - If enabled, collect `--num-mtp-tokens` and confirm model support.
   - In Decode mode with MTP enabled, set `--query-length` to `1 + --num-mtp-tokens`.
6. If the run is Prefill-only, ask whether to set `--prefix-cache-hit-rate`.
   - Do not ask for prefix cache in Decode mode.
7. Ask whether to use recommended quantization defaults or customize them.
8. State that `--compile` is enabled by default and ask only whether the user wants to disable it.
9. Show the candidate command, then always explain optional Debug/Trace switches and ask whether to add any before final confirmation:
   - `--chrome-trace`: export a Chrome trace file for timeline inspection.
   - `--graph-log-url`: dump compiled graph logs; requires `--compile` and may require `pydot`.
   - `--dump-input-shapes`: include input shape grouping in table averages for shape-level debugging.
10. If the user chooses any Debug/Trace switches, update the command and parameter summary.
11. Ask for final execution confirmation.

## Context Reuse And Suggestions

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

## Optional Branches

- MoE: collect EP, MOE-TP, MOE-DP, shared expert, external shared expert, and redundant expert options only when relevant.
- Multimodal: collect image batch size, height, and width together.
- Profiling: collect `--performance-model profiling` and `--profiling-database`.
- Debug: always present `--chrome-trace`, `--graph-log-url`, and `--dump-input-shapes` after candidate command generation; collect `--num-hidden-layers-override` only when the user explicitly asks for hidden-layer debugging.
- Empirical export: collect `--export-empirical-metrics` and ensure profiling mode is enabled.

## Confirmation Format

Before running, present:
- Scenario: model, device, mode, query/context sizes
- Parallelism: TP/DP/EP/MOE settings
- Performance model: analytic or profiling
- Quantization: linear and attention actions
- Optional Debug/Trace choices: selected switches, or `none`
- Command: exact command to execute
- Question: ask for explicit confirmation
