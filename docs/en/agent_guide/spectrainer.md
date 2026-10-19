# SpecTrainer On-Policy Resampling for Speculative Decoding

`SpecTrainer` is an agent for **training data preparation** in speculative decoding scenarios. It discards each assistant response in the existing multi-turn conversation data and uses a verifier model (a vLLM service) to **regenerate** responses round by round (on-policy, called "resampling" in this framework), producing **pretokenized samples** that can be fed directly into training (each line contains `input_ids` and `loss_mask`). You describe your goal in natural language, and the agent orchestrates the entire process of "data pre-check → input normalization → regeneration invocation → artifact validation".

> **Relationship to open-source speculators**: The `scripts/` directory of this agent skill is an **orchestration wrapper around the scripts in the open-source speculators repository** (it only assembles parameters and makes calls, and does not modify upstream source code). The version baseline is **speculators v0.6.0** ([vllm-project/speculators](https://github.com/vllm-project/speculators)). If the upstream script interface changes, you must update the call parameters of the wrapper scripts accordingly.

## 1. Agent Positioning

- Handles only the **data resampling and regeneration** part. It does not start the verifier service, does not perform training, and does not perform acceptance testing on the acceptance rate or precision.
- The input can be raw multi-turn conversation JSONL, or JSONL that already conforms to the regeneration format.
- It confirms key decisions with you first (endpoint, data, number of samples, concurrency, sampling parameters, and output directory) and does not claim resources on its own.

## 2. Core Capabilities

- **Pre-check**: Runs a DFX pre-check on raw multi-turn conversations and reports the results by severity level (high, medium, or low). For high-severity risks, it stops first and confirms with you.
- **Normalization**: Converts raw conversations into the format required for regeneration: `{id, conversations:[{from,value}]}`.
- **Regeneration**: Forwards the request to the `response_regeneration` script in the speculators repository and uses the verifier service to regenerate assistant responses round by round (on-policy).
- **Delivery**: Pretokenized JSONL (each line contains `id`, `primary_id`, `input_ids`, `loss_mask`, and `text`), an error-line file, and ok/errors/truncated statistics.

## 3. Getting Started

**Just state your goal in natural language**, and replace `path/to/…` with the actual path on your machine. For example:

| What You Want to Do | Example Prompt |
| --- | --- |
| Resampling (minimal usage) | `Resample path/to/conversations.jsonl with path/to/Qwen3-8B` |
| Specify the number of samples and concurrency | `Resample this dataset, run only 500 samples with a concurrency of 8` |
| Check the data quality first | `First check whether path/to/raw.jsonl can be used directly for resampling` |
| Specify the output directory | `Resample this data with this model and put the artifacts in path/to/out` |

When key information is missing, the agent asks you item by item (it does not guess or force a run). **You are advised to think through the following in advance**:

| Information | Description | Default Behavior |
| --- | --- | --- |
| verifier service endpoint | An OpenAI-compatible `/v1/chat/completions` endpoint that must return `prompt_token_ids` and `choices[0].token_ids` | The agent asks you (this agent does not start the service for you) |
| Model | Local path of the verifier model (passed as `--model`) | The agent asks you |
| Data | Raw multi-turn conversation JSONL, or JSONL that already conforms to the `{id, conversations}` format | The agent asks you |
| Number of samples/concurrency | How many samples to run this time (you can run a small smoke test first) and the concurrency level | The agent asks you |
| Sampling parameters | Optional, such as a JSON snippet with `temperature` and other settings | Upstream defaults apply if you do not provide them |
| Output directory | Unified location for artifacts and logs | The agent asks you |

## 4. How the Process Works

1. **Alignment**: The agent restates your endpoint, data, number of samples, and output directory. It asks about anything missing and corrects anything wrong.
2. **Pre-check** (for raw conversation data): It runs a DFX pre-check and produces a report. For high-severity risks, it stops first and confirms with you.
3. **Normalization**: When needed, it converts raw conversations into the regeneration input format while keeping the row order and content unchanged.
4. **Service validation**: It first checks the health of `/v1/models`, then sends a minimal chat request to verify that `return_token_ids` takes effect. If validation fails, the process does not proceed to the next step.
5. **Regeneration**: It runs according to your number of samples and concurrency. For large batches, it runs the job in the background and polls the logs.
6. **Validation and delivery**: It validates the consistency of artifact fields and lengths and reports the ok/errors/truncated counts and the artifact and log paths.

You only need to make decisions at key points such as data risks and parameter confirmation. The agent handles the rest and reports the artifact path of each step.

## 5. Prerequisites

- An Ascend inference environment and available NPUs (**you start the service yourself**).
- The speculators repository (v0.6.0, which provides the regeneration script).
- A ready verifier vLLM service endpoint that supports returning token IDs.
- The multi-turn conversation data to be processed.
- If any item is missing, the agent prompts you to provide it before it starts.

## 6. Deliverables

- Pretokenized sample JSONL (`input_ids` and `loss_mask`, ready for training).
- An error-line file (for example, `<outfile>.errors.jsonl`) and logs.
- If a pre-check was performed, the DFX pre-check report and the severity-level conclusion.
- Execution parameters and statistics (endpoint, number of samples, concurrency, and ok/errors/truncated).

## 7. Usage Notes

- The agent does not perform regeneration until the service passes `return_token_ids` validation.
- It does not start the service for you and does not claim devices on its own. Large batch jobs run in the background with polling, and no progress in the logs does not mean the job is stuck.
- It only performs regeneration. It does not change the content or order of the input data, and it does not perform training or acceptance testing.
- If the upstream speculators interface is upgraded (to a version higher than v0.6.0), the wrapper scripts may need corresponding parameter adjustments.
