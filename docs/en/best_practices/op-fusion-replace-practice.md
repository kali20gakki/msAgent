# Best Practices for Model Fusion Operator Identification and Replacement

Taking the deepseek-v4 model code in `transformers` as an example, this section shows how to complete the identification and replacement verification of common fusion operators end-to-end using the `ascend-fusion-op-replace` skill of msagent.

## 1. Demonstration

### 1.1 Evaluating Code Snippets Eligible for Fusion Operators

1. In msagent, select the `ascend-fusion-op-replace` skill and enter the prompt: "Determine which operators in the deepseek-v4 model code can be replaced with `torch_npu` fusion operators".
2. msagent reads and follows the guidance of the `ascend-fusion-op-replace` skill and confirms that it adopts a two-stage process: first output the identification report, then modify and verify the code after you confirm.
3. Locate the model implementation of DeepSeek-V4 and focus on the core computation logic in `src/transformers/models/deepseek_v4/modular_deepseek_v4.py`.
4. Scan the key computation paths such as RMSNorm, RoPE, Attention, SwiGLU, and MoE, and use the API query script provided by the skill to confirm one by one the actual fusion APIs provided by `torch_npu` and their usage constraints.
5. Based on the scan and query results, determine the feasibility of fusion replacement for each candidate operator, output the Stage 1 identification report, and wait for your confirmation.

![Example of fusion operator search](../figures/best_practices/dsv4-fusion-op-find-process.gif)

### 1.2 Verifying Accuracy and Performance After Replacement

1. Based on your feedback, prioritize the fusion replacement of the high-priority operator RmsNorm.
2. First, launch the original code without any modification, run 50 steps, record the loss and duration of each iteration, and save them to `baseline.jsonl`.
3. Integrate the RmsNorm fusion operator implementation into the code, control the fusion switch through a configuration parameter, and retain the original branch logic to support rollback comparison.
4. Launch the code with the fusion operator integrated, run 50 steps as well, record the loss and duration of each iteration, and save them to `rmsnorm.jsonl`.
5. Compare the loss curves and step duration before and after the integration, and ensure that the accuracy passes validation after fusion and that the step duration is consistently lower than the baseline.

![Example of fusion operator validation](../figures/best_practices/dsv4-fusion-op-valid-process.gif)

### 1.3 Generating Validation Result HTML

![Example of fusion operator HTML](../figures/best_practices/dsv4-fusion-op-report-html.gif)
