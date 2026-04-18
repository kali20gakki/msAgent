# Manual Case Pattern

This skill encodes the following expert workflow:

1. Start from `output_4_module_analysis.csv`
2. Prioritize `module_priority_rank` 1 and 2
3. Exclude fused or mapping-artifact suspects first
4. For each real suspect, find the previous aligned module in the same layer/block
5. Compare intermediate modules on both sides between the aligned boundary and the suspect
6. If one side has an extra unmatched activation-like module, treat that as the main clue
7. Explain the mismatch as one of:
   - extra or missing op
   - implementation mismatch
   - parameter sync problem
   - upstream propagation

Canonical example:

- Upstream aligned module:
  - rollout `Module.model.layers.0.mlp.gate_up_proj...`
  - train `Module.module.module.decoder.layers.0.mlp.linear_fc1...`
- Downstream mismatched module:
  - rollout `Module.model.layers.0.mlp.down_proj...`
  - train `Module.module.module.decoder.layers.0.mlp.linear_fc2...`
- Rollout has an extra unmatched intermediate module:
  - `Module.model.layers.0.mlp.act_fn.AscendSiluAndMul.forward.0`

Interpretation:

- the real issue is not the downstream row-parallel linear itself
- the first credible divergence boundary is the extra activation path on rollout
- root cause is likely activation implementation or config mismatch

In the reference case, the expert conclusion was:

- rollout used `vllm_ascend` fused `swiglu`
- train used Megatron-side `megatrongelu`
- config intended `swiglu`, but bridge parameters did not activate it correctly
