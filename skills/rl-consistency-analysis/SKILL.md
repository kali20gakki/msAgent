---
name: rl-consistency-analysis
description: End-to-end root cause analysis for train vs rollout dump mismatches. Use when module mapping/value compare is not enough and you need to trace the first credible divergence boundary, filter out fused or structural false positives, follow producer-consumer chains, and generate a root-cause report with concrete hypotheses and evidence.
---

# Dump Root Cause Investigator

Use this skill when:
- `dump-module-mapping-value-compare` already ran, but the remaining suspects still need expert-style triage
- you need to distinguish real root causes from fused/unfused structural mismatches
- you need a report that explains the most likely divergence boundary, missing or extra ops, and implementation mismatches

## Workflow

1. Run the fixed script below. It will:
   - regenerate `output_1` to `output_4` by calling the sibling `dump-module-mapping-value-compare` skill
   - rank `module_priority_rank` 1 and 2 candidates
   - suppress obvious structural false positives such as fused QKV boundaries
   - inspect the previous aligned module in the same layer/block
   - compare train-only and rollout-only intermediate modules between the aligned boundary and the suspect
   - emit structured root-cause evidence in JSON
2. Read:
   - `output_5_root_cause_report.json` for machine-readable evidence
   - `references/report_template.md` for the final markdown structure
   - `references/manual_case_pattern.md` for the expert reasoning pattern when needed
3. The agent, not the script, must write the final `output_5_root_cause_report.md`:
   - use the template as a skeleton, not as fixed prose
   - explain why top suspects were kept or discarded
   - convert structured evidence into a task-specific narrative
   - state the most likely root cause and concrete next checks
4. If the evidence points to one side having an extra activation or fused op, verify that side’s implementation/config next.

## Analysis Rules

- Prefer candidates with `module_priority_rank` in `1, 2`
- Do not stop at the first high-rank mismatch; first rule out structural mismatches
- If a suspect’s previous aligned module is clean, and one side has an extra unmatched intermediate module before the suspect, treat that extra module as a high-confidence clue
- `output` mismatch matters more than internal mismatch
- If a module is mismatched but later neighboring modules realign, lower confidence unless there is a clear boundary explanation
- Activation-like unmatched modules are especially important:
  - `act_fn`
  - `silu`
  - `swiglu`
  - `gelu`
  - `mul`
  - `activation`
- Do not let the script author the final markdown narrative. The script should stay focused on deterministic evidence extraction, while the agent owns the final explanation.

## Fixed Command

```bash
python3 "<skill_root>/scripts/run_root_cause_analysis.py" \
  --train "<train_dump_json_path>" \
  --rollout "<rollout_dump_json_path>" \
  --out-dir "<output_dir>"
```

`<skill_root>` is the path to this skill directory.  
This skill expects the sibling directory `dump-module-mapping-value-compare/` to exist under the same parent directory.

## Outputs

- `output_1_key_mapping.*`
- `output_2_mapping.*`
- `output_3_value_compare.*`
- `output_4_module_analysis.*`
- `output_5_root_cause_report.json`
- `output_5_root_cause_report.md` (agent-authored from template, not script-authored)

## Interpretation Guide

- `structural_false_positive`
  - likely caused by fused/unfused implementation shape
- `missing_or_extra_op_between_aligned_boundary`
  - one side has unmatched intermediate modules between the last aligned boundary and the current suspect
- `parameter_or_checkpoint_issue`
  - parameters differ while inputs are aligned
- `in_module_impl_difference`
  - inputs and parameters align, but outputs differ
- `upstream_propagation`
  - parameters align, inputs already diverged

If the report identifies an extra rollout activation between an aligned upstream module and a mismatched downstream module, investigate whether train is missing the same activation or is using a different activation implementation/config.

## References

For the concrete reasoning pattern behind this skill, read:
- `references/manual_case_pattern.md`
- `references/report_template.md`
