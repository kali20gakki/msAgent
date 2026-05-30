---
name: msmodelslim-adapter-verification
description: 为 msModelSlim 适配器执行功能性验证。适用于基础适配器开发完成后，自动执行四步验证（测试模型、全回退量化、权重一致性与可加载/保存、实际量化规则校验）并输出通过/失败结论。
---

# msModelSlim 适配器功能性验证 Skill

用于在基础适配器开发完成后，自动帮助用户进行功能性验证。

## 触发条件

- `msmodelslim-model-adapt` 已完成适配器开发与注册安装。
- 用户希望确认适配器是否可用，或要求执行标准验证流程。

## 执行要求

- 必须按顺序执行四步验证，不可跳步。
- 每一步失败都要立即停止并返回失败原因与下一步修复建议。
- 仅当四步全部通过时，返回“功能性验证通过”。

## 四步验证流程

1. Step1：生成随机权重测试模型。
2. Step2：执行全回退量化，验证流程与注册生效。
3. Step3：验证 Step2 与 Step1 的权重严格一致，且产物可完整加载/保存。
4. Step4：执行实际量化（W8A8 静态/动态）并校验描述文件规则。

## Buffer 权重说明（Step3 常见问题）

- 若 Step3 出现“全回退权重缺失/键不一致”，需优先检查缺失项是否来自模型 `buffer`。
- `msmodelslim` 通常不会保存 `buffer` 类型权重，因此可能导致全回退产物缺少对应键。
- 适配器需要主动将这类关键 `buffer` 转为 `nn.Parameter`，以确保量化导出和一致性校验可覆盖该权重。

## 自动化脚本

- `scripts/step1_generate_test_model.py`
- `scripts/step2_run_quantization.py`
- `scripts/step3_verify_weights.py`
- `scripts/step4_verify_quant_description.py`

## 参考资料

- [适配器验证指南](references/verification_guide.md)

## 输出格式要求

- 给出每一步的执行结果（PASS/FAIL）。
- 若失败，标注失败步骤、错误要点、建议修复方向。
- 最后给出总结结论：通过 / 未通过。
