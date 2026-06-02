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
- 修改代码后要执行 `bash install.sh` 重新安装msModelslim
- 若验证过程中出现模型实现文件（如权重目录内 `modeling_*.py`）报错，必须先判断是否为 `transformers` 版本不契合导致。
- 对疑似版本不契合问题，必须先告知用户并确认版本需求（目标版本或可接受版本区间）；未确认前不得切换版本。
- 仅在用户确认后，才可执行 `transformers` 版本切换与重试验证。

## 四步验证流程

1. Step1：生成随机权重测试模型。
2. Step2：执行全回退量化，验证流程与注册生效。
3. Step3：验证 Step2 与 Step1 的权重严格一致，且产物可完整加载/保存。
4. Step4：执行实际量化（W8A8 静态/动态）并校验描述文件规则。

## Buffer 权重说明（Step3 常见问题）

- 若 Step3 出现“全回退权重缺失/键不一致”，需优先检查缺失项是否来自模型 `buffer`。
- `msmodelslim` 通常不会保存 `buffer` 类型权重，因此可能导致全回退产物缺少对应键。
- 适配器需要主动将这类关键 `buffer` 转为 `nn.Parameter`，以确保量化导出和一致性校验可覆盖该权重。

## transformers 版本兼容处理（验证期）

- 触发条件：验证阶段出现模型实现文件导入/模型forward错误，且报错指向 `transformers` API 变更、缺失符号或签名不匹配。
- 必做沟通：向用户说明“当前报错疑似版本兼容问题”、给出关键报错摘要、请求确认目标版本策略（指定版本或版本区间）。
- 搜索策略：获用户确认后，使用二分法在确认范围内搜索可用 `transformers` 版本（每次切换版本后需重装并重跑触发失败的验证步骤）。
- 收敛标准：找到“可成功加载并通过对应验证步骤”的版本后停止搜索，并将最终版本写入msModelslim的config.ini。
- 失败处理：若二分搜索后仍无可用版本，返回阻塞结论并要求用户提供官方建议版本或模型实现修订方案。

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
