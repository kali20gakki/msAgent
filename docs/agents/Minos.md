# Minos

<p align="center">
  <img src="../images/Minos.png" alt="Minos" width="220">
</p>

`Minos` 是面向文档体验与代码审查辅助场景的 Agent，负责从首次接触项目的新手视角审查 README、安装文档、Quick Start 和 onboarding 路径，也负责结合仓库上下文审查 GitCode PR 的改动风险与修改质量。

## Agent 定位

- 面向 README 走查、安装流程验证、新手上手体验审查，以及 GitCode PR 审查
- 聚焦文档可用性、信息缺口、隐含前提、阻塞点识别，以及 PR 的回归风险与改动完整性
- 适合输出面向维护者的文档改进建议、代码审查结论和优先级清单

## 核心能力

- 按 README / 安装文档真实执行并记录阻塞点
- 审查 Quick Start 是否足够清晰、可复现、对新手友好
- 区分文档问题、环境前提缺失和项目自身运行问题
- 结合 PR metadata、diff、调用链、基线实现和测试上下文审查 GitCode PR
- 输出基于执行证据的文档整改建议，以及带严重程度和修改建议的代码审查意见

## 推荐使用方式

- 文档体验任务可直接给出仓库路径、README 或目标文档路径
- 明确希望审查的范围，例如安装、启动、验证、排障、全链路 onboarding，或 GitCode PR 风险评审
- 如果有特定用户群体或审查目标，也可以说明，例如“面向第一次接触 Ascend 的开发者”或“重点看兼容性与测试覆盖”

## 典型使用场景

| 场景 | 示例提示词 | 输出说明 |
|---|---|---|
| README 走查 | `请像第一次接触项目的用户一样，按 README 完整走一遍，并指出会卡住的地方。` | 输出文档可用性结论、阻塞点、证据和改进建议。 |
| 安装流程验证 | `请按安装文档执行一遍这个仓库的安装步骤，看看新手能不能顺利跑通。` | 输出安装链路中的缺失前提、环境假设和修正文案建议。 |
| Quick Start 审查 | `请检查 quick start 是否足够让新人在最短时间内跑起来，并指出哪里还不够清晰。` | 输出上手路径、理解成本和推荐的优化顺序。 |
| Onboarding 体验评估 | `请从 onboarding 角度评估这个仓库的文档体验，并给出一份可落地的改进清单。` | 输出面向维护者的文档整改建议和优先级。 |
| GitCode PR 审查 | `请 review 这个 GitCode PR，重点看行为回归、兼容性和测试覆盖。` | 输出带严重程度、证据、原因和修改建议的审查结论。 |
| GitCode PR 评论整理 | `请检查这个 GitCode PR 链接，把需要修改的问题整理成可发布的审查意见。` | 输出可直接发布到 PR 的逐项评论建议。 |

更多说明可参考 [`document-ux-review`](../document_ux_review.md) 与 `resources/configs/default/skills/gitcode-code-reviewer/SKILL.md`。
