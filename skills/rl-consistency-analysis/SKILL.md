---
name: rl-consistency-analysis
description: 训练与推理数据不一致的端到端根因分析。当模块映射/值比较不足以定位问题时使用，可追踪首个可信的分歧边界，过滤融合或结构性误报，遵循生产者-消费者链，并生成包含具体假设和证据的根因报告。
---

# 数据不一致根因分析器

使用此技能的场景：
- 训推模块不一致时需要进行模块匹配并比对
- 需要区分真正的根因与融合/非融合结构不匹配
- 需要一份解释最可能分歧边界、缺失或多余操作以及实现不匹配的报告

## 工作流程

1. 你是一个训推一致性的专家，知道不同训练和推理框架的层级映射关系，现在需要识别并获取训练和推理数据的key，将他们进行映射，生成`<output_dir>/output_0_key_mapping.json`，如果 `<output_dir>` 不存在，先创建，其中key是训练的key, value是推理的key
2. 运行下面的固定脚本。它将：
   - 调用 `scripts/run_root_cause_analysis.py` 脚本生成 `output_1` 到 `output_5`
   - 对 `module_priority_rank` 为 1 和 2 的候选进行排序
   - 抑制明显的结构性误报（如融合 QKV 边界）
   - 检查同一层/块中前一个对齐的模块
   - 比较对齐边界与可疑模块之间的训练侧和推理侧中间模块
   - 以 JSON 格式输出结构化根因证据
3. 读取：
   - `output_5_root_cause_report.json` 获取机器可读的证据
   - `references/report_template.md` 获取最终 Markdown 报告结构
   - `references/manual_case_pattern.md` 获取必要时的专家推理模式
4. 由 Agent（而非脚本）编写最终的 `output_5_root_cause_report.md`：
   - 使用模板作为框架，而非固定文本
   - 解释为什么保留或排除顶级可疑模块
   - 将结构化证据转换为特定任务的叙述
   - 陈述最可能的根因和具体的下一步检查
   - 在回复“报告已生成”或复用任何历史结论前，必须在当前轮次检查 `<output_dir>` 下的 `output_5_root_cause_report.md` 是否真实存在
   - 如果 `output_5_root_cause_report.md` 缺失，则任务不能视为完成：需要按需重新运行固定脚本，并在当前轮次重新写出该 Markdown 报告，不能仅依赖 thread 中的历史记忆
5. 如果证据表明某一侧有额外的激活或融合操作，请下一步验证该侧的实现/配置。

## 分析规则

- 优先处理 `module_priority_rank` 为 1 和 2 的候选
- 不要在第一个高排名不匹配处停止；首先排除结构性不匹配
- 如果可疑模块的前一个对齐模块是干净的，且某一侧在对齐边界和可疑模块之间有额外的未匹配中间模块，则将该额外模块视为高置信度线索
- 输出不匹配比内部不匹配更重要
- 如果某个模块不匹配但后续相邻模块重新对齐，则降低置信度，除非有明确的边界解释
- 激活类未匹配模块特别重要：
  - `act_fn`
  - `silu`
  - `swiglu`
  - `gelu`
  - `mul`
  - `activation`
- 不要让脚本编写最终的 Markdown 叙述。脚本应专注于确定性证据提取，而 Agent 负责最终解释。
- 同样的提示词在同一个 thread 中再次出现时，不能默认上一次产物仍然有效；必须以当前文件系统中的实际产物是否存在为准。

## 固定命令

```bash
python3 "<skill_root>/scripts/run_root_cause_analysis.py" \
  --train "<train_dump_json_path>" \
  --rollout "<rollout_dump_json_path>" \
  --mapping_key "<mapping_key_json_path>" \
  --out-dir "<output_dir>"
```

`<skill_root>` 是此技能目录的路径。

## 输出文件

- `output_0_key_mapping.json`
- `output_1_key_mapping.*`
- `output_2_mapping.*`
- `output_3_value_compare.*`
- `output_4_module_analysis.*`
- `output_5_root_cause_report.json`
- `output_5_root_cause_report.md`（由 Agent 基于模板编写，非脚本生成）

`output_5_root_cause_report.md` 是任务完成的必需交付物。如果回答时该文件不存在，Agent 必须先补生成或重写，再声明“报告已保存”。

## 解读指南

- `structural_false_positive`（结构性误报）
  - 可能由融合/非融合实现形态引起
- `missing_or_extra_op_between_aligned_boundary`（对齐边界之间存在缺失或多余操作）
  - 某一侧在最后对齐边界和当前可疑模块之间有未匹配的中间模块
- `parameter_or_checkpoint_issue`（参数或检查点问题）
  - 参数不同但输入对齐
- `in_module_impl_difference`（模块内部实现差异）
  - 输入和参数对齐，但输出不同
- `upstream_propagation`（上游传播）
  - 参数对齐，但输入已发散

如果报告识别出在对齐的上游模块和不匹配的下游模块之间有额外的推理侧激活，请调查训练侧是否缺少相同的激活，或者是否使用了不同的激活实现/配置。

## 参考资料

有关此技能背后的具体推理模式，请阅读：
- `references/manual_case_pattern.md`
- `references/report_template.md`
