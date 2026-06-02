# 专家分析模式

此技能编码了以下专家工作流程：

1. 从 `output_4_module_analysis.csv` 开始
2. 优先处理 `module_priority_rank` 为 1 和 2 的候选
3. 首先排除融合或映射伪影引起的可疑模块
4. 对于每个真实可疑模块，在同一层/块中找到前一个对齐的模块
5. 比较两侧在对齐边界和可疑模块之间的中间模块
6. 如果某一侧有额外的未匹配激活类模块，则将其作为主要线索
7. 将不匹配解释为以下之一：
   - 多余或缺失的操作
   - 实现不匹配
   - 参数同步问题
   - 上游传播

典型示例：

- 上游对齐模块：
  - 推理侧 `Module.model.layers.0.mlp.gate_up_proj...`
  - 训练侧 `Module.module.module.decoder.layers.0.mlp.linear_fc1...`
- 下游不匹配模块：
  - 推理侧 `Module.model.layers.0.mlp.down_proj...`
  - 训练侧 `Module.module.module.decoder.layers.0.mlp.linear_fc2...`
- 推理侧有一个额外的未匹配中间模块：
  - `Module.model.layers.0.mlp.act_fn.AscendSiluAndMul.forward.0`

解读：

- 真正的问题不是下游的行并行线性层本身
- 首个可信的分歧边界是推理侧的额外激活路径
- 根因很可能是激活实现或配置不匹配

在参考案例中，专家结论是：

- 推理侧使用 `vllm_ascend` 融合的 `swiglu`
- 训练侧使用 Megatron 侧的 `megatrongelu`
- 配置意图使用 `swiglu`，但桥接参数未正确激活它