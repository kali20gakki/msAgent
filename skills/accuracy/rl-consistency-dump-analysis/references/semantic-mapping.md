# 语义边界映射方法论(架构无关)

比对的前提是**两侧每个被比的张量在数学上是同一个东西**。训练侧与推理侧算子命名差异极大,同名比对没有意义;必须按语义边界对齐,不按名字。

## 什么是语义边界

语义边界 = 数学上同一位置、可在两侧独立计算得到的张量。常用边界清单(沿数据流):

```text
embedding 输出 / embedding 权重
  → 各层输入归一化输出(GEMM 前)
  → Q/K/V 下投影 GEMM 输出(注意推理侧可能融合成一段,切片对齐)
  → q/kv norm 输出(可能在投影里融合)
  → q/kv 上投影 GEMM 输出
  → attention 输出(reshape 前 [seq, heads, head_dim] / reshape 后 [seq, hidden] 任选一致口径)
  → o_proj 输出
  → 残差加和输出(post-attention)
  → MLP fc1/gate+up 输出 / act(silu/swiglu)输出 / fc2/down 输出
  → 残差加和输出(layer 输出)
  → 最终归一化输出
  → logits(注意推理侧可能逐 position) / lm_head 权重
```

MoE 层额外边界:层输入 norm → attention 段 → 层输出(o_proj 后残差)→ MoE 输出 → 共享专家 fc1/fc2。

**权重单独一条线**:embedding / qkv 下投影 / o_proj / lm_head 权重,按语义切好对齐——用于排除「权重不同步」。

**权重验证实操**:训练侧 msprobe dump 只带 forward 输出,不带权重;权重要从模型 checkpoint(HF safetensors)读——`safetensors.safe_open(path, framework="pt")` + `get_tensor(...)` **只读单 tensor**,别整文件加载(1GB+ 分片);用 `model.safetensors.index.json` 的 `weight_map` 定位权重在哪个分片。注意融合投影的行序(`fused[:<dim>,:]` vs `fused[<dim>:,:]`),方向错了会误判 shape mismatch。

## 对齐步骤

1. **划边界清单**(上面的数据流骨架,按实际模型增减)。
2. **逐边界定位两侧算子**:靠 construct 轨迹顺序 + 数据流 + shape 校验。一个边界的「训练侧算子输出」往往等于「推理侧某算子输入/输出」。
   - 先找**同名算子**(probe 已列出):可直接精确比,是最低垂的果实。
   - 再按数据流找语义对应:训练侧独立 Q 投影输出 ↔ 推理侧融合 QKV 投影输出取 q 段。
3. **融合算子切片对齐**:
   - 推理侧融合 QKV 投影输出 `[seq, q_dim + kv_dim]` = q 段 | kv 段;训练侧 Q、KV 投影是独立两个 GEMM。切 q 段、kv 段分别对齐。
   - 推理侧融合 MLP 输入投影(gate+up 合一)输出 `[seq, 2*hidden]`(gate|up 或 up|gate)↔ 训练侧独立 fc1。
   - 融合残差+norm(常见如 `add_rms_norm` 类算子):输出 = 训练侧「残差 add + 下个 norm」的结果,对到训练侧下一层输入 norm。
   - 切片规则进 mapping 的 `transform`(`train_slice` / `infer_slice` / `squeeze` / `reshape`),脚本自动应用。
   - **切完必须逐段分别比对,别只看整段**:融合输出一段里可能只有某一段(如位置编码段)有差异、其余段 bitwise 一致——整段比只得到"非零",不知道差异在哪一段;逐段比才能定位到具体段,进而指向该段的实现(如位置编码算法)。
4. **reshape 对齐 attention 输出**:训练侧 flash attention 常输出 `[seq, heads, head_dim]`,推理侧 o_proj 输入常是 `[seq, hidden]`;numel 一致则 reshape 对齐。
5. **逐 position 拼接**:推理 logits 若按 position 逐个 dump(每个输出 `[1, vocab]`),mapping 用 `infer_concat=True` + 多个 `infer_globs` 拼接成 `[seq, vocab]`。
6. **映射不了的标 `evidence_missing`**,不猜:shape 对不上、语义找不到对应、结构差异过大(MoE 内部路由/permute 两侧算子完全不同)。如实记录,进入报告的 limitations。

## 切片前先做布局相关性扫描

融合输出分段布局(如 fused 权重行序 [q|kv] vs [kv|q])**不能靠猜**。切片对齐前,用一段快速脚本对融合段做相关性/bitwise 检查:

```python
# 假设 q 段在前:比对 fused[:q_dim] 与 train_q 权重
# 若 max_abs==0 → 布局确认;若 >0,试 fused[q_dim:] 或其它候选布局
rep("fused[:q_dim] vs train_q_weight", train_q_weight, fused_w[:q_dim])
```

融合输出的分段布局(如 q|kv 还是 kv|q)**必须现场用 bitwise 验证,不许沿用别人的布局结论**。

## 产出物:语义映射表

每边界一条,直接是 `scripts/analyze_dump_boundaries.py` 的 `--mapping` 输入:

```json
{
  "name": "layer0_q_proj",
  "semantic_boundary": "decoder.layer0.attention.q_proj",
  "execution_order": 40,
  "train_glob": "*<train_prefix>*...q_proj*...output.0.pt",
  "infer_glob": "*<infer_prefix>*...fused_qkv*...output.0.pt",
  "transform": {"squeeze": [1], "infer_slice": "[:, :<q_dim>]"},
  "note": "推理侧融合 GEMM 取 q 段(权重行序已 bitwise 验证)"
}
```

`execution_order` 按语义执行顺序递增(embedding→…→logits),是首差异定位的依据。`infer_globs` 支持多个 glob 拼接(逐 position logits)。

## 判定:什么时候该标 evidence_missing

- 两侧找不到数学等价的张量(MoE 路由/permute、RoPE 融合内部)。
- shape 经合理变换(numel 相等)仍对不上。
- 只有一侧 dump 到某边界(另一侧该算子被融合吃掉且无法拆出)。

**evidence_missing ≠ 差异**:它只是「这层没比到」。把它与 NOT_EXACT 严格区分,写进报告 limitations。
