# 实战失败教训(Pitfalls)

训推一致性**分析侧**(rl-consistency-dump-analysis)踩过的坑,编号用 A 系列,与采集侧(`rl-prefill-consistency-dump/references/pitfalls.md` 的 P 系列)分开。每条给「症状 → 根因 → 修复 → 预防」。后续各框架/版本遇同类症状先对照这里。

## 数据对齐陷阱

### A1. 推理侧 prompt logprob 的 shift-by-one 因果位移 → logits 直接比对假阳性(最重要的分析陷阱)
- **症状**(09-22,glm4.7 R5 后):打桩隔离 MoE 段后,最终 norm 输出两侧 **bitwise=True**,但 lm_head 输出(logits)仍差 max 49.9、dense 0.9987、pearson **0.04**——而两侧 mean/std 却完全相同(-0.2413/4.0653)。
- **根因**:vLLM 计算 prompt logprob 时把**因果位移序列** `[h_{-1}, h_0, …, h_{L-2}]`(roll-by-1)喂给 lm_head,`logits[p]` 预测 `token[p+1]`;训练侧 output_layer 逐位置 `logits[p]=GEMM(h[p])`。同位置直接比对 = **错位比对**,得到"巨大差异"假象,掩盖真实已收敛的结论。
- **判别**(两句脚本,先做再下结论):
  1. 逐行相关性扫描 `corr(h_i[0], h_t[j])`——峰值在 `j=L-1` 且 `corr=1.0000` → 位移 1;
  2. `torch.equal(h_i, torch.roll(h_t, 1, dims=0))` bitwise=True 确认。
- **修复**:语义对齐 `train_logits[p] ↔ infer_logits[(p+1) % L]`(`aligned_i = torch.roll(i_logits, -1)`)。对齐后仅剩 bf16 GEMM 1-2 ULP 舍入(max 0.0625、dense 0.0002),**argmax 100% 一致,top1 prob diff 3.4e-8** → 收敛判定成立。
- **预防**:推理侧"逐 position 序列型输出"(logits/logprob/lm_head 输入)比对前**先做 shift 校验**;发现 norm.out 与 lm_head 输入差异巨大、但 mean/std 一致 → 先查位移,别判"大差异"、别为它打桩。

### A2. 推理侧 lm_head 权重不在命名模块下(融合算子),按 shape 找权重
- **症状**:dump 里 `logits_processor` 只有 `.input.1`(hidden)与 `.output.0`(logits),找不到 lm_head 权重;`(154880, 2048)` 权重实际在 `Functional.linear.34/35.forward.input.1`。
- **根因**:vLLM-ascend 的 `AscendLogitsProcessor` 是**融合算子**,lm_head GEMM 在内部执行(内部不逐层 dump);lm_head 权重被归到通用 `Functional.linear.NN`,不是 `lm_head` 命名模块。
- **修复**:按 shape 扫描全 dump 找权重(`t.shape[0]==vocab_size and t.shape[1]==hidden`),再与训练侧 `output_layer` 权重 `torch.equal`。
- **预防**:找权重别按模块名先猜;融合算子(logits_processor/lm_head)的权重归属现场扫 shape 定位。

## logits 差异定性

### A3. logits 差异快速定性:线性拟合 + 排序集合 + argmax
- **症状**:train vs infer logits 差巨大、pearson 低但 mean/std 相同,疑列重排或后处理。
- **方法**(判别是纯 GEMM / 缩放 / 列重排 / softmax 后处理):
  1. **线性拟合** `y=k·x+b`(采样点):k≈1 且残差舍入级 → 纯 GEMM;k≠1 → 温度/缩放;残差大 → 非仿射(softmax/重排)。
  2. **逐行排序后比较**(集合相等测试)→ 判别列重排。
  3. **argmax 一致性** → 判别语义是否一致。
  4. **逐 position 拟合 k/b** → 判别是否逐位处理。
- **预防**:logits 大差异先做这三件套定性,再决定是否值得打桩;疑似位移的先走 A1 shift 校验。
