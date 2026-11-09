# 分析:精确门禁 / 首差异 / ULP 量化 / 放大链 / 排除假设 / 分类结论

阶段 3-7 的方法细节。比对用 `scripts/analyze_dump_boundaries.py`,本节讲**怎么读结果、怎么下结论**。

## 1 逐边界精确门禁

每个已映射边界跑五连精确门禁:

```
same_shape / same_dtype / no_nonfinite / torch.equal=true / max_abs_diff=0
```

任一不过即非 EXACT。分类(脚本输出 + 语义层补充):

| 分类 | 含义 |
|---|---|
| `EXACT` | 五连门禁全过,bitwise 一致 |
| `NOT_EXACT` | shape/dtype/非有限都过但值不等(`torch.equal=false`,`max_abs_diff>0`) |
| `SHAPE_DIFF` / `DTYPE_DIFF` | 形状/类型不一致(先查变换是否遗漏,再看语义是否真不等) |
| `NONFINITE` | 含 NaN/Inf(真异常,单独报) |
| `MISSING` | 某侧文件缺失 |
| `evidence_missing` | 语义层映射不上,没比(不是脚本分类,是分析者标注) |

**结果必读项**:每个 NOT_EXACT 边界的 `max_abs_diff`、`nz_count/nz_fraction`(非零占比)、`max_rel_diff`。占比极低(<1e-3)提示舍入级,占比接近 1 提示系统性差异。

## 2 首差异定位

首差异 = **沿语义执行顺序第一个非 EXACT 的边界**(脚本按 `execution_order` 排序给出)。

纪律:
- **首差异的所有前序边界必须全部 EXACT**。任一前序非 EXACT,差异真正源头更早,继续前推。
- 若首差异之前有 `evidence_missing`(没比到的层),结论注明「映射缺失,不排除更早来源」,必要时补映射再定论。
- 首差异往往是**极小值**(GEMM 舍入 ULP 级),别因为小就忽略——它是下游所有非零的放大源头。

## 3 ULP 量化:舍入级 vs 逻辑级

对首差异边界,算 `diff / ULP(value)` 的倍数分布(bf16:ULP 相对值 2^-8;fp32:2^-23):

```python
def bf16_ulp(v):  # 1 ULP 绝对值 = 2^(exp-8)
    e = torch.floor(torch.log2(v.clamp_min(1e-30))).clamp_min(-126)
    return torch.pow(2.0, e - 8)
mult = diff / bf16_ulp(a[nz].abs())   # 差异元素 / ULP 倍数
```

判定指纹:

| 指纹 | 舍入级(融合/独立 GEMM、累加顺序差) | 逻辑级(配置/权重/内核错误) |
|---|---|---|
| 非零占比 | 极稀疏(<1‰) | 密集或明显成片 |
| diff/ULP 倍数 | 集中于 ±1~2(实测出现全部恰好 2 ULP) | 远超 ULP、无规律 |
| 分布形态 | position/hidden 稀疏散布,无结构 | 整行/整列/整 position 出错 |
| 前序边界 | 全部 EXACT | 可能有前序污染 |

舍入级 ≠ 错误,但要写清楚「非逻辑/配置/权重错误,属浮点累加顺序差异」。逻辑级才需要上升到修复/排查。

## 4 放大链

记录首差异量级 → 逐边界 max_abs 变化,证明「首差异小,下游逐层放大到最终指标」:

```text
<q_down> ~1e-3 → <q_norm> ~1e-2 → <q_up> ~1e-1 → <o_proj> ~1e-1
  → <post_attn_norm> ~1e0 → <final_norm> ~1e1 → <logits> ~1e1
```

放大链是结论的证据主线:它把「边界级首差异」与「端到端指标非零」连起来。取链上关键边界(不是每个)打印 max_abs 即可。

## 5 排除假设

按序排除,每条记录排除依据:

1. **权重不同步** → 权重边界(embedding/qkv/o_proj/lm_head)是否 `torch.equal`?是 → 排除。
2. **输入未对齐/padding** → 首差异前序输入边界是否 EXACT?训练侧有 padding 导致 shape 不等(此时是映射/采集问题,修正而非报差异)?是 → 排除。
3. **配置差异** → 两侧结构(独立 vs 融合 GEMM、norm 融合位置、MoE 路由)是否语义等价?结构不同但数学等价 → 不算根因。
4. **并行切分不一致** → 两侧 TP/PP/EP 一致?是 → 排除。
5. **fused 内核/累加顺序** → 首差异 ULP 指纹符合舍入级 → 定性浮点差异。
6. **随机性** → 单样本确定性配置 + 前序 EXACT → 排除。

**排除假设的证据与首差异证据同等重要**。结论里必须写清「为什么不是权重/配置/输入问题」——否则「首差异是舍入差异」的结论不成立。

## 6 分类与结论

- 边界级:`EXACT` / `NOT_EXACT` / `evidence_missing`。
- 整体级四分类(映射到报告 `results`):
  | 全部边界 | 整体状态 |
  |---|---|
  | 全 EXACT | `exact` |
  | 任一原生边界 NOT_EXACT | `nonzero` |
  | 未执行 | `not_run` |
  | 证据不足 | `evidence_missing` |
- **结论口径**:首差异位置 + 根因 + 证据链 + 是否可解。绝不只报「最终指标非零」。
- 数值指标只引用实测:
  - prefill 段 logprob diff(训练循环直接算,日志实测 mean/max/std):`mean≈0` 训推一致;非 0 结合逐边界定位;日志没有 → `evidence_missing`,不编。口径见 `prefill-logprob-diff.md`。
  - response 段 E2E(`rollout_probs_diff`):prompt-only 裁剪下无实测(valid=0 是「没算」不是「一致」);要数值需跑含 response 的标准训练实测,否则 `evidence_missing`。与 prefill logprob diff 不同层面,不互替。
