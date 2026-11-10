# 证据与状态管理

> 管理两类文件：`investigation.json`（分析状态，工作过程中维护）与 `evidence.json`（交付物，结构化证据记录）。目标：分析状态不依赖对话上下文——上下文被压缩后从状态文件恢复，不重头复述推理、不重复查询。

## 1. investigation.json（分析状态）

```json
{
  "case": "caseXXX",
  "start_time": "开工 epoch 秒（HC-C11⑦ 下班点计时起点；阶段二初始化时写入，压缩恢复不重置）",
  "box": "轻案45|重案60（HC-C11⑦，开工按两侧 rank 目录总数 <4/≥4 判定）",
  "phenomenon_anchor": {"description": "…", "quantified": "…（比值/差值）"},
  "sides": {"target": "<路径>", "golden": "<路径>"},
  "alignment_limits": ["已确认的映射限制/告警"],
  "anchor_checklist": [
    {"anchor": "速览锚点描述", "status": "pending|checked-evidence|checked-noncausal",
     "note": "核查方式与结论"}
  ],
  "candidates": [
    {
      "candidate_id": "C1",
      "node": "节点标识（dump 原始节点名含调用号 + order）",
      "status": "discovered|pending-L1|pending-L2|verified|downgraded|excluded|merged",
      "discovery_channels": ["速览-守恒违反", "参数梯度分解"],
      "supporting_evidence": ["E1", "E3"],
      "counterevidence": ["E2"],
      "validation_level": "L1|L2|L3",
      "unresolved_questions": [],
      "next_action": null,
      "last_checked": null
    }
  ],
  "evidence_log": [
    {"id": "E1", "source": {"side": "T|G", "file": "…dump.json", "node": "…",
      "slot": "output[0]"},
     "metrics": "四统计量/计数/比值等关键指标",
     "finding": "观察事实", "method": "获取方式（脚本/反查/闭合计算）"}
  ],
  "checked_queries": ["已执行的查询/反查/闭合计算（参数+结论摘要，去重复用）"]
}
```

**状态管理规则**：
1. 新证据追加到已有候选，不重复创建同一候选；同一节点/同一机制的差异实例合并为一个候选（实例清单内枚举，HC-E3）。
2. 每条证据包含可复查的来源定位（侧别+文件+节点+槽位）。
3. 已完成的检查记录结果（`checked_queries`），避免上下文压缩后重复执行。
4. 候选状态显式区分：discovered / pending-L1 / pending-L2 / verified / downgraded / excluded / merged。
5. 只有出现新证据、旧证据被推翻或关键问题仍未解决时，才重新打开已完成的检查。
6. 状态文件记录事实与决策状态，不保存冗长的模型思维过程。
7. **下班点状态（HC-C11⑦）**：超盒冻结取数后，各候选 `next_action` 只允许写作/定稿类动作（誊写、落盘、自检、check）；冻结时刻与冻结原因记入 run_notes，未完成取数意图转 open_questions——冻结不是中止，交付义务不减。

## 2. evidence.json（交付物 schema，必填）

承载全部机器可查结构——`candidates[].qualification`（HC-C8 资格表四格）、`exclusion_ledger`（排除/降权项与速览锚点结案台账）、`open_questions`（覆盖缺口与剩余不确定性）；report.md 只承载人读的判断与证据，不设对应章节。

```json
{
  "case": "caseXXX",
  "phenomenon": {
    "description": "目标异常现象描述（表现、范围）",
    "evidence_sources": ["现象证据来源：用户 Prompt / reference.md 路径与相关内容"],
    "quantified": "量化锚点（比值/差值，谁大谁小）"
  },
  "candidates": [
    {
      "candidate_id": "C1",
      "node": "…（含调用号 + order）",
      "rank": 1,
      "basis": "候选依据（可疑原因）",
      "supporting_evidence_ids": ["E1", "E3"],
      "counterevidence_ids": ["E2"],
      "causal_chain": [
        {"step": "环节描述（节点→中间结果→…→目标异常）",
         "status": "已证实|合理推断|待验证",
         "evidence_ids": ["E1"]}
      ],
      "confidence": {"as_root_cause": "高|中|低", "evidence_strength": "强|中|弱",
                     "rationale": "…"},
      "role": "头号根因候选|较强候选|待验证假设|机制层候选|伪影簇（不承根因）",
      "qualification": {
        "predicate_rewrite_test": "谓语改写测试结论（改写为『载体（或其输入缓冲）X』后语义不变=失格）",
        "anchor_verification": "锚点验证（合并格，分号并列三项：正向生产者锚点节点名或『无』+输入一致性自证；节点级验证类型与已验证/未验证（验证对象=候选主张的机制/差异制造环节，载体值复现不算）；同缺陷类归一判定（组名或『无同类』））",
        "background_unexcluded": "背景级判定与未排除状态（含 HC-C5 迁移改判留痕）",
        "verdict": "资格结论（含机制覆盖内/外）"
      },
      "open_questions": []
    }
  ],
  "evidence_log": [
    {"id": "E1",
     "source": {"side": "T|G", "file": "原始数据位置（绝对或相对 case 的路径）",
                "node": "…", "slot": "…"},
     "metrics": "相关指标（四统计量全精度/计数/比值，标侧别）",
     "usage": "用途与验证结论（一句，如：支持 C1——反查成立 / 反证 C2 / 现象证据）"}
  ],
  "exclusion_ledger": [
    {"item": "被排除/降权项或速览高置信锚点（精确 API 节点名含调用号）",
     "type": "排除|降权|锚点结案",
     "disposition": "处置（如：已核查（有效域=被验证路径）；对现象载体路径未排除）",
     "basis": "依据（证据 ID/程序引用）"}
  ],
  "open_questions": ["尚未解决的问题（区分：可能改变排序的 / 一般性的）"],
  "root_cause_judgment": {
    "status": "已验证根因|较强候选|待验证假设",
    "statement": "当前根因判断（主语 = 候选资格表位序第 1 合格候选）",
    "confidence": "高|中|低",
    "unverified_links": ["因果链中未验证环节"]
  },
  "verification_methods_used": ["本轮使用的关键验证方法（如：分侧守恒检验、
    写-消费三值核查、输出侧双假设闭合…）"],
  "run_notes": "必要的运行状态及异常记录（可选；脚本告警、数据质量事件、
    中断与恢复等）"
}
```

**一致性要求**：evidence.json 与 report.md 的候选数、排序、根因主语、置信度必须一致（交付前自检项）；主报告中的关键结论必须能回溯到 evidence_log 条目（证据 ID 可解析、来源可定位）。

## 3. 查询去重与批量取数

- 同一证据只读取一次；后续引用证据 ID。
- `checked_queries` 登记已执行的查询（反查参数+结果）；重复请求直接复用，不重查。
- 优先合并细粒度查询为批量：一次遍历建立四元组索引后批量溯源；一次脚本调用取多个节点的属性/邻域/统计；抽查原始 dump 用小的 `python3 -I` 片段按节点名精确取（机制见 mechanism §3.6），不整读大文件。

## 4. 上下文恢复（上下文被压缩后）

**恢复只读状态，不重读证据（硬性——恢复时重读已落盘产物是重复劳动）**：

1. 只读两处：`investigation.json`（候选池、证据索引、已查清单、anchor checklist 状态、next_action）+ report.md 的**未完成小节**（按"（待填）"钉与各节标题定位）。**禁止**在恢复时重读 analysis_summary.md 全文、重读任何参考文档全文、重读 report.md 已完成小节——已落盘即已在册，摘要/文档只在**当前步骤确实需要某具体数值/程序**时按下方第 2 条定向取。**禁止**在恢复时重跑 dump_analyzer.py（analysis_out 产物已在盘）；恢复后新增取数仅限各候选 `next_action` 所需，逐条计入 checked_queries（重复查询直接复用不重查）。
2. 参考文档按节定位读：先 `grep -n <关键词>` 拿行号，再定向读该节（一次分析只整读一遍，此后一律按节取）。analysis_summary.md 同理：需要某个锚点数值时 grep 定位读那几行，不通读。
3. 从各候选的 `next_action` / `unresolved_questions` 继续当前阶段，不重头复述整段推理、不重读已查证据（证据 ID 在册即可引）。
4. 报告恢复：只续写未完成小节（先核对首行 case 身份锚，HC-A4）。
5. 恢复后第一步先把 investigation.json 更新一条 recovery 记录（恢复自哪步、当前 next_action），再继续执行——防止二次压缩后再次推导。

## 5. 增量落盘纪律

- report.md 骨架先落；每完成一个阶段/得到一个结论立即写盘；较长批次（批量普查/逐 rank 复核）取数前，先把已有候选草稿写盘。
- **取数输出即取即誊（硬性）**：query.py / survey / first / gain 等批量输出的**承重行**取数后立即誊入 investigation.json / report.md 草稿对应小节（同一数值全文只取一次、后续引证据 ID），Top-N 长输出只截承重行（--top/head 控制单次输出长度），**不整段滞留对话上下文**——落盘即弃（在册即可引，不再贴原文）。
- **investigation.json 检查点粒度（硬性）**：在阶段三结束（候选池定型）、每次 L1/L2 验证完成、统一排序落盘后**必须**更新（candidates 状态+已验值+next_action）——保证上下文被压缩后单次恢复即可续作，不重新推导。
- **候选详析逐条誊写（硬性）**：阶段四每完成一个候选的 L2 验证，立即把该候选的详析小节写盘（8 项结构按已完成项写、未完成项留"（待填）"钉）；排除项每结案一条立即誊一条（落 evidence.json exclusion_ledger）——不把多个候选/多条结案攒到最后统一写。压缩恢复时只需续写未完成候选，已完成候选不重推。
- **承报证据分级（硬性）**：承报级最小充分证据集（锚点槽位直接核验+方向/形态批量普查一遍——按 HC-C11⑥ 普查折叠口径执行：族级一次+代表实例抽样、禁自建普查脚本/逐实例枚举+代表槽对照）先行落盘；深闭合证据（全库逐槽定量拟合/指纹参数估计/全族逐实例画像等 L3 增强项）仅在必要时（L3 触发条件）追加，完成当前决策所需验证后立即停止，未做项记入 open_questions。
- **誊写折叠（硬性）**：普查/census 类结论登记为**族级一条** evidence_log 条目（引工具单次输出计数：调用、覆盖 rank×槽位范围、方向/形态结论与实例计数），不逐实例/逐计数逐条落证据；exclusion_ledger 同类条目（同族溢出、同型锚点结案）合并为族级行，**行数参考上限 5**——超出部分以一行汇总+代表实例证据 ID 引用承报；causal_chain 只写**最小充分环节**（典型 3-5 环节：候选节点→差异制造/传播关键环节→现象载体），非承重中间节点不逐个成环（引证据 ID 即可回查）；候选小节「关键证据」正文**参考上限 3 条承重项**（锚点四统计量并列、一条最承重闭合/反证证据、一条反证锚点）——深挖产出的其余证据全量入 evidence_log（带 ID/出处），正文以 ID 引用+一句结论形态呈现（挖掘深度由位序轮候与盒决定，呈现密度由此条统一）。
- evidence.json 在阶段五集中生成（从 investigation.json 提炼），生成后与 report.md 做一致性核对。
