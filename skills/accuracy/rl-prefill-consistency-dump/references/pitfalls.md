# 实战失败教训(Pitfalls)

训推一致性采集/打桩踩过的坑,按阶段组织。每条给「症状 → 根因 → 修复 → 预防」。后续各框架/版本遇同类症状先对照这里。

## 阶段 4:输入对齐(PROMPTS_ONLY 裁剪)

### P1. 裁剪副本不能替换整个 rollout_data
- **症状**:`compute_advantages_and_returns` 崩 `split_with_sizes ... got split_sizes=[0]`;或 `log_rollout_data` 崩。
- **根因**:整体替换 `rollout_data` 后 `response_lengths` 全 0,下游按它 split 的统计 tensor 变空。
- **修复**:裁剪副本**只喂 `get_data_iterator`**(compute_log_prob 主前向输入),原始 `rollout_data` 留下游;`rollout_corr_helper` 在 `PROMPTS_ONLY=1` 时直接 `return batch, {}` 早退,loss/metrics 同步兜底,训练照常走完(见 P6b);`log_rollout_data` 可跳过。
- **预防**:PROMPTS_ONLY 只影响「log_prob 前向输入」,禁止动 advantages/统计路径。

### P2. loss_masks 裁剪必须 `m[:0]` 再靠 pad 展开
- **症状**:`assert loss_masks.shape == tokens.shape` 不过,shape [1,260] vs [1,256]。
- **根因**:`F.pad(loss_mask, (prompt_length-1, 1), value=0)` 展开成 prompt_length;裁剪时没去掉 response 段则 pad 后超长。
- **修复**:`loss_masks = [m[:0] for m in loss_masks]`(裁空)再靠 pad 展开。校验:`prompt_length = total_length - response_length`,`F.pad(mask, (prompt_length-1, 1))` 长度 = prompt_length。
- **预防**:改裁剪逻辑先把长度数学算一遍,再对照 data.py 的 pad 行。

### P3. response_lengths=0 时 log_prob 返回空 tensor
- **症状**:compute_log_prob 返回空 tensor(`log_prob_full[255:255]`),下游以为有值。
- **修复**:采集模式显式跳过依赖 response 段的下游(log_rollout_data),别把空 tensor 当正常产物。
- **预防**:裁剪方案设计时列出「哪些下游会崩」提前跳过。

### P18. 训练端 padding 必须去掉,输入 shape 两侧对齐(190 vs 256)
- **症状**:逐边界比对时训练侧 shape [256,...]、推理侧 [190,...] 对不上;训练侧 prefill logprob 带 padding 位错位。
- **根因**:训练侧(Megatron 系)把序列 pad 到对齐长度才进 prefill;推理侧是真实 prompt 长度。padding 是**训练侧单边**的。
- **修复**:**在输入对齐阶段去 padding,不是分析阶段再处理**——裁剪成 prompt-only 后按真实 prompt token 数截取训练侧输入与 logprob(`log_probs[:, :-1]` 前先切),保证两侧输入 shape=真实长度、逐 token 1:1;TP>1 时两侧按同一 pad 规则对齐(见 SKILL 阶段 4)。
- **预防**:采完先验两侧同一边界输入 shape 相等且=真实长度;出现 [256,...] vs [190,...] 回阶段 4 修掉重采,别把带 padding 的 dump 交接分析环节。

## 阶段 6:接入补丁(改代码的代码)

### P11. 补丁脚本自身的字符串嵌套陷阱
- **症状**(09-15):`apply_patch.py` 语法错误/字符串提前终止,反复 `PATCH_RESULT=FAIL`(6 次)。
- **根因**:补丁用 `"""` 包 old/new 字符串,而补丁内容本身含 `"""`(docstring)或 f-string 花括号,边界错乱。
- **修复**:定界符用 `'''`(内容里最少出现的引号);应用前 `py_compile` 自检。
- **预防**:"改代码的代码"必须先自检语法;定界符避开目标代码最常见的引号。

### P12. 推理侧 prompt_logprobs 是 top-k 嵌套 dict,不是单层
- **症状**(09-15):训练侧 `KeyError: 'logprob'`(vllm_rollout 解析 prompt_logprobs 处)。
- **根因**:vLLM-ascend 的 `prompt_logprobs` 每 position 是 `{token_id: {"logprob", "rank", "decoded_token"}, ...}`(**top-k 嵌套字典**,逻辑在 Rust 二进制 `vllm_router_rs.abi3.so`),position 0 无前序 token;不是文档的单层 `{logprob}`。
- **修复**:容错解析——entry 是 dict 且含 `"logprob"` 直接取;dict-of-dicts 取概率最大项;`None` 兜底 0;解析前打一行 `[DUMP_DIAG]` 核对结构。
- **预防**:任何"引擎返回体"字段先打诊断确认结构再写解析,别照抄文档。

## 阶段 7:执行采集

### P4. msprobe step 语义:训练侧有效 dump 在 step0,不是 step1
- **症状**:训练 dump 的 step1 `debug.json` 空/不完整。
- **根因**:step0 = compute_log_prob **主前向**(完整),step1 = train 前向(未 flush,空);且只有 `can_reuse_log_probs_in_loss == False` 时主前向才执行,不强制则什么都采不到。
- **修复**:用 `--get-mismatch-metrics --custom-tis-function-path <vanilla_tis_function>` 强制 actor 主前向;验证认准 step0 dump。
- **预防**:采集后先验训练侧 step0 dump 非空再进分析。

### P5. docker exec 卡死 / exit 137(SIGKILL)
- **症状**:`docker exec` 超时/SIGKILL;pkill 清理把自己杀了。
- **根因**:`pkill -f` 匹配串命中执行命令的 bash 自身(命令字符串含同样关键字)。
- **修复**:正则防自匹配 `pkill -f "[r]ay"`;长任务 `docker exec -d` 分离;特定进程(EngineCore)用精确 PID,别宽泛 `pkill -f vllm`。
- **预防**:清理命令写成不自匹配形式;能按 PID 就不按名字。

### P6. 残留 VLLM::EngineCore 占用显存 → 二次采集 OOM
- **症状**:job 崩溃后 EngineCore 残留,第二次 vLLM 引擎 OOM(free 只几百 MiB)。
- **修复**:kill 残留 EngineCore + 清 ray,确认卡 HBM 回基线(910B4 空闲 ~2.9G/32G)再重跑。独立 PID namespace:`npu-smi info` 显示宿主 PID,须进容器按 `comm=VLLM::EngineCor` 找容器内 PID 再 kill。
- **补充(09-17)**:job 崩后 EngineCore 可能**孤儿化(PPID=1)**且不再带 ray 前缀,`ps aux | grep ray` 找不到——但 npu-smi 显示卡 HBM 明显高于基线(如 21G/32G)。识别法:全量扫持 davinci 句柄进程(★ 必须 `readlink` fd:`ls /proc/<pid>/fd` 目录条目是纯数字,`grep davinci` 查不到(漏检两次);正确 `for fd in /proc/<pid>/fd/*; do readlink "$fd"; done | grep davinci`)→ `ps -o pid,ppid,etime,cmd -p <pid>` 确认身份 → 精确 `kill -9`。宽泛 pkill 对孤儿进程未必命中。
- **预防**:采集前/后检查卡 HBM 占用,残留先清;卡基线写进 task_card。

### P19. ray job 没解析工作目录的 vime_plugins → AutoBridge 拒架构(桥早已存在)
- **症状**(09-16,卡 1,2):`ValueError: Model architecture 'XxxLiteForCausalLM' is not yet supported`。但工作目录 `vime_plugins/megatron_bridge/` 里**已有**自定义 bridge 文件,且 09-15 同 job 成功过(workdir `__pycache__` 有编译产物为证)。
- **根因**:vime 是 **editable 安装**,finder 的 `MAPPING` 把 `vime_plugins` 固定映射到已安装路径 `/root/vime/vime_plugins`(只注册 `glm4v_moe`)。job driver 的 `PYTHONPATH` 没工作目录,`import vime_plugins` 落到已安装版 → 找不到自定义架构注册。**import 解析漂移,不是缺桥**。
- **修复**:工作目录放 job `PYTHONPATH` 最前(如 `export PYTHONPATH="/workspace/vime/vime-zjc:${PYTHONPATH:-}"`)。验证:`python3 -c "import vime_plugins; print(vime_plugins.__file__)"`。
- **预防**:① 复现先看 workdir `__pycache__` `.pyc` 时间戳,有即证明不是缺文件;② 遇 "not yet supported" 先核对两侧 `__init__.py` import 列表 + job `PYTHONPATH`;③ 重建脚本把「工作目录在 PYTHONPATH 最前」当必备段。

### P20. msprobe(level=mix)替换 F.silu 后,引用缓存/引用相等校验连锁崩(09-17)
- **症状**:`DUMP_ON=1` 训练侧报错分两层(取决于 debugger 实例化时机):
  1. debugger 晚于模型构造:`mlp.py:198` `Only support fusion of gelu and swiglu`(`self.activation_func == F.silu` 引用相等失败);
  2. debugger 提前到模块导入期后仍崩:`transformer_config.py:1436-1445` `When bias_activation_fusion is True, activation function should be either gelu, swiglu, or quick_geglu`。
- **根因**:msprobe 实例化时经 `_register_api_hook` **全局替换 `torch.nn.functional.silu` 为包装函数**。凡**模块导入期缓存原生 `F.silu` 引用、替换后与动态 `F.silu`(包装)做引用相等/成员判断**的代码都失败。Megatron-Bridge 链:`model_bridge.py:322` `ACTIVATION_MAPPING = {"silu": F.silu}` 导入期缓存原生引用 → `import megatron.bridge` 在 debugger 实例化之前跑完 → 模型构造时对照动态 F.silu(包装)不匹配。**时序敏感:debugger 时机一变,连锁点就变**。
- **修复**(采集态补丁,采完还原):① debugger 提前到模块导入期(训练 `model.py` 顶部,env-gated);② 激活映射改调用期动态求值(`hf_to_megatron_activation` 对 `"silu"/"gelu"` 动态访问 `F.silu`/`F.gelu`,数值与原生一致);`megatron_to_hf_activation` 反向加 `is F.silu` → `"silu"` 兜底。
- **预防**:① 接入前 `grep -rn "F.silu\|F.gelu"` 扫模型/转换代码,列所有"导入期缓存 + 替换后引用相等判断"点;② 改 debugger 时机后重扫调用链;③ 校验崩别改 msprobe 排除 silu(排除了采不到 silu 边界),正确是让校验两侧引用一致。

### P21. Megatron-Bridge 的 TE 硬依赖假阻塞 + Python 3.12 注解求值(09-17)
- **症状**:孤立环境(无 transformer_engine)`import megatron.bridge` 崩 `ModuleNotFoundError: transformer_engine`;连锁假象是 AutoBridge 报 `'XxxLiteForCausalLM' is not yet supported`。
- **根因**:`peft/lora_layers.py:20` 顶层**无条件** `import transformer_engine.pytorch as te`;TE 只服务 LoRA 但 `import megatron.bridge` 必经;TE 缺失 → 导入期崩 → bridge 注册语句从未执行 → AutoBridge 找不到架构。**"不支持"是假象,真因是导入链硬依赖崩在注册之前**。孤立环境无 TE ≠ 作业进程无 TE(modelopt 警告显示作业里 TE 存在但缺 RMSNorm,注入源不明,别耗)。
- **修复**(采集态补丁,采完还原):① TE import 改 `try/except ModuleNotFoundError`(`te=None, HAVE_TE=False`);② `class TELinearAdapter(te.Linear):` → `(te.Linear if HAVE_TE else nn.Module):`;③ 文件顶加 `from __future__ import annotations`——**Python 3.12 无 future 时类体内方法返回注解在类定义期求值**,`-> tuple[te.ops.Sequential, ...]` 在 te=None 时直接崩。
- **预防**:① 遇 "not yet supported" 先查 bridge 导入链有没有导入期硬依赖炸,别只盯 AutoBridge 校验;② 孤立环境报 ModuleNotFoundError 先判断该模块是否真被这条路径用到(TE 只服务 LoRA);③ 加 try/except 时同步检查类定义期会求值的注解(3.12)。

### P6b. PROMPTS_ONLY 裁剪后 loss 崩:跳过训练是错误收尾,官方做法是改 loss 兜底
- **症状**(09-14,卡 6,7):两侧 dump **已完整落盘**,随后训练崩:`policy_loss_function` 里 `torch.cat(batch["advantages"])` 收到 `None` → `TypeError`(或 `split_sizes=[0]`)。
- **根因**:PROMPTS_ONLY 把训练输入裁成 prompt-only,但此前为防崩把 `advantages` 设 `None`、跳过 advantages 路径,却没按官方改 loss 兜底——loss 仍按 response 段切、要 `advantages`/`log_probs`,收到 None 即崩。
- **修复(官方做法,训练照常走完)**:不是跳过训练,改三处:
  1. **裁剪**(`compute_log_prob`):有 `responses` 时按 `response_length` 截 `input_ids/attention_mask/position_ids` 的 response 段,`pop("responses"/"rollout_log_probs"/"response_mask")`;log_probs 分支改 prompt-only(`log_probs[:, :-1]`)。
  2. **改 loss**(`loss_func`):`response_length==0` 时不再按 response 段切 logits,用 prompt 段;`response_mask=None` 用全 True prompt mask 兜底;`rollout_corr_helper` 在 `PROMPTS_ONLY=1` 时直接 `return batch, {}`。
  3. **改 metrics**(`calculate_debug_metrics`):缺 `responses`/`rollout_log_probs`/`old_log_probs` 任一 → 主动返回 `{rollout_probs_diff_valid:0, _max/_mean/_std:0.0, ...}`,不崩。
  训练照常 forward→loss→backward→step;官方明确该模式下 loss/梯度**不代表正常训练结果**。采完取消 `DUMP_ON`/`PROMPTS_ONLY`。
- **预防**:不要在采集开关下跳过训练、也不要把 `advantages` 设 `None` 绕过——正确是让裁剪后 loss 也能算。response 段 E2E 锚定 response 段,裁剪下 metrics 返回 0 值(是"没算"不是"一致",见 `probs-diff-monitoring.md`);`train_rollout_logprob_abs_diff` 已改为裁剪下可算,**不要**再对返回 0 或跳过,见 P17。

### P16. 诊断字段进 log_rollout_data 会崩 `sum(val)`(int + list)
- **症状**(09-15):第 1 个 step 两侧 dump 已落盘、prefill_logprob_diff 已打印,随后 `log_rollout_data` 崩:`data.py:380 sum_value = sum(val)` → `TypeError: unsupported operand type(s) for +: 'int' and 'list'`。
- **根因**:`log_rollout_data` 遍历 `rollout_data.items()` 全量聚合,新增诊断 key `prompt_logprobs`(list of list)不在跳过列表,落入 `sum(val)`——`sum([list, list])` 初始值 0(int) + list 即崩。`lp_train_prompt`(list of tensor)不崩但会被当指标聚合污染日志。
- **修复**:把新增诊断 key `prompt_logprobs`、`lp_train_prompt` 加进 `log_rollout_data` 跳过列表(与 `tokens`/`loss_masks`/`micro_batch_indices` 并列)。
- **预防**:往 `rollout_data` 加"仅供诊断/打印"key 时先查 `log_rollout_data` 跳过列表——它全量遍历且对非 tensor-list 走 `sum(val)`,任何新增非标量 key 都崩。

### P17. PROMPTS_ONLY 下让 `train_rollout_logprob_abs_diff` 可算(替代"跳过")
- **症状**(09-15):PROMPTS_ONLY 下直接跳过该指标,日志没有 `train/train_rollout_logprob_abs_diff`。原始实现 `(old_log_probs - rollout_log_probs).abs()` 在裁剪下崩(tensor 长度不一致,跳过是错误收尾)。
- **根因**:① `old_log_probs` 裁剪后是 prompt-only 段(prompt_len-1 tokens),`rollout_log_probs` 是 response 段,长度不匹配;② 即便对齐,`sum_of_sample_mean` 在 `response_lengths` 全 0 时是 **常量-0 stub**(cp_utils 的 `get_sum_of_sample_mean` 早退分支),任何传入值都返回 0——这就是"算不出来/恒 0"的根子。
- **修复(两处,缺一不可)**:
  1. `model.py` 训练前向 `get_batch` 的 keys 列表**必须加 `"prompt_logprobs"`**——否则 `policy_loss_function` 的 `batch.get("prompt_logprobs")` 恒 None,推理侧 prompt 段 logprob 进不了 loss。
  2. `loss.py` 的 `train_rollout_logprob_abs_diff` 分支:PROMPTS_ONLY 下把 `old_log_probs`(train prompt 段)与 `batch["prompt_logprobs"]`(推理侧同一段)对齐,`torch.cat` 后算 `.abs().mean()`——**绕过 `sum_of_sample_mean`(0 stub)**,直接 `.mean()` 按 prompt token 归约;非 PROMPTS_ONLY 保持原逻辑。
- **数值口径**:该指标在采集配置下是 **raw logprob 空间的 prompt 段均值绝对差**。prompt 段大量低概率 token,小概率差被放大成几个 nats;标准训练 response 段是高概率 token,log 差小。**两个口径数量级不同属预期,不要当作应相等**。
- **预防**:① 加诊断 key 进 loss 先确认进了 `get_batch` keys 白名单;② 看到"某归约器返回恒 0"先查是否有 PROMPTS_ONLY/全 0 分母早退 stub;③ E2E 指标裁剪下要么算 prompt 段对齐值、要么标"没算",禁止用"返回 0"冒充一致。

### P13. 重建脚本漏掉基线里的"不报错就不带"参数
- **症状**(09-15):`NotImplementedError: Rule-based RM type is not specified`(rm_hub)。
- **根因**:重建采集脚本漏了基线里的 `--rm-type deepscaler`。deepscaler 是**本地 rule-based RM**,不需外部服务但参数必须带。
- **修复**:从基线脚本逐段复制完整 RM 参数(不改 RM 语义)。
- **预防**:任何基于基线重建的脚本跑前逐段 `diff` 基线,特别留意 RM/并行大小/dispatcher 这类"觉得没用就省掉"的段。

### P14. `ray job submit -- <cmd>` 会把含空格的参数拆开
- **症状**(09-15):vLLM 收到截断的 `{"dump_config_path":`;additional-config JSON 解析失败。
- **根因**:`ray job submit` 把 entrypoint 参数拼成字符串、driver 端重新 shell 解析;JSON 含空格时在空格处拆成多个 argv、双引号被剥。
- **修复**:过 `ray job submit` 的含空格 JSON 一律写**无空格 compact 形式** `'{"dump_config_path":"/abs/path"}'`。
- **预防**:传给 ray job 的参数先看是否需要空格;启动失败先查日志 argv 是否被截断。

### P15. 容器内慢命令 + docker exec 默认超时/输出换行
- **症状**(09-15):`docker exec` 超时 exit 143;`source set_env.sh` 无输出假死;监控脚本 `[: integer expression expected`。
- **根因**:docker exec 继承 Bash 默认 120s timeout,容器内 source set_env 就要 ~21s,job 启动更久;宿主对容器内进程无 kill 权限,exec 只能等超时。docker exec 输出带换行,`[ $x ...]` 判数值直接炸。
- **修复**:sleep 与 docker exec 拆开、给长 timeout(≥300s);长任务 `docker exec -d` 分离;监控轮询先 `tr -d '\n\r'` 再提取数值。
- **补充(09-17)**:`bash -lc 'cat > /tmp/x.py << "EOF" ... EOF'` 里 heredoc 若含**单引号 f-string**(如 `f"[{'EQ' if ...}]"`)会被外层单引号提前截断/转义炸掉。修复:一律**宿主 Write 脚本 → `docker cp` 进容器 → 容器内执行**,别在 `bash -lc` 里嵌长 heredoc。
- **预防**:容器内 source 环境脚本先实测耗时再定 timeout;跨容器传脚本用 docker cp,不在命令行嵌 heredoc。

### P22. ray start Session name 不匹配 → 必须清 /tmp/ray_* 残留 Redis
- **症状**(09-20 打桩 round2 首启):`ray start` 报 `Session name '...' does not match persisted value b'...'`。
- **根因**:上一轮 ray head 用了同一 `--temp-dir`,Redis/日志残留在 `/tmp/ray_<tag>/`;新一轮读到旧 Session name 不匹配直接拒。只 pkill 进程不够——**持久化数据还在**。
- **修复**:清理连 `/tmp/ray_*` 一起:`pkill -9 -f '<temp-dir 关键字>' && rm -rf /tmp/ray_<tag>`,再 `unset RAY_ADDRESS RAY_REDIS_ADDRESS` 后重启。
- **预防**:每轮采集/打桩脚本开头固定"清本集群残留"段(pkill 旧 temp-dir + rm -rf 该 temp-dir),`RAY_TMPDIR` 每轮换唯一 tag;遇 Session name mismatch 先 `ls /tmp/ray_*` 确认残留再清。

### P23. "No available agent to submit job" → 残留 ray 子进程占 GCS/dashboard 端口
- **症状**(09-20 打桩 round3 两次失败):`ray job submit` 报 `No available agent to submit job` / `Failed to submit job`,dashboard 端口有响应但提交不进。
- **根因**:上一轮 ray head 的**子进程没被上轮 pkill pattern 命中**,一直占 GCS(6382)/dashboard(8267)端口。漏网进程 cmdline 常含 `/ray/` 路径但**首字段不以 ray 开头**:`ray.util.client.server`、`ray/dashboard/dashboard.py`、`ray/core/src/ray/gcs`、`ray/core/src/ray/raylet`、`ray/_private`、`ray/autoscaler`、`ray/job`、`redis-server`。
- **修复**:清理 pattern 扩到含 `/ray/` 的路径段:`pkill -9 -f 'ray/util.client.server'`、`ray/dashboard`、`ray/core/src`、`ray/_private`、`redis-server` 等(用 `[r]ay` 正则防自匹配);配合 P6 全量扫 davinci 句柄进程兜底,再验端口释放(`ss -ltnp | grep <port>` 空)。
- **预防**:清残留别只按"进程名首字段"猜,凡 cmdline 含 `/ray/` 路径段都纳入 pattern;清完必须验证 GCS/dashboard 端口释放;把"本轮用的 ray pattern 全集"沉淀下来复用(本次教训:pattern 是逐步踩出来的)。

### P24. ray 控制面 socket 路径超 107 字节 → MetricsHead 启动失败
- **症状**(09-20 打桩 round1 首启):报 `AF_UNIX socket path length exceeds the maximum socket path length`(或等价)后失败。
- **根因**:ray dashboard/agent socket 落在 temp-dir 下,`<temp-dir>/<job>/<长 id>/...` 拼出的 AF_UNIX socket 绝对路径超过 107 字节(Linux `sun_path` 上限)。tag/路径越长越易触发。
- **修复**:缩短运行时临时路径——`VIME_TAG` 用短名,`RAY_TMPDIR`/`--temp-dir` 放短路径(`/tmp/ray_<短tag>`),必要时缩短 job 名。
- **预防**:tag/路径长度是硬约束——计划 tag 先估 `len(temp_dir)+len(job) < 107`;每次加后缀(roundN 之类)复查总长。

### P25. mcore 默认 MLA rope_type=yarn 误套,训练侧需显式 rope_type="rope"(09-22,glm4.7 R1 隔离)
- **症状**:训练侧 rope 段与推理侧不一致(YaRN 配置**逻辑级**差异);推理侧 HF `rope_scaling=None` 是标准 RoPE。
- **根因**:mcore 默认 `args.rope_type = "yarn" if args.multi_latent_attention else "rope"`(vime `arguments.py`);GLM-4.7 MLA 被误套 yarn+40,污染 rope 段。
- **修复**(隔离或修复都是这一行):bridge provider 显式 `provider.rope_type = "rope"`。
- **预防**:MLA 模型先查 HF `rope_scaling`;为 None 时确认训练侧 `rope_type` 未被默认 yarn 污染;打桩隔离轮改配置时记录为"诊断残留"而非"功能修复"(见 P29)。

### P26. 恢复代码时甄别 [DUMP] 标记的功能修复 vs 诊断残留(09-22,恢复阶段)
- **症状**:恢复时 bridge 里 `qk_layernorm=True`、MLA 权重映射(`q_a_layernorm`)与 `mtp_num_layers=0`、`rope_type="rope"` **都带 [DUMP] 注释**,无法一眼区分该保留还是回退。
- **根因**:采集接入期加的功能修复(模型能跑的前提)与诊断隔离(验证差异点)都顺手标了 [DUMP];恢复阶段靠"有没有标记"分不清。
- **修复**:用 **checkpoint 权重键名**做三方对照(git HEAD / `.bak_dump` 备份 / 磁盘 HF checkpoint 键名)验证——旧映射 `q_layernorm→q_norm`/`k_layernorm→k_norm` 在 HF checkpoint(实为 `q_a_layernorm`/`kv_a_layernorm`)下**必然加载失败** → 这些改动是功能修复(**保留**);`rope_type="rope"` 是隔离轮诊断残留(**回退**)。删除 `.bak_dump` 前先 `diff` 备份确认无其他未覆盖改动。
- **预防**:接入期给"功能修复"与"诊断隔离"用**不同标记前缀**(如 `# FIX:` vs `# ISOLATE:`),恢复时有据可依;恢复不是无脑 checkout,功能修复回退会把训练打回跑不起来。

### P27. vLLM 引擎权重传输挂起 → SKIP_WEIGHT_TRANSFER 直接磁盘加载(09-22,R0 反复启动 12 次)
- **症状**:R0 job 反复启动失败,卡在 vLLM 引擎权重传输(HCCL 慢路径/挂起);加 `export SKIP_WEIGHT_TRANSFER=1` 后跳过即成功。
- **根因**:vLLM 引擎经 HCCL 从训练侧收权重慢/挂起;而两侧本就从**同一磁盘 HF 权重**加载,该传输纯冗余。
- **修复**:采集/打桩脚本加 `export SKIP_WEIGHT_TRANSFER=1`(vime 侧 env-gated:跳过 vLLM 权重传输,两侧各自从磁盘加载;训练 bridge 578/578 加载 + vLLM 6.23GB)。
- **预防**:两侧同源磁盘权重时**默认跳过引擎权重传输**;R0 启动失败先分"引擎起不来" vs "权重传输挂"两类原因,别在后者上反复重试。

## 已单独记录的教训

- **设备分配(09-10,最重的一次)**:改动前先确认卡号来源,先做原生对照,不得凭记忆二次映射卡号。事故细节见 `references/vime.md`;已沉淀为 SKILL.md 阶段 3 通用警示,不重复。
