# uwc-verify 参考

## 1. 采集目录布局

`dump_dir` 由 `UWC_DUMP_DIR`（或配置 `dump_dir`）决定：

```
dump_dir/
├── runs.json                    # 每个阶段命令的 command/rc/log/耗时
├── summary.json                 # compare 的中间汇总（含原始 stage report）
├── report.json                  # 最终报告（可程序化消费）
├── report.md
├── report.html
├── logs/
│   ├── stage1_<ts>.log
│   └── ...
├── stage1_bridge/
│   ├── P0_hf_original_step0_rank0.pt      # load_weights 之前的 HF 权重
│   ├── P2_hf_exported_step0_rank0.pt      # load_weights 之后 export 的 HF 权重
│   ├── export_diag_step0.json             # 前 3 个张量的统计 + vs_P0_max_diff
│   ├── G0_logits_gt.pt                    # HF 前向 logits
│   └── G1_logits_mcore.pt                 # mcore 前向 logits
├── stage2_transfer/
│   ├── P2_sender_step0_rank0.pt           # 发送端实际交给传输的 (name, tensor) 列表
│   ├── P5_receiver_step0_rank0.pt         # vLLM _update_weights 收到的列表
│   └── transport_metadata_step0.json      # 名称/形状/dtype/nbytes
└── stage3_vllm/
    ├── ptr_before_after_step0_rank0.json  # data_ptr 前后对比
    ├── dummy_rl_first_step_output.json    # load_format=dummy 首步输出
    ├── safetensors_rl_first_step_output.json
    └── diff_test_report.json              # 2×2 差分结论
```

比对器会自动选择最新 step 的文件；也可用 `uwc_compare --step N` 指定。

## 2. 环境变量

| 变量 | 作用 |
|------|------|
| `UWC_ENABLE=1` | 总开关；未开启时 `install_module` 直接返回 |
| `UWC_STAGE1/2/3` | 分阶段开关，默认开启 |
| `UWC_DUMP_DIR` | 采集根目录，默认 `/tmp/uwc_dump` |
| `UWC_STEP` | 写入文件名/报告的 step，默认 0 |
| `UWC_PTR_CHECK=1` | 在同步轮前后记录 `data_ptr` + 内容采样 |
| `UWC_DIFF_DUMP=1` | 允许导出首步生成结果 |
| `UWC_DIFF_TAG` | `dummy` / `safetensors`，决定输出文件名 |
| `UWC_LOAD_FORMAT` | `dummy` / `safetensors`：既是 `UWC_DIFF_TAG` 的回退值，也会写入采集记录，供差分判断覆盖是否生效 |
| `VERL_FORCE_USE_SHM=1` | 强制 `use_shm=True`（文本改写 + 运行时兜底）；`patch.use_shm: true` 时 `run` 会自动导出 |
| `RANK` / `LOCAL_RANK` | 只有 rank 0 写盘；所有 rank 仍会消费 export 的 collective |
| `UWC_FAULT_STAGE1/2/4` | 仅负向验证：`<参数名子串>:<增量>`，在 Bridge load / P2_sender 落盘 / mcore 前向之前注入扰动；stage 1/2 改一个元素，stage 4 改整个张量（单元素不足以影响 cosine） |
| `UWC_FAULT_STAGE3_SKIP` | 仅负向验证：逗号分隔的名字子串，记录 P5 后把这些张量从发给 vLLM 的列表里删除，制造漏同步 |
| `UWC_FAULT_STAGE5_SAME_FORMAT=1` | 仅负向验证：两次 stage 5 都强制 `LOAD_FORMAT=dummy` 并记录为 dummy |

`run` 会按阶段自动设置这些变量，通常不需要手工导出。`--stage 1/2/3` 只运行对应的一条命令，`--stage 5` 会连续跑 dummy 与 safetensors 两次。`stop_when` 只认本次启动之后写入（mtime 不早于进程启动时间）的 dump：上一轮遗留的文件会被忽略并打印 `[run] ignoring ...`，避免秒退。

第 3 项的判据是「轮次开始时对每个参数记录 `data_ptr` + 内容采样，整轮结束后再比一次」：指针变化或采样值变化都算已同步。vLLM 的 `model.load_weights` 是 `copy_` 原地写入，指针在成功同步时也不会变（实测 290/290 层 pointer 不变、content 全变），只比指针会全员误报。

第 5 项同理不能把「两次输出相同」直接判失败：两次运行最终拿到同一份 actor 权重，贪心解码下首步输出本就应当一致；只有两条记录里的 `load_format` 也相同（说明 `LOAD_FORMAT` 覆盖没生效、两次其实跑的同一格式）才判失败。veRL 0.9 的 rollout 走 `vLLMHttpServer.generate`，采集记录里只有 token ids / log_probs；`compare` 会用 `model.hf_path`（或 `compare.tokenizer`）的 tokenizer 离线解码 `payload.responses`，再用文本乱码矩阵判定。`compare.trust_remote_code`（默认 `true`）控制 tokenizer / 模型加载是否允许执行模型仓库自带的自定义代码；权重目录来源不受信任时设为 `false`。

所有 `.pt` 落盘都是「先写 `.tmp` 再 `os.replace`」的原子写：`run` 用文件出现判断阶段完成并杀掉进程组，非原子写会让它把写了一半的文件当成完成品。

## 3. 钩子机制

`prepare` 在每个目标文件末尾追加：

```python
# >>> UWC-SKILL-PATCH v1 (managed; uwc-verify restore removes this) >>>
try:
    from verl.utils.uwc.hooks import install_module
    install_module(__name__, globals())
except Exception as _uwc_exc:
    print(f"[UWC] hook install skipped: {_uwc_exc}", flush=True)
# <<< UWC-SKILL-PATCH v1 <<<
```

同时把 `runtime/uwc` 复制到 `verl/utils/uwc`，并写 `.uwc_skill_manifest.json`（记录 patch 目标与 `use_shm` 文本改写）。

`install_module` 根据模块路径选择角色：

| 模块 | 包装对象 | 采集 |
|------|----------|------|
| `workers.engine.megatron.transformer_impl` | `_build_tf_config` / `initialize` / `__init__` | 安装 Bridge 包装；兜底导出 P2 |
| 同上 | `get_per_tensor_param` | 物化生成器 → P2_sender；打印 export 诊断 |
| 同上 | `get_per_tensor_param_shard/delta_shard` | 仅告警：shard 路径不覆盖 |
| `workers.rollout.vllm_rollout.utils` | `update_weights_from_ipc`（轮入口）/ `_update_weights`（每 bucket） | P5 按轮汇总；`UWC_PTR_CHECK=1` 时在轮首/轮末做 ptr + 内容采样快照 |
| `...bucketed_weight_transfer` | `BucketedWeightSender.send` / `async_send_weights` | 打印 `use_shm`/bucket/zmq，tee 前 3 个权重 |
| `...vllm_rollout.vllm_rollout` | `__init__` / `generate_sequences` | `VERL_FORCE_USE_SHM` 兜底；首步输出 |
| `...vllm_rollout.vllm_async_server` | `vLLMHttpServer.generate` | 异步 rollout 的首步输出（veRL 0.9 真实路径） |
| `workers.engine_workers` | `update_weights` | 起止日志 |
| `experimental.agent_loop.agent_loop` | `generate_sequences` | 异步 agent-loop 首步输出（可选） |

Bridge 的两代方法名都会包装：`load_weights`（Megatron-Bridge）与 `load_hf_weights`（新 veRL）。导出同时支持 `export_weights` 与 `export_hf_weights(models, conversion_tasks=...)`。sender 入口同时兼容 veRL ≤0.8 的 `send` 与 0.9+ 的 `async_send_weights`。

所有包装器都用 `functools.wraps`，collector 异常被捕获并打印 `[UWC] WARNING`，不会改变返回值。

## 4. 比对与名称归一化

- 去掉包装前缀：`module.`、`_orig_mod.`、`model.module.`、`model.model.`（保留一个 `model.`）、`language_model.`。
- 融合投影：`qkv_proj` 按 `hidden_size/num_attention_heads/num_key_value_heads/head_dim` 切分；`gate_up_proj` 按行对半或交错切分。显式的 q/k/v、gate/up key 优先。
- 缺少 `config.json` 时融合 key 无法切分，会在 `meta.unresolved_combined` 中列出并导致两侧 key 不匹配。
- 比较支持 NaN（NaN==NaN 视为一致）、形状不符、缺失/多余 key、逐位或 `atol/rtol`。

## 5. 报告

`report.json`：

```json
{
  "schema": "uwc-report/v1",
  "overall": {"status": "pass", "counts": {"pass": 5, "fail": 0, "skip": 0, "error": 0}, "total": 5},
  "items": [
    {
      "id": "bridge_p0_p2",
      "index": 1,
      "name": "Bridge 转换：P0 vs P2",
      "stage": 1,
      "status": "pass",
      "metrics": {},
      "evidence": [".../P0_hf_original_step0_rank0.pt", "..."],
      "details": {"missing_keys": [], "extra_keys": [], "worst": []}
    }
  ],
  "stage_reports": {"stage1": {}, "stage2": {}, "stage3": {}}
}
```

固定 id：`bridge_p0_p2`、`transport_p2sender_p5`、`data_ptr`、`bridge_g0_g1`、`dummy_vs_safetensors`。
`status`：`pass` / `fail` / `skip`（未采集到文件）/ `error`（采集文件损坏）。
`overall.status`：全 pass → `pass`；有 fail/error → `fail`；只有 skip → `skip`；混合 → `partial`。

## 6. 手工调用

准备完成后，可直接调用离线工具：

```bash
# 名称归一化 + 比较（自动查找 dump_dir 中的最新文件）
python -m verl.utils.uwc.offline.uwc_compare --dump_dir /tmp/uwc_dump \
    --model_config /path/to/model/config.json --atol 1e-3

# 只比较某一阶段
python -m verl.utils.uwc.offline.uwc_compare --dump_dir /tmp/uwc_dump --stage 2

# 2×2 差分（--tokenizer 用于解码只有 token ids 的记录）
python -m verl.utils.uwc.offline.stage3_diff_test \
    --dummy /tmp/uwc_dump/stage3_vllm/dummy_rl_first_step_output.json \
    --safetensors /tmp/uwc_dump/stage3_vllm/safetensors_rl_first_step_output.json \
    --dump_dir /tmp/uwc_dump --tokenizer /path/to/model

# 单卡前向（HF 与 mcore 分别跑）
python -m verl.utils.uwc.offline.forward_verify --backend hf \
    --local_path /path/to/model --input_ids 1,2,3,4,5 --dump_dir /tmp/uwc_dump
python -m verl.utils.uwc.offline.forward_verify --backend mcore \
    --local_path /path/to/model --input_ids 1,2,3,4,5 --dump_dir /tmp/uwc_dump

# 重新生成报告
python -m verl.utils.uwc.offline.uwc_summary --dump_dir /tmp/uwc_dump
```

## 7. 排障

| 现象 | 排查 |
|------|------|
| `hook install skipped: No module named verl.utils.uwc` | 未执行 `prepare`，或运行时目录被删除；重新 prepare |
| 日志有 `hooks installed` 但无 dump | 检查 `UWC_ENABLE=1`、`UWC_STAGEn=1`、`DUMP_DIR` 是否同一路径 |
| 只有 rank0 有文件 | 预期行为；比较只看 rank0 |
| P5 缺失 | 检查 vLLM worker 是否走 `_update_weights`；异步路径可能用别的更新函数 |
| stage2 告警 shard/delta | 强制 per-tensor 同步；本 skill 不采集 shard 路径 |
| P0/P2 key 对不上 | 提供 `model.config.json`；检查融合 qkv/gate_up 布局 |
| 第 3 项大量 not_synced | vLLM 未覆盖这些层；结合第 5 项与 vLLM 日志定位 |
| `run` 提前结束 | 旧 dump 已按 mtime 过滤；若仍提前结束，检查命令本身是否立刻退出 |
| 第 5 项把 `completed` 当文本 | 已修复：`stop_reason` 属于元数据，`TokenOutput` 记录按 token 序列比对 |
| 进程不退出 | 调小 `timeouts`；skill 会 SIGINT→SIGTERM→SIGKILL 整个进程组 |

## 8. 已知限制

- 文本改写只处理 `self.use_shm = not is_support_ipc()` 这一种写法；其他版本依赖运行时 `__init__` 兜底与 `VERL_FORCE_USE_SHM` 环境变量（veRL 0.9 的 `ServerAdapter` 直接读它）。
- 同步接口走 `generate_sequences`，veRL 0.9 异步 rollout 走 `vLLMHttpServer.generate`，两者都已打补丁；agent-loop 仅在 `agent_loop.py` 存在且被 `prepare` 打上补丁时覆盖。
- 前向验证默认单卡、fp32；bf16 下 cosine 阈值需按模型放宽。
- demo 报告使用的是随机合成权重，仅用于验证工具链，不代表任何真实模型结论。
