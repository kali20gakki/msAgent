---
name: uwc-verify
description: 自动验证 veRL + Megatron-core + vLLM 训推权重转换一致性（UWC）的 5 项检查：Bridge P0/P2、传输 P2_sender/P5、data_ptr 覆盖、G0/G1 前向、dummy vs safetensors 差分。注入采集钩子、采集权重/指针/日志，生成 report.md / report.html / report.json。当用户要验证 RL 训练与推理权重同步、排查 rollout 输出乱码/漏同步、或复现《RL 权重一致性验证》文档时使用。
---

# uwc-verify

把《权重转换一致性验证》文档里的 5 项手工检查变成可执行的 skill：向 veRL 源码注入 best-effort 采集钩子，按阶段运行用户命令，离线比对并生成报告。

## 适用场景

- veRL（Megatron-core 训练 + vLLM 推理）升级后，验证 HF↔mcore↔推理进程的权重链路。
- RL rollout 输出乱码、怀疑某些层漏同步、shared-memory / IPC 传输异常。
- 需要一份可归档的权重一致性报告（Markdown/HTML/JSON）。
- 仅支持 veRL；vLLM 单机/共置（colocate）与 agent-loop 异步路径均可埋点。

## 五项检查

| # | 检查 | 采集物 | 判定 |
|---|------|--------|------|
| 1 | Bridge 往返 | `P0_hf_original_*.pt`、`P2_hf_exported_*.pt` | 逐张量 `atol=1e-3`（可配），缺 key/形状不符即失败 |
| 2 | 传输一致性 | `P2_sender_*.pt`、`P5_receiver_*.pt` | 默认逐位一致（共享内存/IPC 不应改变数值） |
| 3 | data_ptr 覆盖 | `ptr_before_after_*.json` | 同步轮开始/结束各快照一次：`data_ptr` 或内容采样变化即视为已同步（vLLM 原地 `copy_` 时指针不变，只看指针会全员误报） |
| 4 | 前向一致性 | `G0_logits_gt.pt`、`G1_logits_mcore.pt` | 逐位置 cosine：均值 >0.99 且最小 ≥0.95 |
| 5 | dummy vs safetensors | `dummy_/safetensors_rl_first_step_output.json` | 2×2 乱码矩阵（token ids 用 `model.hf_path` 的 tokenizer 离线解码）：dummy 乱码+safetensors 正常=漏同步；均乱码=值错误；均正常=通过。两次输出完全相同是正常的，只有两次记录的 `load_format` 也相同才判“覆盖未生效” |

报告状态：`pass` / `fail` / `partial`（部分项缺数据）/ `skip`。

## 工作流

```bash
# 0) 环境与配置自检
python scripts/uwc_verify.py doctor --config uwc.yaml

# 1) 注入钩子（幂等；可先 --dry-run）
python scripts/uwc_verify.py prepare --config uwc.yaml --dry-run
python scripts/uwc_verify.py prepare --config uwc.yaml

# 2) 按阶段运行训练/推理命令（命令写在 uwc.yaml 的 commands 里）
python scripts/uwc_verify.py run --config uwc.yaml --stage 1,2,3,4,5
# 已有 dump 时只跑缺失阶段：--reuse；stop_when 只认本次运行新写入的文件

# 3) 离线比对 + 报告
python scripts/uwc_verify.py compare --config uwc.yaml
python scripts/uwc_verify.py report  --config uwc.yaml

# 4) 撤销所有源码改动
python scripts/uwc_verify.py prepare --config uwc.yaml --restore
```

一步式：`python scripts/uwc_verify.py all --config uwc.yaml`。
演示（合成数据，无需 verl/NPU）：`python scripts/uwc_verify.py demo --out /tmp/uwc_demo`。

## 配置

复制 `uwc.example.yaml` 为 `uwc.yaml`，至少填写：

- `verl.path`：veRL 仓库根目录（含 `verl/` 包）。
- `model.hf_path`：HF 权重目录（阶段 1/前向验证用）。
- `dump_dir`：采集输出目录。
- `commands`：五个阶段各自要执行的命令。skill 会用 `UWC_*` 环境变量包裹后执行，并在 `stop_when` 指定的文件出现或超时后停止进程组。

环境变量优先级：`UWC_VERL_PATH`、`UWC_DUMP_DIR`、`UWC_MODEL_PATH` 可覆盖配置。

## 采集开关（由 skill 自动注入）

- `UWC_ENABLE=1`：总开关。
- `UWC_STAGE1/2/3=1`：阶段开关（默认开）。
- `UWC_DUMP_DIR`、`UWC_STEP`、`UWC_DIFF_TAG=dummy|safetensors`。
- `UWC_PTR_CHECK=1`：阶段 3 在同步轮前后快照 `data_ptr` + 内容采样。
- `UWC_LOAD_FORMAT=dummy|safetensors`：阶段 5 写入采集记录，差分时用来判断两次运行的加载格式是否真的不同。
- `VERL_FORCE_USE_SHM=1`：强制走共享内存路径（会同时改写 `self.use_shm = ...`）。

### 故障注入（仅用于验证 skill 自身，默认关闭）

以下变量只在做负向验证时显式设置，正常使用不要设置：

- `UWC_FAULT_STAGE1=<子串>:<增量>`：Bridge load 后扰动 mcore 参数，使 P2 与 P0 不一致（R1）。
- `UWC_FAULT_STAGE2=<子串>:<增量>`：P2_sender 落盘后扰动实际发送的张量，使 P5 与 P2_sender 不一致（R2）。
- `UWC_FAULT_STAGE3_SKIP=<子串,...>`：记录 P5 后从发给 vLLM 的权重里丢弃匹配层，制造漏同步（R3）。
- `UWC_FAULT_STAGE4=<子串>:<增量>`：mcore 前向前对命中参数的**整个张量**做偏移，使 G1 与 G0 的 cosine 下降（R4；只改单个元素通常不足以影响 cosine）。
- `UWC_FAULT_STAGE5_SAME_FORMAT=1`：强制两次 stage 5 都以 dummy 运行并记录 `load_format=dummy`（R5）。

## 关键文件

- `scripts/uwc_verify.py`：CLI（doctor/prepare/run/compare/report/all/demo）。
- `runtime/uwc/hooks.py`：运行时埋点，按模块角色包装 Bridge load/export、`_update_weights`、bucket send、`generate_sequences`。
- `runtime/uwc/collectors/`：P0/P2/P2_sender/P5/data_ptr/首步输出采集。
- `runtime/uwc/offline/`：比对、2×2 差分、前向验证、报告渲染。
- `../../../tests/skills/uwc-verify/`：离线比对、钩子注入/恢复、demo 端到端测试。
- [references/reference.md](references/reference.md)：dump 目录布局、各阶段环境变量、判定阈值与参数说明。

## 真实环境（Ascend + veRL 0.9.1）

在 8×910B4（32 GiB/卡）容器里实测：TP=4/DP=2、`GPU_MEMORY_UTILIZATION=0.25`；显存更紧时可用 4 卡 TP=4/DP=1，并适当降低 vLLM 显存比例。各阶段启动命令直接填用户自己的脚本。

veRL 0.9.1 差异：sender 入口是 `async_send_weights`（0.8 及以前叫 `send`，两个都包装）；Bridge 从 HF 初始化依赖 `use_mbridge=True` + `use_dist_checkpointing=False`，否则阶段 1 不经过 `load_hf_weights`。

## 报告怎么读

1. 先看 `overall.status` 与每项 `status`。
2. 第 1 项失败：Bridge 转换问题，看 `report.json` 的 `worst` / missing keys。
3. 第 2 项失败：传输层改变了数值，优先检查 `use_shm`、bucket 边界、dtype 转换。
4. 第 3 项失败：`not_synced` 列出的参数就是漏同步层；结合第 5 项定位 vLLM 覆盖范围。
5. 第 4 项失败：训练侧 mcore 权重或前向配置不一致（dtype/并行切分/输入不一致）。
6. 第 5 项失败但前几项通过：vLLM 加载格式覆盖参数未生效，或只同步了部分层。

## 限制

- 埋点是 best-effort：collector 异常只打印 `[UWC] WARNING`，不会中断训练。
- 只覆盖 per-tensor 同步路径；若 veRL 使用 shard/delta 路径，会打印警告，需强制切回 per-tensor。
- 阶段 2 会把整份导出权重复制到主机内存用于 `P2_sender` 落盘（Qwen3-4B bf16 约 8 GB/rank，逐个张量拷贝，不额外占用设备显存）；与 vLLM 共置、显存紧张时同步阶段仍可能 OOM，可降低 `GPU_MEMORY_UTILIZATION` 重试。
- 阶段 4 的 mcore 前向验证需要单卡可加载的 mcore 权重与 Megatron-Bridge（自动兼容旧 `mbridge` 包）；多卡并行模型需先转换/合并。
- `run` 通过文件出现判断阶段完成，超时与命令需按实际训练时长配置。
- 修改 veRL 源码前建议提交/备份；`prepare --restore` 可逐字节还原（依赖 manifest）。
