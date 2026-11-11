# 框架画像

## 识别口径

识别按三级优先，`comm` 只作辅助：

1. **`npu-smi info` 卡↔进程**（权威锚点，同卡进程强相关）；
2. **`/proc/<pid>/cmdline`**（框架入口、启动脚本、模块名）；
3. **`/proc/<pid>/comm` 与线程 `comm`**（最弱——`comm` 按 `TASK_COMM_LEN=16` 截断到 15 字符）。

> **通用名不作识别依据**：`python` / `python3` / `python3.x` 只能说明「这是 Python 进程」，torchrun、msrun、vLLM、SGLang、verl 的宿主进程都会收敛成它，无法区分框架，因此**不列入下表、也不进 `discover.py` 关键字**。框架判定只认 `cmdline` 的框架特征或 `ray::` 前缀。

线程名到角色的判定关键字见 `process-roles.md`「线程角色映射」（唯一来源 `scripts/roles.py`），本文件不重复维护。

**确认度**：**高** = 官方行为或已实测；**中** = 版本相关、常见但未逐一实测；**待核** = 需现场采集确认，确认前不进入 `discover.py` / `roles.py` 关键字。

## 框架进程/线程名

| 框架 | 宿主进程 comm | 典型线程 comm | cmdline 特征 | 绑核策略 |
| --- | --- | --- | --- | --- |
| vLLM | `VLLMWorker`（中，v1 worker 进程标题）；`EngineCore`（中，v1 引擎核心进程） | `pt_main_thread`、`vllm`、`pt_autograd`、`acl_thread`、`hccl*`、`zmq`（中） | `vllm serve`、`-m vllm.entrypoints.openai.api_server` | 单实例连续核集；调度/分词线程与计算核分离；网络中断亲和同 node |
| SGLang | `sglang`、`sglang::scheduler*`、`sglang::detokenizer`、`sglang::tokenizer`（待核，进程标题随版本变化） | `scheduler`、`tokenizer`、`detokenizer`、`zmq`、`hccl*`（待核） | `sglang.launch_server`、`-m sglang.launch_server`、`--model-path` | 单实例连续核集；调度/分词线程与计算核分离；网络中断亲和同 node |
| verl（RL） | `ray::<类名>`（高，如 `ray::ActorRollo`——已截断）；`raylet`、`gcs_server`（高） | `ray::` 前缀工作线程（高）；`hccl*`、`pt_autograd`（中） | `verl`、`ray::`、`trainer.main_ppo` | 按角色分核：actor 计算核、rollout 推理核、Ray 调度线程隔离 |
| Megatron / msrun | `torchrun`（中，launcher）；`msrun`（中，昇腾 launcher） | `pt_main_thread`（高，PyTorch 主线程名）、`hccl*`、`dataloader`、`pt_autograd`、`omp`（高/中） | `torchrun`、`msrun`、`pretrain_*.py`、`megatron` | rank 均分物理核，通信线程独立核 |
| MindSpeed（LLM / MM） | 同 Megatron | 同 Megatron，另加 MindSpeed 并行/通信线程（待核） | `mindspeed`、`torchrun`/`msrun` | 同 Megatron；额外关注 MindSpeed 并行/通信线程与 GE/OMP 线程数匹配 |

> `ray::<类名>` 会被截断（如 `ray::ActorRolloutRefWorker` → `ray::ActorRollo`），匹配必须用前缀 `ray::` 而非全名；`sglang::*` 同理（`sglang::detokenizer` → `sglang::detoke`），用前缀 `sglang::` 匹配。
>
> 训练/推理的 rank 进程 comm 普遍回落为 `python`，因此上表对 Megatron / MindSpeed 只列 launcher 与线程名；判定该进程属于哪个框架，以 `cmdline` 为准。

已入码的框架关键字（`discover.py: FRAMEWORK_PATTERNS`）：`vllm` / `vllmworker` / `sglang` / `verl` / `ray::` / `raylet` / `mindspeed` / `megatron` / `torchrun` / `msrun`。表中其余名字为**识别参考**，未经实测确认前不进代码。

覆盖范围：vLLM / SGLang（推理）、verl（RL）、Megatron/msrun / MindSpeed（训练）。
MindIE 暂不纳入；新框架按同名结构扩展。

## 识别命令

```bash
# 进程名与命令行
cat /proc/<pid>/comm
tr '\0' ' ' < /proc/<pid>/cmdline; echo

# 线程名分布（按出现次数排序，用于核对上表与 roles.py 关键字覆盖）
for t in /proc/<pid>/task/*; do cat "$t/comm"; done | sort | uniq -c | sort -rn | head -30

# 卡↔进程权威锚点
npu-smi info

# 进程树：先按 cmdline 特征缩小候选（python 只作兜底枚举，不作为识别依据）
ps -eo pid,ppid,comm,args --forest | grep -E 'vllm|sglang|ray::|torchrun|msrun|mindspeed' | grep -v grep
# 仍找不到时再放宽到 python，并逐个人工核对 cmdline
ps -eo pid,ppid,comm,args --forest | grep -E '[p]ython'
```

## 新增框架或进程名的入码流程

1. 先在本表登记，确认度标「待核」；
2. 现场采集 ≥2 台机器或 ≥2 个版本，确认名字稳定；
3. 框架识别加 `discover.py: FRAMEWORK_PATTERNS`，线程角色加 `roles.py: ROLE_PATTERNS`；
4. 在 `tests/skills/msagent-perf-cpu-bind-tune/scripts/` 下补对应用例（如 `test_discover.py` / `test_roles.py`）；
5. 本表确认度改「高」，并在 `process-roles.md` 同步（若涉及新角色语义）。

## 训练 / 推理侧重

| 维度 | 训练 | 推理 |
| --- | --- | --- |
| 进程模型 | 多 rank、对称；RL 为 actor/rollout 多角色 | 多实例或单引擎多 worker（vLLM / SGLang） |
| 主要压力 | 计算 + HCCL + dataloader；RL 增加 rollout 推理 | 请求调度 + 网络中断 + KV cache |
| 分核重点 | rank 均分、通信/数据独立核；RL 按 actor/rollout 分核 | 实例隔离、调度/分词线程与计算核分离、网络中断亲和 |
