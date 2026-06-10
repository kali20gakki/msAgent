# 环境准备

**Load when:** 进入量化配置调优前，需确认硬件、运行环境就绪。

## msmodelslim 安装（强制）

量化调优通过 **当前 shell 环境** 调用 `msmodelslim` CLI（`analyze` / `quant` 等），编排层脚本通过 Python 直接 `import msmodelslim`。

1. 确认已安装 Ascend 工具链与 CANN（NPU 场景）。
2. 安装 msmodelslim（版本以项目要求为准）：

```bash
pip install msmodelslim
```

3. 验证：

```bash
python -c "import msmodelslim; print('ok')"
```

4. 若模型适配阶段执行过 `bash install.sh`，脚本会在**同一 Python 环境**中读取新 adapter，**无需**重启 MCP 服务；继续调优前确认 `import` 无报错即可。

## Ascend 运行环境变量

原先写在 `config.mcp.json` modelslim 条目中的环境变量，需在**执行量化/评测脚本的 shell** 中配置（或写入用户 `~/.bashrc` / 启动脚本）。典型项包括：

- `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` / `ASCEND_AICPU_PATH` / `ASCEND_OPP_PATH`
- `ATB_HOME_PATH` 及 ATB 相关 tuning 变量
- `LD_LIBRARY_PATH`（含 Ascend driver、ATB、opp 等路径）
- `PYTHONPATH`（含 ascend-toolkit `python/site-packages` 与 tbe 路径）

以本机 CANN 安装路径为准；不同机器路径可能不同，**禁止**假设固定目录存在。

## NPU 资源检查（强制）

环境检查时必须执行以下步骤：

1. 运行 `npu-smi info` 获取全部卡信息，确认**总卡数**（不可假设为 8 卡，不同机器卡数不同）
2. 逐卡检查 HBM 占用和进程列表，判断哪些卡空闲（注意：HBM 有少量占用且有进程运行才表示被占用；仅有少量 HBM 占用但无进程通常为驱动占用，可视为空闲）
3. 将空闲卡号列表回显给用户确认后，才确定为量化/评测设备

回显中必须体现：

- 机器总卡数
- 每张卡的 HBM 占用量和进程情况
- 最终选用的卡号列表及用途（量化卡号 / 评测卡号）

## 命令调用方式

- **敏感层分析 / 量化**：通过 `execute` 运行 `msmodelslim analyze`、`msmodelslim quant`（参数见各 Skill 文档）。
- **编排层 / 校验 / 评测**：通过 `execute` 运行 skill 目录下脚本，例如：

```bash
python skills/quantization-accuracy-tuning-orchestrator/scripts/history_clear.py --save-path /path/to/workdir
```

编排脚本 **stdout** 输出单行 JSON（`{ok: ...}`）；CLI 以 **exit code** 判定成败。

## 环境就绪确认

向用户回显：msmodelslim 可 import、Ascend 环境变量已配置、NPU 卡号已确认。获得用户认可后再进入模型准备阶段。
