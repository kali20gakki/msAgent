# 接口参考（msprobe.pytorch）

本文给出本技能范围内（PyTorch 动态图 eager）`PrecisionDebugger` 相关接口的函数原型、参数说明与调用示例，
内容依据官方文档 [PyTorch场景精度数据采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/pytorch_data_dump_instruct.md)
的“接口介绍”章节。config.json 字段见 [config-json.md](config-json.md)；L2 与 nan_check 的相关约束分别见
[kernel-dump-l2.md](kernel-dump-l2.md) 与 [nan-check.md](nan-check.md)。

## 0. 导入与通用约定

```python
from msprobe.pytorch import PrecisionDebugger, seed_all
```

- `PrecisionDebugger` 的初始化必须与采集目标**在同一个进程中**，否则无法采集目标数据；实例化放在导包后、迭代循环外（官方原话：请勿将初始化流程插入到循环代码中）。
- `start`/`stop`/`step`/`save` 均为**类方法**：多文件工程可以在持有实例的入口文件之外直接 `PrecisionDebugger.start()`（见 §5 跨文件模式）。
- 标准生命周期：`start()` → 前向/反向计算 → `stop()` → `step()`。`stop()` **必须调用**，否则数据落盘不全；`step()` 必须在 `stop()` 之后。

## 1. PrecisionDebugger（构造）

```python
debugger = PrecisionDebugger(config_path=None, task=None, dump_path=None, level=None, step=None)
```

| 参数 | 说明 |
| --- | --- |
| `config_path` | 可选，str，dump 配置文件路径，如 `"./config.json"`。未配置时使用安装目录下默认 `msprobe/config.json` 的默认配置 |
| `task` / `dump_path` / `level` / `step` | 均可选，与 config.json 同名配置含义一致，**优先级高于 config.json 文件**。全部不配置时默认采集所有 rank、所有 step 的 L1 级别统计（statistics）数据 |

返回值：debugger 实例对象，提供 start/stop/step 等采集控制函数。

## 2. start（开启采集）

```python
debugger.start(model=None, token_range=None, rank_id=None, scheduled_tokens=None)
```

模型初始化之后、目标计算之前调用。

| 参数 | 说明 |
| --- | --- |
| `model` | 可选，需采集 Module 级数据的模型：支持 `torch.nn.Module`、`list[torch.nn.Module]`、`Tuple[torch.nn.Module]`。**level 为 "L0"/"mix" 或 `token_range` 非 None 时必须传入**。只监测一部分时传子模块（如 `model.A`）即可——注意**传入的当前层本身不会被 dump，只 dump 其子层级** |
| `token_range` | 可选，`[int, int]`，推理模型采集的 token 循环始末范围（闭区间），如 `[1, 3]` 采集第 1~3 次循环 |
| `rank_id` | 可选，≥0 整数，自定义 rank ID。默认工具经 `torch.distributed.get_rank` 自动获取；仅在该接口无法得到唯一 rank ID 的场景（如 sglang DP 推理各 worker 独立分布式导致 rank 重复、结果目录互相覆盖）使用，且必须保证各进程唯一，如 `debugger.start(rank_id=self.gpu_id)` |
| `scheduled_tokens` | 可选，dict：key 为请求 id（str），value 为该请求占用的 token 数（≥0 int）。配合 config.json 的 `request_id` 实现按请求切片采集，如 `debugger.start(scheduled_tokens={"chatcmpl-req-0": 10})` |

## 3. stop（关闭采集）

```python
debugger.stop()
```

在 `start()` 之后的任意位置添加。

- 放在反向计算（如 `loss.backward()`）**之后**：采集 start 与 stop 之间的前反向数据。
- 放在反向计算**之前**：须将 `step()` 放在反向计算之后，才能采集到前反向数据。
- **必须调用**，否则可能导致精度数据落盘不全。

## 4. step（结束一个 step）

```python
debugger.step()
```

- 在一个 step 结束的位置添加，**必须在 `stop()` 之后**调用。
- 作用：结束本 step 的数据采集、完成所有数据落盘、更新 dump 参数（下一次 start 采集的数据记入下一个 step）。
- **尽量添加在反向计算之后**（如 `loss.backward()` 之后），否则可能导致反向数据丢失。

## 5. 跨文件模式（start/stop 是类方法）

模型定义与训练入口不在同一文件时：入口文件在导包后实例化 debugger（确保所有 API 被封装），模型文件中直接以类方法打点，无需持有实例：

```python
# train.py（入口，导包后）
from msprobe.pytorch import PrecisionDebugger
debugger = PrecisionDebugger(config_path='./config.json')
```

```python
# module.py（模型定义）
from msprobe.pytorch import PrecisionDebugger

class ModuleOP(nn.Module):
    def forward(self, x):
        PrecisionDebugger.start()
        x1 = self.linear_1(x)
        PrecisionDebugger.stop()
        x2 = self.linear_2(x1)
        return x2
```

## 6. set_init_step（起始 step 计数）

```python
debugger.set_init_step(step)
```

step 数默认从 0 开始计数；调用后从指定值开始计数（如训练已跑到 step 100，配置 `step: [101]` 前先 `set_init_step(101)` 对齐）。**必须写在训练迭代循环开始前，不能写在循环内**。

## 7. save（单点保存，level=debug）

```python
debugger.save(variable, name, save_backward=True)
```

| 参数 | 说明 |
| --- | --- |
| `variable` | 必选，需保存的变量：dict / list / tuple / torch.tensor / int / float / str |
| `name` | 必选，str，保存名称 |
| `save_backward` | 可选，bool，是否保存反向数据，默认 True |

注意：**仅 task 为 `tensor` 或 `statistics` 且 level 为 `debug` 时生效**（不满足条件时接口静默返回），用于在网络执行过程中单点保存前反向数值，落盘为统计值数据或张量文件。完整使用见官方[单点保存工具使用介绍](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/debugger_save_instruct.md)。

## 8. register_custom_api / restore_custom_api（自定义 API）

```python
debugger.register_custom_api(module, api, api_prefix=None)
debugger.restore_custom_api(module, api)
```

把工具支持列表之外的用户自定义 API 注册到 API 级 dump。以 `torch.matmul` 为例：

| 参数 | 说明 |
| --- | --- |
| `module` | 必选，API 所属的包（如 `torch`） |
| `api` | 必选，str，API 名称（如 `"matmul"`） |
| `api_prefix` | 可选，dump.json 中 api 名的前缀，默认为包名的字符串格式（如 `"torch"`） |

调用示例（注册自定义算子 `module_a.compute(args)`）：

```python
import module_a
from msprobe.pytorch import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")
debugger.register_custom_api(module=module_a, api="compute")
# ... 正常训练，start/stop/step 打点不变 ...
debugger.restore_custom_api(module_a, "compute")   # 恢复原 API，取消 dump，按需使用
```

## 9. seed_all（固定随机性）

```python
seed_all(seed=1234, mode=False, rm_dropout=False, is_enhanced=False)
```

| 参数 | 说明 |
| --- | --- |
| `seed` | 随机种子，默认 1234。固定范围包括：`PYTHONHASHSEED`、`HCCL_DETERMINISTIC=True`、`np.random.seed`、`torch.manual_seed`、`torch.cuda.manual_seed(_all)`、`torch_npu.npu.manual_seed(_all)`、cudnn 确定性设置 |
| `mode` | 确定性计算模式，默认 False。True 时调用 `torch.use_deterministic_algorithms(True)` 等开启确定性计算，**会降低 API 执行性能**，建议仅在发现模型多次执行结果不同的情况下开启 |
| `rm_dropout` | 默认 False。True 时将 dropout 类接口的 `p` 置 0，避免随机 dropout 带来的网络随机性。**须在初始化 dropout 实例前调用才能生效** |
| `is_enhanced` | 默认 False。True 时进一步固定 PyTorch、NumPy 以及 Python 内置随机数生成器，同一进程内多次执行相同随机性 API 生成的随机值完全相同 |

注意：`seed_all` 不能保证模型输入一致——CPU/GPU/NPU 输入完全一致时 dump 比对才有意义，**数据集的 shuffle 需用户自行关闭**（如 DataLoader `shuffle=False`）。dump 功能默认不固定随机性，若希望每次采集的数据一致，建议在 dump 前调用 `seed_all()`。

## 10. 常用组合速查

| 场景 | 接口组合 |
| --- | --- |
| 常规训练循环 | 导包后 `seed_all()` + `debugger = PrecisionDebugger(config_path=...)`；每迭代 `start(model)` → forward/backward/optimizer → `stop()` → `step()` |
| 推理（无 optimizer） | `start(model, token_range=[a, b])`（可选范围）→ 推理输出 → `stop()` → `step()` |
| 单点保存 | level=debug + task=tensor/statistics，循环内需要处 `debugger.save(var, "name")` |
| 自定义算子 | 实例化后 `register_custom_api(module, api)`，结束按需 `restore_custom_api` |
| 跨文件 | 入口文件实例化；模型文件 `PrecisionDebugger.start()/stop()` 类方法调用 |
| 多进程 rank 目录冲突（如 sglang DP） | `start(rank_id=<进程唯一 id>)` |
