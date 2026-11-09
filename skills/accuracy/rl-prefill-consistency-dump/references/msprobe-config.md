# msprobe PrecisionDebugger 配置与接口

所有训推一致性采集都基于 msprobe 的 `PrecisionDebugger`。本文件是配置模板和接口速查。更完整说明见 msprobe 文档 `pytorch_data_dump_instruct.md` 和 `config_json_introduct.md`。

## 接口速查

```python
from msprobe.pytorch import PrecisionDebugger, seed_all

seed_all()   # 固定随机种子 + 开启确定性计算
debugger = PrecisionDebugger(config_path="./config.json")  # 尽早实例化,在导包之后、模型之前

debugger.start(model=model, token_range=None, rank_id=None, scheduled_tokens=None)
#   model          采集 Module 级数据时必传(L0/mix)。只 dump 传入层的子层。
#   token_range    [start, end] 闭区间,推理逐 token 采集用
#   rank_id        自定义 rank(默认 get_rank;SGLang DP 等重复 rank 场景必须传,如 self.gpu_id)
#   scheduled_tokens  {"request_id": token数},配合 config 的 request_id 按请求切片
#   前向代码
debugger.stop()   # 必须调用,否则落盘不全
debugger.step()   # 结束一个 step,落盘并推进(放 stop 之后);尽量放 loss.backward 之后,否则反向数据可能丢失
```

其他接口:
- `module_dump(module, name)` / `module_dump_end()` 只 dump 指定模块
- `save(variable, name, save_backward=True)` **单点保存,补 L0/L1/mix 盲区**(需 `level="debug"`)
- `set_init_step(n)` 改起始 step;`register_custom_api(module, api, prefix)` / `restore_custom_api` 注册自定义 API
- `seed_all(seed=1234, mode=False, rm_dropout=False)` 固定种子 + 确定性;**不保证模型输入一致**,需自行关 shuffle

## config.json 模板(训推一致性用)

```json
{
  "task": "statistics",
  "dump_path": "/abs/path/to/dump",
  "rank": [],
  "step": [0],
  "level": "L0",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

关键字段:

| 字段 | 取值 | 说明 |
|---|---|---|
| `task` | `statistics` / `tensor` / `nan_check` / `acc_check` | statistics=统计量(轻量,首选);tensor=完整张量;nan_check 仅 PyTorch(L1);acc_check 仅 PyTorch,采集勿选 |
| `level` | `L0` / `L1` / `mix` | L0=Module 级(需 start 传 model);L1=API 级;mix=两者 |
| `step` | 数组 | 采集哪些 step,训推一致性通常 `[0]` |
| `rank` | 数组 | 空=全部(单卡必须为 `[]`) |
| `async_dump` | bool | tensor 异步必须配 `list`;有显存溢出风险 |
| `dump_enable` | bool | 动态启停 dump,建议初始 false |
| `extra_info` | bool | 控制 stack.json / construct.json(L1 时 construct 为空) |
| `statistics.summary_mode` | `statistics` / `md5` / `xor` | 统计量 / CRC-32 校验 / XOR 校验(**xor/md5 都存字段 `md5`,实际是 CRC-32**)。确定性比对可用 md5/xor 提速 |
| `statistics.data_mode` | `["all"]` | 采前反向全部 |
| `statistics.scope` | `[start, stop]` | **长度须为 2**,锁定采集区间 |
| `statistics.request_id` | 字符串 | 配合 `scheduled_tokens` 按请求切片 |

`PrecisionDebugger` 构造参数(`config_path/task/dump_path/level/step`)与 config 等价且**优先级高于 config**;但构造参数可配项少于 config.json(rank/async_dump/extra_info 只能在 config 配)。

## dump 结果目录

```text
{dump_path}/
├── step0/
│   └── rank{ID}/
│       ├── dump.json              # 统计量/校验值(必读)
│       ├── stack.json             # 调用栈
│       ├── construct.json         # 分层结构(L1 时为空)
│       └── dump_tensor_data/      # task=tensor 时的 *.pt 张量文件
└── ...
```

非分布式进程目录名是 `proc{pid}`;大模型场景可能 rank 和 proc 并存(proc 是 CPU 侧预处理,一般不比对)。

## dump.json 字段

```json
{
  "task": "tensor",
  "level": "L0",
  "framework": "pytorch",
  "dump_data_dir": "/dump/path",
  "data": {
    "Module.conv2.Conv2d.forward.0": {
      "input_args": [{"type":"torch.Tensor","dtype":"torch.float32","shape":[8,16,14,14],
                      "Max":1.63,"Min":0.0,"Mean":0.25,"Norm":70.5,"requires_grad":true,"data_name":"...input.0.pt"}],
      "input_kwargs": {},
      "output": [...],
      "parameters": {"weight": {...}, "bias": {...}}
    },
    "Module.conv2.Conv2d.parameters_grad.0": {...},
    "Module.conv2.Conv2d.backward.0": {"input":[...], "output":[...]}
  }
}
```

- L0 命名:`Module.{name}.{class}.forward.{n}` / `.backward.{n}` / `.parameters_grad.{n}`;L1 命名:`{Api}.{name}.{n}.forward/backward`。
- 每个 tensor 含 `Max/Min/Mean/Norm(L2)/dtype/shape/data_name`;`summary_mode=md5/xor` 时字段名为 `md5`。
- NZ 格式 tensor 的 Max/Min/Mean/Norm 为 `null`;非 float16/32/bf16 只算 Max/Min。
- 比对时两侧取相同语义边界的条目(dump.json 内同名字段),shape/dtype 一致才可比。

## 输出前缀与 torch 模块对应

`Tensor`→torch.Tensor,`Torch`→torch,`Functional`→torch.nn.functional,`NPU`→torch_npu,`Aten`→torch.ops.aten,`Distributed`→torch.distributed,`MindSpeed`→mindspeed.ops。

## 注意事项

- 仅支持 PyTorch;PyTorch ≥2.7 的 dynamo 场景不支持。eager 模式才可钩住算子。
- 接入 msprobe 可能改变 loss/gnorm(工具 item 操作引入同步 + hook 机制),比对前确认可接受。
- 可配置 `dump_enable` 动态启停 dump。
- 不要采集不需要的 API/输出:改 `support_wrap_ops.yaml`(完全不采)或 `builtin_ignore_ops.yaml`(采调用但屏蔽输入/输出,常用于通信算子收缓冲区)。
