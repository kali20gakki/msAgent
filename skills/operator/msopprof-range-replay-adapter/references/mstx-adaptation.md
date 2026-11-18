# MSTX 范围接入

## 接口说明

使用 MSTX 接口时应核对与安装版本对应的接口原型和参数语义。

```cpp
mstxRangeId mstxRangeStartA(const char *message, aclrtStream stream);
```

- `message` 为范围标记文字，不能传入空指针。官方说明在 MSPTI 场景下长度不超过 255 字节；这与 custom-input 配置字段的长度上限属于不同约束，需同时满足实际场景要求。
- `stream` 为标记关联的 stream。传入 `nullptr` 时仅记录 Host 侧标记；传入有效 stream 时同时记录 Host 及对应 Device 侧标记。本技能按 msOpProf 范围级重放示例使用 `nullptr`，Python 对应 `None`。
- 标记接口的 `stream=nullptr` 不表示目标 kernel 使用默认 stream，也不会改变 kernel 的执行 stream；范围内目标算子的 stream 仍需保持不变。
- 开始接口返回范围 ID，通用 MSTX 接口约定返回 `0` 表示失败。部分 msOpProf 注入版本的范围 ID 从 `0` 开始分配，并将其登记为有效范围；遇到此情况，须核对当前注入实现、线程范围登记与 capture/replay 产物，不能仅凭 `0` 判断失败，也不能仅凭非零判断成功。结束时应传入本次开始接口返回的 ID，不硬编码或推算 ID；采集已使能时应联合日志与每个预期设备的实际数据检查范围标记有效性。

官方 MSTX 通用示例包含交叉范围，但 msOpProf 的 `--replay-mode=range` 有额外限制。以下同线程、非交叉和单 stream 的适配规则以 msOpProf 的范围级重放文档为依据，不能将通用 API 示例直接用于重放。

## 1. 范围划分

MSTX 标记应位于实际下发目标算子的 Host 线程，普通单卡算子和多卡通算融合算子均可按此方式接入。范围名称统一使用字母、数字和下划线；多卡场景各 rank 的同一逻辑阶段使用相同名称，以兼容流水合并配置。

- 开始和结束接口必须成对调用，不跨线程、不交叉调用；优先使用非嵌套范围。不要仅因为已有嵌套标记就擅自拉平或改变范围粒度。
- 区分交叉与嵌套的调用顺序：交叉为 `start A → start B → end A → end B`；嵌套为 `start A → start B → end B → end A`。已有嵌套范围时，应核对目标 msOpProf 版本及范围使能配置，确认各范围的实际生效与重放行为，不能假定内外层都会独立重放。
- 例如 [msOpProf 典型用例](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/best_practices/typical_cases.md#配合mstx接口实现范围级重放)中 `range3` 嵌套在 `range2` 内；该文档说明全部范围同时使能时仅先开始的 `range2` 生效，`range3` 无效。因此如果任务要求内层独立重放，应先核验目标版本和实际采集结果，再决定如何调整标记；无法满足时说明限制，不擅自更改依赖阶段。
- 单个范围内目标算子的 stream 不能改变。多 stream 工程应按完整依赖阶段划分范围，不能擅自改为单 stream。
- workspace 查询、分配、通信域初始化和 JIT 准备原则上置于范围外；接口内部包含这些操作时，应核验其捕获能力。
- 同步等待、输出拷回及校验置于范围结束后。保留原有通信协议、必要同步和数据依赖，不能批量移动或删除同步语句。
- 识别已有标记、头文件和链接配置，重复执行适配时不得重复添加。

## 2. C++ 调用

### Kernel 直调

在原有 launch 之前开始范围，在返回后结束范围，保留原调用参数和错误处理。以 SHMEM AllGather 为例：

```cpp
#include "mstx/ms_tools_ext.h"

const std::string rangeName = "shmem_allgather_" + std::to_string(trans_size);
const auto rangeId = mstxRangeStartA(rangeName.c_str(), nullptr);
for (int iteration = 0; iteration < perf_times; ++iteration) {
    magic++;
    allgather_demo<T>(
        BLOCK_NUM, stream, fftsAddr,
        static_cast<uint8_t*>(input_ptr), static_cast<uint8_t*>(output_ptr),
        static_cast<uint8_t*>(ptr), trans_size, magic * MAGIC_MULTIPLIER);
}
mstxRangeEnd(rangeId);
status = aclrtSynchronizeStream(stream);
```

上述循环边界适用于已核验的单 stream 调用；范围内算子较多时按完整逻辑迭代划分，遵守目标工具的数量限制。

### ACLNN 两阶段调用

第一阶段和 workspace 分配保持在范围外，标记第二阶段执行接口：

```cpp
// 原有第一阶段、workspace 分配及错误检查已完成。
const auto rangeId = mstxRangeStartA("all_gather_add", nullptr);
ret = aclnnAllGatherAdd(workspaceAddr, workspaceSize, executor, stream);
mstxRangeEnd(rangeId);
// 原有 ret 检查、stream 同步及精度校验。
```

如有异常或提前返回，使用符合工程风格的局部 RAII 等方式保证配对。多次调用应按 API 契约重新准备 executor，不能复用已消费的对象。

### 构建配置

将 MSTX 依赖加到实际编译 Host 调用文件的目标，而不是仅加入 kernel 编译目标：

```cmake
target_include_directories(example PRIVATE "$ENV{ASCEND_HOME_PATH}/include")
target_link_libraries(example PRIVATE
    "$ENV{ASCEND_HOME_PATH}/lib64/libms_tools_ext.so"
    dl
)
```

手工编译入口相应加入 `-I.../include -L.../lib64 -lms_tools_ext -ldl`。头文件、库路径和动态加载方式以安装版本为准。
Host MSTX 接入不要求增加 `ASCENDC_TIME_STAMP_ON`、设备侧同步上报或改变 kernel 优化级别。

## 3. PyTorch 调用

确认当前 Python 环境实际使用 Ascend NPU 后端，并检查 torch/torch_npu 版本及路径。
官方 mstxRangeStartA 文档提供 `mstx.range_start(message, stream)`，并介绍通过 Python 绑定模块或 `ctypes.CDLL("libms_tools_ext.so")` 调用的方式。下例采用已正确配置的 `mstx` Python 绑定模块；应先确认模块来源和配套版本，不假定安装环境自带该模块，也不安装来源不明的同名包。

保留工程原有 torch/torch_npu 导入和初始化，按以下结构添加范围：

```python
import mstx

range_id = mstx.range_start("dispatch", None)
try:
    result = launch_target()  # 工程原有的目标调用
finally:
    mstx.range_end(range_id)

# 执行原有完成等待和输出校验。
```

若工程采用 `torch_npu.npu.mstx`，须另行核对当前 torch_npu 版本提供的接口，不能仅凭上述 MSTX 文档认定该入口存在。采用 ctypes 时，按官方示例为开始接口设置 `argtypes=[ctypes.c_char_p, ctypes.c_void_p]`、`restype=ctypes.c_uint64`，为结束接口设置 `argtypes=[ctypes.c_uint64]`、`restype=None`，使用编码后的字节字符串和真实返回 ID。

- 不将 `.cpu()`、`.numpy()`、`.item()`、输出打印或隐式同步包含在范围内。
- `async_finish`、event、接收 hook 等应追踪到实际下发线程及 stream；函数返回不能证明目标任务已全部下发。
- 保留原有完成等待的作用域，不统一替换成设备级同步。
- 多卡场景每个参与 rank 都应执行标记，不能只在父进程包围 `spawn`、`torchrun`，或只给 rank 0 添加标记。
- Python 调用若包含内部 stream 切换，应先确认 capture 支持或选择更细的完整阶段，不根据外层 current stream 推定兼容性。

## 4. 多卡启动

使用一次 msOpProf 任务包住工程原有的全部 rank 启动入口，保留 `torchrun`、Python spawn 或 shell launcher 的进程组织方式；不要对每个 rank 分别启动独立的 msOpProf 任务。

核对逻辑 rank、物理卡及 `ASCEND_RT_VISIBLE_DEVICES` 的映射，避免与样例已有设备偏移重复叠加。各 rank 的范围名称、阶段顺序和输入用例应对应一致。
保留运行 cwd、共享库路径和输入数据位置；launcher 应等待所有 rank 并传播失败状态。

## 参考资料

- [mstxRangeStartA 接口说明](https://gitcode.com/Ascend/mstx/blob/master/docs/zh/api_reference/Common/mstxRangeStartA.md)：接口原型、参数语义和调用示例。
- [mstxRangeEnd 接口说明](https://gitcode.com/Ascend/mstx/blob/master/docs/zh/api_reference/Common/mstxRangeEnd.md)：范围结束接口。
- msOpProf 对应版本的范围级重放和 MSTX 扩展功能文档：范围划分、采集条件及工具限制。
