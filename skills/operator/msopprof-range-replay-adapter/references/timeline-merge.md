# 自动生成 timeline_merge_config

根据工程调用链及已有采集产物自动推导合并配置，并记录字段来源。

本配置仅用于多卡流水合并，不是范围级重放的必选项。单卡普通算子或不需要多卡合并时，不生成本配置；如已有 custom-input 用于其他功能，应保留其内容。

## 字段来源

| 字段 | 推导规则 |
| --- | --- |
| `kernel_name` | 实际设备 kernel 名或稳定前缀表达式；优先用现有 OpBasicInfo、注册代码或编译产物。host API 名不等于 kernel 名 |
| `mstx_name` | 实际插入的范围名称，各 rank 相同；精确匹配，不支持 `*` |
| `rank_count` | 本组参与的设备数，不是 host 线程数或进程总数；共享设备的多进程不能当成多个卡 |
| `launch_count` | 本组各设备预计匹配的执行次数上限；不是包含多个组的命令行总采集数量 |

当前参考版本约束：

- `timeline_merge_config` 必须是数组，最多 64 组。
- `kernel_name` 非空、最长 1024，仅 `A-Za-z0-9_*`；支持前缀及工具实现的匹配规则。不能使用 `A|B`，多种 kernel 分组配置。
- `mstx_name` 非空、最长 1024，仅 `A-Za-z0-9_`；动态名称必须展开为实际字符串。
- `rank_count` 为整数 `[2,64]`，`launch_count` 为整数 `[1,50]`，不能用布尔值。
- 非空合并配置要求开启 MSTX。

## 生成步骤

1. 分析 launcher 的 world size、设备映射和调用循环，记录每组字段来源。
2. 为多 shape、dispatch/combine 或多个不同 kernel 分别推导独立配置项，并尽量为可独立划分的调用范围使用不同的 MSTX 名称，便于区分；各配置项的 `mstx_name` 必须与源码中的实际标记一致。同一范围因依赖关系需要整体重放、不能拆分时，才按不同 `kernel_name` 配置多个对象并保留相同的 `mstx_name`，不得仅修改 JSON 名称而不修改对应源码标记。
3. 若没有实测 kernel 名，优先从产物推断稳定前缀；只能使用 `*` 时标注为“范围内全 kernel 临时匹配”。一个范围包含多个不同 kernel 时，应分别配置，不能将其当作同一算子的重复执行。
4. 读取用户已有的 custom-input；没有现有文件时创建 JSON 对象。由 Agent 直接将上述四字段对象写入 `timeline_merge_config` 数组，保留其他字段和不属于本任务的配置项，生成最终 `custom_input.json`。
5. 检查配置项满足上述字段约束；同一 `kernel_name` 与 `mstx_name` 的完全相同配置不重复添加，卡数或次数冲突时先核对来源，再调整目标条目。
6. 解析最终 JSON 检查语法，例如执行 `python3 -m json.tool <custom_input.json>`；同时逐项检查名称与源码一致、参与设备数和执行次数有依据、配置组数不超限。语法解析通过不能代替这些语义检查。

配置推导、写入和检查均由 Agent 完成，无需用户提供中间配置列表或执行额外转换步骤。仅生成 patch 时，在独立副本或产物目录完成上述步骤，并将最终配置纳入补丁。

## SHMEM 示例

以 SHMEM AllGather 的两卡调用为例，Host 在每个数据规模的调用循环前后添加 MSTX 标记。以下片段省略输入准备和错误检查，所有参与 rank 均执行相同的范围命名逻辑：

```cpp
#include "mstx/ms_tools_ext.h"

// 外层按 16 * 2^i（i = 0..23）依次设置 trans_size。
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
// 同步完成后执行原有输出回读和精度校验。
```

当启动参数为 `-pes 2`、`trans_size=16`、`perf_times=1` 时，配置字段的对应关系如下：

| 配置字段 | 对应依据 | 取值 |
| --- | --- | --- |
| `kernel_name` | `allgather_demo<T>()` 下发的设备 kernel 为 `ShmemAllGather_<type>`，不是 Host 函数名 | `ShmemAllGather*` |
| `mstx_name` | 传入 `mstxRangeStartA` 的实际字符串 | `shmem_allgather_16` |
| `rank_count` | launcher 启动两个 rank，分别使用两张设备 | `2` |
| `launch_count` | 每张设备在该范围内执行一次目标 kernel | `1` |

据此直接生成单个规模的最终 custom-input：

```json
{
  "timeline_merge_config": [
    {
      "kernel_name": "ShmemAllGather*",
      "mstx_name": "shmem_allgather_16",
      "rank_count": 2,
      "launch_count": 1
    }
  ]
}
```

24 个规模全选时，从源码的 `16 * 2**i`、`i=0..23` 展开成 24 组。
不能用 `mstx_name: "shmem_allgather_*"`，也不能写一组后声称全部规模都合并。
生成文件可同时包含已有的 `mc2_trace_config.descid_display`，不得将其覆盖。

## 合并语义与验证

1. 参考实现以各设备匹配记录的出现顺序配组，`rank_count` 是设备数上限，不会补造缺失 rank；少于预期参与设备也可能有输出。因此验收必须对照期望的设备清单，而不仅检查有无 trace 文件。
2. 合并文件按各组在 `summary/` 下组织，具体路径以工具版本的输出为准，检查所有期望设备前缀、调用次数和非空事件。
3. Default CSV 与通算流水是独立结果：只有实际生成单卡流水，才有有效输入可合并。
4. 若没有 PrintTimeStamp/对应流水能力，配置仍可生成，但合并标为“无可合并流水”；不要填造事件或把 BasicInfo CSV 当成 trace。
5. 若用户另行要求自定义打点，再按工具文档增加打点、编译选项和 descid 映射，不自动扩大本技能的源码修改范围。
