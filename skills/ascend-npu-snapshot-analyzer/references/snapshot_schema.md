# Ascend NPU Memory Snapshot 数据格式参考

本文档定义 `torch_npu.npu.memory._dump_snapshot()` 导出的 pickle 文件数据结构。

## 顶层结构

```python
{
    "segments": List[Segment],
    "device_traces": List[List[TraceEntry]]
}
```

## Segment 结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `device` | int | NPU 设备编号 |
| `address` | int | Segment 起始虚拟地址 |
| `total_size` | int | aclrtMalloc 分配的段总大小 (bytes) |
| `allocated_size` | int | 已分配使用的内存大小 (bytes) |
| `active_size` | int | 正在使用或等待释放的内存大小 (bytes) |
| `requested_size` | int | 用户请求的内存大小 (bytes) |
| `stream` | int | 关联的 NPU stream |
| `segment_type` | str | 段类型：`large` (>1MB) 或 `small` |
| `segment_pool_id` | tuple | 内存池 ID 元组 |
| `is_expandable` | bool | 是否可扩容 |
| `frames` | List[Frame] | 分配时的堆栈跟踪（可能为空） |
| `blocks` | List[Block] | 段内的内存块列表 |

## Block 结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `address` | int | Block 起始地址 |
| `size` | int | 实际占用大小（含对齐） |
| `requested_size` | int | malloc 请求大小（可能小于 size） |
| `state` | str | 状态：`active_allocated` / `active_pending_free` / `inactive` |
| `frames` | List[Frame] | 分配时的堆栈跟踪 |

## TraceEntry 结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `action` | str | 事件类型 |
| `addr` | int | 关联内存地址（OOM 时为 NULL） |
| `device_free` | int | 仅 OOM 事件存在，OOM 时可用内存 |
| `size` | int | 操作的内存大小 |
| `stream` | int | 关联的 NPU stream |
| `frames` | List[Frame] | 事件关联的堆栈跟踪 |

### action 枚举值

| 值 | 说明 |
|------|------|
| `alloc` | 内存分配 |
| `free_requested` | 释放请求 |
| `free_completed` | 释放完成 |
| `segment_alloc` | Segment 分配 |
| `segment_free` | Segment 释放 |
| `segment_map` | Segment 映射 |
| `segment_unmap` | Segment 解映射 |
| `snapshot` | 快照标记 |
| `oom` | Out of Memory 事件 |

## Frame 结构

单个堆栈帧为字符串，格式为 `filename:line_number:function_name`。

## 采集方式

### NPU 环境

```python
import torch
import torch_npu

torch_npu.npu.memory._record_memory_history(max_entries=100000)
# ... 执行训练/推理代码 ...
torch_npu.npu.memory._dump_snapshot("snapshot.pkl")
torch_npu.npu.memory._record_memory_history(enabled=None)
```

### CUDA 环境（兼容）

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=100000)
# ... 执行训练/推理代码 ...
torch.cuda.memory._dump_snapshot("snapshot.pkl")
torch.cuda.memory._record_memory_history(enabled=None)
```