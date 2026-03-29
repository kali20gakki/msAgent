# 上下文压缩与卸载使用指南

本文介绍 `msagent` 的“上下文压缩与卸载”能力：在长时间会话中，自动总结旧消息，把原始消息卸载到后端存储，并释放活跃上下文窗口。

## 1. 这个特性解决什么问题

长会话里，历史消息会不断堆积，常见问题包括：

- 模型输入 token 持续增长，响应变慢、成本变高
- 会话接近上下文窗口上限，容易触发截断或退化
- 旧消息已经不需要逐字保留，但又不能完全丢失

现在的实现会在合适时机做两件事：

1. 把较早的历史消息总结成一条摘要消息，继续保留在当前 thread 中
2. 把被总结掉的原始消息追加保存到会话历史文件，后续需要时仍可回查

也就是说，这不是“直接删除旧消息”，而是“摘要保留 + 原文卸载”。

## 2. 默认行为

默认 agent 配置位于：

- `resources/configs/default/agents/Hermes.yml`

当前默认配置为：

```yaml
compression:
  auto_compress_enabled: true
  auto_compress_threshold: 0.8
  llm: default
  prompt:
    - prompts/shared/general_compression.md
    - prompts/suffixes/environments.md
```

含义如下：

- `auto_compress_enabled: true`
  - 开启自动压缩
- `auto_compress_threshold: 0.8`
  - 当当前输入上下文接近模型窗口的 80% 时，自动触发压缩
- `llm: default`
  - 使用哪个模型来生成摘要；默认复用当前默认模型
- `prompt`
  - 生成摘要时使用的提示词模板

## 3. 作为普通用户，怎么使用

这个能力有两种使用方式：

- 自动触发
- 手动执行 `/offload`

示例：

```bash
msagent
```

进入会话后，持续和 agent 对话。当上下文达到阈值时，CLI 会自动执行一次上下文压缩。

压缩发生后，你会看到类似的终端提示：

- 当前上下文占用达到阈值，开始自动压缩
- 已卸载多少条历史消息
- 保留了多少条最近消息
- 压缩前后大致 token 变化
- 会话历史已保存到哪个路径

注意：

- 压缩后仍然留在同一个 `thread_id` 中，不会切换到新的 thread
- 最近消息会保留在活跃上下文里
- 更早的消息会变成一条摘要消息

### 手动触发 `/offload`

如果你不想等到自动阈值触发，也可以在交互会话中主动执行：

```text
/offload
```

适合场景：

- 你刚结束一段较长的问题排查，准备切到新任务
- 当前会话已经很长，希望立即释放上下文窗口
- 你确认前面的大量中间推理和 tool 输出不需要继续逐字保留

执行 `/offload` 后，会发生以下事情：

1. 当前 thread 中较早的消息会被总结成一条摘要消息
2. 原始消息会被写入 `/conversation_history/<thread_id>.md`
3. 当前对话状态会写入一个摘要占位事件 `_summarization_event`
4. 后续模型主要看到“摘要 + 最近保留消息”，从而减少上下文占用

注意：

- `/offload` 不会新建 thread
- `/offload` 会保留当前会话连续性
- 如果当前对话本来就很短，可能会提示无需压缩

## 4. 压缩后，数据存在哪里

压缩后的原始历史消息会写入当前工作目录下：

```text
<working-dir>/.msagent/conversation_history/<thread_id>.md
```

例如：

```text
/path/to/project/.msagent/conversation_history/8d6c4f3f-....md
```

这个文件是按时间追加写入的。每次压缩都会新增一个段落，类似：

```md
## Offloaded at 2026-04-02T12:34:56+00:00

Human: ...
AI: ...
Tool: ...
```

另外，当前 thread 的 checkpoint 状态里会记录一个 `_summarization_event`，其中包含：

- `cutoff_index`
- `summary_message`
- `file_path`

其中 `file_path` 是逻辑路径：

```text
/conversation_history/<thread_id>.md
```

它对应的实际磁盘目录就是上面的：

```text
<working-dir>/.msagent/conversation_history/<thread_id>.md
```

## 5. 它具体是怎么工作的

一次压缩的大致流程如下：

1. 读取当前 thread 的消息列表
2. 结合已有 `_summarization_event`，恢复“模型实际看到的有效消息”
3. 根据保留策略，决定哪些消息要保留，哪些消息要被总结
4. 用压缩模型生成摘要
5. 把被总结掉的原始消息追加写入 `conversation_history` 文件
6. 把新的摘要事件写回当前 thread 状态
7. 更新当前上下文 token 统计

压缩后，模型后续继续看到的是：

- 一条“之前对话摘要”
- 最近保留的若干条消息

## 6. 如何调整压缩策略

如果你想修改自动压缩行为，可以调整 agent 配置中的 `compression` 段。

常见可调项：

```yaml
compression:
  auto_compress_enabled: true
  auto_compress_threshold: 0.7
  llm: default
  prompt:
    - prompts/shared/general_compression.md
    - prompts/suffixes/environments.md
  messages_to_keep: 4
```

字段说明：

- `auto_compress_threshold`
  - 越小越早压缩；例如 `0.7` 表示上下文使用到 70% 就触发
- `messages_to_keep`
  - 压缩时保留最近多少条非 system 消息不做总结
- `llm`
  - 可以替换成更便宜或更快的总结模型
- `prompt`
  - 可以自定义摘要风格，例如更偏“事实记录”或“任务进度记录”

建议：

- 如果你更关心保留最近上下文细节，增大 `messages_to_keep`
- 如果你更关心尽早控住 token，降低 `auto_compress_threshold`
- 如果你只想手动控制，关闭 `auto_compress_enabled`

## 7. 适合什么场景

这个特性特别适合：

- 长时间排查 profiling 问题
- 一次会话里分析多个目录或多个 case
- tool 输出很多、上下文增长很快的任务
- 需要连续协作但不希望历史消息无限膨胀的场景

## 8. 怎么确认它真的生效了

你可以从三个地方确认：

### 终端提示

当达到阈值时，会看到自动压缩提示和压缩结果摘要。

如果你手动执行 `/offload`，也会看到类似输出，包括：

- 卸载了多少条消息
- 保留了多少条消息
- 压缩前后 token 变化
- 历史消息保存到了哪个路径

### 会话历史文件

检查：

```text
<working-dir>/.msagent/conversation_history/
```

是否已经生成当前 `thread_id` 对应的 `.md` 文件。

### checkpoint 数据

如果你在排查内部状态，可以查看当前 thread 对应 checkpoint 中是否已有 `_summarization_event`。

## 9. 排障建议

如果你觉得没有触发压缩，可以依次检查：

### 1) 当前 agent 是否启用了 compression

检查 agent 配置里是否有：

```yaml
compression:
  auto_compress_enabled: true
```

### 2) 当前模型是否配置了 `context_window`

自动压缩依赖上下文窗口大小来计算阈值。如果模型上下文窗口为空，自动判断可能不会触发。

### 3) 当前对话是否真的达到阈值

如果对话还不够长，压缩不会发生。

### 4) 工作目录下是否能写入 `.msagent/conversation_history/`

如果历史文件写入失败，摘要仍可能生成，但原始消息不会成功落盘。此时终端会出现 warning。

## 10. 对开发者的说明

当前实现的主要入口如下：

- `src/msagent/cli/handlers/compress.py`
  - 当前 CLI 压缩入口
- `src/msagent/utils/offload.py`
  - 摘要生成、原始消息卸载、摘要事件构造
- `src/msagent/agents/factory.py`
  - 为 graph 暴露 `_agent_backend`，并把 `conversation_history` 路由到持久目录

当前设计有两个关键点：

- 压缩是 in-place 的
  - 不再新建 thread，而是在当前 thread 上更新 `_summarization_event`
- 原始历史是可恢复的
  - 通过 `conversation_history/<thread_id>.md` 形成按时间追加的卸载日志

如果后续要继续增强，比较自然的方向有：

- 增加显式手动命令，例如 `/compress`
- 增加“查看历史摘要 / 打开原始历史文件”的 CLI 辅助命令
- 允许把 `conversation_history` 写入 SQLite、对象存储或远端 store

## 11. 一句话总结

对使用者来说，这个特性基本是“开箱即用”的：正常长时间使用 `msagent`，当上下文接近上限时，它会自动把旧消息总结并卸载到 `.msagent/conversation_history/`，从而释放上下文窗口，同时保留可追溯的原始历史。
