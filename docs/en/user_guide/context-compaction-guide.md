# Context Compaction User Guide

This document introduces the context compaction and offloading capability of `msagent`. In long-running sessions, it automatically summarizes old messages, offloads the original messages to backend storage, and frees up the active context window.

## 1. What Problem Does This Feature Solve

In long sessions, historical messages keep accumulating. Common problems include:

- The model input token count keeps growing, slowing down responses and increasing cost
- The session approaches the context window limit, which can easily trigger truncation or degradation
- Old messages no longer need to be preserved verbatim, but they cannot be completely lost either

The current implementation does the following at the right time:

1. Summarize earlier historical messages into a single summary message and keep it in the current thread
2. Append the summarized original messages to the conversation history file so that you can look them up later when needed

In other words, this feature does not directly delete old messages. Instead, it retains summaries and offloads the original messages.

## 2. Default Behavior

The default agent configuration is located at `resources/configs/default/agents/Profiler.yml`.

The current default configuration is:

```yaml
compression:
  auto_compress_enabled: true
  auto_compress_threshold: 0.8
  llm: default
  prompt:
    - prompts/shared/general_compression.md
    - prompts/suffixes/environments.md
```

The meaning is as follows:

- `auto_compress_enabled: true`

  Enables automatic compaction.
- `auto_compress_threshold: 0.8`

  Automatically triggers compaction when the current input context approaches 80% of the model window.
- `llm: default`
  - Model used to generate the summary. By default, it reuses the current default model.
- `prompt`

  Prompt template used to generate the summary.

## 3. How to Use as a Regular User

You can use this capability in the following ways:

- Automatic trigger
- Manual trigger of `/offload`

### 3.1 Automatic Trigger

For example:

```bash
msagent
```

After entering the session, keep talking with the agent. When the context reaches the threshold, the CLI automatically performs a context compaction.

When compaction occurs, you see terminal prompts similar to the following:

- The current context usage has reached the threshold. Therefore, automatic compaction starts
- How many historical messages were offloaded
- How many recent messages were kept
- The approximate token change before and after compaction
- The path where the conversation history was saved

Note:

- After compaction, the session remains in the same `thread_id` and does not switch to a new thread
- Recent messages stay in the active context
- Earlier messages become a single summary message

### 3.2 Manually Triggering `/offload`

If you do not want to wait until the automatic threshold is reached, you can also manually trigger it in the interactive session:

```text
/offload
```

Suitable scenarios:

- You just finished a long troubleshooting session and are ready to switch to a new task
- The current session is already long, and you want to free up the context window immediately
- You confirm that the large amount of intermediate reasoning and tool output from earlier does not need to be preserved verbatim

After you run `/offload`, the following happens:

1. Earlier messages in the current thread are summarized into a single summary message
2. Original messages are written to `/conversation_history/<thread_id>.md`
3. The current conversation state is written to a summary placeholder event, `_summarization_event`
4. Afterward, the model mainly sees the summary plus the recently kept messages, reducing context usage

Note:

- `/offload` does not create a new thread
- `/offload` preserves the continuity of the current session
- If the current conversation is very short, the CLI may indicate that no compaction is needed

## 4. Where the Compressed Data Is Stored

After compaction, the original historical messages are written to the global state directory of the current project:

```text
~/.msagent/state/projects/<project-id>/conversation_history/<thread_id>.md
```

For example:

```text
~/.msagent/state/projects/my-project-a1b2c3d4e5f6/conversation_history/8d6c4f3f-....md
```

This file is appended to over time. Each compaction adds a new section, similar to the following:

```md
## Offloaded at 2026-04-02T12:34:56+00:00

Human: ...
AI: ...
Tool: ...
```

In addition, the checkpoint state of the current thread records a `_summarization_event`, which contains:

- `cutoff_index`
- `summary_message`
- `file_path`

The `file_path` is a logical path:

```text
/conversation_history/<thread_id>.md
```

The actual drive directory it corresponds to is the one shown earlier:

```text
~/.msagent/state/projects/<project-id>/conversation_history/<thread_id>.md
```

## 5. How It Works in Detail

A compaction run follows this general process:

1. Read the message list of the current thread
2. Restore the effective messages that the model actually sees, based on the existing `_summarization_event`
3. Decide which messages to keep and which to summarize, based on the retention policy
4. Generate the summary with the compaction model
5. Append the summarized original messages to the `conversation_history` file
6. Write the new summary event back to the current thread state
7. Update the current context token statistics

After compaction, the model continues to see:

- A "previous conversation summary"
- A few recently kept messages

## 6. How to Adjust the Compaction Strategy

If you want to modify the automatic compaction behavior, adjust the `compression` section in the agent configuration.

Commonly adjustable items:

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

Fields:

- `auto_compress_threshold`

  A smaller value triggers compaction earlier. For example, 0.7 triggers it when the context usage reaches 70%.
- `messages_to_keep`

  How many of the most recent non-system messages to keep without summarizing during compaction
- `llm`

  Can be replaced with a cheaper or faster summarization model.
- `prompt`

  Customizes the summary style, for example, leaning more toward fact recording or task progress recording.

Recommendations:

- If you care more about preserving the details of the recent context, increase `messages_to_keep`
- If you care more about keeping tokens under control early, lower `auto_compress_threshold`
- If you want manual control only, disable `auto_compress_enabled`

## 7. Suitable Scenarios

This feature is especially suitable for:

- Long troubleshooting sessions for profiling problems
- Analyzing multiple directories or cases in a single session
- Tasks where tool output is heavy and the context grows quickly
- Scenarios that require continuous collaboration but do not want historical messages to grow without limit

## 8. How to Confirm That It Actually Took Effect

You can confirm it in the following places:

### 8.1 Terminal Prompts

When the threshold is reached, you see the automatic compaction prompt and a summary of the compaction results.

If you run `/offload` manually, you also see similar output, including:

- How many messages were offloaded
- How many messages were kept
- The token change before and after compaction
- The path where the historical messages were saved

### 8.2 Conversation History Files

Check:

```text
~/.msagent/state/projects/<project-id>/conversation_history/
```

Check whether the `.md` file corresponding to the current `thread_id` has been generated.

### 8.3 Checkpoint Data

If you are investigating internal state, check whether the checkpoint for the current thread already contains a `_summarization_event`.

## 9. Troubleshooting Suggestions

If you think compaction was not triggered, check the following in order:

### 9.1 Whether the Current Agent Has Compression Enabled

Check whether the agent configuration contains:

```yaml
compression:
  auto_compress_enabled: true
```

### 9.2 Whether the Current Model Has `context_window` Configured

Automatic compaction depends on the context window size to calculate the threshold. If the context window of the model is empty, the automatic decision may not trigger.

### 9.3 Whether the Conversation Has Actually Reached the Threshold

If the conversation is not long enough, compaction does not happen.

### 9.4 Whether the Project State Directory in `MSAGENT_HOME` Is Writable

If writing the history file fails, the summary may still be generated, but the original messages are not successfully written to the drive. In this case, a warning appears in the terminal.

## 10. Notes for Developers

The main entry points of the current implementation are:

- `src/msagent/cli/handlers/compress.py`

  Current CLI compaction entry point
- `src/msagent/utils/offload.py`

  Summary generation, original message offloading, and summary event construction
- `src/msagent/agents/factory.py`

  Exposes `_agent_backend` to the graph and routes `conversation_history` to a persistent directory.

The current design has the following key points:

- Compaction is in-place

  Instead of creating a new thread, it updates `_summarization_event` on the current thread
- The original history is recoverable

  The `conversation_history/<thread_id>.md` file forms an append-only offload log over time

If you want to enhance the feature further, the natural directions are:

- Add an explicit manual command, for example, `/compress`.
- Add CLI helper commands for viewing the historical summary or opening the original history file.
- Allow writing `conversation_history` to SQLite, object storage, or a remote store.

## 11. One-Sentence Summary

For you, this feature is essentially out of the box. During normal long-running use of `msagent`, when the context approaches its limit, the feature automatically summarizes old messages and offloads them to the project state area in the global directory, freeing up the context window while keeping the original history traceable.
