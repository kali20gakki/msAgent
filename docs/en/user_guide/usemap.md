# msAgent User Guide

Once you enter the interactive session, you can type your questions directly, or combine slash commands and keyboard shortcuts to work more efficiently.

## 1. Command Reference

### 1.1 Basic Commands

| Command | Description |
|---|---|
| `/hotkeys` | View keyboard shortcut help. |
| `/agents` | Open the agent selector. Switching starts a new session and records the current agent in the workspace project state. |
| `/model` | Open the model selector. |
| `/threads` | Browse and resume historical conversation threads. Threads that have been offloaded show `[history offloaded]`. |
| `/tools` | View the currently available tools. |
| `/skills` | Browse the currently available skills. |
| /mcp | Manage the enabled state of MCP services. |
| `/remember <content>` | Save long-term memory, such as user preferences, project background, or facts that subsequent sessions need to keep referencing. |
| `/showmemory` | View the long-term memory saved for the current project. |
| `/offload` | Compress and offload earlier conversation messages. |
| `/tool-output` | Open the most recent expandable tool output. |
| `/clear` | Clear the screen and start a new thread. |
| `/exit` | Exit the current session. |

### 1.2 Feature Commands

When you start `msagent`, specify the `--agent` parameter to launch a sub-feature, as shown in the following example:

| Agent | Description | Startup Command |
| --------------------------------------- | ------------------ | --------------------------- |
| [Profiler](../agent_guide/profiler.md)   | Performance tuning | msagent --agent Profiler |
| [Accuracy](../agent_guide/accuracy.md)   | Accuracy debugging | msagent --agent Accuracy |
| [Quantizer](../agent_guide/quantizer.md) | Model quantization | msagent --agent Quantizer |
| [Modeling](../agent_guide/modeling.md)   | Simulation modeling and automatic optimization | msagent --agent Modeling |
| [Operator](../agent_guide/operator.md)   | Operator tuning | msagent --agent Operator |
| [Minos](../agent_guide/minos.md)         | Documentation assistance | msagent --agent Minos |
| [SpecTrainer](../agent_guide/spectrainer.md) | Speculative decoding data resampling | msagent --agent SpecTrainer |

## 2. Input Area Shortcuts

| Shortcut | Description |
|---|---|
| Ctrl+C | Clears the input box when it has content. Pressing it twice exits the session. |
| Ctrl+J | Inserts a line break for multi-line input. |
| Shift+Tab | Cycles through approval modes. |
| Ctrl+B | Toggles bash mode. |
| Ctrl+K | Opens the keyboard shortcut help directly. |
| Ctrl+O | Opens the most recent expandable tool output. |
| Tab | Applies the first completion item. |
| Enter | Submits the input. If a completion item is selected, it applies the completion first. |

## 3. Tool Output Viewer

When a tool call supports expandable viewing, press `Ctrl+O` or use `/tool-output` to open the tool output viewer. Inside the viewer, you can:

- Left and right arrow keys: switch between the outputs of different tool calls.
- Up and down arrow keys, `PageUp`/`PageDown`, `Home`/`End`: scroll the content.
- Click, `Ctrl+O`, or `Enter`: expand or collapse the full output.
- `Esc`: close the viewer.

## 4. Long-Term Memory

`msagent` can save information that needs to persist across sessions as project memory. The memory content is written to `~/.msagent/state/projects/<project-id>/memory.md`. When subsequent sessions start and process messages, they automatically read any non-empty memory and inject it into the context.

| Command | Description |
|---|---|
| `/remember <content>` | Appends a long-term memory entry, for example `/remember the user wants all answers to use Chinese by default`. |
| `/showmemory` | Displays the memory saved for the current project. If no valid memory exists yet, it prompts that there is no content. |

Suitable content to save includes user preferences, project conventions, long-lived path instructions, troubleshooting conclusions, or background facts that later tasks need to keep referencing. Avoid saving sensitive information such as keys, tokens, and passwords.

## 5. Deep Dive by Topic

For more detailed configuration or troubleshooting, consult the corresponding document for each topic:

| Topic | Description | Document |
|---|---|---|
| Configuration and extension | Local configuration directory, model providers, MCP configuration, skills extensions and load order | [Configuration and Extension](configuration-and-extension.md) |
| Context compaction | Configuration and behavior of the context compaction and offloading mechanism (`/offload` and so on) for long sessions | [Context Compaction User Guide](context-compaction-guide.md) |
| Retry and timeout | Retry, backoff, and timeout configuration of the Retry Middleware for model calls | [Retry Middleware User Guide](retry-middleware-guide.md) |
| Capability boundaries | Filtering and matching rules for agent/Tool/skill | [Agent YAML Tool/Skills Filter Rules](agent-tool-skill-filter-rules.md) |
| Documentation experience review | Use the built-in `document-ux-review` skill to walk through the README and the getting-started process. | [`document-ux-review` Usage](document-ux-review.md) |
| FAQ | Troubleshooting for frequently encountered issues during installation, configuration, and usage | [FAQ](../support/faq.md) |
