# Configuration and Extension

This document summarizes the global configuration, project state, MCP extensions, and skills extensions in the current `msAgent` code implementation.

## 1. Global Configuration Directory

`msAgent` stores user configuration centrally in `MSAGENT_HOME`. When this environment variable is not set, the default directory is:

```text
~/.msagent/
```

The working directory serves only as the execution root for agents and tools, and a project-local `.msagent/` is no longer generated there. Default configuration in the installation package is read directly from the wheel. The global directory stores only user overrides.

At this stage, an existing `<working-dir>/.msagent/` is not imported automatically. Keep the old directory in place for now and wait for the migration feature in a later release.

## 2. Default Templates and Runtime Files

The default templates contain the following main content:

| Path | Description |
|---|---|
| ~/.msagent/config/config.llms.yml | Default model override configuration |
| ~/.msagent/config/llms/*.yml | Additional model alias override configuration |
| ~/.msagent/config/agents/*.yml | Minimal agent overrides or new definitions for advanced extension. Normal switching does not generate them. |
| ~/.msagent/config/subagents/*.yml | Minimal subagent overrides or new definitions for advanced extension |
| ~/.msagent/config/checkpointers/*.yml | Checkpointer override configuration |
| ~/.msagent/config/sandboxes/*.yml | Sandbox override configuration |
| ~/.msagent/prompts/ | User prompt files |
| ~/.msagent/skills/ | Skills installed by the user |
| ~/.msagent/config/config.mcp.json | MCP server override configuration |
| ~/.msagent/config/config.approval.json | Tool approval rule override configuration |

The runtime generates the following files or directories on demand:

| Path | Description |
|---|---|
| `~/.msagent/state/projects/<project-id>/project.json` | Working directory identifier, along with the current agent of the project and the model preferences of each agent |
| `~/.msagent/state/projects/<project-id>/checkpoints.sqlite` | Checkpoint database of the current project |
| `~/.msagent/state/projects/<project-id>/history` | Input history of the current project |
| `~/.msagent/state/projects/<project-id>/memory.md` | Long-term memory of the current project |
| `~/.msagent/state/projects/<project-id>/conversation_history/` | Conversation history of the current project |
| `~/.msagent/state/projects/<project-id>/audit_log/` | Audit log of the current project |
| `~/.msagent/cache/mcp/` | Global MCP runtime cache |
| `~/.msagent/oauth/mcp/` | Global MCP OAuth data |
| `~/.msagent/logs/` | Global log directory |

## 3. Project Long-Term Memory

`~/.msagent/state/projects/<project-id>/memory.md` stores the long-term memory of the current project. You can maintain it in an interactive session with the following commands:

| Command | Description |
|---|---|
| /remember \<content\> | Appends a piece of long-term memory. The first use automatically creates the project memory file. |
| /showmemory | Displays the long-term memory saved for the current project. |

Later sessions automatically read the memory file of the corresponding project and provide it to the agent as `<user-memory>` context. The default template content is not injected as valid memory.

Long-term memory is suitable for stable information. You are advised not to write sensitive data such as API keys, passwords, or tokens into it.

## 4. Configuration Loading

The current implementation supports both single-file configuration and directory-based configuration:

- LLM: Built-in defaults overlaid with `~/.msagent/config/config.llms.yml` and `config/llms/*.yml`.
- Agent: Built-in defaults overlaid with `~/.msagent/config/config.agents.yml` and `config/agents/*.yml`.
- Subagent: Built-in defaults overlaid with `~/.msagent/config/config.subagents.yml` and `config/subagents/*.yml`.
- Checkpointer: Built-in defaults overlaid with `~/.msagent/config/config.checkpointers.yml` and `config/checkpointers/*.yml`.
- Sandbox: Built-in defaults overlaid with `~/.msagent/config/sandboxes/*.yml`.

The default templates currently use directory-based configuration as the primary approach.

The complete definitions of built-in agents, such as the prompt, tools, skills, and subagent, are maintained by the installation package, and regular users do not need to copy or edit them. The `/agents` switching result is saved in the `project.json` of the current project. The `/models` selection is also saved per agent in the same project state. At startup, the selection priority is the explicit `--agent`/`--model` argument, the current project preference, and the installation package default, in that order.

`config/agents/*.yml` is generated only for advanced extension or when you explicitly modify agent fields. Generated files use minimal field-level overrides, and fields that do not appear continue to inherit the default definitions in the installation package.

## 5. MCP Configuration

Built-in MCP definitions come from `resources/configs/default/config.mcp.json` in the installation package, and user files override the built-in definitions by server name. The following is an example of a local stdio service:

```json
{
  "mcpServers": {
    "msprof-mcp": {
      "command": "msprof-mcp",
      "args": [],
      "transport": "stdio",
      "env": {},
      "include": [],
      "exclude": [],
      "enabled": true,
      "stateful": true,
      "repair_timeout": 30,
      "invoke_timeout": 3600.0
    }
  }
}
```

The common fields are as follows:

| Field | Description |
|---|---|
| command | The command used to start the local MCP service |
| url | The address of the remote MCP service |
| headers | The request headers of the remote MCP service |
| args | The argument list of `command` |
| transport | The transport protocol supports `stdio`, `sse`, `http`, and `websocket`. |
| env | The environment variables injected when the local MCP process starts |
| include/exclude | The tool list to allow or exclude |
| enabled | Whether to enable this MCP service |
| stateful | Whether to keep the connection alive to avoid restarting it on each invocation |
| repair_command | The optional repair command list used when initialization fails |
| repair_timeout | The timeout of the repair command, in seconds |
| timeout | The timeout for establishing a connection |
| sse_read_timeout | The SSE read timeout |
| invoke_timeout | The timeout of a single tool invocation |

Notes:

- `streamable_http`/`streamable-http` is normalized to `http` at runtime.
- When `repair_command` is configured but `repair_timeout` is not explicitly set, it defaults to 30.
- For local `stdio` MCP services such as `msprof-mcp`, pay attention primarily to `stateful` and `invoke_timeout`.

Built-in MCP services:

- `msprof-mcp`: Ascend Profiling data analysis service (local stdio, enabled by default).
- `ascend-doc-mcp`: Query service for Ascend community documentation and open-source repository materials (enabled by default). It uses the `msagent-ascend-doc-mcp` launcher to start `@opencxd/ascend-doc-mcp` with `npx`, and by default it queries Ascend community documentation (products, versions, documentation catalogs, search, open-source repository files, and installation downloads). It requires Node.js (≥22) and npm/npx, and the first startup pulls the package from npm. The installation script has preinstalled the package to `~/.msagent/ascend-doc-mcp`. When preinstallation succeeds, the launcher runs the local copy directly. Therefore, no npm is required at runtime, and it can work offline. The npm registry resolution order is `MSAGENT_NPM_REGISTRY` → the registry configured in the local npm itself (`npm config get registry`, including the enterprise-issued `.npmrc`, which makes intranet machines use the intranet source automatically) → `registry.npmmirror.com` → `mirrors.huaweicloud.com/repository/npm` → `registry.npmjs.org`. It probes each entry and uses the first available one (for intranet scenarios, see item 9 in the [FAQ](../support/faq.md)). `MSAGENT_NPM_REGISTRY_ONLY=1` forces the use of only the explicitly specified registry. When this service is unavailable, session startup is not affected. It only prevents the `ascend-knowledge` subagent from performing documentation and repository queries, and causes it to report the unavailability to the main agent. You can also disable the service with `/mcp` in a session.
- `tavily-mcp`: Web search service (disabled by default, and you can enable it in `/mcp` after configuring `TAVILY_API_KEY`).

Common usage:

- Use `/mcp` in a session to toggle the enabled state of an existing MCP service.
- Edit `~/.msagent/config/config.mcp.json` directly to add, delete, or adjust service definitions.

## 6. Skills Extensions

The current skills are scanned in the following order:

1. `<working-dir>/skills`
2. `~/.msagent/skills`
3. Built-in skills directory: prefers `skills/` in the repository root, and uses the packaged resources when a wheel is installed.

Skills with the same name follow the first-loaded-wins rule. Therefore, the current priority is:

1. `skills/` in the project root directory
2. Global user skills
3. Built-in skills

## 7. Skill Directory Structure

The following two directory structures are supported:

```text
skills/
  my-skill/
    SKILL.md
```

```text
skills/
  profiler/
    my-skill/
      SKILL.md
```

You are advised to include frontmatter in `SKILL.md` with at least:

```yaml
---
name: my-skill
description: What this skill does
---
```

## 8. Built-in Skills When Running from Source

The built-in skills are merged directly into the `msagent` main repository. When running from source, `msagent` uses the repository root by default:

```text
skills/
```

When building a wheel, the preceding directory is packaged into:

```text
resources/configs/default/skills/
```

Therefore, running from source and running from an installation share the same skills content. You no longer need to run `git submodule` synchronization separately.

(custom-skill-guide)=

## 9. Adding a Custom Skill

If you want to add a new skill to the current project, you are advised to place it directly in the repository root:

```text
skills/
  my-skill/
    SKILL.md
```

A categorized structure is also supported:

```text
skills/
  profiler/
    my-skill/
      SKILL.md
```

In this case:

- `profiler` is the category.
- `my-skill` is the skill name.

### 9.1 Writing `SKILL.md`

Minimal example:

```md
---
name: my-skill
description: A custom skill used to handle a certain type of fixed task
---

# My Skill

Use this skill when the user raises a request of this type:

- Analyze logs
- Generate reports

Execution requirements:

1. First check whether the input is complete.
2. Read the existing configuration and examples in the project first.
3. Output conclusions, evidence, and suggestions.
```

Notes:

- When `name` is not specified, the directory name is used by default.
- You are advised to fill in `description` so the skill can be identified in `/skills`.
- If you need scripts or templates, you can place them in the skill directory, for example, `scripts/` or `templates/`.

### 9.2 Making the Agent See the Skill

Creating the file alone is not enough. The current agent must also enable this skill in its configuration.

For example:

```yaml
skills:
  patterns:
    - default:my-skill
  use_catalog: false
```

For a skill with a category:

```yaml
skills:
  patterns:
    - profiler:my-skill
  use_catalog: false
```

The rule format is:

```text
<category>:<name_pattern>
```

For more complete matching semantics, refer to [Agent YAML Tool/Skills Filter Rules](agent-tool-skill-filter-rules.md).

### 9.3 Verifying Whether the Skill Takes Effect

After starting `msagent`, check it as follows:

```text
/skills
```

Or specify the skill directly:

```text
/skills my-skill
```

If skills share the same name, you are advised to write the full name:

```text
/skills profiler/my-skill
```

(custom-skill-faq)=

## 10. Skill FAQ

### 10.1 The New Skill Does Not Appear in `/skills`

In most cases, check the following items first:

- Check whether the path is correct and whether the file name is `SKILL.md`.
- Check whether the current agent has the corresponding `skills.patterns` configured.
- Check whether a skill with the same name in a higher-priority directory overrides it.

### 10.2 The Skill Exists, but the Agent Does Not Use It Automatically

This is usually caused by one of the following reasons:

- The `description` is too generic, making it difficult for the model to determine the trigger scenario.
- The `skills.patterns` of the agent does not enable the corresponding skill.
- The current task matches another built-in skill better. Therefore, this skill is not selected.

You are advised to first confirm visibility with `/skills`, then add a more specific `description` and trigger instructions.
