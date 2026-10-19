# Adding a Custom Agent

Agents in msAgent are defined through declarative configuration. Therefore, no Python code is required. To add a new agent, you only need to create a YAML configuration file and the corresponding prompt file in `resources/configs/default/`.

## 1. Overview

A complete agent consists of two parts:

- **YAML configuration file**: defines the agent name, model, tools, skills, subagents, and so on
- **Prompt file**: defines the agent system prompt, which determines its behavior and capabilities

Definitions in `resources/configs/default/` are loaded directly as read-only built-in defaults and are not copied as a whole directory on first run. User overrides are located in `~/.msagent/config/`, and user prompts are located in `~/.msagent/prompts/`.

## 2. Directory Structure

```text
resources/configs/default/
├── agents/                          # Agent configuration (one Agent per file)
│   ├── Profiler.yml
│   └── MyAgent.yml                  # Newly added Agent
├── subagents/                       # Subagent configuration (optional)
│   └── my-subagent.yml
├── prompts/
│   └── agents/                      # Prompt files for agents
│       └── MyAgent.md
└── ...
```

## 3. Creating an Agent Configuration File

Create `MyAgent.yml` in `resources/configs/default/agents/` (the file name must match `name`):

```yaml
version: __APP_VERSION__
name: MyAgent
description: My custom Agent for the XXX scenario
prompt:
  - prompts/agents/MyAgent.md
  - prompts/suffixes/environments.md
llm: default
checkpointer: sqlite
default: false
subagents:
  - explorer
recursion_limit: 1000
tools:
  patterns:
    - impl:deepagents:*
  use_catalog: false
  output_max_tokens: 10000
skills:
  patterns:
    - default:ascend-computation-analysis
  use_catalog: false
compression:
  auto_compress_enabled: true
  auto_compress_threshold: 0.8
  llm: default
  prompt:
    - prompts/shared/general_compression.md
    - prompts/suffixes/environments.md
retry:
  enabled: true
  model:
    enabled: true
    max_retries: 5
    timeout: 120.0
  tool:
    enabled: true
    max_retries: 5
```

### 3.1 Key Fields

| Field | Required | Description |
|------|------|------|
| version | Yes | Configuration version. Uses the `__APP_VERSION__` placeholder, which is replaced automatically at runtime. |
| name | Yes | Unique agent identifier. **Must match the file name** (for example, `MyAgent.yml` corresponds to `name: MyAgent`). |
| description | Yes | Brief description of the agent, shown in the `/agents` list. |
| prompt | Yes | List of prompt file paths. Built-in configurations are relative to the default resource directory, and user override configurations are relative to `~/.msagent/`. |
| llm | Yes | LLM alias to use. `default` corresponds to the default model. You can also specify `sonnet`, `opus`, and so on. |
| checkpointer | Yes | Session persistence backend, usually `sqlite`. |
| default | Yes | Whether this is the default agent. **The entire system must have exactly one `default: true`**. |
| subagents | No | Names of referenced subagents, which can come from built-in configurations or `~/.msagent/config/subagents/`. |
| tools.patterns | Yes | Tool filtering rules in the `category:module:name` format. The `*` wildcard is supported, and the `!` prefix excludes a tool. |
| skills.patterns | Yes | Skill filtering rules in the `category:name` format. For the complete list, see [Skills](../../../skills/README.md). |

### 3.2 `tools.patterns`

The format is `category:module:name`:

| Category | Description | Example |
|------|------|------|
| impl | Built-in implementation tools | `impl:deepagents:*` (all built-in tools) |
| mcp | MCP service tools | `mcp:msprof-mcp:*` (all tools of the msprof-mcp service) |
| internal | Internal tools | `internal:*:*` |

To exclude a tool, use the `!` prefix: `!impl:deepagents:some_tool`

## 4. Creating a Prompt File

Create `MyAgent.md` in `resources/configs/default/prompts/agents/` to define the agent system prompt:

```markdown
# MyAgent - XXX Assistant

You are MyAgent, an AI assistant focused on the XXX scenario.

## Hard Rules

1. **Rule 1**: Describe the behavioral constraints
2. **Rule 2**: Describe the output requirements

## Skill Invocation Rules

When a task matches one of the following scenarios, call `get_skill(name="<skill-name>")`:

| Skill | Applicable Scenario |
|------------|----------|
| ascend-computation-analysis | Computation bottleneck analysis |

## Output Guidelines

Prefer the following structure:

Problem / Evidence / Root Cause / Recommendation
```

For how to write prompt files, refer to the prompts of existing agents (for example, [Profiler.md](../../../resources/configs/default/prompts/agents/Profiler.md)).

## 5. Verification

### 5.1 Quick Verification with `uv run`

```bash
uv run msagent --agent MyAgent
```

### 5.2 Verification After Rebuilding and Installing the Package

After rebuilding and reinstalling the package, start msAgent to verify:

```bash
msagent --agent MyAgent
```

For the build and installation process, see [Building and Packaging](./build-and-package.md).
