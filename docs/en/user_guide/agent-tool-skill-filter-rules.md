# Agent YAML Tool/Skills Filter Rules

This document describes how to configure `tools` and `skills` in `agent yml`, the matching semantics, and a complete example you can test directly.

## 1. Tools Configuration

`tools.patterns` uses a three-part format:

```text
<category>:<module>:<name_pattern>
```

- `category`:
  - `impl`: Built-in implementation tools (currently all grouped into the `deepagents` module)
  - `mcp`: MCP tools
- `module`:
  - For `impl`, always use `deepagents`.
  - For `mcp`, use the MCP server name (for example, `msprof-mcp`).
- `name_pattern`:
  - Tool name, which supports `fnmatch` wildcards (`*`, `?`, `[abc]`)
- Negative rules: prefix with `!`, for example `!impl:deepagents:run_tool`.

Example:

```yaml
tools:
  patterns:
    - impl:deepagents:get_tool
    - impl:deepagents:run_tool
    - impl:deepagents:fetch_skills
    - impl:deepagents:get_skill
    - mcp:msprof-mcp:*
    - "!mcp:msprof-mcp:*_debug"
  use_catalog: false
```

## 2. Skills Configuration

`skills.patterns` uses a two-part format:

```text
<category>:<name_pattern>
```

- `category`: Usually `default`, but it can also match other categories.
- `name_pattern`: skill name, which supports `fnmatch` wildcards.
- Negative rules: prefix with `!`, for example `!default:op-mfu-calculator`.

Example:

```yaml
skills:
  patterns:
    - profiler:ascend-profiler-data-validation
    - profiler:ascend-profiler-db-explorer
    - "!default:op-*"
  use_catalog: false
```

## 3. Filter Rules (Strictly Enforced Semantics)

### 3.1 Tools Rules

- First, collect the positive rules (without `!`) and the negative rules (with `!`).
- A tool is available only when it matches at least one positive rule and does not match any negative rule.
- If there are no positive rules, all tools are disabled.
- Entries with an invalid format (not three parts) are ignored, and a warning is printed.
- Filtering takes effect at two levels:
  - `AgentFactory.create` filters once when building the tool list.
  - The model-call middleware filters again to prevent the tools that `deepagents` injects by default from bypassing the filter.

### 3.2 Skills Rules

- A skill is retained only when it matches at least one positive rule and does not match any negative rule.
- If there are no positive rules, no skills are available.
- The filtering result for skills goes into the runtime cache and serves as the preferred source for `fetch_skills` and `get_skill`.

## 4. Compatibility Strategy (Current Constraints)

- It provides no runtime compatibility aliases (for example, `execute <-> run_command`).
- It provides no legacy module compatibility (for example, `file_system`, `grep_search`, or `terminal`).
- It matches only the latest `deepagents` interface naming.

## 5. Complete YAML Configuration Ready for Direct Testing

For reference:

- `resources/configs/default/agents/msagent-filter-smoke.yml`

The key part is as follows:

```yaml
tools:
  patterns:
    - impl:deepagents:get_tool
    - impl:deepagents:run_tool
    - impl:deepagents:fetch_skills
    - impl:deepagents:get_skill
    - mcp:msprof-mcp:*
  use_catalog: false

skills:
  patterns:
    - profiler:ascend-profiler-data-validation
    - profiler:ascend-profiler-db-explorer
  use_catalog: false
```

## 6. Testing Recommendations

After startup, ask:

```text
What MCP, tools, and skills do you have?
```

Expected results:

- skills returns only the set allowed by `patterns`.
- tools returns only the allowed `impl:deepagents:*` and `mcp:msprof-mcp:*` tools.
- Excluded skills and tools no longer appear.
