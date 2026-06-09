# msAgent Local Config

This directory stores project-local runtime configuration for `msagent`.

- `config.agents.yml`: agent selection and defaults
- `config.llms.yml`: LLM aliases and provider settings
- `config.mcp.json`: MCP server configuration, including `msprof-mcp`
- `config.approval.json`: deepagents Human-in-the-Loop (`interrupt_on`) plus fine-grained `decision_rules`
- `skills/`: project-local skills loaded in addition to the bundled default skills
- `sandboxes/`: sandbox profiles used by tools and MCP servers

These files are copied into `./.msagent/` on first run.

## Tavily API key setup

This README is the single source of truth for Tavily MCP configuration in the default local config template.

If you enable `tavily-mcp` in `config.mcp.json`, the recommended default is to load `TAVILY_API_KEY` from an existing environment variable instead of writing the key directly into the file.

Example:

```json
{
  "mcpServers": {
    "tavily-mcp": {
      "command": "npx",
      "args": ["-y", "tavily-mcp@latest"],
      "transport": "stdio",
      "env": {
        "TAVILY_API_KEY": "${TAVILY_API_KEY}"
      },
      "enabled": true,
      "stateful": true,
      "repair_timeout": 30,
      "invoke_timeout": 120.0
    }
  }
}
```

This keeps the secret out of the repo and out of copied local config files. If you must use an inline value temporarily, prefer changing it only in your private local environment and avoid committing it.

Recommended steps:

1. Get your Tavily API key from the Tavily dashboard.
2. Export `TAVILY_API_KEY` in your shell or user environment.
3. Open `.msagent/config.mcp.json`.
4. Find the `tavily-mcp` server entry.
5. Keep `TAVILY_API_KEY` under `env` as `"${TAVILY_API_KEY}"`.
6. Restart the `msagent` session so the MCP server is started again with the new environment.

Notes:

- If `tavily-mcp` is enabled but no `TAVILY_API_KEY` is available, Tavily behavior depends on the MCP server implementation and runtime environment.
- In the default template, `TAVILY_API_KEY` intentionally references the process environment so secrets do not need to live in `config.mcp.json`.
- In this project, when Tavily is enabled and the key validates successfully, Tavily tools are preferred over the built-in `web_search`.
- When the key is missing, invalid, or cannot be validated, the built-in `web_search` is kept as a fallback tool.
