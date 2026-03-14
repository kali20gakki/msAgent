# msAgent Local Config

This directory stores project-local runtime configuration for `msagent`.

- `config.agents.yml`: agent selection and defaults
- `config.llms.yml`: LLM aliases and provider settings
- `config.mcp.json`: MCP server configuration, including `msprof-mcp`
- `config.approval.json`: tool approval defaults, including high-risk `run_command` prompts
- `skills/`: project-local skills loaded in addition to default `mindstudio-skills`
- `sandboxes/`: sandbox profiles used by tools and MCP servers

These files are copied into `./.msagent/` on first run.
