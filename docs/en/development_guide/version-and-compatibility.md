# Version and Compatibility

This document summarizes the version, runtime requirements, and compatibility information of the current code repository.

## Current Version Information

| Item | Description |
|---|---|
| Current version | 0.1.2 |
| Package name | mindstudio-agent |
| CLI entry | msagent |
| Python requirement | >=3.11 |
| Default built-in MCP | msprof-mcp==0.1.6 |

## Currently Supported LLM Providers

The runtime currently supports the following providers:

- `openai`
- `anthropic`
- `google`

Compatibility notes:

- For compatibility, the runtime maps `gemini` in legacy configurations to `google`.
- The `custom` provider is no longer provided separately. For self-deployed services, reuse the same three providers based on protocol compatibility and point `--llm-base-url` to your custom address.

## Current Capabilities Overview

- Supports the default interactive session and launching a specified agent.
- Provides the following seven built-in agents: `Profiler`, `Accuracy`, `Quantizer`, `Modeling`, `Operator`, `Minos`, and `SpecTrainer`.
- Supports scenarios such as Ascend performance tuning, profiling analysis, accuracy analysis, model quantization, msModeling simulation modeling, operator tuning, documentation walkthrough, speculative decoding training data resampling, and tool consultation.
- Supports MCP extensions, with `msprof-mcp` enabled by default.
- Supports skills extensions. The built-in skills in the source repository are loaded directly from the `skills/` directory at the repository root.

## Viewing the Locally Installed Version

```bash
msagent --version
```

## Version History

## Versioning Policy

The project follows semantic versioning (SemVer):

- Patch versions focus on compatibility fixes.
- Minor versions add features while maintaining backward compatibility.
- Major versions contain incompatible changes.
