# 快速入门指导

本文承接 [入门安装指南](install_guide.md)，介绍如何配置模型、选择 Agent，并进入最小可用的交互流程。

## 配置 LLM

当前 `config` 子命令直接支持 `openai`、`anthropic`、`google` 三类 Provider。对于自部署服务、企业网关或代理层，请根据接口协议兼容性复用上述 Provider，并通过 `--llm-base-url` 指定服务地址。

OpenAI 兼容接口示例：

```bash
export OPENAI_API_KEY="your-key"
msagent config --llm-provider openai --llm-base-url "https://api.deepseek.com/v1" --llm-model "deepseek-chat"
```

本地 OpenAI 兼容服务示例：

```bash
export OPENAI_API_KEY="dummy"
msagent config --llm-provider openai --llm-base-url "http://127.0.0.1:8000/v1" --llm-model "your-model"
```

Anthropic 兼容服务示例：

```bash
export ANTHROPIC_API_KEY="your-key"
msagent config --llm-provider anthropic --llm-base-url "https://example.com/anthropic" --llm-model "claude-sonnet-4-20250514"
```

Google / Gemini 服务示例：

```bash
export GOOGLE_API_KEY="your-key"
msagent config --llm-provider google --llm-base-url "https://example.com/google" --llm-model "gemini-2.5-pro"
```

查看当前配置：

```bash
msagent config --show
```

## 启动会话

进入默认交互式会话：

```bash
msagent
```

手动指定启动 Agent：

```bash
msagent --agent Hermes
msagent --agent Accuracy
msagent --agent Zephyr
msagent --agent Minos
msagent --agent Icarus
```

更多命令，请参见[《使用指南》](../user_guide/usemap.md)
