# msAgent快速入门

本文介绍如何配置模型、选择 Agent 功能、启动并进入msAgent最小可用的交互流程。

## 1. 环境准备

安装昇腾NPU驱动和配套版本的CANN软件（包含Toolkit和ops包）并配置环境变量，具体请参见《[CANN 快速安装](https://www.hiascend.com/cann/download)》。

## 2. 配置 LLM

1. 准备一个可用的 LLM API Key。

   需要用户自行登录模型服务商网站进行创建。

2. 配置 LLM。

   包括 LLM 服务的环境变量（*_API_KEY）、通过`msagent config`命令的`--llm-provider`参数配置LLM 服务的协议类型、`--llm-base-url`参数配置模型服务商地址、`--llm-model`参数配置模型名称（模型名称从模型服务商网站的模型广场获取）。

   - OpenAI 兼容接口示例：

     ```bash
     export OPENAI_API_KEY="your-key"
     msagent config --llm-provider openai --llm-base-url "https://api.deepseek.com/v1" --llm-model "deepseek-v4-flash"
     ```

   - 本地 OpenAI 兼容服务示例：

     ```bash
     export OPENAI_API_KEY="dummy"
     msagent config --llm-provider openai --llm-base-url "http://127.0.0.1:8000/v1" --llm-model "your-model"
     ```

   - Anthropic 兼容服务示例：

     ```bash
     export ANTHROPIC_API_KEY="your-key"
     msagent config --llm-provider anthropic --llm-base-url "https://example.com/anthropic" --llm-model "claude-sonnet-4-20250514"
     ```

   - Google / Gemini 服务示例：

     ```bash
     export GOOGLE_API_KEY="your-key"
     msagent config --llm-provider google --llm-base-url "https://example.com/google" --llm-model "gemini-2.5-pro"
     ```

3. 查看当前配置。

   ```bash
   msagent config --show
   ```

   显示步骤2配置的参数值则表示配置成功。

## 3. 启动会话

- 启动并进入默认交互式会话。

  ```bash
  msagent
  ```

- 启动并进入[Profiler 性能调优](../agent_guide/Profiler.md)功能交互式会话。

  ```bash
  msagent --agent Profiler
  ```

- 启动并进入[Accuracy 精度调试](../agent_guide/Accuracy.md)功能交互式会话。

  ```bash
  msagent --agent Accuracy
  ```

- 启动并进入[Quantizer 模型量化](../agent_guide/Quantizer.md)功能交互式会话。

  ```bash
  msagent --agent Quantizer
  ```

- 启动并进入[Operator 算子调优](../agent_guide/Operator.md)功能交互式会话。

  ```bash
  msagent --agent Operator
  ```

- 启动并进入[Minos 文档辅助](../agent_guide/Minos.md)功能交互式会话。

  ```bash
  msagent --agent Minos
  ```

- 更多命令请参见《[msAgent使用指南](../user_guide/usemap.md)》。
