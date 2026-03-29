# Refact to deepagents

## 总体原则
使用deepagents替代现有工程里面用langchain实现的agent，尽可能降低使用langchain造轮子的事情，优先使用deepagents的组件，tests原有测试用例不用管
可以参考官方deepagents cli（https://github.com/langchain-ai/deepagents/tree/main/libs/cli）的实现，学习使用deepagents来写CLI

## 目标
1. src/msagent/cli 里面关于TUI相关代码尽可能不动，只修改后端，保证前端用户感知不变
2. resources 里配置agents的形式不变，可以做合适的修改
3. src/msagent/tools 里面的工具如果deepagent有，就删掉，使用deepagent内置的
4. src/msagent/sandboxes 直接替换词deepagents的实现
5. src/msagent/agents 不要使用langchain再造轮子，直接对接deepagent的接口，精简代码量
6. mcp、skills的配置方式和原来保持一样，后端使用deepagents的接口
7. 上下文压缩直接替换deepagents的实现
8. subagents、server先删除，这版本不需要
9. src/msagent/utils 里面无用代码删除

## 判断
如果重构工作量大，不如直接重写，保留前端

## TODO
1. deepagents CLI 通过**工具结果驱逐（Tool Result Eviction）**机制处理超长的工具返回。当工具返回的内容超过令牌限制时，系统会自动将其保存到文件系统，并向LLM返回一个指针和预览，而不是完整内容。 filesystem.py:1303-1334