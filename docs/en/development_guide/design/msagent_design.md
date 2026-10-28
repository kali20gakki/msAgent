# msAgent Design Document

## Revision History

| Date | Revision | Description | Author | RFC Document |
| -- | -- | -- | -- | -- |
| 2026-06-03 | 1.0 | Added the msAgent detailed design document, covering the architecture, interaction chain, extension mechanisms, and test design. | kali20gakki1 |  |

## Background Description

### 1. Product Positioning

msAgent is a one-stop debugging and tuning solution for the Ascend development process. Rather than a single-point tool, it integrates the following capabilities into a unified interactive CLI:

- Specialized agents for different problem domains, such as performance tuning, accuracy analysis, model quantization, operator tuning, documentation experience, and code review.
- An LLM adaptation layer for multiple model providers, supporting model integration from OpenAI, Anthropic, Google/Gemini, and others.
- An MCP (Model Context Protocol) tool integration mechanism for extending external capabilities.
- A skill assembly capability for workflow reuse.
- Checkpoint, context compression, approval, timeout, and retry mechanisms for stable operation of long-running sessions.

### 2. Pain Points

Using agents to debug and tune in the Ascend ecosystem faces several typical difficulties:

- Diverse problem domains: Performance, accuracy, quantization, operator optimization, documentation experience, and other problems require different knowledge systems.
- Fragmented toolchains: In MindStudio scenarios, multiple CLI tools are often involved at the same time. For example, performance tuning involves msprof and Profiling data, model scripts, configuration files, documentation, and auxiliary analysis scripts. Therefore, locating problems often requires switching back and forth between multiple tool entry points and multiple data carriers.
- Complex context: Tuning tasks often span multiple conversation rounds and involve large amounts of context such as logs, Profiler data, code, configurations, and historical conclusions.
- Non-negligible risks: Tool execution may involve shell commands, external connections, and long-running tasks. Therefore, protection mechanisms such as approval, timeout, and retry are required.
- High extension costs: Without a clear assembly layer, integrating new agents, new skills, and new MCP services quickly erodes the system boundaries.

### 3. Core Value

| Value Point | Description |
| -- | -- |
| Unified entry point | Expose a unified entry point through the `msagent` CLI, reducing learning and switching costs. |
| Domain separation | Use the agent + subagent + skill combination to carry knowledge from different domains, instead of stuffing all logic into a single prompt. |
| Configuration-driven | LLM, agent, checkpointer, MCP, sandbox, and approval are driven by a combination of the installation package defaults and global user overrides. |
| Controlled extensibility | Tool pattern, skill pattern, and MCP include/exclude jointly define the capability boundaries. |
| Stable operation | Checkpoints, retry, timeout, approval, middleware, and context compression keep long-chain conversations sustainable. |

### 4. Design Goals and Non-Goals

#### 4.1 Design Goals

- Support multiple specialized agents while maintaining a unified runtime model.
- Support global configuration, read-only default templates, project state isolation, and directory-based extension.
- Support CLI/TUI interaction using a unified graph runtime.
- Support composable assembly and fine-grained filtering of tool, skill, and MCP.
- Support context governance for long-running sessions, including checkpoints, memory injection, compression offloading, and externalization of large results.
- Provide stable boundaries for testing, packaging, and documentation release.

#### 4.2 Non-Goals

- This design document does not elaborate on the internal algorithm implementation details of each skill.
- msAgent is not designed as a GUI-first frontend product. The current primary entry point remains CLI/TUI.
- The runtime does not directly host hardware Profiling/quantization algorithms. Instead, these are accomplished collaboratively through skills, MCP, and external tools.

## Solution Design

### 1. Design Principles

- **Configuration-first**: Models, agents, MCP, approval, and checkpoints are all resolved through configuration, without hard-coding policies into service logic.
- **Centralized assembly**: Use the `Initializer` to centrally assemble the graph runtime, avoiding CLI and tests each assembling dependencies separately.
- **Clear boundaries**: CLI handles interaction, `ConfigRegistry` handles configuration, `AgentFactory` handles graph construction, and `MessageDispatcher` handles the message flow.
- **Incremental extension**: When adding agents, skills, or MCP services, extend along the existing directory structure and filtering rules first, rather than modifying the core main flow.
- **Runtime controllability**: Control risks in complex scenarios through approval, timeout, retry, context compression, and externalization of tool results.

### 2. Overall Architecture

#### 2.1 Layered Architecture Diagram

```mermaid
flowchart TB
    User["User / Tuning Engineer"] --> CLI["CLI / TUI session layer<br/>app.py + chat.py + Session"]

    CLI --> INIT["Initializer<br/>Runtime assembly and cache center"]

    INIT --> REG["ConfigRegistry<br/>Built-in defaults + global overrides"]
    INIT --> AF["AgentFactory<br/>deepagents graph construction"]
    INIT --> MF["MCPFactory / MCPClient"]
    INIT --> SF["SkillFactory"]
    INIT --> LF["LLMFactory"]
    INIT --> CPF["Checkpointer creation"]

    AF --> DAG["deepagents / LangGraph Compiled Graph"]
    DAG --> MW["Middlewares<br/>Memory / Skills / Retry / ToolResultEviction / SystemMessage"]
    DAG --> TOOLS["Runtime Tools<br/>deepagents built-ins + catalog tools + web_search + MCP tools"]
    DAG --> CTX["AgentContext / AgentState"]

    REG --> GLOBAL["MSAGENT_HOME<br/>default ~/.msagent/"]
    REG --> DEFAULTS["resources/configs/default/"]

    MF --> MCP["External MCP Servers<br/>default msprof-mcp"]
    SF --> SKILLS["Project skills/ + global skills + built-in skills"]
    LF --> MODELS["OpenAI / Anthropic / Google(Gemini)"]
    CPF --> CHECKPOINT["Memory / SQLite Checkpointer"]

    DAG --> BACKEND["CompositeBackend<br/>LocalShell + Filesystem routes"]
    BACKEND --> HIST["conversation_history/"]
    BACKEND --> LARGE["large_tool_results/"]
```

#### 2.2 Architecture Interpretation

Overall, msAgent adopts an approach of "**unified entry layer, centralized runtime assembly, and decoupled capability modules**":

- CLI/TUI serves as the unified entry point, and all interaction ultimately relies on `Initializer.create_graph()` to assemble the same kind of graph runtime.
- `ConfigRegistry` merges the installation package built-in defaults with the `MSAGENT_HOME` user overrides and organizes them into strongly typed configuration objects.
- `AgentFactory` is responsible for actually assembling the LLM, tools, skills, middleware, backends, and checkpoints into a `CompiledStateGraph`.
- `MessageDispatcher` is responsible for the message flow, streaming rendering, approval resume, automatic compression, and token statistics during the runtime phase.
- The three capability sources of tools, skills, and MCP jointly determine the "capability boundary" of each agent through pattern filtering.

### 3. Module Responsibility Breakdown

| Module | Representative Path | Main Responsibilities | Design Highlights |
| -- | -- | -- | -- |
| CLI startup layer | src/msagent/cli/bootstrap/ | Parse commands, start sessions, and route to chat/config modes. | `normalize_argv()` automatically routes bare invocations to the default interactive session. |
| Session layer | src/msagent/cli/core/ | Save thread context, session state, hot-switching of Agent/Model, and tool output memory. | `Context` and `Session` isolate runtime UI state from the graph lifecycle. |
| Message dispatch layer | src/msagent/cli/dispatchers/ | Handle slash commands, regular messages, streaming output, and approval resume. | `MessageDispatcher` is the main entry point for single-turn conversation execution. |
| Configuration layer | src/msagent/configs/ | Define configuration models for agent/LLM/MCP/Approval/checkpointer and so on. | Modeled with Pydantic, with version migration logic retained. |
| Runtime assembly layer | src/msagent/cli/bootstrap/initializer.py | Aggregate all configuration, create the graph, and cache the visible capability catalog. | Serves as the true dependency injection center of the system. |
| Agent construction layer | src/msagent/agents/factory.py | Call `create_deep_agent()` to compile the graph and inject middleware and backends. | Tool filtering, system prompt rendering, result externalization, and other capabilities are centralized here. |
| LLM layer | src/msagent/llms/factory.py | Parse model Provider, API Key, Base URL, timeout, and HTTP client parameters. | Compatible with both official and OpenAI-compatible gateways. |
| MCP layer | src/msagent/mcp/ | Parse MCP connections, fetch MCP tools, and attach timeout wrappers. | Uses `tool_name_prefix=True` by default to resolve tool namespace issues. |
| Tool/skill layer | src/msagent/tools/, src/msagent/skills/ | Unify tool wrapping, capability cataloging, and skill scanning and reading. | `fetch_tools/get_tool/run_tool` and `fetch_skills/get_skill` form a self-describing capability interface. |
| Middleware layer | src/msagent/middlewares/ | Cross-cutting capabilities such as tool result externalization, token statistics, and approval | Allows runtime enhancements without intruding on the core service chain. |

### 4. Startup and Assembly Design

#### 4.1 Startup Sequence Diagram

```mermaid
sequenceDiagram
    participant User as User
    participant App as app.py / legacy parser
    participant Chat as handle_chat_command
    participant Ctx as Context.create
    participant Init as Initializer
    participant Reg as ConfigRegistry
    participant MCP as MCPFactory/MCPClient
    participant Skill as SkillFactory
    participant AgentF as AgentFactory
    participant Graph as deepagents graph

    User->>App: msagent [-a Agent] [-m Model] [message]
    App->>Chat: Parse into a default session command
    Chat->>Ctx: Create a runtime Context
    Ctx->>Init: load_agent_config / load_llm_config
    Init->>Reg: Resolve global paths and project state, and load the layered configuration
    Reg-->>Init: agent/LLM/MCP/Approval/checkpointer configuration
    Init->>MCP: Create MCPClient
    MCP-->>Init: Visible MCP tools + module_map
    Init->>Skill: Scan and filter Skills
    Init->>AgentF: create(config, llm, mcp, skills, checkpointer)
    AgentF->>Graph: create_deep_agent(...)
    Graph-->>Init: CompiledStateGraph
    Init-->>Chat: graph + cleanup
    Chat->>User: Enter the one-shot or interactive loop
```

#### 4.2 Key Points of the Startup Chain

1. **CLI compatibility routing**
   `legacy.py` converges the command surface into two categories, `config` and the default session, while retaining runtime parameters such as `--agent`, `--model`, and `--approval-mode`.

2. **Configuration and working directory separation**
   `AppPaths` resolves `MSAGENT_HOME`, and `ConfigRegistry` reads the wheel defaults directly and layers the global overrides on top. `working_dir` is retained only as the tool execution root directory.

3. **Cache-based assembly**
   After graph construction, the `Initializer` caches:
   - `cached_llm_tools`
   - `cached_tools_in_catalog`
   - `cached_agent_skills`
   - `cached_mcp_server_names`

   These caches serve interactive commands such as `/tools`, `/skills`, and `/mcp`, as well as subsequent prompt template rendering.

### 5. Agent Architecture Design

#### 5.1 Specialized Agent Design

The default template includes 7 main agents, all assembled through YAML configuration. The core differences lie in prompt, visible tools, visible skills, subagent combinations, and default status.

| Agent | Domain Positioning | Default Status | Typical Tool Pattern | Typical Skill Pattern | Subagent |
| -- | -- | -- | -- | -- | -- |
| Profiler | Ascend Profiling/performance analysis | Default agent | `impl:deepagents:*` + `mcp:msprof-mcp:*` | Profiler DB analysis, fast/slow device diagnosis, MFU calculation | explorer + general-purpose + ascend-knowledge |
| Accuracy | Model accuracy analysis | No | `impl:deepagents:*` | RL consistency, NaN/overflow, determinism analysis | explorer + general-purpose |
| Quantizer | Model quantization and adaptation | No | `impl:deepagents:*` | msModelSlim analysis, adaptation, quantization | explorer + general-purpose |
| Modeling | `msmodeling` simulation modeling | No | `impl:deepagents:*` | `text_generate`/`throughput_optimizer`/device profiling/model integration preparation | explorer + general-purpose |
| Operator | Operator performance optimization | No | `impl:deepagents:*` + specific MCP patterns | AscendC operator optimization, operator profiler | explorer + general-purpose |
| Minos | Documentation experience and code review | No | `impl:deepagents:*` | `document-ux-review`, `atomgit-code-reviewer` | explorer |
| SpecTrainer | Speculative decoding on-policy resampling (response regeneration) | No | `impl:deepagents:*` | `speculators:*` (precheck → input normalization → regenerate pre-tokenized samples) | explorer |

This design implies:

- Domain capabilities are not hard-coded into Python branch logic, but are orchestrated through agent configuration.
- Most domain differences are pushed down into prompt and skill combinations, keeping the underlying runtime unified.
- Each agent can precisely control its authorization boundary through tool pattern and skill pattern, reducing the risk of misuse caused by overly broad capabilities.

#### 5.2 Subagent Design

The default template includes three subagents:

- `explorer`: focuses on code/repository structure exploration, suitable for locating files, implementation points, and dependencies.
- `general-purpose`: focuses on multi-step research, comprehensive analysis, and reasoning.
- `ascend-knowledge`: a subagent for Ascend material and knowledge queries. It locates and verifies Ascend official documentation, community knowledge, examples, and open-source code from the Ascend organization, and returns conclusions with an evidence level (`sources`/`ref`/`facts`/`inferences`/`unverified`). Profiler currently integrates it.

They essentially reuse the same runtime as the main agent, with only these differences:

- Different prompts.
- Lighter default model aliases (`haiku-4.5` in the template).
- Broader tool patterns, used to assist the division of labor of the main agent.

#### 5.3 Domain Capability Overview

Default agents define their runtime behavior through the sequence of "problem domain -> prompt constraints -> tool boundaries -> skill combinations -> subagent collaboration methods".

```mermaid
flowchart LR
    U["User question"] --> Q1["Performance / Profiling"]
    U --> Q2["Accuracy anomalies / NaN / consistency"]
    U --> Q3["Quantization / model adaptation"]
    U --> Q4["msmodeling modeling / deployment planning"]
    U --> Q5["Operator performance bottlenecks"]
    U --> Q6["Documentation experience / code review"]
    U --> Q7["Speculative decoding data resampling"]

    Q1 --> H["Profiler"]
    Q2 --> A["Accuracy"]
    Q3 --> Z["Quantizer"]
    Q4 --> O["Modeling"]
    Q5 --> I["Operator"]
    Q6 --> M["Minos"]
    Q7 --> S["SpecTrainer"]

    H --> H1["Prompt: Performance analysis methodology"]
    H --> H2["Tools: deepagents + msprof-mcp"]
    H --> H3["Skills: profiler DB / fast and slow cards / MFU"]
    H --> H4["Subagents: explorer + general-purpose + ascend-knowledge"]

    A --> A1["Prompt: Accuracy diagnosis methodology"]
    A --> A2["Tools: deepagents"]
    A --> A3["Skills: RL consistency / NaN detection"]
    A --> A4["Subagents: explorer + general-purpose"]

    Z --> Z1["Prompt: Quantization and adaptation methodology"]
    Z --> Z2["Tools: deepagents"]
    Z --> Z3["Skills: msModelSlim analysis / adaptation / quantization"]
    Z --> Z4["Subagents: explorer + general-purpose"]

    O --> O1["Prompt: Simulation modeling methodology"]
    O --> O2["Tools: deepagents"]
    O --> O3["Skills: reserved msmodeling integration slots"]
    O --> O4["Subagents: explorer + general-purpose"]

    I --> I1["Prompt: Operator tuning methodology"]
    I --> I2["Tools: deepagents + specific MCP"]
    I --> I3["Skills: AscendC / operator profiler"]
    I --> I4["Subagents: explorer + general-purpose"]

    M --> M1["Prompt: Documentation walkthrough / review methodology"]
    M --> M2["Tools: deepagents"]
    M --> M3["Skills: document-ux-review / PR review"]
    M --> M4["Subagents: explorer"]

    S --> S1["Prompt: Resampling criteria and data specifications"]
    S --> S2["Tools: deepagents"]
    S --> S3["Skills: precheck / normalization / response regeneration"]
    S --> S4["Subagents: explorer"]
```

The 7 default agents share the same runtime skeleton and differ through configuration along the following dimensions:

| Dimension | Profiler | Accuracy | Quantizer | Modeling | Operator | Minos | SpecTrainer |
| -- | -- | -- | -- | -- | -- | -- | -- |
| Primary problem domain | Profiling/performance bottlenecks | Accuracy anomalies | Quantization and adaptation | msmodeling simulation modeling | Operator optimization | Documentation and code review | Speculative decoding training data resampling |
| Prompt focus | Scheduling, hotspots, communication, MFU | Consistency, NaN, overflow | Model structure, quantization risks, adaptation costs | Simulation parameters, deployment modes, input constraints, validation paths | Operator hotspots, end-to-end performance | Getting-started experience, documentation usability, PR risks | Endpoint readiness and `return_token_ids`, data parseability, record count/concurrency/sampling parameters, artifact and report paths |
| Tool boundaries | deepagents + msprof-mcp | deepagents | deepagents | deepagents | deepagents + specific MCP | deepagents | deepagents |
| Skill combination characteristics | Strongly relies on Profiling data analysis skills. | Strongly relies on diagnostic skills. | Strongly relies on quantization/adaptation skills. | First version reserves dedicated msmodeling skill extension slots. | Strongly relies on operator tuning skills. | Strongly relies on process review skills. | Strongly relies on on-policy regeneration orchestration skills. |
| Collaboration approach | Main agent makes decisions, subagents supplement with exploration and comprehensive analysis. | Same as left | Same as left | Same as left | Same as left | Leans toward explorer for auxiliary information collection. | Same as left |

The default agent architecture maps different problem domains to different analysis strategies and capability boundaries through configuration. When adding agents, follow the same extension approach: define capabilities through prompt, skill, tool pattern, and subagent combinations, without duplicating the runtime implementation.

### 6. Configuration System Design

#### 6.1 Configuration Sources and Priority

msAgent configuration follows a "read-only installation package defaults + global user overrides" organization. `ConfigRegistry` no longer copies the entire set of default templates, and runtime state such as project memory, checkpoints, and history is uniformly written to a project-isolated directory under the global home.

A typical structure is as follows:

```text
resources/configs/default/
├─ config.llms.yml
├─ config.mcp.json
├─ config.approval.json
├─ agents/
│  ├─ Profiler.yml
│  ├─ Accuracy.yml
│  ├─ Quantizer.yml
│  ├─ Modeling.yml
│  ├─ Operator.yml
│  ├─ Minos.yml
│  └─ SpecTrainer.yml
├─ subagents/
│  ├─ explorer.yml
│  ├─ general-purpose.yml
│  └─ ascend-knowledge.yml
├─ llms/
├─ checkpointers/
├─ sandboxes/
├─ prompts/
└─ skills/

~/.msagent/
├─ config/
│  ├─ config.llms.yml
│  ├─ config.mcp.json
│  ├─ config.approval.json
│  ├─ agents/
│  ├─ subagents/
│  ├─ llms/
│  ├─ checkpointers/
│  └─ sandboxes/
├─ prompts/
├─ skills/
├─ cache/
├─ oauth/
├─ logs/
└─ state/projects/<project-id>/
   ├─ project.json
   ├─ checkpoints.sqlite
   ├─ memory.md
   ├─ history
   ├─ conversation_history/
   └─ audit_log/
```

The configuration directory is divided into three layers:

- `resources/configs/default/`: the **read-only default layer** distributed with the source code or installation package.
- `~/.msagent/config/`: the **user override layer** shared by all projects.
- `~/.msagent/state/projects/<project-id>/`: the **project state layer** isolated by canonical working directory.

The configuration reading and initialization rules are as follows:

| Scenario | Behavior |
| -- | -- |
| Running `msagent` in a working directory for the first time | Create the global directory and the state directory for the current project, without modifying the working directory. |
| Reading LLM/agent/subagent/checkpointer/sandbox | Overlay the user items in `~/.msagent/config/` onto the built-in defaults by semantic key. |
| Switching agent or model | Write the current selection to the project `project.json`, without modifying or copying the built-in agent definitions. |
| Reading MCP | Merge the built-in definitions and `config/config.mcp.json` by server name. |
| Reading approval configuration | If the user file exists, read `config/config.approval.json`. Otherwise, read the built-in defaults directly. |
| Advanced modification of a built-in agent | Write only the name, version, and modified fields. All other fields continue to inherit the installation package default definitions. |

The key features of the configuration system are as follows:

- **No copying of defaults**: A wheel update can directly provide new default items, and the user directory stores only actual overrides.
- **Directory-based extension**: Support directory-organized formats such as `agents/*.yml`, `llms/*.yml`, `checkpointers/*.yml`, and `sandboxes/*.yml`, making it easy to maintain them one by one.
- **Explicit path injection**: `AppPaths` and `ProjectPaths` uniformly control the storage locations of configuration, global resources, and project state.
- **Strong-type validation**: Everything converges to Pydantic configuration objects, ensuring that key fields such as Provider, pattern, timeout, approval rules, and MCP transport are validated.

#### 6.2 Key Configuration Objects

| Configuration Object | Key Fields | Design Purpose |
| -- | -- | -- |
| LLMConfig | provider, model, alias, base_url, api_key_env, extended_reasoning | Decouple "model aliases" from "real models", compatible with official APIs and compatible gateways. |
| AgentConfig | prompt, llm, tools, skills, subagents, compression, retry | Define the complete runtime profile of a single agent. |
| ToolsConfig | patterns, output_max_tokens, execution_timeout_seconds | Determine the visible tool scope, the large output handling threshold, and the execution timeout. |
| SkillsConfig | patterns | Determine the visible skill scope. |
| MCPServerConfig | command/url, transport, include/exclude, invoke_timeout | Safely incorporate external MCP services into the tool plane. |
| ToolApprovalConfig | interrupt_on, decision_rules | Bring high-risk tools into the approval and automatic decision framework. |
| CheckpointerConfig | type, connection_string | Determine the session state persistence backend. |

#### 6.3 Path Responsibilities

- `working_dir`: The root directory for agent file operations, shell, and project-level `skills/`.
- `AppPaths`: The global configuration, prompt, skill, cache, OAuth, and log directories.
- `ProjectPaths`: The memory, history, checkpoint, conversation history, and audit of the current project.

Different projects share the user configuration but do not share session state.

### 7. Tool/Skill/MCP Capability Surface Design

#### 7.1 Tool Design

`ToolFactory` is responsible for unified tool adaptation and timeout wrapping. The current capability surface includes three sources:

- **Runtime built-in tools**: the filesystem/command execution capabilities of deepagents, as well as catalog tools such as `fetch_tools`, `get_tool`, `run_tool`, `fetch_skills`, `get_skill`, and `web_search`.
- **MCP tools**: from `MCPClient.tools()`, mapped back to their owning namespaces through the service name prefix.
- **Middleware-enhanced tool behavior**: for example, timeout wrapping and externalization of large results.

Design highlights:

- `fetch_tools/get_tool/run_tool` let the agent first learn about tools, then selectively call them, reducing blind usage.
- `ToolFactory.wrap_tool_with_timeout()` uniformly applies timeout protection to synchronous and asynchronous tools.
- When tool output is too large, `ToolResultEvictionMiddleware` externalizes the results to a virtual filesystem, preventing the context from being blown up by large text blocks.

#### 7.2 Skill Design

Skills are lightweight assets for workflow reuse. The current scan order is:

1. `<working-dir>/skills`
2. `~/.msagent/skills`
3. Built-in `skills/` in the repository root or in the packaged distribution

This means:

- Project-level and global user skills can override default skills.
- Packaged and source forms share a unified built-in skill distribution approach.
- Skills describe their trigger semantics and execution requirements through `SKILL.md`, instead of being tightly bound to Python code entry points.

As of the current repository state, 19 built-in skills with `SKILL.md` can be scanned in the root directory, reflecting the dual-layer positioning of msAgent as a "platform + domain knowledge base".

#### 7.3 MCP Design

The external MCP service enabled in the default template is `msprof-mcp`. MCP integration adopts the following design:

- Declare connection information through `config.mcp.json` instead of scattering CLI parameters in code.
- Use `include/exclude` to control the subset of tools exposed by the service.
- Use `invoke_timeout` and `default_invoke_timeout` to handle long-running tools.
- Use `tool_name_prefix=True` to maintain namespace isolation for tools from multiple services on the unified tool plane.

#### 7.4 Capability Surface Assembly Relationship Diagram

```mermaid
flowchart TB
    AgentCfg["AgentConfig"] --> ToolPattern["tools.patterns"]
    AgentCfg --> SkillPattern["skills.patterns"]
    AgentCfg --> Timeout["execution_timeout_seconds / output_max_tokens"]

    ToolPattern --> RuntimeTools["Runtime built-in Tools"]
    ToolPattern --> MCPTools["MCP Tools"]
    SkillPattern --> SkillCatalog["Skill Catalog"]

    RuntimeTools --> AgentGraph["Compiled Graph"]
    MCPTools --> AgentGraph
    SkillCatalog --> AgentGraph
    Timeout --> Middleware["Timeout Wrapper / ToolResultEvictionMiddleware"]
    Middleware --> AgentGraph
```

### 8. Conversation Execution Chain Design

#### 8.1 Single-Turn Request Sequence Diagram

```mermaid
sequenceDiagram
    participant User as User
    participant Session as Session
    participant Msg as MessageDispatcher
    participant Graph as LangGraph / deepagents graph
    participant LLM as LLM
    participant HITL as InterruptHandler
    participant Tool as Tool / MCP Tool
    participant Backend as LocalShell / Filesystem / MCP

    User->>Session: Input a message or slash command
    Session->>Msg: dispatch(content)
    Msg->>Msg: MessageBuilder parses text/file/image references
    Msg->>Graph: astream(messages, thread_id, AgentContext)
    Graph->>LLM: Generate a response or tool plan
    alt High-risk tool approval triggered
        Graph-->>Msg: interrupt_on payload
        Msg->>HITL: Show the approval panel
        User->>HITL: approve / reject / edit
        HITL->>Graph: Command(resume=...)
    end
    alt Tool call needed
        Graph->>Tool: ainvoke()
        Tool->>Backend: Execute shell / file / MCP / Web retrieval
        Backend-->>Tool: Return the result
        Tool-->>Graph: ToolMessage
        Graph->>LLM: Continue reasoning based on the tool result
    end
    Graph-->>Msg: AIMessage / ToolMessage / token usage
    Msg->>Session: Update token, tool output cache, and thread state
    Session-->>User: Stream and render the final result
```

#### 8.2 Key Design Points of the Execution Chain

1. **AgentContext injection**
   `MessageDispatcher._build_agent_context()` constructs:
   - The current working directory
   - Platform and OS version
   - Current time
   - A local environment snapshot
   - The currently enabled MCP list
   - The project `memory.md`
   - The currently visible tool catalog and skill catalog

   `_SystemMessageMiddleware` then renders these variables into the system prompt template. This design gives the prompt "environment awareness" without hard-coding environment information into prompt files.

2. **Streaming rendering and the tool activity area**
   `MessageDispatcher` processes two types of output from `astream()`:
   - `messages` stream: aggregates the AI token stream, thought previews, and tool call previews.
   - `updates` stream: renders the final AIMessage, ToolMessage, token statistics, and tool results.

   This design lets the CLI provide real-time feedback on "what is being done", instead of only emitting a long text block at the end.

3. **Approval resume**
   `ToolApprovalConfig` is converted into an `interrupt_on` payload injected into the graph runtime. When a high-risk tool (especially `execute`) hits the policy, the graph interrupts, the CLI pops up the approval interface, and the user's decision resumes execution through `Command(resume=...)`.

4. **Large result governance**
   `ToolResultEvictionMiddleware` uses the `FilesystemMiddleware` of deepagents to externalize large tool outputs as virtual file content and normalizes ToolMessages into text, reducing context bloat and rendering complexity.

### 9. Long-Running Sessions, State, and Compression Design

#### 9.1 State Model

`AgentState` adds the following fields on top of the deepagents/LangGraph base state:

- Token tracking: `input_tokens`, `output_tokens`, `current_input_tokens`, `current_output_tokens`
- Interrupt tracking: `interrupts`
- Files and Todos: `files`, `todos`

The Reducer design ensures that different fields have different merge semantics during graph updates, for example:

- Token increment fields use replacement or summation.
- `files` uses a dictionary union update.
- `todos` uses latest-value replacement.

#### 9.2 Checkpoint Design

Two types of checkpointer are currently supported:

- `memory`
- `sqlite`

The default main agent template uses `sqlite`. This means:

- Session threads can be resumed across processes.
- Slash commands such as `/threads` work based on persisted state.
- Long-running sessions are not completely lost when the CLI exits.

#### 9.3 Context Compression Sequence Diagram

```mermaid
sequenceDiagram
    participant Msg as MessageDispatcher
    participant Graph as Graph State
    participant CH as CompressionHandler
    participant LLM as Compression LLM
    participant FS as conversation_history backend

    Msg->>Msg: Check the threshold based on token usage
    alt Compression threshold reached or the user runs /offload
        Msg->>CH: handle()
        CH->>Graph: aget_state(thread_id)
        Graph-->>CH: Current message list and summary events
        CH->>LLM: Summarize the historical messages to offload
        LLM-->>CH: Structured summary
        CH->>FS: Write to the conversation_history file
        CH->>Graph: aupdate_state(_summarization_event)
        CH-->>Msg: Return compressed token statistics
    end
```

The significance of the compression design is:

- Instead of simply deleting history, the raw history is offloaded to the backend filesystem, preserving traceability.
- Compression shares the same graph state as the main conversation thread. Therefore, subsequent conversations can perceive "what was previously compressed".
- The compression prompt also supports local environment context injection, improving the effectiveness of summaries.

### 10. Interaction and Experience Design

#### 10.1 CLI/TUI Design

The CLI layer consists of `Session + InteractivePrompt + Renderer + Dispatcher/Handler`, with the core goal of giving "complex agent behavior" a clear and operable terminal experience.

The main interaction capabilities include:

- Regular message input and one-shot mode.
- Slash commands: `/agents`, `/model`, `/threads`, `/tools`, `/skills`, `/mcp`, `/offload`, `/tool-output`, and so on.
- Keyboard shortcuts: approval mode switching, bash mode, the tool output viewer, and quick help.
- Unified SIGINT behavior: the first press gives priority to interrupting the current stream, and only then exits the session.

### 11. Security, Reliability, and Maintainability Design

#### 11.1 Security Design

- **Approval mechanism**: High-risk tools implement HITL through `interrupt_on` and `decision_rules`.
- **Timeout control**: LLM request timeouts, MCP invoke timeouts, and tool timeout wrapping jointly prevent long-term hangs.
- **Local configuration isolation**: Sensitive information is carried through `api_key_env` and environment variables, avoiding writes to repository files.
- **Tool boundary filtering**: agents only expose tools/skills matched by patterns, avoiding "default full authorization".

#### 11.2 Reliability Design

- **Checkpoints**: Support persistent recovery.
- **Model/tool retry**: `retry.model` and `retry.tool` map to model call and tool call retry capabilities, respectively.
- **Fake Backend**: `MSAGENT_FAKE_BACKEND=1` provides a controllable backend for testing and offline validation.
- **Output externalization**: Prevents large tool results from directly blowing up the context.

#### 11.3 Maintainability Design

- Core assembly logic is centralized in the `Initializer`, reducing duplicate implementations across entry points.
- Configuration models are independent of runtime logic, making incremental evolution and migration easier.
- Directory-organized resources (agent/prompt/skill/MCP) reduce the comprehension cost for maintainers.
- CLI and tests share the same graph construction logic, reducing behavioral divergence.

## Usage Guide

### 1. Basic Usage Paths

#### 1.1 Initialization and Viewing Configuration

On first run, msAgent creates the global directory and the current project state directory in `MSAGENT_HOME` (default `~/.msagent/`), without modifying the project working directory. You can view the current configuration with the following command:

```bash
msagent config --show
```

To view it in a specified working directory:

```bash
msagent config --show -w /path/to/project
```

#### 1.2 Entering an Interactive Session

```bash
msagent
```

Common startup parameters:

```bash
msagent -a Profiler -m default
msagent -a Minos "Check the README onboarding process of this repository"
msagent -a SpecTrainer "Resample this dataset with Qwen3-8B"
msagent --approval-mode active
```

#### 1.3 Common Interactive Commands

| Command | Purpose |
| -- | -- |
| /agents | Switch agents. |
| /model | Switch model aliases. |
| /threads | Browse and resume historical threads. |
| /tools | View the currently visible tools. |
| /skills | View or run the currently visible skills. |
| /mcp | Enable/disable MCP services. |
| /offload | Manually trigger context compression. |
| /tool-output | View the most recent expandable tool output. |
| /clear | Clear the screen and start a new thread. |
| /exit | Exit the session. |

### 2. Configuration Guide

#### 2.1 LLM Configuration

`config.llms.yml` in the default template defines the primary model entry point. Additional model aliases can be placed in `~/.msagent/config/llms/*.yml`. The current schema supports:

- `openai`
- `anthropic`
- `google` (also compatible with the `gemini` alias)

Example:

```yaml
llms:
  - version: 26.0.0
    provider: openai
    alias: default
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    max_tokens: 0
    temperature: 0.7
    streaming: true
```

#### 2.2 Agent Capability Boundary Configuration

The tool and skill boundaries of an agent are determined by patterns:

- Tool pattern: `category:module:name`
- Skill pattern: `category:name`
- Negative exclusion: prefix the pattern with `!`

Example:

```yaml
tools:
  patterns:
    - impl:deepagents:*
    - mcp:msprof-mcp:*
    - "!impl:deepagents:run_tool"

skills:
  patterns:
    - profiler:ascend-profiler-db-explorer
    - "!profiler:op-mfu-calculator"
```

#### 2.3 MCP Configuration

The MCP user override configuration is located in `~/.msagent/config/config.mcp.json`. The built-in service definitions come from the installation package resources. For example, a local stdio service can be written as follows:

```json
{
  "mcpServers": {
    "msprof-mcp": {
      "command": "msprof-mcp",
      "args": [],
      "transport": "stdio",
      "enabled": true,
      "stateful": true,
      "invoke_timeout": 3600.0
    }
  }
}
```

Before enabling MCP, ensure that the local command or remote service is actually available.

### 3. Extension Guide

#### 3.1 Adding a Skill

You are advised to place skills in one of the following locations first:

- `<working-dir>/skills/<category>/<skill-name>/SKILL.md`
- `~/.msagent/skills/<category>/<skill-name>/SKILL.md`

At the same time, you need to open the corresponding `skills.patterns` in the target agent configuration.

#### 3.2 Adding an Agent

You can add YAML files in the `~/.msagent/config/agents/` directory to define:

- `name`
- `prompt`
- `llm`
- `tools`
- `skills`
- `subagents`
- `compression`
- `retry`

After adding the file, you can use the agent through `/agents` or `-a <agent>`.

### 4. Usage Constraints and Limitations

- The core capability boundary of msAgent depends on the local configuration and working directory. Switching directories is equivalent to switching the project-level context.
- Whether MCP tools are available is determined not only by configuration, but also by the local command, network, and remote service status.
- Although long-running sessions support compression and checkpoints, when external tool output is exceptionally large, you still need to control the scope through tool patterns and problem decomposition.

## Test Design

### 1. Test Goals

The test design goal of msAgent is not simply to increase coverage, but to establish regression guarantees around the following risk surfaces:

- Whether configuration compatibility and template migration are stable.
- Whether runtime graph assembly meets expectations.
- Whether Tool/skill/MCP filtering boundaries are correct.
- Whether key chains such as CLI interaction, approval resume, and context compression work.
- Whether multi-Provider/multi-backend compatibility logic still runs after regression.

As of the current repository state, the local test directory contains 129 test files with about 293 test cases, forming a test system of "unit-first, integration-supplemented, E2E fallback".

### 2. Test Layering Strategy

| Layer | Goal | Representative Objects | Typical Test Files |
| -- | -- | -- | -- |
| Unit tests | Validate pure logic, configuration parsing, tool adaptation, and auxiliary function behavior. | Config model, LLM/MCP/Tool Factory, UI components, tool helper functions | `test_llm_factory.py`, `test_mcp_client.py`, `test_path_utils.py` |
| Integration tests | Validate multi-module collaboration, graph assembly, context injection, and CLI handler behavior. | `Initializer`, `AgentFactory`, Dispatcher/Handler | `test_agent_factory_runtime.py`, `test_initializer_runtime_context.py` |
| End-to-end tests | Validate real entry commands, stdout presentation, and user paths under the fake backend. | `msagent` executable entry point | `tests/e2e/test_msagent_entrypoint_e2e.py` |
| Documentation/packaging validation | Ensure resource packaging, version numbers, and documentation engineering remain sustainably buildable. | `hatch_build`, package imports and version compatibility | `test_hatch_build.py`, `test_package_imports.py`, `test_config_versions.py` |

### 3. Core Test Targets and Test Case Design

#### 3.1 Configuration System Tests

**Goal**: Ensure that global path resolution, loading of built-in values and user overrides, project state isolation, and directory-based extension behavior are stable.

The core test cases covered are as follows:

- On first run, directories are created only in `MSAGENT_HOME`, and no `.msagent/` is generated in the working directory.
- The project id is stable for the same project path, and project states with the same name but different paths are isolated from each other.
- When `config.llms.yml` and `llms/*.yml` are loaded together, aliases are not duplicated and Providers are normalized correctly.
- agent/subagent/checkpointer/sandbox configurations correctly resolve reference relationships.
- Historical field migration works correctly, for example:
  - Legacy `tools: [list]` automatically migrates to `ToolsConfig`
  - Missing `skills`, `retry`, and `compression.prompt` get default values filled in
- MCP configuration can be correctly parsed from JSON for the transport, include/exclude, and timeout fields.

Existing representative tests:

- `tests/ut/configs/test_config_registry.py`
- `tests/ut/configs/test_config_versions.py`
- `tests/ut/cli/bootstrap/test_legacy_defaults.py`
- `tests/ut/configs/test_provider_constraints.py`

#### 3.2 Runtime Assembly Tests

**Goal**: Ensure that the graph assembled by `Initializer` and `AgentFactory` and the visible capability catalog are correct.

Key test cases:

- `AgentFactory.create()` correctly injects:
  - `_llm_tools`
  - `_tools_in_catalog`
  - `_agent_backend`
- Tool pattern supports:
  - Positive matching
  - Negative exclusion
  - Matching across different impl/mcp categories
- `Initializer.create_graph()` passes `AgentContext` into the graph as the `context_schema`.
- Default MCP service name caching can be correctly derived from tool name prefixes.
- `retry.model`/`retry.tool` map to underlying runtime parameters.
- `ToolResultEvictionMiddleware` is mounted when enabled in configuration.

Existing representative tests:

- `tests/ut/agents/test_agent_factory_runtime.py`
- `tests/ut/cli/bootstrap/test_initializer_runtime_context.py`
- `tests/ut/cli/bootstrap/test_checkpointer_factory.py`

#### 3.3 Tool/Skill/MCP Collaboration Tests

**Goal**: Ensure that capability boundaries are correct, catalog capabilities can be discovered, and MCP tool timeouts and filtering work.

Key test cases:

- `fetch_tools/get_tool/run_tool` work in both the runtime directory and the fallback directory.
- `fetch_skills/get_skill` handle multi-directory scanning, duplicate name conflicts, and specified categories.
- MCP tools are filtered according to `include/exclude`.
- When a service-level `invoke_timeout` is missing, it falls back to the agent-level default timeout.

Existing representative tests:

- `tests/ut/mcp/test_mcp_client.py`
- `tests/ut/mcp/test_mcp_factory_and_memory.py`
- `tests/ut/tools/catalog/test_catalog_interfaces.py`
- `tests/ut/cli/handlers/test_skills_handler.py`
- `tests/ut/agents/test_skill_script_guidance.py`

#### 3.4 CLI/TUI Interaction Tests

**Goal**: Ensure a stable interactive user experience, including slash commands, completion, rendering, threads, and interrupt behavior.

Key test cases:

- Commands such as `/agents`, `/model`, `/threads`, `/skills`, `/mcp`, and `/tool-output` are dispatched correctly.
- Prompt completion, reference completion, the welcome page, the bottom bar, the Todo panel, and the tool output viewer render as expected.
- SIGINT correctly interrupts the streaming task on the first press and exits on the second.
- Bash mode and approval mode switching synchronously update the context.

Existing representative tests:

- `tests/ut/cli/dispatchers/test_commands.py`
- `tests/ut/cli/completers/test_completer_router.py`
- `tests/ut/cli/completers/test_reference_completer.py`
- `tests/ut/cli/ui/test_renderer_welcome.py`
- `tests/ut/cli/ui/test_bottom_toolbar.py`
- `tests/ut/cli/handlers/test_interrupt_handler.py`
- `tests/ut/cli/handlers/test_threads_handler.py`

#### 3.5 Long-Session and Compression Tests

**Goal**: Ensure that token statistics, automatic compression threshold determination, history offloading, and the recovery chain are stable.

Key test cases:

- `input_tokens/output_tokens` can be extracted from various response structures.
- Automatic compression marking is triggered when the threshold is reached.
- `/offload` correctly performs summarization, externalization, and state updates.
- Externalized large tool results can still be viewed through the tool output viewer.

Existing representative tests:

- `tests/ut/cli/handlers/test_compression_handler.py`
- `tests/ut/middlewares/test_tool_result_eviction_middleware.py`
- `tests/ut/middlewares/test_token_cost_extraction.py`
- `tests/ut/configs/test_timeout_controls.py`

#### 3.6 End-to-End Tests

**Goal**: Validate from the real entry point whether users can complete the key shortest paths.

Key test cases:

- `msagent --version`
- `msagent config --show`
- One-shot messages can complete tool calls and Todo rendering under the fake backend.

Existing representative tests:

- `tests/e2e/test_msagent_entrypoint_e2e.py`

### 4. Test Data and Test Doubles Design

Many of the capabilities of msAgent depend on the external environment. Therefore, tests should make heavy use of doubles and isolation strategies:

- `FakeGraph`: replaces the real deepagents graph runtime, preventing tests from being blocked by model calls.
- `monkeypatch`/`AsyncMock`: replace time-consuming dependencies such as MCP services, skill catalog loading, and agent construction.
- Temporary directory `tmp_path`: isolates `.msagent/` local configuration and test data.
- Environment variable switches: for example, `MSAGENT_FAKE_BACKEND=1` controls the fake backend behavior.

This design separates the verification of "whether platform behavior is correct" from "whether external dependencies themselves are available".

### 5. Test Execution Levels

Test execution can be organized in layers by the scope of change impact:

| Scope of Change | Key Execution Content |
| -- | -- |
| Configuration models and templates | Tests related to `ConfigRegistry`, version compatibility, and template migration |
| Runtime assembly | Tests related to `AgentFactory`, `Initializer`, and `MCPClient` |
| CLI/TUI interaction | Tests related to commands, renderer, completer, and interrupt |
| Release entry fallback | E2E entry tests |

## Appendix

### 1. References

- DeepWiki: [msAgent](https://deepwiki.com/kali20gakki/msAgent)
- deepagents official documentation: [Overview](https://docs.langchain.com/oss/python/deepagents/overview)
- MCP official documentation: [Introduction](https://modelcontextprotocol.io/docs/getting-started/intro)

### 2. Terminology Notes

| Term | Description |
| -- | -- |
| Agent | A main agent configuration unit that carries the work goal of a specific domain. |
| Subagent | A helper agent invoked by the main agent, responsible for exploration or general analysis tasks. |
| Tool | An execution capability that the graph runtime can call directly. |
| Skill | A workflow knowledge asset described through `SKILL.md`. |
| MCP | Model Context Protocol, used to uniformly bring external tools/resources into the agent capability surface. |
| Checkpointer | A backend responsible for persisting graph state and thread sessions. |
| AgentContext | A context object that renders prompts, constructs runtime environment snapshots, and builds the visible capability catalog. |
