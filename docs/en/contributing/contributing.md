# Contribution Guide

Thank you for your interest in MindStudio-Agent (msAgent). Whether you want to fix a bug, improve the documentation, add a new skill, or extend the capabilities of agents and the framework, you are welcome to contribute through Issues or Pull Requests.

## What You Can Contribute

MindStudio-Agent is an agent workbench for Ascend NPU scenarios. Its core consists of the CLI, configuration system, MCP tools, built-in skills, and domain agents. Common contribution types include:

| Contribution Type | Description | Recommended Reading |
|---|---|---|
| **Issue feedback** | Bugs, feature requests, documentation errors, or experience issues | [FAQ](../support/faq.md), [GitCode Issues](https://atomgit.com/Ascend/msagent/issues) |
| **Documentation improvements** | Installation instructions, usage guides, agent descriptions, architecture documentation | [Installation Guide](../install_guide/msagent_install_guide.md), [ReadTheDocs local verification](../development_guide/readthedocs-local-build.md) |
| **Skills extension** | Adding or improving domain diagnostic SOPs, scripts, and trigger descriptions | [Configuration and Extension](../user_guide/configuration-and-extension.md), [skills/README.md](../../../skills/README.md) |
| **Agents/configuration** | Adjusting agent YAML, prompts, and Tool/skill filter rules | [Agent/Tool/Skill Filter Rules](../user_guide/agent-tool-skill-filter-rules.md) |
| **Framework code** | Core capabilities such as the CLI, middleware, MCP integration, and configuration loading | [Architecture Overview](../development_guide/arch_overview.md) |
| **Testing and building** | Unit tests, integration tests, and wheel build verification | [Building and Packaging](../development_guide/build-and-package.md) |

The current built-in agents and their domain positioning are as follows. Before contributing, you can first understand the boundaries of each:

- [Profiler](../agent_guide/profiler.md): performance tuning
- [Accuracy](../agent_guide/accuracy.md): accuracy tuning
- [Quantizer](../agent_guide/quantizer.md): model quantization
- [Modeling](../agent_guide/modeling.md): simulation modeling
- [Operator](../agent_guide/operator.md): operator tuning
- [Minos](../agent_guide/minos.md): documentation experience and code review
- [SpecTrainer](../agent_guide/spectrainer.md): speculative decoding data resampling

## Before You Start

### Method 1: One-Click devcontainer Development Environment (Recommended)

msAgent includes a built-in [devcontainer](https://containers.dev/) development environment configuration. After opening the repository in VS Code, you can enter a standardized container with one click, without manually installing any dependencies.

**Prerequisites:**

| Environment | Requirement |
|---|---|
| PC | VS Code with the Dev Containers and Remote-SSH extensions installed |
| Linux server | Docker service running |

**Procedure:**

1. Clone the msAgent repository to the Linux server.
2. Connect to the server through Remote-SSH in VS Code and open the repository directory.
3. VS Code automatically detects the `.devcontainer` configuration. Click **"Reopen in Container"** in the bottom-left corner.
4. The container automatically runs `post-create.sh` to complete the initialization (Python 3.11+, pip dependencies, pre-commit, and project dev dependencies).
5. Press `Ctrl+Shift+P` → `Tasks: Run Task` and select a build or test task.

**VS Code built-in tasks:**

| Task | Shortcut | Description |
|---|---|---|
| Build: Install Dependencies | `Ctrl+Shift+B` | Installs the project development dependencies |
| Test: Run Unit Tests | — | Runs all unit tests |
| Clean: All Workspace | — | Cleans cache files |

**Code navigation and debugging:** For Python, Pylance provides semantic navigation (F12) and type inference. Press `F5` and select `Python: Debug Active File` to start debugpy debugging.

### Method 2: Manual Environment Configuration

Before contributing code or performing local verification, confirm that your environment meets the following requirements:

- Python `3.11+`
- You are advised to use [uv](https://docs.astral.sh/uv/) to manage dependencies.
- Prepare at least one usable LLM API key (for interactive verification).

For details, see [Installation Guide](../install_guide/msagent_install_guide.md) and [Version and Compatibility](../development_guide/version-and-compatibility.md).

### Cloning and Running from Source

```bash
git clone https://atomgit.com/Ascend/msagent.git
cd msagent
uv sync --dev
uv run msagent --version
```

When running from source, you can replace `msagent` with `uv run msagent` in commands. For the first startup and model configuration, see [Quick Start Guide](../quick_start/msagent_quick_start.md).

### Recommended Reading Order

1. [Architecture Overview](../development_guide/arch_overview.md): Understand the layered relationships among the CLI, agent factory, tools, skills, MCP, and middleware.
2. [Configuration and Extension](../user_guide/configuration-and-extension.md): Understand global configuration, project state, and the loading order of MCP and skills.
3. [Agent/Tool/Skill Filter Rules](../user_guide/agent-tool-skill-filter-rules.md): Read this before modifying agent capability boundaries.
4. The agent guides or user guides relevant to your changes.

## Development Workflow

### 1. Creating an Issue or Claiming a Task

- Submit bugs or feature suggestions through [GitCode Issues](https://atomgit.com/Ascend/msagent/issues).
- For larger changes, you are advised to open an Issue first to discuss the approach and avoid conflicting with the maintainers' direction.
- For security issues, follow the [Security Statement](../legal/SECURITY.md) and avoid publicly disclosing exploitable details.

### 2. Creating a Branch and Developing

Create a feature branch from the latest main branch, keeping the commit granularity clear and descriptions accurate.

If you need to understand the runtime behavior during development, refer to the following:

- Session commands and shortcuts: [Usage Guide](../user_guide/usemap.md)
- Context compaction mechanism: [Context Compaction Usage Guide](../user_guide/context-compaction-guide.md)
- Retry and timeout configuration: [Retry Middleware Usage Guide](../user_guide/retry-middleware-guide.md)

### 3. Performing a Local Self-Check

Before submitting a PR, you are advised to complete at least the following checks:

```bash
# Install development dependencies
uv sync --dev

# Run tests
python3 build.py test

# Verify that the lockfile is in sync (required after modifying pyproject.toml)
uv lock --check

# Build the wheel (run when packaging or Skills changes are involved)
python3 build.py
```

To verify the wheel installation, use the following command:

```bash
python3 build.py --extra VERIFY_WHEEL_INSTALL=1
```

For build details, see [Building and Packaging](../development_guide/build-and-package.md).

### 4. Code Quality Checks

The project uses pre-commit to run checks before committing, including Ruff, pylint, Bandit, typos, and so on. Install the hook before the first use:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The configuration files are in `.pre-commit-config.yaml` and the `pre-commit/` directory. For Python code style and static check rules, see `pre-commit/pyproject.toml`.

### 5. Submitting a Pull Request

In the PR, you are advised to describe the following:

- The motivation for the change and its scope of impact
- The related Issue, if any
- How the change was verified locally (test commands and manual verification steps)
- Whether the documentation has been updated accordingly

For documentation PRs, you can build a Sphinx preview locally. For details, see [ReadTheDocs local build instructions](../development_guide/readthedocs-local-build.md).

## Guidelines by Contribution Type

### Documentation Contributions

All documentation is maintained in the `docs/zh/` directory, organized by Quick Start, Agent Guide, User Guide, and Development Guide. When modifying the README, installation steps, or the Quick Start, you are advised to:

1. Walk through the minimal workflow by following the [Quick Start Guide](../quick_start/msagent_quick_start.md).
2. Use the built-in skill [document-ux-review](../user_guide/document-ux-review.md) to review the documentation onboarding experience.
3. Build the documentation locally to confirm that there are no errors: [ReadTheDocs local build instructions](../development_guide/readthedocs-local-build.md).

For common questions, first check the [FAQ](../support/faq.md).

### Skills Contributions

Skills are one of the most important extension surfaces of the project. The `skills/` directory at the repository root is the source of built-in skills and is packaged into `resources/configs/default/skills/` when the wheel is built.

The recommended steps for adding a new skill are as follows:

1. Create a kebab-case directory in `skills/` and provide a `SKILL.md` (with `name` and `description` in the frontmatter).
2. Put stable, repeatable steps into `scripts/` (if needed).
3. Enable the skill in the `skills.patterns` of the corresponding agent.
4. After starting `msagent`, verify its visibility through `/skills`.
5. Update the skill list in [skills/README.md](../../../skills/README.md).

For detailed conventions, see [Configuration and Extension · Adding Custom Skills](../user_guide/configuration-and-extension.md#9-adding-a-custom-skill) and [skills/README.md · Recommended Process for Adding New Skills](../../../skills/README.md).

For skill matching and filter rules, see [Agent/Tool/Skill Filter Rules](../user_guide/agent-tool-skill-filter-rules.md).

### Agent and Configuration Contributions

Built-in agent definitions are located in `resources/configs/default/agents/` and are read directly at runtime; user overrides are located in `~/.msagent/config/agents/`. When modifying the visible scope of tools or skills:

- `tools.patterns` uses the three-part format `<category>:<module>:<name_pattern>`.
- `skills.patterns` uses the two-part format `<category>:<name_pattern>`.
- Negative rules with the `!` prefix and `fnmatch` wildcards are supported.

For the complete semantics and smoke examples, see [Agent/Tool/Skill Filter Rules](../user_guide/agent-tool-skill-filter-rules.md) and `resources/configs/default/agents/msagent-filter-smoke.yml`.

For MCP integration and field descriptions, see [Configuration and Extension · MCP Configuration](../user_guide/configuration-and-extension.md#5-mcp-configuration).

### Framework Code Contributions

The framework is based on the deepagents runtime and is modularly divided into the interaction layer, scheduling layer, core layer, and infrastructure layer. Before development, you are advised to read the following sections of [Architecture Overview](../development_guide/arch_overview.md):

- System architecture and the responsibilities of core modules
- The startup process and the agent execution process
- Test design (the test directory is `tests/`)

When modifying dependencies or Python version requirements, update [Version and Compatibility](../development_guide/version-and-compatibility.md) accordingly and describe the compatibility impact in the PR.

### Test Contributions

- Test directory: `tests/`
- How to run: `python3 build.py test`
- Integration test marker: `@pytest.mark.integration`

For changes involving agent behavior, configuration loading, and Tool/skill filtering, you are advised to add or update the corresponding unit tests.

## License and Community

- License: [Mulan PSL v2](http://license.coscl.org.cn/MulanPSL2)
- Issue feedback: [GitCode Issues](https://atomgit.com/Ascend/msagent/issues)
- Online documentation: [ReadTheDocs](https://mindstudio-docs-master.readthedocs.io)
- Ascend community: [MindStudio software portal](https://www.hiascend.com/eng/developer/software/mindstudio)

Thank you again for your contribution. Whether you fix a line of documentation, add a skill, or improve the core framework, your contribution helps more Ascend developers debug and tune faster.
