# `document-ux-review` Usage

`document-ux-review` is a built-in skill for reviewing the onboarding experience of documentation. It verifies whether the `README` of a repository, installation documentation, or quick start can actually walk new users through the entire process.

## 1. What the Agent Actually Does

When you hand the agent a repository URL or a local repository path, it does not merely read the documentation statically. Instead, it executes the steps in the actual `README` one by one.

- Starts with the `README` and identifies the installation, quick start, and runtime documentation directly associated with it
- Reuses the prepared base environment first based on the environment information you provide, rather than reinstalling it
- Executes the real steps in the documentation and checks whether the commands, dependencies, paths, configurations, and startup process work
- Records these issues explicitly if the documentation contains missing steps, incorrect commands, implicit prerequisites, or platform differences
- Also records such cases as documentation completeness issues when continuing requires reading additional scripts, source code, CI files, or Dockerfiles
- Finally, outputs a structured report describing which steps succeeded, which were blocked, and the corresponding issue locations and improvement suggestions

## 2. Prerequisites

- When you install `mindstudio-agent` with `pip install`, the version must be `>= 0.1.3`
- If you run from source code, you can generally use the skill directly. The built-in skills are already placed in the `skills/` directory at the repository root. Therefore, you do not need to sync the submodule.

## 3. msAgent Installation and Configuration

`document-ux-review` is a built-in skill. In most cases, as long as msAgent is installed correctly and the LLM is configured, you can use it directly, and you do not need to maintain separate installation steps here.

For the specific installation, LLM configuration, and startup methods, refer directly to the [Quick Start Guide](../quick_start/msagent_quick_start.md).

## 4. Verification Before Use

To confirm that `document-ux-review` is enabled with the default configuration, check it directly in the TUI:

- Method 1: Start `msagent` to enter the TUI, and check the `Skills` section on the welcome page to confirm that it contains `document-ux-review`.
- Method 2: Enter `/skills` in the TUI, and confirm that `document-ux-review` appears in the list of currently available skills.

## 5. Prompt Examples

**Example 1: Hands-On Example with `msmonitor`**

```text
Please experience and review the onboarding experience of this repository's documentation: https://atomgit.com/Ascend/msmonitor.

Local environment:
- Ubuntu 20.04
- CANN is installed, environment script: `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- conda virtual environment is ready, please use it first: `msmonitor_ux_review`

Please output a detailed Chinese HTML report to `/home/msmonitor`, focusing on whether new users can complete the installation and reach a runnable state by following the documentation in the preceding environment.
```

## 🖼️ Sample Results

You can refer to the sample output: [msmonitor-review-run.html](../example/msmonitor-review-run.html).

Because GitHub repository pages do not render this type of HTML report directly, you are advised to download the file to your local machine and open it directly in a browser to view the result.
