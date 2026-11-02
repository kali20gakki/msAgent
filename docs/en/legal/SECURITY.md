# Security Statement

If you discover a security issue related to MindStudio-Agent, report it first through the project Issue tracker or a private communication channel recognized by the maintainers, and provide reproducible information, the impact scope, and a suggested fix.

Before the security issue is confirmed and fixed, avoid publicly disclosing exploitable details.

`msagent` stores user configuration, approval rules, session state, and some runtime artifacts in local files under `MSAGENT_HOME` (default `~/.msagent/`). These files may contain user prompts, tool call parameters, audit records, trace content, and other runtime context.

You are responsible for access control and backup protection of the local storage directory. Protect `MSAGENT_HOME` and the host where it resides according to your actual security requirements. Do not write unnecessary sensitive information, such as API keys, tokens, passwords, or Bearer credentials, in prompts, long-term memory, configuration files, or tool parameters.

## Command Approval and Authorization Risks

`msagent` supports the shell command approval modes Safe Mode and Convenience Mode, which you can view or switch with `/permissions`. In Safe Mode, commands on the allowlist are approved by default, while commands on the denylist and regular commands require confirmation one by one. In Convenience Mode, regular commands are approved by default, while commands on the denylist require confirmation one by one.

`approve`/`reject` take effect only once. The `always` rules for the denylist are saved only within the session, while project-level `always` rules for regular commands in Safe Mode are persisted to `~/.msagent/state/projects/<project-id>/config.approval.json`. You are advised to authorize commands with caution, including those that delete files, modify Git history, push code, elevate privileges, install dependencies, run remote scripts, access the network, or handle sensitive data. You bear the execution risks of any command you authorize.

The command allowlist and denylist are used only for heuristic classification. They are not a security boundary. Commands can indirectly perform operations through interpreters, scripts, aliases, environment variables, command substitution, or other wrappers. Therefore, even when a command is not classified as denylisted, you must still review each command to be executed. Manual confirmation is the final line of defense.

The current version of `msagent` does not generate `.msagent/` in your working directory, nor does it automatically modify `.git/info/exclude` or `.gitignore` to ignore these files. If you need to configure Git ignore rules, set them up yourself as needed.

This repository provides reference configurations for VS Code and devcontainer to facilitate development and debugging. Use them only to develop and debug this repository in a secure debugging environment. In addition, check the security of the configurations yourself and adjust them according to your actual needs.

## MCP Services and Third-Party Tool Risks

An MCP server is not a trusted sandbox. Local `stdio` MCP usually runs with the current user's permissions, while remote MCP may send the prompts, file content, tool parameters, or authentication information required by the call to third-party services.

Before you enable a service, review its source, commands, URL, request headers, and tool scope. Grant only the necessary permissions, and do not write unnecessary sensitive information. Timeout settings cannot replace isolation and permission control. You are responsible for the execution, data leakage, and compliance risks arising from configuring or calling MCP servers yourself.
