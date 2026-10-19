# Security Statement

If you discover a security issue related to MindStudio-Agent, report it first through the project Issue tracker or a private communication channel recognized by the maintainers, and provide reproducible information, the impact scope, and a suggested fix.

Before the security issue is confirmed and fixed, avoid publicly disclosing exploitable details.

`msagent` stores user configuration, approval rules, session state, and some runtime artifacts in local files under `MSAGENT_HOME` (default `~/.msagent/`). These files may contain user prompts, tool call parameters, audit records, trace content, and other runtime context.

You are responsible for access control and backup protection of the local storage directory. Protect `MSAGENT_HOME` and the host where it resides according to your actual security requirements. Do not write unnecessary sensitive information, such as API keys, tokens, passwords, or Bearer credentials, in prompts, long-term memory, configuration files, or tool parameters.

The current version of `msagent` does not generate `.msagent/` in your working directory, nor does it automatically modify `.git/info/exclude` or `.gitignore` to ignore these files. If you need to configure Git ignore rules, set them up yourself as needed.

This repository provides reference configurations for VS Code and devcontainer to facilitate development and debugging. Use them only to develop and debug this repository in a secure debugging environment. In addition, check the security of the configurations yourself and adjust them according to your actual needs.
