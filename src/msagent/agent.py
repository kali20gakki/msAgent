"""Core agent logic for msagent."""

import asyncio
import json
import os
import re
import sys
import time
from collections.abc import AsyncGenerator
from pathlib import Path, PurePosixPath
from typing import Any

from .config import AppConfig, config_manager
from .interfaces import AgentBackend, AgentEvent, AgentStatus, UsageSnapshot
from .llm import Message, create_llm_client
from .mcp_client import mcp_manager
from .tools import create_tool_invoker


class Agent(AgentBackend):
    """msagent - Core agent implementation."""

    _LOCAL_SHELL_MODE = "local_shell"
    _FILESYSTEM_MODE = "filesystem"
    _SUPPORTED_BACKEND_MODES = (_FILESYSTEM_MODE, _LOCAL_SHELL_MODE)
    _BACKEND_ENV = "MSAGENT_DEEPAGENTS_BACKEND"
    _ENABLE_LOCAL_SHELL_ENV = "MSAGENT_ENABLE_LOCAL_SHELL"
    _LOCAL_SHELL_WARNING = (
        "⚠️ LocalShellBackend 已启用：msAgent 现在可以直接在当前机器上执行 shell 命令。\n"
        "   - 该能力没有沙箱隔离，命令以当前用户权限在宿主机执行\n"
        "   - 仅建议在受信任的本地开发环境使用\n"
        "   - 不要在生产环境、多租户服务或不受信任输入场景启用\n"
        "   - 如需关闭，请移除环境变量 "
        "MSAGENT_DEEPAGENTS_BACKEND=local_shell / MSAGENT_ENABLE_LOCAL_SHELL=1\n"
    )
    _LOCAL_SHELL_PROMPT_WARNING = """
[LocalShellBackend 安全约束]
- 当前已启用 deepagents LocalShellBackend。`execute` 工具会直接在用户机器上执行 shell 命令，没有沙箱隔离。
- 默认优先使用内置文件工具和已注册工具；只有在确实需要时才使用 `execute`。
- 禁止主动读取敏感信息，如 `.env`、SSH key、云凭证、系统账户文件、用户主目录凭据等，除非用户明确要求。
- 禁止执行破坏性或高风险命令，如 `rm -rf`、`sudo`、后台守护进程、批量删除、系统配置修改、包安装、网络下载/上传，除非用户明确要求并确认风险。
- 涉及文件修改、依赖安装、进程管理、网络访问或跨工作区访问时，必须先向用户说明风险并获得明确确认。
- 命令应尽量限制在当前工作区内，默认保持只读、最小化、可回滚。
""".strip()

    def __init__(self, config: AppConfig | None = None):
        self.config = config or config_manager.get_config()
        self.llm_client = None
        self.messages: list[Message] = []
        self._session_number = 1
        self._initialized = False
        self._error_message = ""
        self._workspace_root = Path.cwd().resolve()
        self._deepagents_backend_mode = self._resolve_deepagents_backend_mode()
        self._tool_invoker = create_tool_invoker(self._workspace_root, mcp_manager)
        self._file_index_cache: tuple[float, list[str]] | None = None
        self._loaded_skill_sources: list[str] = []
        self._loaded_skills: list[str] = []
        self._system_prompt_template: str | None = None
        self._session_usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._context_tokens: int | None = None

    _AT_REF_PATTERN = re.compile(r"(?<!\S)@([^\s]+)")
    _MAX_ATTACHED_FILES = 5
    _MAX_FILE_CHARS = 4000
    _FILE_INDEX_TTL_S = 30.0
    _QUICK_SCAN_MAX_DEPTH = 2
    _SYSTEM_PROMPT_FILE = "prompts/system_prompt.txt"
    _PACKAGE_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
    _SKIP_DIRS = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
    }

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def error_message(self) -> str:
        return self._error_message

    @property
    def session_number(self) -> int:
        return self._session_number

    async def initialize(self) -> bool:
        """Initialize the agent."""
        try:
            if not self.config.llm.is_configured():
                self._error_message = (
                    "⚠️ LLM not configured. Please set up your API key:\n"
                    "   • Environment: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY\n"
                    "   • Config file stores only api_key_env (not the secret itself)\n"
                    "   • Use: msagent config --help"
                )
                return False

            skill_sources = self._resolve_skill_sources()
            skill_directories = self._resolve_skill_directories(skill_sources)
            self.llm_client = self._create_llm_client(skill_sources)
            self._loaded_skill_sources = skill_sources
            self._loaded_skills = self._discover_skills(skill_directories)

            if self._uses_local_shell_backend():
                print(self._LOCAL_SHELL_WARNING, file=sys.stderr, flush=True)

            for mcp_config in self.config.mcp_servers:
                if mcp_config.enabled:
                    await mcp_manager.add_server(mcp_config)

            self._refresh_context_tokens()
            self._initialized = True
            return True

        except Exception as e:
            self._error_message = f"❌ Failed to initialize agent: {e}"
            return False

    def _resolve_skill_sources(self) -> list[str]:
        """Resolve deepagents skill sources as POSIX paths relative to backend root."""
        sources: list[str] = []
        project_skills_dir = (self._workspace_root / "skills").resolve()
        if project_skills_dir.is_dir():
            sources.append("/skills")

        for source in self.config.deepagents.skills:
            normalized = self._normalize_skill_source(source)
            if normalized is None or normalized in sources:
                continue
            sources.append(normalized)
        return sources

    def _normalize_skill_source(self, source: str) -> str | None:
        text = (source or "").strip()
        if not text:
            return None

        if text.startswith("/") and not (len(text) > 1 and text[1] == ":"):
            normalized = PurePosixPath(text).as_posix()
            return normalized if normalized != "." else "/"

        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = (self._workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()

        try:
            relative = candidate.relative_to(self._workspace_root)
        except Exception:
            return None

        if str(relative) in {"", "."}:
            return "/"
        return f"/{relative.as_posix()}"

    def _resolve_skill_directories(self, sources: list[str]) -> list[Path]:
        """Resolve on-disk skill directories for discoverable backend-relative sources."""
        directories: list[Path] = []
        for source in sources:
            parts = [part for part in PurePosixPath(source).parts if part not in {"/", ""}]
            directory = self._workspace_root.joinpath(*parts).resolve()
            if directory.is_dir() and directory not in directories:
                directories.append(directory)
        return directories

    def _discover_skills(self, directories: list[Path]) -> list[str]:
        """Discover skill names from source directories."""
        discovered: list[str] = []
        seen: set[str] = set()

        for source_path in directories:
            try:
                children = sorted(source_path.iterdir(), key=lambda p: p.name)
            except Exception:
                continue

            for child in children:
                if not child.is_dir():
                    continue
                if not (child / "SKILL.md").is_file():
                    continue
                skill_name = child.name
                if skill_name in seen:
                    continue
                seen.add(skill_name)
                discovered.append(skill_name)

        return discovered

    def get_loaded_skills(self) -> list[str]:
        """Get loaded skill names."""
        return self._loaded_skills.copy()

    def get_status(self) -> AgentStatus:
        """Return a frontend-safe status snapshot."""
        llm_cfg = self.config.llm
        return AgentStatus(
            is_initialized=self._initialized,
            error_message=self._error_message,
            session_number=self._session_number,
            provider=(llm_cfg.provider or "unknown").strip(),
            model=(llm_cfg.model or "unknown").strip(),
            backend_mode=self._deepagents_backend_mode,
            connected_servers=tuple(mcp_manager.get_connected_servers()),
            loaded_skills=tuple(self._loaded_skills),
            usage=self._build_usage_snapshot(),
            cumulative_usage=self._build_cumulative_usage_snapshot(),
            context_tokens=self._context_tokens,
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        mcp_servers = mcp_manager.get_connected_servers()
        server_text = ", ".join(mcp_servers) if mcp_servers else "None"
        template = self._load_system_prompt_template()
        prompt = template.replace("__MCP_SERVERS__", server_text)
        if self._uses_local_shell_backend():
            prompt = f"{prompt}\n\n{self._LOCAL_SHELL_PROMPT_WARNING}"
        return prompt

    def _load_system_prompt_template(self) -> str:
        if self._system_prompt_template is not None:
            return self._system_prompt_template

        template_path = Path(__file__).resolve().parent / self._SYSTEM_PROMPT_FILE
        self._system_prompt_template = template_path.read_text(encoding="utf-8").strip()
        return self._system_prompt_template

    def _create_llm_client(self, skill_sources: list[str] | None = None):
        resolved_skill_sources = skill_sources
        if resolved_skill_sources is None:
            resolved_skill_sources = self._resolve_skill_sources()
        return create_llm_client(
            self.config.llm,
            skills=resolved_skill_sources,
            memory=self.config.deepagents.memory,
            recursion_limit=self.config.deepagents.recursion_limit,
            workspace_root=self._workspace_root,
            tool_invoker=self._tool_invoker.call_tool,
            backend_mode=self._deepagents_backend_mode,
        )

    def _resolve_deepagents_backend_mode(self) -> str:
        configured = (os.getenv(self._BACKEND_ENV, "") or "").strip().lower()
        if configured in self._SUPPORTED_BACKEND_MODES:
            return configured
        if self._env_flag(self._ENABLE_LOCAL_SHELL_ENV):
            return self._LOCAL_SHELL_MODE
        return self._FILESYSTEM_MODE

    def _uses_local_shell_backend(self) -> bool:
        return self._deepagents_backend_mode == self._LOCAL_SHELL_MODE

    def _env_flag(self, name: str) -> bool:
        value = os.getenv(name, "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def switch_deepagents_backend(self, mode: str) -> str:
        normalized = self._normalize_backend_mode(mode)
        if normalized is None:
            supported = ", ".join(self._SUPPORTED_BACKEND_MODES)
            return (
                f"❌ 不支持的 deepagents backend: {mode or '<empty>'}。\n"
                f"可选值：{supported}\n"
                "可用命令：/backend filesystem | /backend local_shell | /shell on | /shell off"
            )

        if normalized == self._deepagents_backend_mode:
            return self._format_backend_status_message(
                prefix="当前已启用",
                include_command_hint=True,
            )

        previous_mode = self._deepagents_backend_mode
        previous_client = self.llm_client
        self._deepagents_backend_mode = normalized

        if self._initialized:
            try:
                self.llm_client = self._create_llm_client(self._loaded_skill_sources)
            except Exception as exc:
                self._deepagents_backend_mode = previous_mode
                self.llm_client = previous_client
                return f"❌ 切换 deepagents backend 失败：{exc}"

        self._refresh_context_tokens()
        return self._format_backend_status_message(
            prefix="已切换当前会话的",
            include_command_hint=True,
        )

    def _normalize_backend_mode(self, mode: str) -> str | None:
        normalized = (mode or "").strip().lower()
        if normalized in self._SUPPORTED_BACKEND_MODES:
            return normalized
        return None

    def _backend_display_name(self, mode: str | None = None) -> str:
        resolved = mode or self._deepagents_backend_mode
        if resolved == self._LOCAL_SHELL_MODE:
            return "LocalShellBackend"
        return "FilesystemBackend"

    def _format_backend_status_message(
        self,
        *,
        prefix: str,
        include_command_hint: bool = False,
    ) -> str:
        mode = self._deepagents_backend_mode
        lines = [
            f"{prefix} deepagents backend：{self._backend_display_name(mode)} ({mode})。"
        ]
        if mode == self._LOCAL_SHELL_MODE:
            lines.append("⚠️ `execute` 将直接在当前机器上执行 shell 命令，没有沙箱隔离。")
            lines.append("仅建议在受信任的本地开发环境中临时开启；请避免处理不受信任输入。")
        else:
            lines.append("当前为默认文件工具模式，不提供 LocalShellBackend 的宿主机 shell 执行能力。")
        if include_command_hint:
            lines.append("命令：/backend status | /backend filesystem | /backend local_shell | /shell on | /shell off")
        return "\n".join(lines)

    async def chat(self, user_input: str) -> str:
        """Process a chat message and return the response."""
        if not self._initialized or not self.llm_client:
            return "Error: Agent not initialized. Please check your configuration."

        self.messages.append(Message("user", self._inject_file_context(user_input)))
        all_messages = [Message("system", self.get_system_prompt())] + self.messages
        tools = self._tool_invoker.get_tools()

        timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "600"))
        try:
            response = await asyncio.wait_for(
                self.llm_client.chat(all_messages, tools=tools if tools else None),
                timeout=timeout_s,
            )

            self.messages.append(Message("assistant", response))
            self._record_latest_usage()
            return response
        except asyncio.TimeoutError:
            return f"❌ Error: LLM response timed out after {timeout_s:.0f}s"
        except Exception as e:
            return f"❌ Error: {e}"
        finally:
            self._refresh_context_tokens()

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Process a chat message and stream the response."""
        async for event in self.stream_chat_events(user_input):
            if event.type == "text" and event.content:
                yield event.content
            elif event.type == "error" and event.content:
                yield event.content

    async def chat_stream_events(
        self, user_input: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Backward-compatible dict events for existing callers."""
        async for event in self.stream_chat_events(user_input):
            yield event.to_dict()

    async def stream_chat_events(
        self, user_input: str
    ) -> AsyncGenerator[AgentEvent, None]:
        """Process a chat message and stream response + tool call events."""
        if not self._initialized or not self.llm_client:
            yield AgentEvent(
                type="error",
                content="Error: Agent not initialized. Please check your configuration.",
            )
            return

        self.messages.append(Message("user", self._inject_file_context(user_input)))
        all_messages = [Message("system", self.get_system_prompt())] + self.messages
        tools = self._tool_invoker.get_tools()

        timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "3600"))
        start = time.monotonic()
        full_response = ""
        stream = None
        saw_tool_event = False

        try:
            stream = self.llm_client.chat_stream_events(
                all_messages, tools=tools if tools else None
            )

            while True:
                elapsed = time.monotonic() - start
                remaining = max(timeout_s - elapsed, 0.001)
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    try:
                        await stream.aclose()
                    except Exception:
                        pass
                    yield AgentEvent(
                        type="error",
                        content=f"❌ Error: LLM stream timed out after {timeout_s:.0f}s",
                    )
                    return

                if not isinstance(chunk, dict):
                    continue

                event_type = chunk.get("type")
                if event_type == "text":
                    text = chunk.get("content")
                    if not isinstance(text, str):
                        continue
                    cleaned = self._filter_info_logs(text)
                    if cleaned:
                        full_response += cleaned
                        yield AgentEvent(type="text", content=cleaned)
                    continue

                if event_type == "tool_start":
                    tool_name = chunk.get("name")
                    if isinstance(tool_name, str) and tool_name:
                        saw_tool_event = True
                        tool_input = chunk.get("input")
                        split = self._split_tool_name(tool_name)
                        yield AgentEvent(
                            type="tool_call",
                            full_name=tool_name,
                            server=split["server"],
                            tool=split["tool"],
                            payload=tool_input,
                        )
                    continue

                if event_type == "tool_end":
                    tool_name = chunk.get("name")
                    if isinstance(tool_name, str) and tool_name:
                        saw_tool_event = True
                        split = self._split_tool_name(tool_name)
                        yield AgentEvent(
                            type="tool_result",
                            full_name=tool_name,
                            server=split["server"],
                            tool=split["tool"],
                            payload=chunk.get("output"),
                        )
                    continue

            if not full_response:
                if saw_tool_event:
                    dt = time.monotonic() - start
                    self._record_latest_usage()
                    yield AgentEvent(type="done", duration_s=dt)
                    return

                fallback = await asyncio.wait_for(
                    self.llm_client.chat(all_messages, tools=tools if tools else None),
                    timeout=timeout_s,
                )
                if fallback:
                    cleaned_fallback = self._filter_info_logs(fallback)
                    if cleaned_fallback:
                        full_response = cleaned_fallback
                        yield AgentEvent(type="text", content=cleaned_fallback)
                else:
                    last_msgs = [m.to_dict() for m in all_messages[-3:]]
                    yield AgentEvent(
                        type="error",
                        content=(
                            "❌ Error: LLM returned empty response.\n"
                            f"Last messages: {json.dumps(last_msgs, ensure_ascii=False)}"
                        ),
                    )
                    return

            dt = time.monotonic() - start
            if full_response:
                self.messages.append(Message("assistant", full_response))
            self._record_latest_usage()
            yield AgentEvent(type="done", duration_s=dt)
        except asyncio.CancelledError:
            try:
                if stream is not None:
                    await stream.aclose()
            except Exception:
                pass
            return
        except Exception as e:
            yield AgentEvent(type="error", content=f"❌ Error: {e}")
        finally:
            self._refresh_context_tokens()

    def _filter_info_logs(self, text: str) -> str:
        """Remove verbose INFO log lines from streamed text."""
        if not text:
            return text

        filtered_lines: list[str] = []
        for line in text.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("INFO"):
                continue
            if re.match(r"^\d{4}-\d{2}-\d{2}.*\bINFO\b", stripped):
                continue
            if re.match(r"^\[?INFO\]?", stripped):
                continue
            filtered_lines.append(line)
        return "".join(filtered_lines)

    def _split_tool_name(self, full_tool_name: str) -> dict[str, str]:
        if "__" in full_tool_name:
            server, tool = full_tool_name.split("__", 1)
            return {"server": server, "tool": tool}
        return {"server": "unknown", "tool": full_tool_name}

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self._reset_session_usage()
        self._reset_last_usage()
        self._refresh_context_tokens()

    def start_new_session(self) -> int:
        """Start a brand new session and clear all previous context."""
        self._session_number += 1
        self.clear_history()
        return self._session_number

    def get_history(self) -> list[Message]:
        """Get conversation history."""
        return self.messages.copy()

    async def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        await mcp_manager.disconnect_all()
        self._initialized = False

    def _build_usage_snapshot(self) -> UsageSnapshot | None:
        return self._usage_snapshot_from_mapping(getattr(self.llm_client, "last_usage", None))

    def _build_cumulative_usage_snapshot(self) -> UsageSnapshot | None:
        return self._usage_snapshot_from_mapping(self._session_usage_totals)

    def _usage_snapshot_from_mapping(self, usage: Any) -> UsageSnapshot | None:
        if not isinstance(usage, dict):
            return None
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if all(isinstance(v, int) for v in (prompt, completion, total)):
            return UsageSnapshot(
                prompt_tokens=int(prompt),
                completion_tokens=int(completion),
                total_tokens=int(total),
            )
        return None

    def _record_latest_usage(self) -> None:
        snapshot = self._build_usage_snapshot()
        if snapshot is None:
            return
        self._session_usage_totals["prompt_tokens"] += snapshot.prompt_tokens
        self._session_usage_totals["completion_tokens"] += snapshot.completion_tokens
        self._session_usage_totals["total_tokens"] += snapshot.total_tokens

    def _reset_session_usage(self) -> None:
        self._session_usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _reset_last_usage(self) -> None:
        if self.llm_client is None:
            return
        if hasattr(self.llm_client, "last_usage"):
            self.llm_client.last_usage = None

    def _refresh_context_tokens(self) -> None:
        if self.llm_client is None or not hasattr(self.llm_client, "count_tokens"):
            self._context_tokens = None
            return

        tools = self._tool_invoker.get_tools()
        messages = [Message("system", self.get_system_prompt()), *self.messages]
        try:
            counted = self.llm_client.count_tokens(messages, tools=tools if tools else None)
        except Exception:
            self._context_tokens = None
            return

        self._context_tokens = counted if isinstance(counted, int) and counted >= 0 else None

    def find_local_files(self, query: str, limit: int = 8) -> list[str]:
        """Find workspace files by fuzzy path query."""
        raw_query = query.strip().lstrip("@").replace("\\", "/")
        if raw_query.startswith("~"):
            raw_query = os.path.expanduser(raw_query)
        if raw_query.startswith("/") or Path(raw_query).is_absolute():
            return self._find_absolute_paths(raw_query, limit=limit)

        needle = raw_query.lstrip("./")
        if not needle:
            return self._list_workspace_files_quick(limit=limit)

        direct_match = self._resolve_direct_workspace_file(needle)
        if direct_match is not None:
            return [direct_match]

        files = self._list_workspace_files()
        if not files:
            return []

        q = needle.casefold()
        scored: list[tuple[int, int, str]] = []
        for rel_path in files:
            rp = rel_path.casefold()
            base = rel_path.rsplit("/", 1)[-1].casefold()
            score = self._path_match_score(q, rp, base)
            if score is None:
                continue
            scored.append((score, len(rel_path), rel_path))

        scored.sort()
        return [item[2] for item in scored[:limit]]

    def _find_absolute_paths(self, query: str, limit: int = 8) -> list[str]:
        normalized = query or "/"
        path = Path(normalized)

        if normalized.endswith("/"):
            base_dir = path
            prefix = ""
        else:
            base_dir = path.parent if str(path.parent) else Path("/")
            prefix = path.name

        try:
            entries = list(base_dir.iterdir())
        except Exception:
            return []

        prefix_fold = prefix.casefold()
        candidates: list[tuple[int, str]] = []
        for entry in entries:
            name = entry.name
            name_fold = name.casefold()
            if prefix and not (name_fold.startswith(prefix_fold) or prefix_fold in name_fold):
                continue
            full = entry.as_posix()
            candidates.append((0 if name_fold.startswith(prefix_fold) else 1, full))

        candidates.sort(key=lambda item: (item[0], len(item[1]), item[1]))
        return [item[1] for item in candidates[:limit]]

    def _path_match_score(self, query: str, rel_path: str, basename: str) -> int | None:
        if rel_path == query:
            return 0
        if rel_path.endswith("/" + query) or rel_path.endswith(query):
            return 1
        if basename == query:
            return 2
        if basename.startswith(query):
            return 3
        if rel_path.startswith(query):
            return 4
        if query in rel_path:
            return 5
        if self._matches_path_segments(query, rel_path):
            return 6
        return None

    def _matches_path_segments(self, query: str, rel_path: str) -> bool:
        parts = [p for p in query.split("/") if p]
        if not parts:
            return False
        target_parts = rel_path.split("/")
        idx = 0
        for part in parts:
            found = False
            while idx < len(target_parts):
                if part in target_parts[idx]:
                    found = True
                    idx += 1
                    break
                idx += 1
            if not found:
                return False
        return True

    def _list_workspace_files(self) -> list[str]:
        now = time.monotonic()
        if self._file_index_cache and now - self._file_index_cache[0] < self._FILE_INDEX_TTL_S:
            return self._file_index_cache[1]

        files: list[str] = []
        for root, dirs, filenames in os.walk(self._workspace_root):
            dirs[:] = [d for d in dirs if d not in self._SKIP_DIRS and not d.startswith(".")]
            root_path = Path(root)
            for name in filenames:
                if name.startswith("."):
                    continue
                abs_path = root_path / name
                rel = abs_path.relative_to(self._workspace_root).as_posix()
                files.append(rel)

        files.sort()
        self._file_index_cache = (now, files)
        return files

    def _list_workspace_files_quick(self, limit: int) -> list[str]:
        if limit <= 0:
            return []

        files: list[str] = []
        queue: list[tuple[Path, int]] = [(self._workspace_root, 0)]

        while queue and len(files) < limit:
            current_dir, depth = queue.pop(0)
            try:
                entries = sorted(current_dir.iterdir(), key=lambda p: p.name)
            except Exception:
                continue

            for entry in entries:
                name = entry.name
                if name.startswith("."):
                    continue
                if entry.is_file():
                    rel = entry.relative_to(self._workspace_root).as_posix()
                    files.append(rel)
                    if len(files) >= limit:
                        break
                    continue
                if (
                    entry.is_dir()
                    and depth < self._QUICK_SCAN_MAX_DEPTH
                    and name not in self._SKIP_DIRS
                ):
                    queue.append((entry, depth + 1))

        return files

    def _resolve_direct_workspace_file(self, query: str) -> str | None:
        target = (self._workspace_root / query).resolve()
        if not target.is_file():
            return None
        try:
            rel = target.relative_to(self._workspace_root.resolve())
        except Exception:
            return None
        return rel.as_posix()

    def _inject_file_context(self, user_input: str) -> str:
        matches = self._AT_REF_PATTERN.findall(user_input)
        if not matches:
            return user_input

        resolved_paths: list[str] = []
        seen: set[str] = set()
        for ref in matches:
            best = self.find_local_files(ref, limit=1)
            if not best:
                continue
            rel_path = best[0]
            if rel_path in seen:
                continue
            seen.add(rel_path)
            resolved_paths.append(rel_path)
            if len(resolved_paths) >= self._MAX_ATTACHED_FILES:
                break

        if not resolved_paths:
            return user_input

        context_parts: list[str] = []
        for rel_path in resolved_paths:
            abs_path = self._workspace_root / rel_path
            if not abs_path.is_file():
                continue
            try:
                raw = abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            truncated = raw[: self._MAX_FILE_CHARS]
            suffix = "\n...[truncated]..." if len(raw) > self._MAX_FILE_CHARS else ""
            context_parts.append(f"<file path=\"{rel_path}\">\n{truncated}{suffix}\n</file>")

        if not context_parts:
            return user_input

        file_context = "\n\n".join(context_parts)
        return (
            f"{user_input}\n\n"
            "[Attached file context]\n"
            "Use these referenced local files as context when helpful:\n"
            f"{file_context}"
        )

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Return the current merged tool catalog for UI inspection."""
        return self._tool_invoker.get_tools()
