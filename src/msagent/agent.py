"""Core agent logic for msagent."""

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncGenerator
from pathlib import Path

from rich.console import Console

from .config import AppConfig, config_manager
from .llm import Message, create_llm_client
from .mcp_client import mcp_manager

console = Console()


class Agent:
    """msagent - Core agent implementation."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or config_manager.get_config()
        self.llm_client = None
        self.messages: list[Message] = []
        self._initialized = False
        self._error_message = ""
        self._workspace_root = Path.cwd()
        self._file_index_cache: tuple[float, list[str]] | None = None

    _AT_REF_PATTERN = re.compile(r"(?<!\S)@([^\s]+)")
    _MAX_ATTACHED_FILES = 5
    _MAX_FILE_CHARS = 4000
    _FILE_INDEX_TTL_S = 3.0
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

    async def initialize(self) -> bool:
        """Initialize the agent."""
        try:
            if not self.config.llm.is_configured():
                self._error_message = (
                    "⚠️ LLM not configured. Please set up your API key:\n"
                    "   • Environment: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY\n"
                    "   • Config file: ~/.config/msagent/config.json\n"
                    "   • Use: msagent config --help"
                )
                return False

            self.llm_client = create_llm_client(self.config.llm)

            for mcp_config in self.config.mcp_servers:
                if mcp_config.enabled:
                    await mcp_manager.add_server(mcp_config)

            self._initialized = True
            return True

        except Exception as e:
            self._error_message = f"❌ Failed to initialize agent: {e}"
            return False

    def get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        mcp_servers = mcp_manager.get_connected_servers()
        prompt = (
            "You are msagent, a helpful AI assistant that can use tools to help users.\n\n"
            "When you need to use a tool, call the tool directly.\n"
            "When you receive tool results, incorporate them into your response naturally.\n\n"
            f"Available MCP servers: {', '.join(mcp_servers) if mcp_servers else 'None'}\n\n"
            "Be concise, helpful, and friendly in your responses."
        )
        return prompt

    async def chat(self, user_input: str) -> str:
        """Process a chat message and return the response."""
        if not self._initialized or not self.llm_client:
            return "Error: Agent not initialized. Please check your configuration."

        self.messages.append(Message("user", self._inject_file_context(user_input)))
        all_messages = [Message("system", self.get_system_prompt())] + self.messages
        tools = mcp_manager.get_all_tools()

        timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
        try:
            console.print("[dim]⏳ Waiting for LLM response...[/dim]")
            t0 = time.monotonic()
            response = await asyncio.wait_for(
                self.llm_client.chat(all_messages, tools=tools if tools else None),
                timeout=timeout_s,
            )
            dt = time.monotonic() - t0
            console.print(f"[grey50]⚡ LLM response took {dt:.2f}s[/grey50]")

            self.messages.append(Message("assistant", response))
            self._print_usage()
            return response
        except asyncio.TimeoutError:
            return f"❌ Error: LLM response timed out after {timeout_s:.0f}s"
        except Exception as e:
            return f"❌ Error: {e}"

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Process a chat message and stream the response."""
        if not self._initialized or not self.llm_client:
            yield "Error: Agent not initialized. Please check your configuration."
            return

        self.messages.append(Message("user", self._inject_file_context(user_input)))
        all_messages = [Message("system", self.get_system_prompt())] + self.messages
        tools = mcp_manager.get_all_tools()

        timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
        start = time.monotonic()
        full_response = ""

        try:
            stream = self.llm_client.chat_stream(all_messages, tools=tools if tools else None)

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
                    yield f"❌ Error: LLM stream timed out after {timeout_s:.0f}s"
                    return

                if chunk:
                    full_response += chunk
                    yield chunk

            if not full_response:
                fallback = await asyncio.wait_for(
                    self.llm_client.chat(all_messages, tools=tools if tools else None),
                    timeout=timeout_s,
                )
                if fallback:
                    full_response = fallback
                    yield fallback
                else:
                    last_msgs = [m.to_dict() for m in all_messages[-3:]]
                    yield (
                        "❌ Error: LLM returned empty response.\n"
                        f"Last messages: {json.dumps(last_msgs, ensure_ascii=False)}"
                    )
                    return

            dt = time.monotonic() - start
            yield f"\n\n⚡ LLM response took {dt:.2f}s\n"
            self.messages.append(Message("assistant", full_response))
        except Exception as e:
            yield f"❌ Error: {e}"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    def get_history(self) -> list[Message]:
        """Get conversation history."""
        return self.messages.copy()

    async def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        await mcp_manager.disconnect_all()
        self._initialized = False

    def _print_usage(self) -> None:
        usage = getattr(self.llm_client, "last_usage", None)
        if not isinstance(usage, dict):
            return
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if all(isinstance(v, int) for v in (prompt, completion, total)):
            console.print(
                "[dim]🧮 Tokens used: "
                f"prompt={prompt} completion={completion} total={total}[/dim]"
            )

    def find_local_files(self, query: str, limit: int = 8) -> list[str]:
        """Find workspace files by fuzzy path query."""
        raw_query = query.strip().lstrip("@").replace("\\", "/")
        if raw_query.startswith("~"):
            raw_query = os.path.expanduser(raw_query)
        if raw_query.startswith("/"):
            return self._find_absolute_paths(raw_query, limit=limit)

        needle = raw_query.lstrip("./")
        files = self._list_workspace_files()
        if not files:
            return []

        if not needle:
            return files[:limit]

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
