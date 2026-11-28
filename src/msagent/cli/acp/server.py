# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""ACP server exposing msagent agents over stdio JSON-RPC."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import acp
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    ConfigOptionUpdate,
    DeleteSessionResponse,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionDeleteCapabilities,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionInfo,
    SessionListCapabilities,
    SessionMode,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
)
from langchain_core.runnables import RunnableConfig

from msagent.audit import AuditWriter, SubagentAuditTracker
from msagent.cli.acp.mapping import AcpTurnRunner, replay_session_history
from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.core.context import Context
from msagent.cli.theme import console
from msagent.configs import ApprovalMode, BatchLLMConfig, ToolDecisionRule
from msagent.configs.registry import ConfigRegistry
from msagent.core.logging import get_logger
from msagent.core.paths import AppPaths
from msagent.llms.factory import (
    _DEFAULT_PROVIDER_API_KEY_ENV,
    _SUPPORTED_PROVIDER_MAP,
)

logger = get_logger(__name__)

_SESSION_INDEX_FILE = "acp_sessions.json"
_SESSION_PAGE_SIZE = 100
_SESSION_CLOSE_TIMEOUT_S = 5.0


def _read_session_index(state_dir: Path) -> dict[str, dict[str, str]]:
    """Read the per-workspace ACP session index, ignoring malformed data."""
    path = state_dir / _SESSION_INDEX_FILE
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning("Failed to read ACP session index: %s", path, exc_info=True)
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        logger.warning("Ignoring malformed ACP session index: %s", path, exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    records: dict[str, dict[str, str]] = {}
    for key, record in data.items():
        if not isinstance(record, dict):
            continue
        records[key] = {field: value for field, value in record.items() if isinstance(value, str)}
    return records


def _write_session_index(state_dir: Path, sessions: dict[str, dict[str, str]]) -> None:
    """Persist the ACP session index atomically."""
    path = state_dir / _SESSION_INDEX_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)
    except Exception:
        logger.warning("Failed to write ACP session index: %s", path, exc_info=True)


def _iter_project_state_dirs() -> list[Path]:
    """Return every project state directory that may hold an ACP session index."""
    try:
        return [path for path in AppPaths.resolve().projects_dir.iterdir() if path.is_dir()]
    except OSError:
        return []


def _build_session_title(text: str) -> str | None:
    """Derive a short session title from the first user message."""
    stripped = text.strip()
    if not stripped:
        return None
    return stripped.splitlines()[0].strip()[:60] or None


def _build_graph_context(context: Context) -> Any:
    """Create the one-shot graph context manager for a session."""
    return initializer.get_graph(
        agent=context.agent,
        model=context.model,
        working_dir=context.working_dir,
    )


def _warn_unsupported_session_params(
    *,
    additional_directories: list[str] | None,
    mcp_servers: list[Any] | None,
) -> None:
    """Surface client parameters that msagent cannot honor instead of dropping them silently."""
    if additional_directories:
        logger.warning("Ignoring ACP additional_directories: %s", additional_directories)
    if mcp_servers:
        logger.warning("Ignoring ACP mcp_servers; MCP servers must be configured in msagent")


_MODE_LABELS: dict[ApprovalMode, tuple[str, str]] = {
    ApprovalMode.ACTIVE: (
        "Manual Approval",
        "Ask before running tools that change the workspace",
    ),
    ApprovalMode.AGGRESSIVE: (
        "Auto Approve",
        "Run tools without asking for confirmation",
    ),
}


def _build_modes(context: Context) -> SessionModeState:
    """Describe the approval modes the client can switch between."""
    current_mode = context.approval_mode if context.approval_mode in _MODE_LABELS else ApprovalMode.ACTIVE
    return SessionModeState(
        current_mode_id=current_mode.value,
        available_modes=[
            SessionMode(id=mode.value, name=name, description=description)
            for mode, (name, description) in _MODE_LABELS.items()
        ],
    )


async def _user_llm_aliases(working_dir: Path) -> set[str]:
    """Return aliases the user configured themselves, empty when no user config exists."""
    registry = ConfigRegistry(working_dir)
    try:
        user_config = await BatchLLMConfig.from_yaml(registry.llms_file, registry.llms_dir)
    except Exception:
        logger.warning("Ignoring unreadable user LLM config", exc_info=True)
        return set()
    return {str(llm.alias) for llm in user_config.llms}


def _option_value(llm: Any) -> str:
    """Return the selector value for one LLM entry, preferring the model name."""
    return str(getattr(llm, "model", "") or getattr(llm, "alias", ""))


def _option_description(llm: Any) -> str:
    """Describe one LLM entry, keeping the alias visible when it differs."""
    provider = getattr(llm.provider, "value", str(llm.provider))
    alias = str(getattr(llm, "alias", ""))
    value = _option_value(llm)
    if not alias or alias == value:
        return provider
    return f"{provider} · alias {alias}"


def _current_option_value(llms: list[Any], alias: str) -> str:
    """Map the session's model alias to the selector value shown to the client."""
    for llm in llms:
        if str(getattr(llm, "alias", "")) == alias:
            return _option_value(llm)
    return alias


def _missing_api_key_env(llm: Any) -> str | None:
    """Return the env var name when no API key can be resolved for this model."""
    provider = str(getattr(llm.provider, "value", llm.provider)).strip().lower()
    normalized = _SUPPORTED_PROVIDER_MAP.get(provider, provider)
    candidates = [llm.api_key_env] if getattr(llm, "api_key_env", None) else []
    default_env = _DEFAULT_PROVIDER_API_KEY_ENV.get(normalized)
    if default_env and default_env not in candidates:
        candidates.append(default_env)
    if not candidates:
        return "API key"
    for name in candidates:
        if os.getenv(name, "").strip():
            return None
    return candidates[0]


def _resolve_llm_alias(llms: list[Any], value: str) -> str:
    """Map a selector value (usually a model name) back to its LLM alias."""
    for llm in llms:
        if str(getattr(llm, "alias", "")) == value:
            return value
    for llm in llms:
        if _option_value(llm) == value:
            return str(getattr(llm, "alias", "")) or value
    return value


async def _build_config_options(context: Context) -> list[Any]:
    """Expose the configured models as a client-side selector."""
    try:
        llms_config = await initializer.load_llms_config(context.working_dir)
    except Exception as exc:
        logger.warning("Failed to load LLM config for ACP session options", exc_info=True)
        raise acp.RequestError.internal_error({"message": "Failed to load LLM configuration"}) from exc
    user_aliases = await _user_llm_aliases(context.working_dir)
    selectable = [
        llm for llm in llms_config.llms if not user_aliases or llm.alias in user_aliases or llm.alias == context.model
    ]
    options: list[SessionConfigSelectOption] = []
    seen: set[str] = set()
    for llm in selectable:
        value = _option_value(llm)
        if not value or value in seen:
            continue
        missing_env = _missing_api_key_env(llm)
        if missing_env and str(getattr(llm, "alias", "")) != context.model:
            continue
        seen.add(value)
        description = _option_description(llm)
        if missing_env:
            description = f"{description} · missing {missing_env}"
        options.append(
            SessionConfigSelectOption(
                value=value,
                name=value,
                description=description,
            )
        )
    if not options:
        return []
    return [
        SessionConfigOptionSelect(
            id="model",
            name="Model",
            category="model",
            current_value=_current_option_value(llms_config.llms, context.model),
            options=options,
        )
    ]


def _session_sort_key(info: SessionInfo) -> tuple[str, str]:
    """Order sessions by recency, breaking ties on session id."""
    return info.updated_at or "", info.session_id


def _encode_session_cursor(info: SessionInfo) -> str:
    """Encode the last entry of a page as an opaque pagination cursor."""
    return f"{info.updated_at or ''}|{info.session_id}"


def _decode_session_cursor(cursor: str) -> tuple[str, str] | None:
    """Decode a pagination cursor, returning None when it is malformed."""
    updated_at, separator, session_id = cursor.partition("|")
    if not separator:
        return None
    return updated_at, session_id


class AcpSession:
    """Duck-typed session providing only the state shared with InterruptHandler."""

    def __init__(self, context: Context) -> None:
        self.context = context
        self.approval_session_rules: list[ToolDecisionRule] = []
        self.execute_approval_mode = context.execute_approval_mode
        self.prompt = SimpleNamespace(reset_interrupt_state=lambda: None)
        self.audit_writer = AuditWriter(
            state_dir=context.state_dir,
            working_dir=context.working_dir,
            thread_id=context.thread_id,
            agent_name=context.agent,
            enabled=context.audit_log_enabled,
        )
        self.subagent_audit = SubagentAuditTracker(self.audit_writer)


@dataclass
class _SessionEntry:
    """Server-side state for one ACP session."""

    session: AcpSession
    context: Context
    graph_stack: AsyncExitStack | None = None
    graph: Any = None
    prompt_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_task: asyncio.Task | None = None
    cancelled_task: asyncio.Task | None = None
    pending_model: str | None = None
    rebuild_task: asyncio.Task | None = None
    dispose_task: asyncio.Task | None = None
    closed: bool = False


class MsagentAcpAgent(acp.Agent):
    """ACP agent implementation backed by msagent graph sessions."""

    def __init__(self, args: Any) -> None:
        super().__init__()
        self._cli_args = args
        self._conn: Any | None = None
        self._sessions: dict[str, _SessionEntry] = {}
        self._index_lock = asyncio.Lock()

    def on_connect(self, conn: Any) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **_kwargs: Any,
    ) -> InitializeResponse:
        logger.info(
            "ACP initialize: protocol_version=%s, client=%s",
            protocol_version,
            client_info,
        )
        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    close=SessionCloseCapabilities(),
                    delete=SessionDeleteCapabilities(),
                ),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **_kwargs: Any,
    ) -> NewSessionResponse:
        _warn_unsupported_session_params(
            additional_directories=additional_directories,
            mcp_servers=mcp_servers,
        )
        context = await self._create_context(Path(cwd))
        session_id = context.thread_id
        entry = self._build_entry(context)
        self._sessions[session_id] = entry
        await self._record_session(entry, touch=False)
        logger.info("ACP session created: %s (agent=%s, cwd=%s)", session_id, context.agent, cwd)
        return NewSessionResponse(
            session_id=session_id,
            modes=_build_modes(context),
            config_options=await _build_config_options(context),
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[str] | None = None,
        **_kwargs: Any,
    ) -> LoadSessionResponse:
        _warn_unsupported_session_params(
            additional_directories=additional_directories,
            mcp_servers=mcp_servers,
        )
        working_dir = Path(cwd)
        target_cwd = working_dir.expanduser().resolve()
        recorded_session = _read_session_index(initializer.get_project_paths(working_dir).root).get(session_id)
        if recorded_session is not None:
            recorded_cwd = recorded_session.get("cwd")
            if recorded_cwd and Path(recorded_cwd).resolve() != target_cwd:
                raise acp.RequestError.invalid_params(
                    {"message": f"Session {session_id} does not belong to {target_cwd}"}
                )

        thread_exists, messages = await self._read_checkpoint_messages(session_id, working_dir)
        if not thread_exists and recorded_session is None:
            raise acp.RequestError.invalid_params({"message": f"Unknown session: {session_id}"})

        existing_entry = self._sessions.get(session_id)
        if existing_entry is not None and existing_entry.prompt_lock.locked():
            raise acp.RequestError.invalid_params({"message": f"Session is busy: {session_id}"})
        self._sessions.pop(session_id, None)
        if existing_entry is not None:
            # A prompt that is queued behind the lock also holds this entry; mark
            # it closed so it bails out instead of running against the new entry.
            existing_entry.closed = True
            await self._dispose_entry(existing_entry)

        context = await self._create_context(
            working_dir,
            thread_id=session_id,
            model=(recorded_session or {}).get("model") or None,
        )
        entry = self._build_entry(context)
        self._sessions[session_id] = entry
        await self._ensure_graph(entry)
        await replay_session_history(conn=self._conn, session_id=session_id, messages=messages)
        await self._record_session(entry, touch=False)
        logger.info(
            "ACP session loaded: %s (cwd=%s, messages=%d)",
            session_id,
            cwd,
            len(messages),
        )
        return LoadSessionResponse(
            modes=_build_modes(context),
            config_options=await _build_config_options(context),
        )

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **_kwargs: Any,
    ) -> ListSessionsResponse:
        working_dir = Path(cwd) if cwd else Path(getattr(self._cli_args, "working_dir", Path.cwd()))
        records = _read_session_index(initializer.get_project_paths(working_dir).root)

        target_cwd = working_dir.expanduser().resolve() if cwd else None
        sessions: list[SessionInfo] = []
        for session_id, record in records.items():
            recorded_cwd = record.get("cwd")
            if target_cwd is not None and recorded_cwd and Path(recorded_cwd).resolve() != target_cwd:
                continue
            sessions.append(
                SessionInfo(
                    session_id=session_id,
                    cwd=recorded_cwd or str(working_dir),
                    title=record.get("title"),
                    updated_at=record.get("updated_at"),
                )
            )

        sessions.sort(key=_session_sort_key, reverse=True)
        if cursor:
            cursor_key = _decode_session_cursor(cursor)
            if cursor_key is not None:
                sessions = [info for info in sessions if _session_sort_key(info) < cursor_key]
        page = sessions[:_SESSION_PAGE_SIZE]
        next_cursor = _encode_session_cursor(page[-1]) if len(sessions) > _SESSION_PAGE_SIZE else None
        logger.info(
            "ACP session list: %d/%d sessions (cwd=%s, paged=%s)",
            len(page),
            len(sessions),
            working_dir,
            next_cursor is not None,
        )
        return ListSessionsResponse(sessions=page, next_cursor=next_cursor)

    async def prompt(self, session_id: str, prompt: list[Any], **_kwargs: Any) -> PromptResponse:
        entry = self._sessions.get(session_id)
        if entry is None:
            raise acp.RequestError.invalid_params({"message": f"Unknown session: {session_id}"})

        user_text = self._extract_prompt_text(prompt)
        if not user_text:
            raise acp.RequestError.invalid_params({"message": "Prompt must contain text content"})
        if entry.prompt_lock.locked():
            raise acp.RequestError.invalid_params({"message": f"Session is busy: {session_id}"})
        runner: AcpTurnRunner | None = None
        async with entry.prompt_lock:
            turn_task = asyncio.current_task()
            entry.active_task = turn_task
            entry.cancelled_task = None
            try:
                if entry.closed:
                    raise acp.RequestError.invalid_params({"message": f"Unknown session: {session_id}"})
                await self._record_session(entry, user_text=user_text)
                await self._settle_model_switch(entry)
                graph = await self._ensure_graph(entry)
                runner = AcpTurnRunner(
                    conn=self._conn,
                    session_id=session_id,
                    session=entry.session,
                    graph=graph,
                )
                await runner.run(user_text)
            except asyncio.CancelledError:
                if turn_task is None or entry.cancelled_task is not turn_task:
                    raise
                logger.info("ACP session prompt cancelled: %s", session_id)
                if runner is not None:
                    try:
                        await runner.abort_pending_tool_calls()
                    except Exception:
                        logger.warning("Failed to report cancelled ACP tool calls: %s", session_id, exc_info=True)
                return PromptResponse(stop_reason="cancelled")
            finally:
                entry.active_task = None
                if entry.cancelled_task is turn_task:
                    entry.cancelled_task = None
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str, **_kwargs: Any) -> None:
        entry = self._sessions.get(session_id)
        if entry is None:
            return
        task = entry.active_task
        if task is None or task.done():
            return
        entry.cancelled_task = task
        task.cancel()
        logger.info("ACP cancel requested: session=%s", session_id)

    async def _stop_active_turn(self, entry: _SessionEntry, session_id: str) -> bool:
        """Cancel a running turn and report whether it stopped within the close timeout."""
        task = entry.active_task
        if task is None or task.done():
            return True
        task.cancel()
        try:
            await asyncio.wait_for(entry.prompt_lock.acquire(), timeout=_SESSION_CLOSE_TIMEOUT_S)
        except TimeoutError:
            logger.warning("Timed out waiting for ACP session turn to stop: %s", session_id)
            return False
        else:
            entry.prompt_lock.release()
            return True

    @staticmethod
    def _defer_dispose_until_turn_finishes(entry: _SessionEntry) -> None:
        """Release graph resources once a turn that outlived the close timeout finishes."""
        task = entry.active_task
        if task is None or task.done():
            return

        def _dispose(_task: asyncio.Task) -> None:
            entry.dispose_task = asyncio.create_task(MsagentAcpAgent._dispose_entry(entry))

        task.add_done_callback(_dispose)

    async def close_session(self, session_id: str, **_kwargs: Any) -> None:
        """Stop a running turn and release the session's graph and runtime resources."""
        entry = self._sessions.pop(session_id, None)
        if entry is None:
            logger.info("ACP session close ignored for unknown session: %s", session_id)
            return
        entry.closed = True
        if await self._stop_active_turn(entry, session_id):
            await self._dispose_entry(entry)
        else:
            self._defer_dispose_until_turn_finishes(entry)
        logger.info("ACP session closed: %s", session_id)

    async def delete_session(self, session_id: str, **_kwargs: Any) -> DeleteSessionResponse:
        """Remove a session's index entry and stored checkpoints."""
        entry = self._sessions.pop(session_id, None)
        working_dir: Path | None = None
        state_dirs: list[Path] = []
        if entry is not None:
            entry.closed = True
            if await self._stop_active_turn(entry, session_id):
                await self._dispose_entry(entry)
            else:
                self._defer_dispose_until_turn_finishes(entry)
            working_dir = entry.context.working_dir
            state_dir = entry.context.state_dir or initializer.get_project_paths(working_dir).root
            state_dirs.append(state_dir)
        else:
            state_dirs.extend(_iter_project_state_dirs())

        index_removed = False
        recorded_cwd: str | None = None
        async with self._index_lock:
            for state_dir in state_dirs:
                records = _read_session_index(state_dir)
                record = records.get(session_id)
                if record is None:
                    continue
                recorded_cwd = record.get("cwd") or recorded_cwd
                del records[session_id]
                _write_session_index(state_dir, records)
                index_removed = True

        target_dir = working_dir or (Path(recorded_cwd) if recorded_cwd else None)
        checkpoints_removed = await self._delete_checkpoints(session_id, target_dir)
        logger.info(
            "ACP session deleted: %s (index_removed=%s, checkpoints_removed=%s)",
            session_id,
            index_removed,
            checkpoints_removed,
        )
        return DeleteSessionResponse()

    async def _delete_checkpoints(self, session_id: str, working_dir: Path | None) -> bool:
        """Best-effort removal of the thread's stored checkpoints."""
        if working_dir is None:
            return False
        try:
            async with initializer.get_checkpointer(self._cli_args.agent, working_dir) as checkpointer:
                await checkpointer.setup()
                await checkpointer.adelete_thread(session_id)
        except Exception:
            logger.warning(
                "Failed to delete ACP session checkpoints: %s",
                session_id,
                exc_info=True,
            )
            return False
        return True

    @staticmethod
    def _extract_prompt_text(content_blocks: list[Any]) -> str:
        texts: list[str] = []
        for block in content_blocks or []:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(text, str) and text:
                texts.append(text)
        return "\n".join(texts).strip()

    async def set_session_mode(
        self,
        session_id: str,
        mode_id: str,
        **_kwargs: Any,
    ) -> SetSessionModeResponse:
        entry = self._sessions.get(session_id)
        if entry is None:
            raise acp.RequestError.invalid_params({"message": f"Unknown session: {session_id}"})
        try:
            approval_mode = ApprovalMode(mode_id)
        except ValueError as exc:
            raise acp.RequestError.invalid_params({"message": f"Unknown mode: {mode_id}"}) from exc
        if approval_mode not in _MODE_LABELS:
            raise acp.RequestError.invalid_params({"message": f"Unsupported mode: {mode_id}"})
        entry.context.approval_mode = approval_mode
        logger.info("ACP session mode changed: %s -> %s", session_id, mode_id)
        return SetSessionModeResponse()

    async def set_config_option(
        self,
        session_id: str,
        config_id: str,
        value: str | bool,
        **_kwargs: Any,
    ) -> SetSessionConfigOptionResponse:
        entry = self._sessions.get(session_id)
        if entry is None:
            raise acp.RequestError.invalid_params({"message": f"Unknown session: {session_id}"})
        if config_id != "model":
            raise acp.RequestError.invalid_params({"message": f"Unknown config option: {config_id}"})
        if not isinstance(value, str) or not value:
            raise acp.RequestError.invalid_params({"message": "Model must be a non-empty string"})
        if entry.prompt_lock.locked():
            raise acp.RequestError.invalid_params({"message": f"Session is busy: {session_id}"})
        try:
            llms_config = await initializer.load_llms_config(entry.context.working_dir)
        except Exception as exc:
            logger.warning("Failed to load LLM config: %s", value, exc_info=True)
            raise acp.RequestError.internal_error({"message": f"Failed to load LLM config for model: {value}"}) from exc
        alias = _resolve_llm_alias(llms_config.llms, value)
        if alias not in {str(llm.alias) for llm in llms_config.llms}:
            raise acp.RequestError.invalid_params({"message": f"Unknown model: {value}"})
        # A prompt may have started while the LLM config was loading; rebuilding
        # the graph now would tear down the running turn.
        if entry.prompt_lock.locked():
            raise acp.RequestError.invalid_params({"message": f"Session is busy: {session_id}"})

        if entry.context.model == alias:
            entry.pending_model = None
            return SetSessionConfigOptionResponse(config_options=await _build_config_options(entry.context))

        # Rebuilding the graph also restarts MCP servers, which is slow enough to
        # be noticed in the client, so answer the selector right away and finish
        # the rebuild in the background.
        entry.pending_model = alias
        self._start_model_rebuild(entry)
        options = await _build_config_options(entry.context)
        if options:
            options[0].current_value = _current_option_value(llms_config.llms, alias)
        return SetSessionConfigOptionResponse(config_options=options)

    def _start_model_rebuild(self, entry: _SessionEntry) -> None:
        """Apply the pending model without blocking the client response."""
        task = entry.rebuild_task
        if task is not None and not task.done():
            return
        task = asyncio.create_task(self._apply_pending_model(entry))
        task.add_done_callback(lambda finished: self._handle_rebuild_failure(entry, finished))
        entry.rebuild_task = task

    def _handle_rebuild_failure(self, entry: _SessionEntry, task: asyncio.Task) -> None:
        """Roll back a failed model switch so the session stays usable."""
        if task.cancelled() or entry.closed:
            return
        exc = task.exception()
        if exc is None:
            return
        if entry.pending_model is None:
            logger.warning("ACP graph rebuild after model switch failed", exc_info=exc)
            return
        # The context was never replaced; drop the pending switch and tell the
        # client to fall back to the active model. Otherwise every subsequent
        # prompt retries the broken switch and fails.
        entry.pending_model = None
        logger.warning("ACP model switch failed; keeping model %s", entry.context.model, exc_info=exc)
        try:
            asyncio.create_task(self._notify_config_options(entry))
        except RuntimeError:
            logger.warning("Unable to notify ACP client about config options: event loop closed")

    async def _notify_config_options(self, entry: _SessionEntry) -> None:
        """Push the current config options so the client selector matches reality."""
        try:
            options = await _build_config_options(entry.context)
            await self._conn.session_update(
                entry.context.thread_id,
                update=ConfigOptionUpdate(config_options=options),
            )
        except Exception:
            logger.warning("Failed to notify ACP client about config options", exc_info=True)

    async def _apply_pending_model(self, entry: _SessionEntry) -> None:
        """Rebuild the session graph in place for the pending model."""
        while entry.pending_model and entry.pending_model != entry.context.model and not entry.closed:
            alias = entry.pending_model
            context = await self._create_context(
                entry.context.working_dir,
                thread_id=entry.context.thread_id,
                model=alias,
                approval_mode=entry.context.approval_mode,
            )
            replacement = self._build_entry(context)
            replacement.session.approval_session_rules = list(entry.session.approval_session_rules)
            replacement.session.execute_approval_mode = entry.session.execute_approval_mode
            await self._dispose_entry(entry)
            entry.context = replacement.context
            entry.session = replacement.session
            entry.graph = None
            if entry.pending_model == alias:
                entry.pending_model = None
            if not entry.closed:
                await self._record_session(entry, touch=False)
            logger.info(
                "ACP session model switched: %s -> %s (%s)",
                context.thread_id,
                context.model,
                alias,
            )
        if entry.pending_model is None and entry.graph is None:
            await self._ensure_graph(entry)

    async def _settle_model_switch(self, entry: _SessionEntry) -> None:
        """Finish any in-flight model switch before the next turn starts."""
        task = entry.rebuild_task
        try:
            if task is not None:
                await task
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except Exception as exc:
            raise acp.RequestError.internal_error({"message": f"Failed to switch model: {exc}"}) from exc
        finally:
            if entry.rebuild_task is task:
                entry.rebuild_task = None
        if entry.pending_model:
            try:
                await self._apply_pending_model(entry)
            except Exception as exc:
                raise acp.RequestError.internal_error({"message": f"Failed to switch model: {exc}"}) from exc

    async def _create_context(
        self,
        working_dir: Path,
        thread_id: str | None = None,
        model: str | None = None,
        approval_mode: ApprovalMode | None = None,
    ) -> Context:
        return await Context.create(
            agent=self._cli_args.agent,
            model=model or self._cli_args.model,
            approval_mode=approval_mode or ApprovalMode(self._cli_args.approval_mode),
            working_dir=working_dir,
            execute_approval_mode=self._cli_args.execute_approval_mode,
            trace_jsonl=Path(self._cli_args.trace_jsonl) if self._cli_args.trace_jsonl else None,
            thread_id=thread_id,
        )

    @staticmethod
    def _build_entry(context: Context) -> _SessionEntry:
        return _SessionEntry(
            session=AcpSession(context),
            context=context,
        )

    async def _record_session(self, entry: _SessionEntry, *, user_text: str | None = None, touch: bool = True) -> None:
        """Track the session in the workspace index so it survives restarts."""
        context = entry.context
        state_dir = context.state_dir or initializer.get_project_paths(context.working_dir).root
        async with self._index_lock:
            index_records = _read_session_index(state_dir)
            record = index_records.get(context.thread_id, {})
            record["cwd"] = str(context.working_dir.expanduser().resolve())
            if context.model:
                record["model"] = context.model
            if touch or not record.get("updated_at"):
                record["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if user_text and not record.get("title"):
                title = _build_session_title(user_text)
                if title:
                    record["title"] = title
            index_records[context.thread_id] = record
            _write_session_index(state_dir, index_records)

    async def _read_checkpoint_messages(self, thread_id: str, working_dir: Path) -> tuple[bool, list[Any]]:
        """Return whether the thread exists and its stored messages."""
        async with initializer.get_checkpointer(self._cli_args.agent, working_dir) as checkpointer:
            await checkpointer.setup()
            checkpoint_tuple = await checkpointer.aget_tuple(RunnableConfig(configurable={"thread_id": thread_id}))
        if checkpoint_tuple is None:
            return False, []
        checkpoint = checkpoint_tuple.checkpoint or {}
        messages = (checkpoint.get("channel_values") or {}).get("messages") or []
        return True, list(messages)

    @staticmethod
    async def _dispose_entry(entry: _SessionEntry) -> None:
        task = entry.rebuild_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            entry.rebuild_task = None
            task.cancel()
        stack = entry.graph_stack
        if stack is None:
            return
        entry.graph_stack = None
        entry.graph = None
        try:
            await stack.aclose()
        except Exception:
            logger.warning("Failed to close ACP session graph", exc_info=True)

    @staticmethod
    async def _ensure_graph(entry: _SessionEntry) -> Any:
        if entry.graph_stack is None:
            stack = AsyncExitStack()
            try:
                entry.graph = await stack.enter_async_context(_build_graph_context(entry.context))
            except BaseException:
                await stack.aclose()
                raise
            entry.graph_stack = stack
        return entry.graph

    async def _dispose_all(self) -> None:
        """Release every session's graph and runtime resources."""
        entries = list(self._sessions.values())
        self._sessions.clear()
        cancelled: list[asyncio.Task] = []
        for entry in entries:
            entry.closed = True
            task = entry.active_task
            if task is not None and not task.done():
                task.cancel()
                cancelled.append(task)
        if cancelled:
            await asyncio.wait(cancelled, timeout=_SESSION_CLOSE_TIMEOUT_S)
        for entry in entries:
            await self._dispose_entry(entry)


async def handle_acp_command(args: Any) -> int:
    """Run the ACP server loop over stdio."""
    console.reconfigure_quiet()
    logger.info("Starting ACP server over stdio")
    agent = MsagentAcpAgent(args)
    try:
        await acp.run_agent(agent)  # type: ignore[abstract]
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("ACP server terminated with an error")
        return 1
    finally:
        await agent._dispose_all()
    return 0
