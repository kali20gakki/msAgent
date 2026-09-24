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

"""Application and project path layout."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _project_slug(path: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", path.name).strip("-.")
    return slug or "project"


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Global msAgent directories rooted at ``MSAGENT_HOME``."""

    home: Path

    @classmethod
    def resolve(cls, env: Mapping[str, str] | None = None) -> AppPaths:
        values = os.environ if env is None else env
        configured = values.get("MSAGENT_HOME", "").strip()
        home = Path(configured).expanduser() if configured else Path.home() / ".msagent"
        return cls(home=home.resolve())

    @classmethod
    def from_home(cls, home: Path) -> AppPaths:
        return cls(home=home.expanduser().resolve())

    @property
    def config_dir(self) -> Path:
        return self.home / "config"

    @property
    def metadata_file(self) -> Path:
        return self.home / "metadata.json"

    @property
    def prompts_dir(self) -> Path:
        return self.home / "prompts"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def state_dir(self) -> Path:
        return self.home / "state"

    @property
    def projects_dir(self) -> Path:
        return self.state_dir / "projects"

    @property
    def mcp_cache_dir(self) -> Path:
        return self.home / "cache" / "mcp"

    @property
    def mcp_oauth_dir(self) -> Path:
        return self.home / "oauth" / "mcp"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    def for_project(self, working_dir: Path) -> ProjectPaths:
        return ProjectPaths.resolve(self, working_dir)

    def ensure(self) -> None:
        for path in (
            self.home,
            self.config_dir,
            self.prompts_dir,
            self.skills_dir,
            self.projects_dir,
            self.mcp_cache_dir,
            self.mcp_oauth_dir,
            self.logs_dir,
        ):
            _ensure_private_dir(path)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Project-isolated runtime state stored below the global home."""

    app: AppPaths
    working_dir: Path
    project_id: str

    @classmethod
    def resolve(cls, app: AppPaths, working_dir: Path) -> ProjectPaths:
        canonical = working_dir.expanduser().resolve()
        identity = os.path.normcase(str(canonical)).encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()[:12]
        return cls(
            app=app,
            working_dir=canonical,
            project_id=f"{_project_slug(canonical)}-{digest}",
        )

    @property
    def root(self) -> Path:
        return self.app.projects_dir / self.project_id

    @property
    def metadata_file(self) -> Path:
        return self.root / "project.json"

    @property
    def memory_file(self) -> Path:
        return self.root / "memory.md"

    @property
    def history_file(self) -> Path:
        return self.root / "history"

    @property
    def checkpoints_db(self) -> Path:
        return self.root / "checkpoints.sqlite"

    @property
    def conversation_history_dir(self) -> Path:
        return self.root / "conversation_history"

    @property
    def audit_dir(self) -> Path:
        return self.root / "audit_log"

    def ensure(self) -> None:
        self.app.ensure()
        _ensure_private_dir(self.root)
        if self.metadata_file.exists():
            return

        self._write_metadata(self._default_metadata())

    def get_current_agent(self) -> str | None:
        """Return the Agent selected for this workspace, if one was persisted."""
        data = self._read_metadata()
        current_agent = data.get("current_agent")
        if not isinstance(current_agent, str):
            return None
        current_agent = current_agent.strip()
        return current_agent or None

    def set_current_agent(self, agent_name: str) -> None:
        """Persist the Agent selected for this workspace."""
        normalized = agent_name.strip()
        if not normalized:
            raise ValueError("Agent name must not be empty")

        self.app.ensure()
        _ensure_private_dir(self.root)
        payload = self._read_metadata()
        payload.update(self._default_metadata())
        payload["current_agent"] = normalized
        self._write_metadata(payload)

    def get_current_model(self, agent_name: str) -> str | None:
        """Return the workspace model preference for an Agent, if present."""
        models = self._read_metadata().get("agent_models")
        if not isinstance(models, dict):
            return None
        model_name = models.get(agent_name)
        if not isinstance(model_name, str):
            return None
        model_name = model_name.strip()
        return model_name or None

    def set_current_model(self, agent_name: str, model_name: str) -> None:
        """Persist a workspace model preference without changing Agent definitions."""
        normalized_agent = agent_name.strip()
        normalized_model = model_name.strip()
        if not normalized_agent or not normalized_model:
            raise ValueError("Agent and model names must not be empty")

        self.app.ensure()
        _ensure_private_dir(self.root)
        payload = self._read_metadata()
        payload.update(self._default_metadata())
        existing_models = payload.get("agent_models")
        models = dict(existing_models) if isinstance(existing_models, dict) else {}
        models[normalized_agent] = normalized_model
        payload["agent_models"] = models
        self._write_metadata(payload)

    def _default_metadata(self) -> dict[str, str]:
        return {
            "project_id": self.project_id,
            "working_dir": str(self.working_dir),
        }

    def _read_metadata(self) -> dict[str, object]:
        if not self.metadata_file.is_file():
            return {}
        try:
            payload = json.loads(self.metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_metadata(self, payload: Mapping[str, object]) -> None:
        temp_file = self.metadata_file.with_name(f".{self.metadata_file.name}.{os.getpid()}.{id(self)}.tmp")
        try:
            temp_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temp_file.replace(self.metadata_file)
        finally:
            if temp_file.exists():
                temp_file.unlink()
