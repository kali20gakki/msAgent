"""Sandbox backends using deepagents."""

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import SandboxBackendProtocol

# Re-export deepagents backend types
BackendProtocol = SandboxBackendProtocol
SandboxBackend = SandboxBackendProtocol

__all__ = ["LocalShellBackend", "SandboxBackendProtocol", "BackendProtocol", "SandboxBackend"]
