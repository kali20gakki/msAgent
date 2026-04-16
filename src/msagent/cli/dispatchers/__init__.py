"""Dispatchers for routing user inputs."""

from msagent.cli.dispatchers.commands import CommandDispatcher
from msagent.cli.dispatchers.messages import MessageDispatcher

__all__ = ["CommandDispatcher", "MessageDispatcher"]
