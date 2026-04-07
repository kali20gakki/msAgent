"""LangGraph-exported graph factory for `msagent web`."""

from msagent.web.runtime import load_web_graph


def graph():
    """Return the lazily constructed msAgent graph for LangGraph server."""
    return load_web_graph()


__all__ = ["graph"]
