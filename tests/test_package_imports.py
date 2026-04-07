import importlib


def test_core_package_surfaces_import() -> None:
    modules = [
        "msagent.agents",
        "msagent.cli.bootstrap.app",
        "msagent.cli.bootstrap.web",
        "msagent.configs",
        "msagent.core.settings",
        "msagent.llms.factory",
        "msagent.mcp.factory",
        "msagent.middlewares",
        "msagent.skills.factory",
        "msagent.tools.factory",
        "msagent.web.graph",
        "msagent.web.launcher",
        "msagent.web.runtime",
        "msagent.web.ui",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name) is not None
