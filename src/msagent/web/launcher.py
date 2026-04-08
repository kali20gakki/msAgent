"""Thin wrapper around the official LangGraph development server."""

from __future__ import annotations

import asyncio
import http.client
import importlib.util
import json
import os
import signal
import socket
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from msagent.cli.theme import console
from msagent.core.constants import APP_NAME
from msagent.web import ui
from msagent.web.runtime import ENV_AGENT, ENV_MODEL, ENV_WORKING_DIR

LANGGRAPH_GRAPH_ID = APP_NAME
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2024
DEFAULT_UI_PORT = ui.DEFAULT_UI_PORT


def project_root() -> Path:
    """Return the repository root when running from source."""
    return Path(__file__).resolve().parents[3]


def graph_export_path() -> str:
    """Return the graph export path used by LangGraph config."""
    return "./src/msagent/web/graph.py:graph"


def packaged_graph_export_path() -> str:
    """Return the import path fallback for installed-package usage."""
    return "msagent.web.graph:graph"


def build_langgraph_config_payload() -> dict[str, object]:
    """Build a minimal LangGraph config compatible with deep-agents-ui."""
    root = project_root()
    if (root / "pyproject.toml").exists():
        dependencies: list[str] = ["."]
        graph_path = graph_export_path()
    else:
        dependencies = ["mindstudio-agent"]
        graph_path = packaged_graph_export_path()

    return {
        "dependencies": dependencies,
        "graphs": {
            LANGGRAPH_GRAPH_ID: graph_path,
        },
    }


def resolve_langgraph_runner() -> list[str]:
    """Resolve the preferred LangGraph CLI launcher."""
    root = project_root()
    if (root / "pyproject.toml").exists() and shutil.which("uv"):
        return [
            "uv",
            "run",
            "--offline",
            "--with",
            "langgraph-cli[inmem]",
            "langgraph",
        ]
    if shutil.which("langgraph"):
        return ["langgraph"]
    if _has_langgraph_cli_module():
        return [sys.executable, "-m", "langgraph_cli.cli"]
    if shutil.which("uvx"):
        return ["uvx", "--offline", "--from", "langgraph-cli[inmem]", "langgraph"]
    raise RuntimeError(
        "LangGraph CLI not found locally. Install `langgraph-cli[inmem]` once, or rerun with network access so uv can cache it."
    )


def build_langgraph_dev_command(
    *,
    config_path: Path,
    host: str,
    port: int,
) -> list[str]:
    """Build the LangGraph dev command."""
    return [
        *resolve_langgraph_runner(),
        "dev",
        "--no-browser",
        "--allow-blocking",
        "--host",
        host,
        "--port",
        str(port),
        "--config",
        str(config_path),
    ]


def build_web_environment(
    *,
    working_dir: Path,
    agent: str | None,
    model: str | None,
) -> dict[str, str]:
    """Build environment variables consumed by the exported graph."""
    env = dict(os.environ)
    env[ENV_WORKING_DIR] = str(working_dir.resolve())
    if agent:
        env[ENV_AGENT] = agent
    else:
        env.pop(ENV_AGENT, None)
    if model:
        env[ENV_MODEL] = model
    else:
        env.pop(ENV_MODEL, None)
    return env


@contextmanager
def langgraph_config_file() -> Iterator[Path]:
    """Yield a LangGraph config file, preferring the repository file when present."""
    root_config = project_root() / "langgraph.json"
    if root_config.exists():
        yield root_config
        return

    payload = build_langgraph_config_payload()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="msagent-langgraph-",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)

    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


async def launch_langgraph_dev_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    ui_port: int = DEFAULT_UI_PORT,
    start_ui: bool = True,
    open_browser_on_start: bool = True,
    working_dir: Path,
    agent: str | None,
    model: str | None,
) -> int:
    """Launch the LangGraph API server and optionally the official deep-agents-ui."""
    api_process: subprocess.Popen[bytes] | None = None
    ui_process: subprocess.Popen[bytes] | None = None
    api_env = build_web_environment(
        working_dir=working_dir,
        agent=agent,
        model=model,
    )
    display_host = _display_host(host)
    api_url = f"http://{display_host}:{port}"
    ui_url = f"http://{display_host}:{ui_port}"
    ui_env = ui.build_ui_environment(
        deployment_url=api_url,
        assistant_id=LANGGRAPH_GRAPH_ID,
        host=host,
        port=ui_port,
    )

    _print_startup_status(
        api_url=api_url,
        ui_url=ui_url,
        working_dir=working_dir,
        start_ui=start_ui,
    )

    try:
        with langgraph_config_file() as config_path:
            api_process = await _spawn_api_server_process(
                config_path=config_path,
                host=host,
                port=port,
                env=api_env,
            )
            await _wait_for_http_service(
                process=api_process,
                host=host,
                port=port,
                service_name="LangGraph API",
            )

            if start_ui:
                ui_process = await _spawn_ui_server_process(
                    host=host,
                    port=ui_port,
                    env=ui_env,
                )
                await _wait_for_http_service(
                    process=ui_process,
                    host=host,
                    port=ui_port,
                    service_name="deep-agents-ui",
                )
                if open_browser_on_start:
                    await asyncio.to_thread(_open_browser, ui_url)

            return await _wait_for_processes(
                api_process=api_process,
                ui_process=ui_process,
            )
    except BaseException:
        await _cleanup_processes(api_process=api_process, ui_process=ui_process)
        raise


def _print_startup_status(
    *,
    api_url: str,
    ui_url: str,
    working_dir: Path,
    start_ui: bool,
) -> None:
    """Print the startup summary shown before launching the web stack."""
    console.print_success("Starting msagent web stack")
    console.print(f"UI URL: [bold]{ui_url}[/bold]" if start_ui else "UI disabled: running API only")
    console.print(
        f"API URL: [bold]{api_url}[/bold] | Graph ID: [bold]{LANGGRAPH_GRAPH_ID}[/bold]"
    )
    console.print(f"Working dir: {working_dir.resolve()}")
    if start_ui:
        console.print(
            f"On first open, set Deployment URL to {api_url} and Assistant ID to {LANGGRAPH_GRAPH_ID}."
        )
    else:
        console.print("API root returns health JSON; it does not serve the UI.")
    console.print("")


async def _spawn_api_server_process(
    *,
    config_path: Path,
    host: str,
    port: int,
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    """Spawn the LangGraph API process after validating its port."""
    if _is_port_open(host, port):
        raise RuntimeError(
            f"API port {port} is already in use. Stop the existing process or pass a different `--port`."
        )

    api_command = build_langgraph_dev_command(
        config_path=config_path,
        host=host,
        port=port,
    )
    return await _spawn_process(
        command=api_command,
        cwd=project_root(),
        env=env,
    )


async def _spawn_ui_server_process(
    *,
    host: str,
    port: int,
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    """Prepare and spawn the cached deep-agents-ui server."""
    if _is_port_open(host, port):
        raise RuntimeError(
            f"UI port {port} is already in use. Stop the existing process or pass a different `--ui-port`."
        )

    if ui.has_bundled_ui_standalone():
        ui_checkout = await asyncio.to_thread(ui.ensure_ui_standalone_checkout)
        ui_command = ui.build_ui_standalone_command()
    else:
        ui_checkout = await asyncio.to_thread(ui.ensure_ui_checkout)
        await asyncio.to_thread(ui.ensure_ui_customizations, ui_checkout)
        await asyncio.to_thread(ui.ensure_ui_dependencies, ui_checkout)
        await asyncio.to_thread(ui.clear_stale_dev_lock, ui_checkout)
        ui_command = ui.build_ui_dev_command(host=host, port=port)

    return await _spawn_process(
        command=ui_command,
        cwd=ui_checkout,
        env=env,
    )


async def _spawn_process(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    """Spawn a long-lived child process sharing the current terminal."""
    return await asyncio.to_thread(
        subprocess.Popen,
        command,
        cwd=str(cwd),
        env=env,
        start_new_session=True,
    )


async def _wait_for_processes(
    *,
    api_process: subprocess.Popen[bytes],
    ui_process: subprocess.Popen[bytes] | None,
) -> int:
    """Wait for the web stack until one process exits, then clean up the rest."""
    try:
        while True:
            api_code = api_process.poll()
            ui_code = ui_process.poll() if ui_process is not None else None

            if api_code is not None:
                await _cleanup_processes(api_process=api_process, ui_process=ui_process)
                return int(api_code)
            if ui_process is not None and ui_code is not None:
                await _cleanup_processes(api_process=api_process, ui_process=ui_process)
                return int(ui_code)

            await asyncio.sleep(0.5)
    except KeyboardInterrupt:
        await _cleanup_processes(api_process=api_process, ui_process=ui_process)
        return 0


async def _terminate_process(process: subprocess.Popen[bytes] | None) -> None:
    """Terminate a spawned child process tree if it is still running."""
    if process is None:
        return

    pid = getattr(process, "pid", None)
    if pid is None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 10)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, 5)
        return

    _signal_process_group(pid, signal.SIGTERM)
    try:
        await asyncio.to_thread(process.wait, 10)
    except subprocess.TimeoutExpired:
        _signal_process_group(pid, signal.SIGKILL)
        await asyncio.to_thread(process.wait, 5)


async def _cleanup_processes(
    *,
    api_process: subprocess.Popen[bytes] | None,
    ui_process: subprocess.Popen[bytes] | None,
) -> None:
    """Clean up any spawned web-stack child processes."""
    await _terminate_process(ui_process)
    await _terminate_process(api_process)


def _display_host(host: str) -> str:
    """Map bind-all hosts to a browser-friendly local hostname."""
    if host == "0.0.0.0":
        return "127.0.0.1"
    return host


def _is_port_open(host: str, port: int) -> bool:
    """Return whether a TCP port is already accepting connections."""
    target_host = _display_host(host)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((target_host, port)) == 0:
            return True

    if shutil.which("lsof"):
        completed = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            return True

    return False


async def _wait_for_http_service(
    *,
    process: subprocess.Popen[bytes],
    host: str,
    port: int,
    service_name: str,
    timeout_seconds: float = 30.0,
) -> None:
    """Wait until a spawned HTTP service is reachable or fail early if it exits."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    target_host = _display_host(host)

    while True:
        if process.poll() is not None:
            raise RuntimeError(
                f"{service_name} failed to start. Check the logs above for the underlying error."
            )

        if _is_port_open(target_host, port):
            try:
                await asyncio.to_thread(_http_healthcheck, target_host, port)
                return
            except OSError:
                pass

        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for {service_name} on {target_host}:{port}."
            )

        await asyncio.sleep(0.5)


def _http_healthcheck(host: str, port: int) -> None:
    """Perform a minimal HTTP probe against a local service."""
    conn = http.client.HTTPConnection(host, port, timeout=2)
    try:
        conn.request("GET", "/")
        response = conn.getresponse()
        response.read()
    finally:
        conn.close()


def _open_browser(url: str) -> None:
    """Open the UI URL in the user's default browser."""
    try:
        webbrowser.open(url, new=2)
    except Exception:
        # Keep startup resilient even if the platform browser integration fails.
        return


def _signal_process_group(pid: int, sig: int) -> None:
    """Signal a spawned child process group, falling back to the direct process."""
    try:
        os.killpg(pid, sig)
        return
    except (AttributeError, ProcessLookupError, PermissionError):
        pass

    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        return


def _has_langgraph_cli_module() -> bool:
    """Return whether langgraph_cli is importable in the current interpreter."""
    try:
        return importlib.util.find_spec("langgraph_cli.cli") is not None
    except ModuleNotFoundError:
        return False
