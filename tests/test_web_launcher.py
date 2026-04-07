from __future__ import annotations

from pathlib import Path

import pytest

from msagent.web import launcher
from msagent.web import ui as web_ui
from msagent.web.runtime import ENV_AGENT, ENV_MODEL, ENV_WORKING_DIR, resolve_web_graph_options


def test_build_langgraph_config_payload_uses_repo_graph_path(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='mindstudio-agent'\n")

    monkeypatch.setattr(launcher, "project_root", lambda: project_root)

    payload = launcher.build_langgraph_config_payload()

    assert payload == {
        "dependencies": ["."],
        "graphs": {
            launcher.LANGGRAPH_GRAPH_ID: "./src/msagent/web/graph.py:graph",
        },
    }


def test_build_web_environment_sets_runtime_variables(tmp_path: Path) -> None:
    env = launcher.build_web_environment(
        working_dir=tmp_path,
        agent="general",
        model="default",
    )

    assert env[ENV_WORKING_DIR] == str(tmp_path.resolve())
    assert env[ENV_AGENT] == "general"
    assert env[ENV_MODEL] == "default"


def test_build_ui_environment_sets_default_config_env() -> None:
    env = web_ui.build_ui_environment(
        deployment_url="http://127.0.0.1:2024",
        assistant_id="msagent",
    )

    assert env[web_ui.ENV_UI_DEPLOYMENT_URL] == "http://127.0.0.1:2024"
    assert env[web_ui.ENV_UI_ASSISTANT_ID] == "msagent"


def test_ensure_ui_default_config_support_patches_page(tmp_path: Path) -> None:
    page_path = tmp_path / "src" / "app" / "page.tsx"
    page_path.parent.mkdir(parents=True)
    page_path.write_text(
        """function HomePageContent() {
  const [config, setConfig] = useState<StandaloneConfig | null>(null);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [assistantId, setAssistantId] = useQueryState("assistantId");

  // On mount, check for saved config, otherwise show config dialog
  useEffect(() => {
    const savedConfig = getConfig();
    if (savedConfig) {
      setConfig(savedConfig);
      if (!assistantId) {
        setAssistantId(savedConfig.assistantId);
      }
    } else {
      setConfigDialogOpen(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
""",
        encoding="utf-8",
    )

    web_ui.ensure_ui_default_config_support(tmp_path)
    patched = page_path.read_text(encoding="utf-8")

    assert web_ui.DEFAULT_CONFIG_MARKER in patched
    assert "NEXT_PUBLIC_MSAGENT_DEPLOYMENT_URL" in patched
    assert "saveConfig(defaultConfig);" in patched


def test_ensure_ui_branding_patches_page_copy(tmp_path: Path) -> None:
    page_path = tmp_path / "src" / "app" / "page.tsx"
    page_path.parent.mkdir(parents=True)
    page_path.write_text(
        """function HomePageContent() {
  return (
    <>
      <h1>Deep Agent UI</h1>
      <p>Welcome to Standalone Chat</p>
      <p>Configure your deployment to get started</p>
    </>
  );
}
""",
        encoding="utf-8",
    )

    web_ui.ensure_ui_branding(tmp_path)
    patched = page_path.read_text(encoding="utf-8")

    assert "Deep Agent UI" not in patched
    assert "msAgent" in patched
    assert "Welcome to msAgent" in patched
    assert "Connect to msAgent and start chatting" in patched
    assert web_ui.BRANDING_MARKER in patched


def test_ensure_ui_customizations_apply_both_patches(tmp_path: Path) -> None:
    page_path = tmp_path / "src" / "app" / "page.tsx"
    page_path.parent.mkdir(parents=True)
    page_path.write_text(
        """function HomePageContent() {
  const [config, setConfig] = useState<StandaloneConfig | null>(null);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [assistantId, setAssistantId] = useQueryState("assistantId");

  // On mount, check for saved config, otherwise show config dialog
  useEffect(() => {
    const savedConfig = getConfig();
    if (savedConfig) {
      setConfig(savedConfig);
      if (!assistantId) {
        setAssistantId(savedConfig.assistantId);
      }
    } else {
      setConfigDialogOpen(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <h1>Deep Agent UI</h1>
      <p>Welcome to Standalone Chat</p>
      <p>Configure your deployment to get started</p>
    </>
  );
}
""",
        encoding="utf-8",
    )

    web_ui.ensure_ui_customizations(tmp_path)
    patched = page_path.read_text(encoding="utf-8")

    assert web_ui.DEFAULT_CONFIG_MARKER in patched
    assert web_ui.BRANDING_MARKER in patched
    assert "NEXT_PUBLIC_MSAGENT_DEPLOYMENT_URL" in patched
    assert "Welcome to msAgent" in patched


def test_ensure_ui_customizations_are_idempotent(tmp_path: Path) -> None:
    page_path = tmp_path / "src" / "app" / "page.tsx"
    page_path.parent.mkdir(parents=True)
    page_path.write_text(
        """function HomePageContent() {
  const [config, setConfig] = useState<StandaloneConfig | null>(null);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [assistantId, setAssistantId] = useQueryState("assistantId");

  useEffect(() => {
    const savedConfig = getConfig();
    if (savedConfig) {
      setConfig(savedConfig);
      if (!assistantId) {
        setAssistantId(savedConfig.assistantId);
      }
    } else {
      setConfigDialogOpen(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <h1>Deep Agent UI</h1>
      <p>Welcome to Standalone Chat</p>
      <p>Configure your deployment to get started</p>
    </>
  );
}
""",
        encoding="utf-8",
    )

    web_ui.ensure_ui_customizations(tmp_path)
    first_pass = page_path.read_text(encoding="utf-8")

    web_ui.ensure_ui_customizations(tmp_path)
    second_pass = page_path.read_text(encoding="utf-8")

    assert first_pass == second_pass


def test_resolve_web_graph_options_reads_env(tmp_path: Path) -> None:
    options = resolve_web_graph_options(
        {
            ENV_WORKING_DIR: str(tmp_path),
            ENV_AGENT: "general",
            ENV_MODEL: "default",
        }
    )

    assert options.working_dir == tmp_path.resolve()
    assert options.agent == "general"
    assert options.model == "default"


def test_build_langgraph_dev_command_uses_runner(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        launcher,
        "resolve_langgraph_runner",
        lambda: ["uv", "run", "--offline", "--with", "langgraph-cli[inmem]", "langgraph"],
    )

    command = launcher.build_langgraph_dev_command(
        config_path=tmp_path / "langgraph.json",
        host="127.0.0.1",
        port=2024,
    )

    assert command == [
        "uv",
        "run",
        "--offline",
        "--with",
        "langgraph-cli[inmem]",
        "langgraph",
        "dev",
        "--no-browser",
        "--allow-blocking",
        "--host",
        "127.0.0.1",
        "--port",
        "2024",
        "--config",
        str(tmp_path / "langgraph.json"),
    ]


def test_resolve_langgraph_runner_falls_back_to_installed_module(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "installed"
    project_root.mkdir()

    monkeypatch.setattr(launcher, "project_root", lambda: project_root)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(launcher, "_has_langgraph_cli_module", lambda: True)

    assert launcher.resolve_langgraph_runner() == [
        launcher.sys.executable,
        "-m",
        "langgraph_cli.cli",
    ]


def test_build_ui_dev_command_uses_npm() -> None:
    command = web_ui.build_ui_dev_command(host="127.0.0.1", port=3000)

    assert command == [
        web_ui.shutil.which("npx"),
        "next",
        "dev",
        "--turbopack",
        "--hostname",
        "127.0.0.1",
        "--port",
        "3000",
    ]


def test_ensure_ui_dependencies_uses_resolved_npm_command(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def _fake_run(command, **kwargs):
        del kwargs
        commands.append(command)

    monkeypatch.setattr(web_ui.shutil, "which", lambda name: f"C:/tools/{name}.cmd")
    monkeypatch.setattr(web_ui.subprocess, "run", _fake_run)

    web_ui.ensure_ui_dependencies(tmp_path)

    assert commands == [
        [
            "C:/tools/npm.cmd",
            "install",
            "--no-fund",
            "--no-audit",
            "--legacy-peer-deps",
        ]
    ]


def test_clear_stale_dev_lock_removes_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / ".next" / "dev" / "lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("", encoding="utf-8")

    web_ui.clear_stale_dev_lock(tmp_path)

    assert not lock_path.exists()


def test_display_host_maps_bind_all_to_localhost() -> None:
    assert launcher._display_host("0.0.0.0") == "127.0.0.1"
    assert launcher._display_host("127.0.0.1") == "127.0.0.1"


@pytest.mark.asyncio
async def test_launch_langgraph_dev_server_invokes_subprocess(
    monkeypatch,
    tmp_path: Path,
) -> None:
    spawned: list[dict[str, object]] = []
    terminated: list[str] = []

    class _Proc:
        def __init__(self, name: str):
            self.name = name
            self.returncode: int | None = None

        def poll(self):
            return self.returncode

        def terminate(self):
            terminated.append(self.name)
            self.returncode = 0

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = 1

    async def _fake_spawn_process(*, command, cwd, env):
        name = "ui" if command[1:3] == ["next", "dev"] else "api"
        proc = _Proc(name)
        spawned.append(
            {
                "name": name,
                "command": command,
                "cwd": cwd,
                "env": env,
                "proc": proc,
            }
        )
        return proc

    async def _fake_wait_for_processes(*, api_process, ui_process):
        del ui_process
        api_process.returncode = 0
        return 0

    async def _fake_wait_for_http_service(*, process, host, port, service_name, timeout_seconds=30.0):
        del process, host, port, service_name, timeout_seconds
        return None

    monkeypatch.setattr(
        launcher,
        "resolve_langgraph_runner",
        lambda: ["uv", "run", "--offline", "--with", "langgraph-cli[inmem]", "langgraph"],
    )
    monkeypatch.setattr(launcher, "project_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "_is_port_open", lambda host, port: False)
    (tmp_path / "langgraph.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(launcher, "_spawn_process", _fake_spawn_process)
    monkeypatch.setattr(launcher, "_wait_for_http_service", _fake_wait_for_http_service)
    monkeypatch.setattr(launcher, "_wait_for_processes", _fake_wait_for_processes)
    opened: list[str] = []
    monkeypatch.setattr(launcher, "_open_browser", opened.append)
    monkeypatch.setattr(web_ui, "ensure_ui_checkout", lambda: tmp_path / "ui")
    monkeypatch.setattr(web_ui, "ensure_ui_customizations", lambda checkout_dir: None)
    monkeypatch.setattr(web_ui, "ensure_ui_dependencies", lambda checkout_dir: None)
    monkeypatch.setattr(web_ui, "clear_stale_dev_lock", lambda checkout_dir: None)

    exit_code = await launcher.launch_langgraph_dev_server(
        host="127.0.0.1",
        port=2024,
        ui_port=3000,
        working_dir=tmp_path / "workspace",
        agent="general",
        model="default",
    )

    assert exit_code == 0
    api_spawn = next(item for item in spawned if item["name"] == "api")
    ui_spawn = next(item for item in spawned if item["name"] == "ui")

    assert api_spawn["cwd"] == tmp_path
    assert api_spawn["command"] == [
        "uv",
        "run",
        "--offline",
        "--with",
        "langgraph-cli[inmem]",
        "langgraph",
        "dev",
        "--no-browser",
        "--allow-blocking",
        "--host",
        "127.0.0.1",
        "--port",
        "2024",
        "--config",
        str(tmp_path / "langgraph.json"),
    ]
    env = api_spawn["env"]
    assert isinstance(env, dict)
    assert env[ENV_AGENT] == "general"
    assert env[ENV_MODEL] == "default"
    assert env[ENV_WORKING_DIR] == str((tmp_path / "workspace").resolve())
    assert ui_spawn["cwd"] == tmp_path / "ui"
    ui_env = ui_spawn["env"]
    assert isinstance(ui_env, dict)
    assert ui_env[web_ui.ENV_UI_DEPLOYMENT_URL] == "http://127.0.0.1:2024"
    assert ui_env[web_ui.ENV_UI_ASSISTANT_ID] == "msagent"
    assert opened == ["http://127.0.0.1:3000"]
    assert ui_spawn["command"] == [
        web_ui.shutil.which("npx"),
        "next",
        "dev",
        "--turbopack",
        "--hostname",
        "127.0.0.1",
        "--port",
        "3000",
    ]


@pytest.mark.asyncio
async def test_wait_for_processes_cleans_up_both_children_on_keyboard_interrupt() -> None:
    terminated: list[str] = []

    class _Proc:
        def __init__(self, name: str):
            self.name = name
            self.pid = 123 if name == "api" else 456

        def poll(self):
            return None

    async def _fake_sleep(_seconds: float) -> None:
        raise KeyboardInterrupt

    async def _fake_terminate_process(process) -> None:
        terminated.append(process.name)

    original_sleep = launcher.asyncio.sleep
    launcher.asyncio.sleep = _fake_sleep
    try:
        api = _Proc("api")
        ui = _Proc("ui")
        original_terminate = launcher._terminate_process
        launcher._terminate_process = _fake_terminate_process
        try:
            exit_code = await launcher._wait_for_processes(
                api_process=api,
                ui_process=ui,
            )
        finally:
            launcher._terminate_process = original_terminate
    finally:
        launcher.asyncio.sleep = original_sleep

    assert exit_code == 0
    assert terminated == ["ui", "api"]


@pytest.mark.asyncio
async def test_launch_langgraph_dev_server_errors_when_ui_port_is_busy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _Proc:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            del timeout
            return 0

        def kill(self):
            return None

    async def _fake_spawn_process(*, command, cwd, env):
        del command, cwd, env
        return _Proc()

    async def _fake_wait_for_http_service(*, process, host, port, service_name, timeout_seconds=30.0):
        del process, host, port, service_name, timeout_seconds
        return None

    monkeypatch.setattr(launcher, "project_root", lambda: tmp_path)
    (tmp_path / "langgraph.json").write_text("{}", encoding="utf-8")

    def _fake_is_port_open(host: str, port: int) -> bool:
        del host
        return port == 3000

    monkeypatch.setattr(launcher, "_is_port_open", _fake_is_port_open)
    monkeypatch.setattr(launcher, "_spawn_process", _fake_spawn_process)
    monkeypatch.setattr(launcher, "_wait_for_http_service", _fake_wait_for_http_service)
    monkeypatch.setattr(launcher, "_open_browser", lambda url: None)
    monkeypatch.setattr(web_ui, "ensure_ui_checkout", lambda: tmp_path / "ui")
    monkeypatch.setattr(web_ui, "ensure_ui_customizations", lambda checkout_dir: None)
    monkeypatch.setattr(web_ui, "ensure_ui_dependencies", lambda checkout_dir: None)

    with pytest.raises(RuntimeError, match="UI port 3000 is already in use"):
        await launcher.launch_langgraph_dev_server(
            host="127.0.0.1",
            port=2024,
            ui_port=3000,
            working_dir=tmp_path / "workspace",
            agent=None,
            model=None,
        )


@pytest.mark.asyncio
async def test_launch_langgraph_dev_server_stops_before_ui_when_api_start_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    spawned: list[str] = []

    class _Proc:
        def __init__(self, name: str):
            self.name = name
            self.returncode: int | None = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = 1

    async def _fake_spawn_process(*, command, cwd, env):
        del cwd, env
        name = "ui" if command[1:3] == ["next", "dev"] else "api"
        spawned.append(name)
        return _Proc(name)

    async def _fake_wait_for_http_service(*, process, host, port, service_name, timeout_seconds=30.0):
        del process, host, port, timeout_seconds
        if service_name == "LangGraph API":
            raise RuntimeError("LangGraph API failed to start. Check the logs above for the underlying error.")

    monkeypatch.setattr(launcher, "project_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "_is_port_open", lambda host, port: False)
    monkeypatch.setattr(launcher, "_spawn_process", _fake_spawn_process)
    monkeypatch.setattr(launcher, "_wait_for_http_service", _fake_wait_for_http_service)
    monkeypatch.setattr(launcher, "_open_browser", lambda url: None)
    (tmp_path / "langgraph.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(web_ui, "ensure_ui_checkout", lambda: tmp_path / "ui")
    monkeypatch.setattr(web_ui, "ensure_ui_customizations", lambda checkout_dir: None)
    monkeypatch.setattr(web_ui, "ensure_ui_dependencies", lambda checkout_dir: None)
    monkeypatch.setattr(web_ui, "clear_stale_dev_lock", lambda checkout_dir: None)

    with pytest.raises(RuntimeError, match="LangGraph API failed to start"):
        await launcher.launch_langgraph_dev_server(
            host="127.0.0.1",
            port=2024,
            ui_port=3000,
            working_dir=tmp_path / "workspace",
            agent=None,
            model=None,
        )

    assert spawned == ["api"]


@pytest.mark.asyncio
async def test_launch_langgraph_dev_server_cleans_started_processes_on_startup_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cleaned: list[str] = []

    class _Proc:
        def __init__(self, name: str):
            self.name = name
            self.pid = 100 if name == "api" else 200
            self.returncode: int | None = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = 1

    async def _fake_spawn_process(*, command, cwd, env):
        del cwd, env
        name = "ui" if command[1:3] == ["next", "dev"] else "api"
        return _Proc(name)

    async def _fake_wait_for_http_service(*, process, host, port, service_name, timeout_seconds=30.0):
        del process, host, port, timeout_seconds
        if service_name == "deep-agents-ui":
            raise RuntimeError("deep-agents-ui failed to start. Check the logs above for the underlying error.")

    async def _fake_terminate_process(process) -> None:
        cleaned.append(process.name)

    monkeypatch.setattr(launcher, "project_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "_is_port_open", lambda host, port: False)
    monkeypatch.setattr(launcher, "_spawn_process", _fake_spawn_process)
    monkeypatch.setattr(launcher, "_wait_for_http_service", _fake_wait_for_http_service)
    monkeypatch.setattr(launcher, "_terminate_process", _fake_terminate_process)
    monkeypatch.setattr(launcher, "_open_browser", lambda url: None)
    (tmp_path / "langgraph.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(web_ui, "ensure_ui_checkout", lambda: tmp_path / "ui")
    monkeypatch.setattr(web_ui, "ensure_ui_customizations", lambda checkout_dir: None)
    monkeypatch.setattr(web_ui, "ensure_ui_dependencies", lambda checkout_dir: None)
    monkeypatch.setattr(web_ui, "clear_stale_dev_lock", lambda checkout_dir: None)

    with pytest.raises(RuntimeError, match="deep-agents-ui failed to start"):
        await launcher.launch_langgraph_dev_server(
            host="127.0.0.1",
            port=2024,
            ui_port=3000,
            working_dir=tmp_path / "workspace",
            agent=None,
            model=None,
        )

    assert cleaned == ["ui", "api"]


@pytest.mark.asyncio
async def test_launch_langgraph_dev_server_can_skip_browser_open(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _Proc:
        def __init__(self):
            self.returncode: int | None = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            del timeout
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = 1

    async def _fake_spawn_process(*, command, cwd, env):
        del command, cwd, env
        return _Proc()

    async def _fake_wait_for_http_service(*, process, host, port, service_name, timeout_seconds=30.0):
        del process, host, port, service_name, timeout_seconds
        return None

    async def _fake_wait_for_processes(*, api_process, ui_process):
        del ui_process
        api_process.returncode = 0
        return 0

    opened: list[str] = []
    monkeypatch.setattr(launcher, "project_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "_is_port_open", lambda host, port: False)
    monkeypatch.setattr(launcher, "_spawn_process", _fake_spawn_process)
    monkeypatch.setattr(launcher, "_wait_for_http_service", _fake_wait_for_http_service)
    monkeypatch.setattr(launcher, "_wait_for_processes", _fake_wait_for_processes)
    monkeypatch.setattr(launcher, "_open_browser", opened.append)
    (tmp_path / "langgraph.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(web_ui, "ensure_ui_checkout", lambda: tmp_path / "ui")
    monkeypatch.setattr(web_ui, "ensure_ui_customizations", lambda checkout_dir: None)
    monkeypatch.setattr(web_ui, "ensure_ui_dependencies", lambda checkout_dir: None)
    monkeypatch.setattr(web_ui, "clear_stale_dev_lock", lambda checkout_dir: None)

    exit_code = await launcher.launch_langgraph_dev_server(
        host="127.0.0.1",
        port=2024,
        ui_port=3000,
        open_browser_on_start=False,
        working_dir=tmp_path / "workspace",
        agent=None,
        model=None,
    )

    assert exit_code == 0
    assert opened == []
