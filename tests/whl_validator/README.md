# msagent whl 自动化验证

验证 `mindstudio-agent` wheel 的安装一致性与端到端功能。支持本地 Conda 脚本、Nox，以及 GitHub Actions 的跨平台验证。

## 1. 快速开始

```bash
cd tests/whl_validator

export LLM_API_KEY='<your-api-key>'

./scripts/run_validation.sh \
  --wheel /path/to/mindstudio_agent-<version>-py3-none-any.whl
```

脚本按顺序执行四个阶段：

1. 创建干净 Conda 环境
2. 安装 whl 及其依赖
3. `pip check` 依赖一致性检查
4. 安装测试依赖并运行 pytest（`-n auto` 并行）
5. 删除创建的临时 Conda 环境

任意阶段失败脚本立即退出，错误信息打屏，conda 环境自动清理。

验证器中的真实 `msagent` 子进程会通过 `--execute-approval-mode auto` 预设 shell 命令审批模式，避免 Mock LLM 触发 `execute` 工具时等待首次模式选择。

### 脚本参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--wheel PATH` | 是 | — | 待验证的 whl 路径 |
| `--python-version VERSION` | 否 | `3.11` | Conda 环境 Python 版本 |
| `--run-dir PATH` | 否 | `artifacts/install-<时间戳>-<PID>` | 运行目录，用于存放环境和日志 |
| `--keep-env` | 否 | 不保留 | 运行后保留 Conda 环境（调试用） |
| `--skip-tests` | 否 | 不跳过 | 跳过 Stage 4，仅做安装门禁 |

查看帮助：

```bash
./scripts/run_validation.sh --help
```

## 2. Nox 本地验证

Nox 提供与 CI 对应的 Conda 和 uv 隔离环境验证。运行前安装 `nox`，并确保执行的 session 所需的 Conda 或 uv 可用。

```bash
cd tests/whl_validator
export WHL_PATH=/absolute/path/to/mindstudio_agent-<version>-py3-none-any.whl
export LLM_API_KEY='<your-api-key>'

nox -s test_conda
nox -s test_uv
```

`WHL_PATH` 必须指向待验证的 wheel 文件。两个 session 都会安装 wheel 和测试依赖，然后执行 `pytest test_case/ -n auto`；`test_conda` 还会运行 `pip check` 检查依赖一致性。

## 3. GitHub Actions

在 GitHub Actions 页面手动运行 `WHL Validator` 工作流。工作流先通过 `scripts/build_whl.sh` 构建并上传 wheel artifact，再下载该 artifact，在 Linux/Windows 与 Conda/uv 的矩阵环境中安装并执行测试。

真实 LLM 用例从仓库 Secret `LLM_API_KEY` 读取密钥。CI 当前直接编排安装和 pytest，不调用 `noxfile.py`。

## 4. 源码结构

```text
tests/whl_validator/
├── README.md                      # 本文档
├── noxfile.py                     # 本地 Conda / uv 验证 session
├── pytest.ini                     # pytest 配置（testpaths=test_case）
├── requirements-test.txt          # 测试依赖
├── config/
│   └── test_config.yaml           # LLM/产物/超时等配置
├── scripts/
│   └── run_validation.sh          # 四阶段编排脚本
├── test_case/                     # pytest 用例与 conftest
│   ├── conftest.py                # 共享 fixture（test_workspace 从种子复制）
│   ├── test_01_llm_conn.py
│   ├── test_02_mcp_tools.py
│   ├── test_03_skills.py
│   ├── test_04_sys_prompt.py
│   ├── test_05_local_env.py
│   ├── test_06_threads.py
│   └── test_07_approval_modes.py
├── test_fixtures/                 # 测试种子数据（每个用例复制到隔离 workspace）
│   └── workspace_seed/
│       ├── read_marker.txt        # 内容：MSAGENT_FILESYSTEM_VALIDATION_OK
│       └── print_marker.sh        # 输出：MSAGENT_LOCALSHELL_VALIDATION_OK
└── validator_core/                # 测试支撑库（运行时、断言、trace 解析等）
```

## 5. 测试用例

pytest 通过 `pytest-xdist` 的 `-n auto` 按 CPU 核数并行执行。测试间无共享状态，每个用例有独立的 workspace 和 MSAGENT_HOME。

| 测试文件 | 测试要点 | 依赖/备注 |
|----------|----------|-----------|
| `test_01_llm_conn.py` | 1. 模型真实返回内容（Ping → Pong）<br>2. 进程退出码、会话状态、应用日志无异常 | 需要真实 API Key<br>覆盖 OpenAI 和 Anthropic 两种协议 |
| `test_02_mcp_tools.py` | 1. Profiler 初始化后发现 `analyze_overlap` 和 `find_slices` 工具<br>2. `analyze_overlap` 对合成 trace 返回精确耗时分类和占比<br>3. `find_slices` 按进程名、主线程、精确匹配过滤，排除干扰数据 | Mock LLM<br>真实启动 msprof-mcp 和 trace_processor_shell |
| `test_03_skills.py` | 1. 显式快捷方式（`/op-mfu-calculator`）触发正确 Skill<br>2. 自然语言（不含 Skill 名）按语义触发同一 Skill | 需要真实 API Key |
| `test_04_sys_prompt.py` | 1. Profiler 和 Accuracy 各自身份、Skill、工具边界不串用<br>2. 运行环境信息（工作目录、OS、Python）已注入且占位符已替换 | Mock LLM 捕获请求 payload<br>不访问真实模型 |
| `test_05_local_env.py` | 1. `read_file` 正常读取和缺失文件错误处理<br>2. `execute` 相对路径和绝对路径执行脚本<br>3. `msprof-analyze` CLI 可用性 | Mock LLM<br>不需要 API Key<br>依赖 `test_fixtures/workspace_seed/` |
| `test_06_threads.py` | `/threads` 能从持久化 checkpoint 中发现之前 Mock LLM 创建的会话，并展示其提示词预览 | Mock LLM<br>不需要 API Key |
| `test_07_approval_modes.py` | 1. Manual / Auto 模式下危险、只读和普通命令的决策<br>2. 普通命令的项目级规则持久化和危险命令的单次审批 | Mock session<br>不需要 API Key |

## 6. 产物结构

每次运行在 `run_dir` 下产生以下内容：

```text
artifacts/install-<UTC时间>-<PID>/
├── conda-env/                    # Conda 环境（默认运行后删除）
├── conda-create.log              # Stage 1 日志
├── pip-install.log               # Stage 2 日志
├── pip-check.log                 # Stage 3 日志
├── pip-install-test-deps.log     # Stage 4 测试依赖安装日志
└── test-artifacts/               # pytest 产物目录
    └── cases/<test-node-id>/     # 按用例隔离
        └── runtime-XX/
            └── invocation-XX/
                ├── trace.jsonl
                ├── app.log
                ├── stdout.txt
                ├── stderr.txt
                └── command.json
```

- pytest 输出直接打屏，不单独存日志文件。
- conda 环境默认运行后自动删除（`--keep-env` 可保留）。
- pytest 产物按 `config/test_config.yaml` 中的 `retention` 策略保留：默认仅保留失败用例产物。

## 7. 配置

配置文件：`config/test_config.yaml`

```yaml
llm:
  model: deepseek-v4-flash
  api_key_env: LLM_API_KEY
  protocols:
    openai:
      base_url: https://api.deepseek.com
      provider_api_key_env: OPENAI_API_KEY
    anthropic:
      base_url: https://api.deepseek.com/anthropic
      provider_api_key_env: ANTHROPIC_API_KEY

msagent:
  executable: msagent
  timeout_seconds: 180

artifacts:
  root_dir: artifacts
  retention: failed
  retain_workspace_on_failure: true
```

运行前需要设置的环境变量：

| 环境变量 | 说明 |
|----------|------|
| `LLM_API_KEY` | LLM API 密钥（真实 LLM 用例需要） |

fixture 会根据协议将 Key 映射到 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY` 并注入 Base URL。API Key 不会写入 trace 或请求捕获文件。

### 产物保留配置

```yaml
artifacts:
  root_dir: artifacts
  retention: failed
  retain_workspace_on_failure: true
```

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `root_dir` | `artifacts` | pytest 产物根目录 |
| `retention` | `failed` | 产物保留策略：`failed` 仅保留失败用例、`all` 保留全部、`none` 全部清理 |
| `retain_workspace_on_failure` | `true` | 失败用例是否额外保存 workspace 快照用于诊断 |

Conda 环境和安装日志不受此策略影响，由 `run_validation.sh` 的 `--keep-env` 控制。
