<#
.SYNOPSIS
  msAgent installer for Windows (PowerShell 5.1+).

.DESCRIPTION
  Installs mindstudio-agent (latest from PyPI) as an isolated uv tool, so it
  will not conflict with any existing Python environment. Bootstraps uv when
  missing, prefers domestic PyPI mirrors, and adds the tool bin directory to
  the user PATH.

.EXAMPLE
  irm https://raw.gitcode.com/Ascend/msagent/raw/master/scripts/install.ps1 | iex

.EXAMPLE
  $env:MSAGENT_VERSION = '26.1.0'
  irm https://raw.gitcode.com/Ascend/msagent/raw/master/scripts/install.ps1 | iex

.NOTES
  Optional env:
    MSAGENT_VERSION           exact version to install (default: latest from PyPI)
    MSAGENT_PYTHON            Python request for the isolated tool env (default: >=3.11,<3.15, reuses local Python)
    MSAGENT_INDEX             PyPI index override (default: official PyPI, domestic mirrors as fallback)
    MSAGENT_NO_MODIFY_PATH    set to a non-empty value to skip PATH modification
    MSAGENT_PLAIN_UI          set to a non-empty value for ASCII-only output (no colours/glyphs)
    MSAGENT_YES               set to '1' to accept prompts without asking
    MSAGENT_NO_FALLBACK       set to '1' to disable the venv fallback
    MSAGENT_FALLBACK_VENV     venv path used by the fallback install (default: %USERPROFILE%\.msagent-venv)
    MSAGENT_NO_ASCEND_DOC_MCP  set to any value to skip Node provisioning and the ascend-doc-mcp pre-install
    MSAGENT_ASCEND_DOC_MCP_REQUIRE  abort install when Node/ascend-doc-mcp prep fails
    MSAGENT_NODE_HOME          user-local Node root (default: %USERPROFILE%\.msagent\node)
    MSAGENT_NODE_MIRROR        Node dist mirror base (default: probed huaweicloud -> npmmirror -> nodejs.org)
    MSAGENT_NPM_REGISTRY       npm registry for the ascend-doc-mcp pre-install
                               (default: the registry this machine's npm is
                               configured with, then probed npmmirror ->
                               huaweicloud -> npmjs)
    MSAGENT_NPM_REGISTRY_ONLY  set to 1 to use MSAGENT_NPM_REGISTRY exclusively
    MSAGENT_NPM_REGISTRY_FALLBACKS  extra last-resort npm sources (comma/semicolon/
                               space separated) for corporate intranets
    MSAGENT_NPM_MIRRORS_FILE   file with one fallback source per line (defaults:
                               %USERPROFILE%\.msagent\npm-mirrors, then
                               %ProgramData%\msagent\npm-mirrors)
    MSAGENT_NPM_CACHE          npm cache directory
    MSAGENT_ASCEND_DOC_MCP_PREFIX  local ascend-doc-mcp install prefix (default: %USERPROFILE%\.msagent\ascend-doc-mcp)
    UV_DEFAULT_INDEX / UV_INDEX_URL / PIP_INDEX_URL   used as a last-resort fallback
    UV_PYTHON_INSTALL_MIRROR  mirror for managed CPython downloads (domestic candidates validated for real binary content)
    UV_NATIVE_TLS             use the system certificate store for uv (default: 1, set 0 to disable)

  Uninstall:
    uv tool uninstall mindstudio-agent

  Upgrade:
    Re-run the installer; it always upgrades to the latest release.
#>

$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 renders the Invoke-WebRequest progress bar very slowly.
$ProgressPreference = 'SilentlyContinue'

# PowerShell 5.1 on older Windows may not negotiate TLS 1.2 by default.
try {
  [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
  # TLS 1.2 is already the default on newer systems; ignore failures here.
}

# ---------------------------------------------------------------------------
# Logging / UI
#
# Colour plus bracketed ASCII tags only: no emoji and no box drawing, so the
# output stays readable on every console and code page. MSAGENT_PLAIN_UI=1
# drops the colours as well.
# ---------------------------------------------------------------------------
$script:Fancy = -not [bool]$env:MSAGENT_PLAIN_UI

# English tags padded to 5 columns so every message starts at the same column.
function Write-Line([string]$tag, [string]$msg, [string]$colour) {
  if ($script:Fancy) { Write-Host ("  {0} {1}" -f $tag, $msg) -ForegroundColor $colour }
  else { Write-Host ("  {0} {1}" -f $tag, $msg) }
}

function Write-Step([string]$msg)    { Write-Line 'INFO ' $msg 'Gray' }
function Write-Success([string]$msg) { Write-Line 'OK   ' $msg 'Green' }
function Write-Warn([string]$msg)    { Write-Line 'WARN ' $msg 'Yellow' }
function Die([string]$msg)           { Write-Line 'ERROR' $msg 'Red'; exit 1 }

# ---------------------------------------------------------------------------
# Progress reporting: numbered phases keep a long install scannable, and the
# collected summary values are printed once at the end instead of scrolling
# away in the middle of the log.
# ---------------------------------------------------------------------------
$script:TotalPhases = 6
$script:Phase = 0
$script:Summary = [ordered]@{}
$script:SummaryColours = @{}
$script:RuleWidth = 42

function Write-Phase([string]$msg) {
  $script:Phase++
  Write-Host ''
  if ($script:Fancy) {
    # Reverse-video badge so each section stands out while scrolling.
    Write-Host (" [{0}/{1}] {2} " -f $script:Phase, $script:TotalPhases, $msg) -ForegroundColor White -BackgroundColor DarkBlue
  } else {
    Write-Host ("[{0}/{1}] {2}" -f $script:Phase, $script:TotalPhases, $msg)
  }
}

function Write-Key([string]$msg) {
  # Highlighted line for the few facts that matter most.
  if ($script:Fancy) { Write-Host ("  {0} {1}" -f 'INFO ', $msg) -ForegroundColor White }
  else { Write-Host ("  {0} {1}" -f 'INFO ', $msg) }
}

$script:BannerArt = @(
  '                                        _',
  '   _ __ ___  ___  __ _  __ _  ___ _ __ | |_',
  '  | ''_ ` _ \/ __|/ _` |/ _` |/ _ \ ''_ \| __|',
  '  | | | | | \__ \ (_| | (_| |  __/ | | | |_',
  '  |_| |_| |_|___/\__,_|\__, |\___|_| |_|\__|',
  '                       |___/'
)

function Write-Banner {
  Write-Host ''
  if ($script:Fancy) {
    foreach ($line in $script:BannerArt) { Write-Host $line -ForegroundColor Cyan }
    Write-Host '  MindStudio Agent Installer' -ForegroundColor DarkGray
  } else {
    foreach ($line in $script:BannerArt) { Write-Host $line }
    Write-Host '  MindStudio Agent Installer'
  }
}

function Write-CardTop([string]$title) {
  $total = $script:RuleWidth - $title.Length - 2
  if ($total -lt 4) { $total = 4 }
  $left = [int][Math]::Floor($total / 2)
  $right = $total - $left
  Write-Host ''
  if ($script:Fancy) {
    Write-Host ("  " + ('-' * $left) + " ") -NoNewline -ForegroundColor DarkGray
    Write-Host (" $title ") -NoNewline -ForegroundColor White -BackgroundColor DarkBlue
    Write-Host (" " + ('-' * $right)) -ForegroundColor DarkGray
  } else {
    Write-Host ("  " + ('-' * $left) + " $title " + ('-' * $right))
  }
}

function Write-CardRow([string]$key, [string]$value, [string]$colour = 'Gray') {
  if ($script:Fancy) {
    Write-Host ("  {0,-18}" -f $key) -NoNewline -ForegroundColor White
    Write-Host (" {0}" -f $value) -ForegroundColor $colour
  } else {
    Write-Host ("  {0,-18} {1}" -f $key, $value)
  }
}

function Write-CardBottom {
  $line = "  " + ('-' * $script:RuleWidth)
  if ($script:Fancy) { Write-Host $line -ForegroundColor DarkGray } else { Write-Host $line }
}

function Write-Summary {
  Write-CardTop 'Summary'
  foreach ($entry in $script:Summary.GetEnumerator()) {
    $colour = if ($script:SummaryColours.ContainsKey($entry.Key)) { $script:SummaryColours[$entry.Key] } else { 'Gray' }
    Write-CardRow $entry.Key "$($entry.Value)" $colour
  }
  Write-CardBottom
}

function Write-NextSteps {
  Write-CardTop 'Next steps'
  Write-CardRow '1) configure' 'set the API key, then point msagent at a model' 'White'
  Write-Host "       `$env:OPENAI_API_KEY = '<your key>'      # Anthropic: ANTHROPIC_API_KEY, Google: GOOGLE_API_KEY"
  Write-Host '       msagent config --llm-provider openai --llm-base-url "<your endpoint>" --llm-model "<model>"'
  Write-Host '       # omit --llm-base-url when using the provider''s official endpoint'
  Write-Host '       msagent config --show                  # verify the configuration'
  Write-CardRow '2) start' 'msagent                                # msagent --help for more' 'White'
  Write-CardRow '3) not found?' "open a new window, then run 'msagent --version'" 'Yellow'
  Write-Host ("       # this session only:  set PATH={0};%PATH%" -f $script:ToolBin)
  Write-Host '       # permanently:        re-run this installer (it writes the user PATH)'
  Write-CardBottom
}

# The CLI prints a banner; keep only the version token ("26.1.2").
# ANSI colour codes are stripped first: with FORCE_COLOR (or a colour-capable
# host) they sit between "msagent" and the version and break the match.
function Get-MsagentVersionToken([string[]]$Output) {
  foreach ($line in $Output) {
    $clean = [regex]::Replace("$line", '\x1b\[[0-9;]*[A-Za-z]', '')
    if ($clean -match 'msagent\s+([0-9][0-9A-Za-z.\-+]*)') { return $Matches[1] }
  }
  return $null
}

function Invoke-CaptureOutput([string]$Exe, [string[]]$Arguments = @()) {
  # Run a native tool and capture stdout+stderr safely.
  # With $ErrorActionPreference='Stop' PowerShell 5.1 turns *redirected native
  # stderr* into a terminating NativeCommandError, and tools like msagent print
  # pydantic/PyPI warnings on stderr - that must not abort the installer.
  $previousEap = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    $output = @(& $Exe @Arguments 2>&1)
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousEap
  }
  # Drop PowerShell's own decoration around native stderr lines (ASCII patterns
  # only: this file must stay pure ASCII, see the logging section above).
  $clean = $output | Where-Object {
    "$_" -notmatch 'CategoryInfo|FullyQualifiedErrorId|NativeCommandError|^\s*\+|^\s*At line'
  }
  return [pscustomobject]@{ Output = @($clean); ExitCode = $code }
}

function Test-VersionGreater([string]$A, [string]$B) {
  # $A > $B for dotted versions ("26.1.3" > "26.1.2").
  # Note: local names must not collide with the typed parameters - PowerShell is
  # case-insensitive, so assigning to $a would re-coerce the array back to the
  # [string]$A parameter.
  $partsA = @(($A -split '[.\-+]') | Where-Object { $_ -match '^\d+$' })
  $partsB = @(($B -split '[.\-+]') | Where-Object { $_ -match '^\d+$' })
  $count = [Math]::Max($partsA.Count, $partsB.Count)
  for ($i = 0; $i -lt $count; $i++) {
    $left = if ($i -lt $partsA.Count) { [int]$partsA[$i] } else { 0 }
    $right = if ($i -lt $partsB.Count) { [int]$partsB[$i] } else { 0 }
    if ($left -gt $right) { return $true }
    if ($left -lt $right) { return $false }
  }
  return $false
}

function Get-NpmProxyArgs {
  # npm does not read the Windows system proxy, so an intranet machine can have
  # working HTTP(S) access while npm still times out. Reuse the proxy from the
  # environment, or the WinHTTP proxy that PowerShell itself is using.
  $proxy = if ($env:HTTPS_PROXY) { $env:HTTPS_PROXY }
  elseif ($env:https_proxy) { $env:https_proxy }
  elseif ($env:HTTP_PROXY) { $env:HTTP_PROXY }
  elseif ($env:http_proxy) { $env:http_proxy }
  elseif ($env:ALL_PROXY) { $env:ALL_PROXY }
  else { $null }
  if (-not $proxy) {
    try {
      $netsh = (Invoke-CaptureOutput 'netsh.exe' @('winhttp', 'show', 'proxy')).Output | Out-String
      if ($netsh -match 'Proxy Server\(s\)\s*:\s*(\S+)') { $proxy = $Matches[1] }
    } catch {
      $proxy = $null
    }
  }
  if (-not $proxy) { return @() }
  if ($proxy -notmatch '^[a-zA-Z]+://') { $proxy = "http://$proxy" }
  return @('--proxy', $proxy, '--https-proxy', $proxy)
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
Write-Banner
Write-Phase 'Checking the environment and selecting sources'
$pipCheck = Get-Command pip -ErrorAction SilentlyContinue
if ($pipCheck) {
  # Note: with $ErrorActionPreference='Stop', PowerShell 5.1 turns redirected
  # native stderr into a terminating NativeCommandError, so guard this probe.
  $pipHasMsagent = $false
  try {
    & pip show mindstudio-agent 2>$null | Out-Null
    $pipHasMsagent = ($LASTEXITCODE -eq 0)
  } catch {
    $pipHasMsagent = $false
  }
  if ($pipHasMsagent) {
    Write-Warn "mindstudio-agent is already installed via pip in this environment."
    Write-Warn "The uv tool install below is isolated and will not modify it, but the"
    Write-Warn "two executables may shadow each other on PATH. Consider: pip uninstall mindstudio-agent"
  }
}

# ---------------------------------------------------------------------------
# Index selection: explicit overrides first, then a domestic mirror chain.
# ---------------------------------------------------------------------------
function Test-Url([string]$url) {
  try {
    Invoke-WebRequest -Uri $url -Method Head -UseBasicParsing -TimeoutSec 8 | Out-Null
    return $true
  } catch { }
  try {
    Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 8 | Out-Null
    return $true
  } catch { }
  return $false
}

function Select-Index {
  # Domestic mirrors first (fast in China), official PyPI as fallback,
  # generic env vars as last resort. Mirror sync may lag weekly releases,
  # so the install is retried with official PyPI on failure (see below),
  # and the latest version is pinned from PyPI beforehand.
  if ($env:MSAGENT_INDEX) {
    $script:MsagentIndex = $env:MSAGENT_INDEX
    Write-Step "Using index from MSAGENT_INDEX: $script:MsagentIndex"
    return
  }
  $candidates = @(
    'https://mirrors.huaweicloud.com/repository/pypi/simple',
    'https://mirrors.aliyun.com/pypi/simple',
    'https://pypi.tuna.tsinghua.edu.cn/simple',
    'https://pypi.org/simple'
  )
  foreach ($c in $candidates) {
    if (Test-Url ($c + '/pip/')) {
      $script:MsagentIndex = $c
      Write-Step "Selected PyPI index: $c"
      return
    }
    Write-Warn "Index unreachable: $c (trying the next candidate)"
  }
  # All candidates are unreachable; fall back to a user-configured
  # generic index if one exists, otherwise use official PyPI.
  if ($env:UV_DEFAULT_INDEX) {
    $script:MsagentIndex = $env:UV_DEFAULT_INDEX
    Write-Step "Fallback to UV_DEFAULT_INDEX: $script:MsagentIndex"
    return
  }
  if ($env:UV_INDEX_URL) {
    $script:MsagentIndex = $env:UV_INDEX_URL
    Write-Step "Fallback to UV_INDEX_URL: $script:MsagentIndex"
    return
  }
  if ($env:PIP_INDEX_URL) {
    $script:MsagentIndex = $env:PIP_INDEX_URL
    Write-Step "Fallback to PIP_INDEX_URL: $script:MsagentIndex"
    return
  }
  $script:MsagentIndex = 'https://pypi.org/simple'
  Write-Warn "No reachable index found; defaulting to $script:MsagentIndex."
}
Select-Index

# ---------------------------------------------------------------------------
# Resolve and announce the msagent version to install. Mirror sync may lag
# weekly releases, so the latest version is pinned straight from PyPI and
# shown up front; the install is retried with official PyPI if the selected
# mirror does not have that version yet.
# ---------------------------------------------------------------------------
# Resolve the version to install and announce it up front:
#   1) MSAGENT_VERSION when set;
#   2) otherwise the newest version PyPI reports;
#   3) otherwise the newest version the selected index serves, so the announced
#      version always matches what is actually installed.
# When nothing can be resolved, say so instead of claiming an unknown "latest".
# ---------------------------------------------------------------------------
function Get-LatestVersionFromIndex([string]$Index) {
  if (-not $Index) { return $null }
  try {
    $html = (Invoke-WebRequest -Uri ($Index.TrimEnd('/') + '/mindstudio-agent/') -UseBasicParsing -TimeoutSec 20).Content
  } catch {
    return $null
  }
  if (-not $html) { return $null }
  $versions = [regex]::Matches($html, 'mindstudio[_-]agent-([0-9][0-9A-Za-z.+-]*)\.(?:whl|tar\.gz)') |
    ForEach-Object { $_.Groups[1].Value -replace '-(py3|py2|cp3).*$', '' } |
    Sort-Object -Unique
  $best = $null
  foreach ($candidate in $versions) {
    if (-not $best) { $best = $candidate; continue }
    try { if ([version]$candidate -gt [version]$best) { $best = $candidate } }
    catch { if ($candidate -gt $best) { $best = $candidate } }
  }
  return $best
}

$script:MsagentSpec = 'mindstudio-agent'
$latestVersion = $null
$versionSource = "latest from $($script:MsagentIndex)"
if ($env:MSAGENT_VERSION) {
  $latestVersion = $env:MSAGENT_VERSION
  $versionSource = 'pinned by MSAGENT_VERSION'
} else {
  if (-not $env:MSAGENT_INDEX) {
    try {
      $latestVersion = (Invoke-RestMethod -Uri 'https://pypi.org/pypi/mindstudio-agent/json' -UseBasicParsing -TimeoutSec 15).info.version
    } catch {
      $latestVersion = $null
    }
  }
  if (-not $latestVersion) { $latestVersion = Get-LatestVersionFromIndex $script:MsagentIndex }
}
if ($latestVersion) {
  $script:MsagentSpec = 'mindstudio-agent==' + $latestVersion
  Write-Key "Will install msagent $latestVersion ($versionSource)"
} else {
  Write-Warn "Could not resolve a version number from $($script:MsagentIndex); the newest build that source provides will be installed."
}
$script:Summary['PyPI index'] = $script:MsagentIndex

# Record what is currently installed, so the summary can report an update
# instead of a bare version ("26.1.2 -> 26.1.3").
$script:Summary['msagent'] = 'not installed yet'
$prevMsagent = Get-Command msagent -ErrorAction SilentlyContinue
if ($prevMsagent) {
  # The CLI prints a banner (and warnings on stderr) before the version line.
  $script:Summary['msagent'] = (Get-MsagentVersionToken (Invoke-CaptureOutput $prevMsagent.Source @('--version')).Output)
  if (-not $script:Summary['msagent']) { $script:Summary['msagent'] = 'not installed yet' }
}
$script:PreviousVersion = if ($script:Summary['msagent'] -eq 'not installed yet') { $null } else { $script:Summary['msagent'] }

# ---------------------------------------------------------------------------
# uv bootstrap: existing uv, official installer, then pip as fallback.
# ---------------------------------------------------------------------------
Write-Phase 'Bootstrapping uv'
$script:UvType = $null          # 'bin' or 'module'
$script:UvPath = $null
$script:UvModulePrefix = $null

function Invoke-Uv {
  param([string[]]$UvArgs)
  # Native stderr redirects become terminating errors under 'Stop' in
  # PowerShell 5.1; run native calls with 'Continue' and check exit codes.
  $oldPref = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    if ($script:UvType -eq 'bin') {
      & $script:UvPath @UvArgs
    } else {
      $head = $script:UvModulePrefix[0]
      $tail = @($script:UvModulePrefix[1..($script:UvModulePrefix.Count - 1)])
      & $head @tail @UvArgs
    }
  } finally {
    $ErrorActionPreference = $oldPref
  }
}

function Get-Uv {
  $cmd = Get-Command uv -ErrorAction SilentlyContinue
  if ($cmd) {
    $script:UvType = 'bin'
    $script:UvPath = $cmd.Source
    return
  }
  $fixed = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
  if (Test-Path $fixed) {
    $script:UvType = 'bin'
    $script:UvPath = $fixed
    return
  }

  Write-Step 'uv is not installed. Bootstrapping uv...'
  try {
    Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' -UseBasicParsing -TimeoutSec 60 | Invoke-Expression
  } catch {
    Write-Warn ('Official uv installer failed: ' + $_.Exception.Message)
  }
  if (Test-Path $fixed) {
    $script:UvType = 'bin'
    $script:UvPath = $fixed
    return
  }
  $cmd = Get-Command uv -ErrorAction SilentlyContinue
  if ($cmd) {
    $script:UvType = 'bin'
    $script:UvPath = $cmd.Source
    return
  }

  Write-Warn "Official uv installer failed or timed out; trying pip + $script:MsagentIndex ..."
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    $script:UvModulePrefix = @('py', '-3', '-m', 'uv')
    & py -3 -m pip install -q -U uv -i $script:MsagentIndex
    if ($LASTEXITCODE -ne 0) {
      Die "uv could not be installed. Install it manually (irm https://astral.sh/uv/install.ps1 | iex), then re-run this installer."
    }
  } else {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
      Die 'python is required to bootstrap uv via pip. Install Python 3.11+, or install uv manually, then re-run.'
    }
    $script:UvModulePrefix = @('python', '-m', 'uv')
    & python -m pip install -q -U uv -i $script:MsagentIndex
    if ($LASTEXITCODE -ne 0) {
      Die "uv could not be installed. Install it manually (irm https://astral.sh/uv/install.ps1 | iex), then re-run this installer."
    }
  }
  # Prefer the uv.exe binary if pip put it somewhere discoverable.
  $scripted = Get-ChildItem (Join-Path $env:APPDATA 'Python\Python*\Scripts\uv.exe') -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($scripted) {
    $script:UvType = 'bin'
    $script:UvPath = $scripted.FullName
  } else {
    $script:UvType = 'module'
  }
}
Get-Uv

# ---------------------------------------------------------------------------
# Use the platform's native TLS store (Windows Schannel / system CA bundle).
# Corporate proxies re-sign HTTPS with their own CA: system tools trust it,
# but uv's bundled roots reject it with "UnknownIssuer". Only an explicitly
# set UV_NATIVE_TLS wins.
# ---------------------------------------------------------------------------
if (-not $env:UV_NATIVE_TLS) {
  $env:UV_NATIVE_TLS = '1'
  Write-Step 'UV_NATIVE_TLS=1 (use the system certificate store for uv)'
}

# ---------------------------------------------------------------------------
# Managed CPython downloads: try Huawei Cloud first, then NJU, but only when
# the mirror actually serves real binary content with a trusted certificate
# (some public github-release mirrors answer HTML with HTTP 200, and corporate
# proxies may reject certificates). Otherwise leave uv's defaults (official
# source or an existing local Python) to handle it.
# ---------------------------------------------------------------------------
function Get-PythonCanary([string]$mirror) {
  # Discover one real python-build-standalone file from the mirror's own
  # directory listing (newest tag) so no tag/version is hardcoded.
  $platform = if ($env:PROCESSOR_ARCHITECTURE -match 'ARM64|ARM') { 'aarch64-pc-windows-msvc' } else { 'x86_64-pc-windows-msvc' }
  try {
    $top = (Invoke-WebRequest -Uri ($mirror + '/') -UseBasicParsing -TimeoutSec 15).Content
    $tags = @([regex]::Matches($top, '[0-9]{8}/') | ForEach-Object { $_.Value.TrimEnd('/') })
    if ($tags.Count -eq 0) { return $null }
    $tag = $tags[$tags.Count - 1]
    $dir = (Invoke-WebRequest -Uri ($mirror + '/' + $tag + '/') -UseBasicParsing -TimeoutSec 15).Content
    $pattern = 'cpython-[0-9.]+%2B[0-9]{8}-' + $platform + '-install_only[^"<]*\.tar\.gz'
    $file = [regex]::Match($dir, $pattern).Value
    if (-not $file) { return $null }
    return ($tag + '/' + $file)
  } catch {
    return $null
  }
}

function Test-PythonMirror([string]$mirror) {
  $canary = Get-PythonCanary $mirror
  if (-not $canary) { return $false }
  try {
    $resp = Invoke-WebRequest -Uri ($mirror + '/' + $canary) -Method Head -UseBasicParsing -TimeoutSec 12
    if ($resp.StatusCode -eq 200 -and $resp.Headers['Content-Type'] -notmatch 'text/html') {
      return $true
    }
  } catch { }
  return $false
}

if (-not $env:UV_PYTHON_INSTALL_MIRROR) {
  $pyMirrorOk = $false
  foreach ($mirror in @(
    'https://mirrors.huaweicloud.com/github-release/astral-sh/python-build-standalone',
    'https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone'
  )) {
    if (Test-PythonMirror $mirror) {
      $env:UV_PYTHON_INSTALL_MIRROR = $mirror
      Write-Step "UV_PYTHON_INSTALL_MIRROR=$env:UV_PYTHON_INSTALL_MIRROR (set it yourself to override)"
      $pyMirrorOk = $true
      break
    }
    Write-Warn "Python mirror unreachable or not serving binary content: $mirror"
  }
  if (-not $pyMirrorOk) {
    Write-Warn 'Python download mirrors are not usable with a trusted certificate;'
    Write-Warn "using uv's default Python source or an existing local Python."
    Write-Warn 'To use an internal mirror, set UV_PYTHON_INSTALL_MIRROR yourself.'
    Write-Warn 'If the proxy still rejects the mirror certificate, set'
    Write-Warn '  UV_INSECURE_HOST=<mirror host>   (insecure, last resort)'
  }
}

# ---------------------------------------------------------------------------
# PATH helper
# ---------------------------------------------------------------------------
function Add-ToUserPath([string]$dir) {
  if ($env:MSAGENT_NO_MODIFY_PATH) {
    Write-Warn "Skipping PATH update (MSAGENT_NO_MODIFY_PATH set). Add $dir to PATH yourself."
    return
  }
  $current = [Environment]::GetEnvironmentVariable('Path', 'User')
  if (-not $current) {
    [Environment]::SetEnvironmentVariable('Path', $dir, 'User')
    Write-Success "Added $dir to the user PATH (open a new terminal to use msagent)."
    return
  }
  $parts = @($current.Split(';') | Where-Object { $_ -ne '' })
  if ($parts -contains $dir) {
    Write-Step "$dir is already on the user PATH."
    return
  }
  # Prepend so the freshly installed msagent wins over older pip installs.
  [Environment]::SetEnvironmentVariable('Path', ($dir + ';' + $current.TrimStart(';')), 'User')
  Write-Success "Added $dir to the user PATH (prepended; open a new terminal to use msagent)."
}

function Get-ToolBinDir {
  $out = Invoke-Uv @('tool', 'dir', '--bin') 2>$null
  $line = $out | Select-Object -Last 1
  if ($line) {
    return $line.Trim()
  }
  return (Join-Path $env:USERPROFILE '.local\bin')
}

# ---------------------------------------------------------------------------
# venv fallback (last resort when uv tool install fails)
# ---------------------------------------------------------------------------
function Invoke-VenvFallback {
  $venvDir = if ($env:MSAGENT_FALLBACK_VENV) { $env:MSAGENT_FALLBACK_VENV } else { Join-Path $env:USERPROFILE '.msagent-venv' }
  $py = Get-Command py -ErrorAction SilentlyContinue
  if (-not $py) {
    Write-Warn 'py launcher not found; cannot create the fallback venv.'
    return $false
  }
  Write-Step "Creating isolated venv at $venvDir ..."
  if (-not (Test-Path $venvDir)) {
    & py -3.11 -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
      & py -3 -m venv $venvDir
    }
    if ($LASTEXITCODE -ne 0) {
      Write-Warn 'Could not create the fallback venv.'
      return $false
    }
  }
  $pip = Join-Path $venvDir 'Scripts\pip.exe'
  if (-not (Test-Path $pip)) {
    Write-Warn 'pip not found inside the fallback venv.'
    return $false
  }
  Write-Step "Installing $script:MsagentSpec from $script:MsagentIndex into the venv..."
  & $pip install -q -U -i $script:MsagentIndex $script:MsagentSpec
  if ($LASTEXITCODE -ne 0) {
    Write-Warn 'The venv fallback install failed.'
    return $false
  }
  $script:ToolBin = Join-Path $venvDir 'Scripts'
  Add-ToUserPath $script:ToolBin
  return $true
}

# uv sometimes only links the primary tool's executables into the tool bin
# directory. The msprof-mcp MCP server is spawned by name from PATH, so make
# sure its executable (and msprof-analyze) are reachable. Best effort only:
# create a small .cmd shim that forwards to the real executable in the tool env.
function Expose-ToolExecutables {
  if (-not $script:ToolBin) { return }
  $toolsRoot = Invoke-Uv @('tool', 'dir') 2>$null | Select-Object -Last 1
  if (-not $toolsRoot) { $toolsRoot = Join-Path $env:USERPROFILE '.local\share\uv\tools' }
  $envBin = Join-Path $toolsRoot.Trim() 'mindstudio-agent\Scripts'
  if (-not (Test-Path $envBin)) { return }
  foreach ($exe in @('msprof-mcp', 'msprof-analyze')) {
    $real = Join-Path $envBin "$exe.exe"
    if (Test-Path $real) {
      $shim = Join-Path $script:ToolBin "$exe.cmd"
      if (-not (Test-Path $shim)) {
        "@echo off`r`n`"$real`" %*" | Set-Content -Path $shim -Encoding Ascii
        Write-Success "Exposed $exe via $shim"
      }
    }
  }
}

# ---------------------------------------------------------------------------
# Optional stage: provision a user-local Node.js (npmmirror mirror) and
# pre-install @opencxd/ascend-doc-mcp into %USERPROFILE%\.msagent\ascend-doc-mcp.
# The msagent ascend-doc-mcp launcher then runs the pre-installed copy directly,
# so no global Node/npx and no on-demand npm fetch are needed on first use.
#
#   MSAGENT_NO_ASCEND_DOC_MCP=1        skip this stage entirely
#   MSAGENT_ASCEND_DOC_MCP_REQUIRE=1   abort the installer if this stage fails
#   MSAGENT_NODE_HOME                  user-local Node root (default %USERPROFILE%\.msagent\node)
#   MSAGENT_NODE_MIRROR                Node dist mirror base (default: probed
#                                      huaweicloud -> npmmirror -> nodejs.org)
#   MSAGENT_NPM_REGISTRY               npm registry (default: the machine's
#                                      configured npm registry, then probed
#                                      npmmirror -> huaweicloud -> npmjs)
#   MSAGENT_NPM_CACHE                  npm cache dir
#   MSAGENT_ASCEND_DOC_MCP_PREFIX      local install prefix (%USERPROFILE%\.msagent\ascend-doc-mcp)
#
# Every mirror below is probed before use, so the stage works on the public
# internet and inside a corporate intranet (the Huawei Cloud mirrors are usually
# reachable where registry.npmmirror.com / registry.npmjs.org are not).
# ---------------------------------------------------------------------------
$script:NodeMinMajor = 22
$script:NodeMirrorCandidates = @(
  'https://mirrors.huaweicloud.com/nodejs',
  'https://registry.npmmirror.com/-/binary/node',
  'https://nodejs.org/dist'
)
$script:NpmRegistryCandidates = @(
  'https://registry.npmmirror.com',
  'https://mirrors.huaweicloud.com/repository/npm',
  'https://registry.npmjs.org'
)
# ---------------------------------------------------------------------------
# Huawei intranet npm mirror, used only as the LAST-RESORT candidate:
#   * it comes after the public chain and any site-injected fallbacks, so it is
#     contacted only when everything else failed (i.e. inside the intranet);
#     public users never pay for it and it is not part of the probe used to pick
#     the preferred source;
#   * set it to '' to drop the built-in fallback (e.g. in a public build);
#   * change this single line when the intranet address changes;
#   * the generic, hostname-free alternative is MSAGENT_NPM_REGISTRY_FALLBACKS or
#     %USERPROFILE%\.msagent\npm-mirrors (see Get-ExtraNpmRegistries).
# ---------------------------------------------------------------------------
$script:IntranetNpmRegistry = 'http://cmc-cd-mirror.rnd.huawei.com/npm'
$script:HwCloudNodeMirror = 'https://mirrors.huaweicloud.com/nodejs'
$script:HwCloudNpmRegistry = 'https://mirrors.huaweicloud.com/repository/npm'

function Select-NodeMirror {
  # MSAGENT_NODE_MIRROR wins; otherwise probe the domestic mirrors first.
  if ($env:MSAGENT_NODE_MIRROR) { return $env:MSAGENT_NODE_MIRROR }
  foreach ($candidate in $script:NodeMirrorCandidates) {
    if (Test-Url ($candidate + '/latest-v22.x/')) { return $candidate }
    Write-Warn "Node mirror unreachable: $candidate (trying the next candidate)"
  }
  return $script:NodeMirrorCandidates[0]
}

function Test-PublicRegistry([string]$registry) {
  # Registries the probe chain below already covers; used to ignore an npm
  # configuration that only points at a default, keeping the public path clean.
  $normalized = $registry.TrimEnd('/')
  return @(
    'https://registry.npmjs.org',
    'http://registry.npmjs.org',
    'https://registry.npmmirror.com',
    'http://registry.npmmirror.com',
    'https://mirrors.huaweicloud.com/repository/npm',
    'http://mirrors.huaweicloud.com/repository/npm'
  ) -contains $normalized
}

function Get-ConfiguredNpmRegistry([string]$NpmCmd) {
  # The registry npm itself would use (env vars, project/user/global .npmrc).
  # Corporate intranets usually point npm at an internal mirror there, so
  # honoring it makes "use the intranet registry when we are on the intranet"
  # automatic without hardcoding any internal host in this script.
  if (-not $NpmCmd -or -not (Test-Path -LiteralPath $NpmCmd)) { return $null }
  $previousEap = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    $configured = (& $NpmCmd config get registry 2>$null | Select-Object -First 1)
  } catch {
    $configured = $null
  } finally {
    $ErrorActionPreference = $previousEap
  }
  if (-not $configured) { return $null }
  $configured = "$configured".Trim()
  if (-not $configured -or $configured -in @('undefined', 'null')) { return $null }
  if (Test-PublicRegistry $configured) { return $null }
  return $configured
}

function Get-ExtraNpmRegistries {
  # Extra fallback npm sources for corporate intranets / offline sites. The
  # script ships no internal hostname itself; the site injects them via:
  #   1) MSAGENT_NPM_REGISTRY_FALLBACKS (comma/semicolon/space separated)
  #   2) a config file, one source per line, '#' starts a comment:
  #        $env:MSAGENT_NPM_MIRRORS_FILE > %USERPROFILE%\.msagent\npm-mirrors
  #        > %ProgramData%\msagent\npm-mirrors
  # They are appended after the public chain, so a machine that can reach the
  # public mirrors never touches them.
  $extra = New-Object System.Collections.Generic.List[string]
  if ($env:MSAGENT_NPM_REGISTRY_FALLBACKS) {
    foreach ($part in ($env:MSAGENT_NPM_REGISTRY_FALLBACKS -split '[,;\s]+')) {
      $value = "$part".Trim()
      if ($value) { $extra.Add($value) }
    }
  }
  $files = @(
    $env:MSAGENT_NPM_MIRRORS_FILE,
    (Join-Path $HOME '.msagent\npm-mirrors'),
    (Join-Path $env:ProgramData 'msagent\npm-mirrors')
  ) | Where-Object { $_ }
  foreach ($file in $files) {
    if (-not (Test-Path -LiteralPath $file)) { continue }
    foreach ($line in (Get-Content -LiteralPath $file -ErrorAction SilentlyContinue)) {
      $value = ("$line" -replace '#.*$', '').Trim()
      if ($value) { $extra.Add($value) }
    }
  }
  return $extra
}

function Get-NpmRegistryCandidates([string]$NpmCmd) {
  # Ordered, de-duplicated registries worth trying:
  #   explicit -> the machine's own npm configuration -> the public chain
  #   -> site-injected fallbacks (see Get-ExtraNpmRegistries)
  #
  # MSAGENT_NPM_REGISTRY is a preference, not an exclusive choice: an intranet
  # mirror that is unreachable from the public internet must fall through to the
  # public mirrors. Set MSAGENT_NPM_REGISTRY_ONLY=1 to pin it to that one source.
  $explicit = $env:MSAGENT_NPM_REGISTRY
  if ($explicit -and $env:MSAGENT_NPM_REGISTRY_ONLY -eq '1') {
    $only = New-Object System.Collections.Generic.List[string]
    $only.Add($explicit)
    return $only
  }
  $configured = Get-ConfiguredNpmRegistry $NpmCmd
  $list = New-Object System.Collections.Generic.List[string]
  $seen = @{}
  foreach ($candidate in (@($explicit, $configured) + @($script:NpmRegistryCandidates) + @(Get-ExtraNpmRegistries) + @($script:IntranetNpmRegistry))) {
    if (-not $candidate) { continue }
    if ($seen.ContainsKey($candidate)) { continue }
    $seen[$candidate] = $true
    $list.Add($candidate)
  }
  return $list
}


function Get-NodeMajorVersion([string]$nodeExe) {
  # Major version of a node binary ("22" for v22.23.2); $null when unusable.
  try {
    $raw = & $nodeExe --version 2>$null
  } catch {
    return $null
  }
  if ("$raw" -match '^v(\d+)\.') { return [int]$Matches[1] }
  return $null
}

# Actionable fallback hints, printed only when the optional stage failed.
function Write-AscendDocMcpGuidance {
  Write-Warn "ascend-doc-mcp needs Node.js >= $($script:NodeMinMajor) and a reachable npm registry."
  Write-Warn 'Fix either one and re-run, or skip the feature with MSAGENT_NO_ASCEND_DOC_MCP=1:'
  Write-Warn "  Node : download node-v22.x.y-win-x64.zip from $($script:HwCloudNodeMirror)/latest-v22.x/"
  Write-Warn '         Expand-Archive node-v22.x.y-win-x64.zip -DestinationPath C:\node'
  Write-Warn '         setx PATH "C:\node\node-v22.x.y-win-x64;%PATH%"'
  Write-Warn "  npm  : `$env:MSAGENT_NPM_REGISTRY = '$($script:HwCloudNpmRegistry)'"
}

function Get-NodeArchiveCandidates([string]$Listing, [string]$Platform) {
  # Node archive names from a mirror directory listing, newest version first.
  # A directory carries several patch releases, so callers can step down to the
  # next version when one archive is missing or corrupt.
  if (-not $Listing -or -not $Platform) { return @() }
  $rx = [regex]::new('node-v(\d+\.\d+\.\d+)-' + $Platform + '\.zip')
  $parsed = foreach ($match in $rx.Matches($Listing)) {
    $version = $null
    try { $version = [version]::new($match.Groups[1].Value) } catch { $version = $null }
    if ($version) { [pscustomobject]@{ Name = $match.Groups[0].Value; Version = $version } }
  }
  return @($parsed | Sort-Object -Property Version -Descending | ForEach-Object { $_.Name } | Select-Object -Unique)
}

function Install-UserNode([string]$base) {
  # Downloads the newest Node 22 LTS win-x64 zip from the given mirror base
  # into the user-local Node home and returns its path on success.
  if (-not $base) { return $null }
  $nodeHome = $script:MsagentNodeHome
  $platform = if ([Environment]::Is64BitOperatingSystem) { 'win-x64' } else { 'win-x86' }
  $base = $base.TrimEnd('/')
  try {
    $listing = (Invoke-WebRequest -Uri ($base + '/latest-v22.x/') -UseBasicParsing -TimeoutSec 25).Content
  } catch {
    return $null
  }
  # Rank newest-first and try in order: one missing or corrupt archive then falls
  # back to the next version instead of failing the whole stage.
  $candidates = @(Get-NodeArchiveCandidates $listing $platform)
  if (-not $candidates) { return $null }
  $tmp = Join-Path ([IO.Path]::GetTempPath()) ("msagent-node-" + [guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Path $tmp | Out-Null
  $installed = $false
  foreach ($candidate in $candidates) {
    $url = $base + '/latest-v22.x/' + $candidate
    $script:NodeTarballUrl = $url
    Write-Step "Downloading $candidate from $base..."
    $zip = Join-Path $tmp $candidate
    try {
      Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -TimeoutSec 600
      Expand-Archive -Path $zip -DestinationPath $tmp
      $nodeRoot = Get-ChildItem -Path $tmp -Directory -Filter 'node-v*' | Select-Object -First 1
      if (-not $nodeRoot) { throw 'node archive did not contain a node-v* directory' }
      if (Test-Path $nodeHome) { Remove-Item -Recurse -Force $nodeHome }
      New-Item -ItemType Directory -Path $nodeHome | Out-Null
      Get-ChildItem -Path $nodeRoot.FullName | Move-Item -Destination $nodeHome
      $installed = $true
      break
    } catch {
      Write-Warn "Node $candidate could not be used: $($_.Exception.Message)"
      Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
      Get-ChildItem -Path $tmp -Directory -Filter 'node-v*' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
  if (-not $installed) { return $null }
  if (-not (Test-Path (Join-Path $nodeHome 'node.exe'))) { return $null }
  return $nodeHome
}

function Get-NpmCacheDir {
  # Returns a writable npm cache directory. The default location may be owned by
  # another account (for example created by an elevated shell), in which case npm
  # fails outright, so fall back to a location this user can write.
  $candidates = @(
    $env:MSAGENT_NPM_CACHE,
    (Join-Path $HOME '.cache\msagent\npm-cache'),
    (Join-Path $HOME '.msagent\npm-cache'),
    (Join-Path ([IO.Path]::GetTempPath()) 'msagent-npm-cache')
  ) | Where-Object { $_ }
  foreach ($candidate in $candidates) {
    try {
      New-Item -ItemType Directory -Force -Path $candidate -ErrorAction Stop | Out-Null
      $probe = Join-Path $candidate ('.msagent-write-test-' + [guid]::NewGuid().ToString('N'))
      [IO.File]::WriteAllText($probe, '')
      Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
      return $candidate
    } catch {
      continue
    }
  }
  return $null
}

function Get-NpmMetadataVersion([string]$Registry) {
  # Newest version of @opencxd/ascend-doc-mcp on that registry, or $null.
  if (-not $Registry) { return $null }
  try {
    $response = Invoke-WebRequest -Uri ($Registry.TrimEnd('/') + '/@opencxd%2Fascend-doc-mcp/latest') -UseBasicParsing -TimeoutSec 6
    if ($response.Content -match '"version"\s*:\s*"([^"]+)"') { return $Matches[1] }
  } catch { }
  return $null
}

function Get-ReachableNpmRegistries([string[]]$Candidates) {
  # Probe every candidate (in parallel, ~6s cap) and return the reachable ones in
  # candidate order, each with the version it reports. Probing first matters on a
  # corporate intranet: otherwise every unreachable public mirror burns npm's own
  # timeout before the intranet mirror is reached.
  $found = [ordered]@{}
  $parallel = $true
  try {
    Add-Type -AssemblyName System.Net.Http -ErrorAction Stop
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.UseProxy = $true
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(6)
  } catch {
    $parallel = $false
  }

  if ($parallel) {
    try {
      $tasks = [ordered]@{}
      foreach ($candidate in $Candidates) {
        if (-not $candidate) { continue }
        try {
          $tasks[$candidate] = $client.GetStringAsync($candidate.TrimEnd('/') + '/@opencxd%2Fascend-doc-mcp/latest')
        } catch {
          # invalid URI and similar: treat as unreachable
        }
      }
      foreach ($candidate in @($tasks.Keys)) {
        try {
          $body = $tasks[$candidate].GetAwaiter().GetResult()
          if ($body -match '"version"\s*:\s*"([^"]+)"') { $found[$candidate] = $Matches[1] }
        } catch { }
      }
    } finally {
      $client.Dispose()
    }
  } else {
    # Constrained environments may not allow HttpClient; fall back to serial.
    foreach ($candidate in $Candidates) {
      $version = Get-NpmMetadataVersion $candidate
      if ($version) { $found[$candidate] = $version }
    }
  }

  $list = New-Object System.Collections.Generic.List[object]
  foreach ($candidate in @($found.Keys)) {
    $list.Add([pscustomobject]@{ Registry = $candidate; Version = $found[$candidate] })
  }
  return $list
}

function Prepare-AscendDocMcp {
  if ($env:MSAGENT_NO_ASCEND_DOC_MCP) {
    Write-Step 'Skipping ascend-doc-mcp preparation (MSAGENT_NO_ASCEND_DOC_MCP is set).'
    $script:Summary['ascend-doc-mcp'] = 'skipped (MSAGENT_NO_ASCEND_DOC_MCP=1)'
    $script:SummaryColours['ascend-doc-mcp'] = 'DarkGray'
    return $true
  }
  $script:MsagentNodeHome = if ($env:MSAGENT_NODE_HOME) { $env:MSAGENT_NODE_HOME } else { Join-Path $HOME '.msagent\node' }
  $nodeHome = $script:MsagentNodeHome
  $nodeBinDir = $null
  $nodeExe = Join-Path $nodeHome 'node.exe'
  if (Test-Path $nodeExe) {
    $nodeBinDir = $nodeHome
  } else {
    $systemNode = Get-Command node -ErrorAction SilentlyContinue
    if ($systemNode) {
      $systemNodeExe = $systemNode.Source
      $systemMajor = Get-NodeMajorVersion $systemNodeExe
      if ($null -ne $systemMajor -and $systemMajor -lt $script:NodeMinMajor) {
        Write-Warn "Found Node $(& $systemNodeExe --version), but ascend-doc-mcp needs >= $($script:NodeMinMajor); provisioning a user-local Node."
      } else {
        $nodeBinDir = Split-Path $systemNodeExe -Parent
      }
    }
  }
  if (-not $nodeBinDir) {
    $nodeMirror = Select-NodeMirror
    Write-Step "Downloading a user-local Node $($script:NodeMinMajor) LTS from $nodeMirror ..."
    $nodeHome = Install-UserNode $nodeMirror
    if (-not $nodeHome) {
      Write-Warn "Could not provision Node.js from $nodeMirror."
      if ($script:NodeTarballUrl) {
        Write-Warn "The archive is reachable through a browser/downloader: $($script:NodeTarballUrl)"
      }
      return $false
    }
    $nodeBinDir = $nodeHome
    Write-Success "Provisioned user-local Node $(& (Join-Path $nodeHome 'node.exe') --version) at $nodeHome."
    Write-Step "Use it in your own shell with: set PATH=$nodeHome;%PATH%"
  } else {
    Write-Step "Using Node $(& (Join-Path $nodeBinDir 'node.exe') --version) ($nodeBinDir)."
  }
  if ($env:PATH -notlike "*$nodeBinDir*") {
    $env:PATH = $nodeBinDir + [IO.Path]::PathSeparator + $env:PATH
  }
  $script:Summary['Node.js'] = (& (Join-Path $nodeBinDir 'node.exe') --version)
  # Prefer npm.cmd: PowerShell may otherwise resolve 'npm' to npm.ps1, which
  # the machine ExecutionPolicy can block. npm.cmd runs through cmd.exe and is
  # independent of the PowerShell ExecutionPolicy.
  $npmCmd = Join-Path $nodeBinDir 'npm.cmd'
  if (-not (Test-Path -LiteralPath $npmCmd)) {
    $npmFromPath = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($npmFromPath) { $npmCmd = $npmFromPath.Source }
  }
  if (-not $npmCmd -or -not (Test-Path -LiteralPath $npmCmd)) {
    Write-Warn 'npm.cmd is unavailable in the provisioned Node; skipping the ascend-doc-mcp pre-install.'
    return $false
  }
  # Probe every candidate first (in parallel, ~6s cap) and install only on the
  # sources that answered: on an intranet, trying npm against each unreachable
  # public mirror serially costs minutes, while probing costs one timeout period.
  # The probe request is the package metadata request, so it also yields the
  # version to install.
  $candidates = @(Get-NpmRegistryCandidates $npmCmd)
  Write-Step "Probing $($candidates.Count) npm sources (in parallel, up to 6s each)..."
  $reachable = @(Get-ReachableNpmRegistries $candidates)
  if ($reachable.Count -gt 0) {
    $registry = $reachable[0].Registry
    $packageVersion = $reachable[0].Version
    $attempts = @($reachable | ForEach-Object { $_.Registry })
    Write-Step ("Reachable npm sources: " + ($attempts -join ', '))
    $configuredRegistry = Get-ConfiguredNpmRegistry $npmCmd
    if ($configuredRegistry -and $registry -eq $configuredRegistry) {
      Write-Step "Using the machine's configured npm registry: $registry"
    }
  } else {
    # A registry may serve packages without answering metadata requests, so fall
    # back to trying each candidate in order (the pre-probe behaviour).
    Write-Warn 'No npm source answered the metadata probe; falling back to trying each source in order.'
    $attempts = $candidates
    $registry = if ($candidates.Count -gt 0) { $candidates[0] } else { $null }
    $packageVersion = $null
  }
  $packageSpec = if ($packageVersion) { "@opencxd/ascend-doc-mcp@$packageVersion" } else { '@opencxd/ascend-doc-mcp@latest' }
  $prefix = if ($env:MSAGENT_ASCEND_DOC_MCP_PREFIX) { $env:MSAGENT_ASCEND_DOC_MCP_PREFIX } else { Join-Path $HOME '.msagent\ascend-doc-mcp' }
  # npm fails outright when its cache directory is not writable (common when the
  # directory was created by an elevated shell), so probe first and move on.
  $defaultCache = if ($env:MSAGENT_NPM_CACHE) { $env:MSAGENT_NPM_CACHE } else { Join-Path $HOME '.cache\msagent\npm-cache' }
  $cache = Get-NpmCacheDir
  if (-not $cache) {
    $cache = Join-Path ([IO.Path]::GetTempPath()) 'msagent-npm-cache'
    New-Item -ItemType Directory -Force -Path $cache -ErrorAction SilentlyContinue | Out-Null
    Write-Warn "No writable npm cache directory; using $cache."
  } elseif ($cache -ne $defaultCache) {
    Write-Warn "npm cache directory is not writable ($defaultCache); using $cache."
  }
  New-Item -ItemType Directory -Force -Path $prefix | Out-Null
  # Install on the probed-and-reachable sources (normally just one). If nothing
  # answered the probe, $attempts holds every candidate and we keep the old
  # fall-through behaviour.
  $installed = $false
  $lastFailed = $null
  $npmProxyArgs = Get-NpmProxyArgs
  if ($npmProxyArgs.Count -gt 0) {
    Write-Step "npm will use the system HTTP proxy: $($npmProxyArgs[1])"
  }
  $npmLog = Join-Path ([IO.Path]::GetTempPath()) ("msagent-npm-" + [guid]::NewGuid().ToString('N') + '.log')
  foreach ($candidate in $attempts) {
    if (-not $candidate) { continue }
    Write-Step "Pre-installing $packageSpec (registry: $candidate)..."
    # npm writes notices/warnings to stderr. Under $ErrorActionPreference='Stop'
    # those become terminating NativeCommandError records, so temporarily relax
    # EAP and merge stderr into the pipeline before capturing it.
    $npmExitCode = $null
    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
      & $npmCmd install --prefix $prefix --registry $candidate --cache $cache @npmProxyArgs --fetch-retries=1 --no-audit --no-fund --no-package-lock $packageSpec 2>&1 | Out-File -FilePath $npmLog -Encoding utf8
      $npmExitCode = $LASTEXITCODE
    } finally {
      $ErrorActionPreference = $previousEap
    }
    if ($npmExitCode -eq 0 -and (Test-Path (Join-Path $prefix 'node_modules\@opencxd\ascend-doc-mcp\package.json'))) {
      $registry = $candidate
      $installed = $true
      break
    }
    $lastFailed = $candidate
    Write-Warn "npm pre-install failed on $candidate (exit $npmExitCode); trying the next source."
  }
  if (-not $installed) {
    Write-Warn "ascend-doc-mcp could not be pre-installed from any source (last: $lastFailed)."
    # The npm error itself is the only useful diagnostic when a proxy, a
    # corporate CA, or a blocked host is the real cause: show the last tail once.
    # Show the real error only: npm's log-file / --loglevel / chown hints would
    # otherwise bury the network or certificate failure that actually matters.
    if (Test-Path -LiteralPath $npmLog) {
      $tail = Get-Content -LiteralPath $npmLog |
        Where-Object { "$_" -match 'npm (error|ERR!|warn)' } |
        Where-Object { "$_" -notmatch 'Log files were not written|loglevel=verbose|A complete log|sudo chown' } |
        Select-Object -First 3
      foreach ($line in $tail) { Write-Warn "  npm: $line" }
    }
    Remove-Item -LiteralPath $npmLog -Force -ErrorAction SilentlyContinue
    Write-Warn 'On a corporate intranet point npm at the internal mirror (any one of these), then re-run:'
    Write-Warn '  npm config set registry <internal npm registry>        # permanent, recommended'
    Write-Warn "  `$env:MSAGENT_NPM_REGISTRY = '<internal npm registry>' # this session; use setx to persist"
    Write-Warn '  or have IT drop %USERPROFILE%\.msagent\npm-mirrors or %ProgramData%\msagent\npm-mirrors'
    Write-Warn 'Behind a proxy, set HTTPS_PROXY or: npm config set proxy/https-proxy <proxy>.'
    Write-Warn 'The launcher will retry on demand; it uses MSAGENT_NPM_REGISTRY too.'
    return $false
  }
  Remove-Item -LiteralPath $npmLog -Force -ErrorAction SilentlyContinue
  Write-Success "ascend-doc-mcp pre-installed locally at $prefix."
  if (Test-Path $nodeExe) {
    $nodeVersion = & (Join-Path $nodeHome 'node.exe') --version
    Write-Step "Node version: $nodeVersion"
  }
  $resolved = if ($packageVersion) { "v$packageVersion" } else { 'latest' }
  $script:Summary['ascend-doc-mcp'] = "$resolved at $prefix"
  $script:SummaryColours['ascend-doc-mcp'] = 'Green'
  $script:Summary['npm registry'] = $registry
  return $true
}

# ---------------------------------------------------------------------------
# Install mindstudio-agent as an isolated uv tool (always latest, no version pin)
# ---------------------------------------------------------------------------
Write-Phase 'Installing mindstudio-agent'
# Prefer any existing local Python >= 3.11 and < 3.15 so a managed CPython is only
# downloaded when needed (downloads may fail on restricted networks).
# Set MSAGENT_PYTHON to pin an exact version, e.g. "3.11".
$pythonVersion = if ($env:MSAGENT_PYTHON) { $env:MSAGENT_PYTHON } else { '>=3.11,<3.15' }

# Never silently downgrade: a machine may carry a newer local/dev build while the
# index still serves an older release. Keep the newer one unless asked to force.
$skipInstall = $false
if ($latestVersion -and $script:PreviousVersion -and -not $env:MSAGENT_VERSION -and
    (Test-VersionGreater $script:PreviousVersion $latestVersion)) {
  Write-Warn "Installed msagent $script:PreviousVersion is newer than $latestVersion on the index; skipping the install to avoid a downgrade."
  Write-Warn "To force $latestVersion anyway: set MSAGENT_VERSION=$latestVersion and re-run."
  $skipInstall = $true
} elseif (Get-Command msagent -ErrorAction SilentlyContinue) {
  Write-Step 'Updating existing msagent installation...'
} else {
  Write-Step "Installing $script:MsagentSpec into an isolated uv tool environment..."
}

$installArgs = @('tool', 'install', '-U', '--python', $pythonVersion, '--default-index', $script:MsagentIndex)
$withExe = if ($env:MSAGENT_WITH_EXECUTABLES_FROM) { $env:MSAGENT_WITH_EXECUTABLES_FROM } else { 'msprof-mcp' }
$installArgs += @('--with-executables-from', $withExe)
$installArgs += $script:MsagentSpec

if ($skipInstall) {
  $installOk = $true
} else {
  Invoke-Uv $installArgs
  $installOk = ($LASTEXITCODE -eq 0)
}

# Mirror sync may lag the latest weekly release; retry with official PyPI.
if (-not $installOk -and -not $env:MSAGENT_INDEX -and $script:MsagentIndex -ne 'https://pypi.org/simple') {
  Write-Warn "Install failed with $script:MsagentIndex; retrying once with official PyPI (mirror sync may lag)..."
  $installArgs = @('tool', 'install', '-U', '--python', $pythonVersion, '--default-index', 'https://pypi.org/simple')
  $installArgs += @('--with-executables-from', $withExe)
  $installArgs += $script:MsagentSpec
  Invoke-Uv $installArgs
  $installOk = ($LASTEXITCODE -eq 0)
}

if (-not $installOk) {
  # Retry once with a freshly upgraded uv: a pre-existing uv may be old or in
  # a broken state (e.g. failing managed-Python resolution on Windows).
  Write-Warn 'First install attempt failed; upgrading uv and retrying once...'
  $retryRunner = $null
  $pyRetry = Get-Command py -ErrorAction SilentlyContinue
  if ($pyRetry) {
    & py -3 -m pip install -q -U uv -i $script:MsagentIndex | Out-Null
    $retryRunner = @('py', '-3', '-m', 'uv')
  } else {
    $pythonRetry = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonRetry) {
      & python -m pip install -q -U uv -i $script:MsagentIndex | Out-Null
      $retryRunner = @('python', '-m', 'uv')
    }
  }
  if ($retryRunner) {
    $script:UvType = 'module'
    $script:UvModulePrefix = $retryRunner
    Invoke-Uv $installArgs
    $installOk = ($LASTEXITCODE -eq 0)
  }
}

if (-not $installOk) {
  # The tool may exist in a broken/stale state (e.g. its Python interpreter
  # no longer matches). Remove it and retry the install from scratch once.
  Write-Warn 'Install still failing; removing any existing msagent tool and retrying once more...'
  Invoke-Uv @('tool', 'uninstall', 'mindstudio-agent') | Out-Null
  Invoke-Uv $installArgs
  $installOk = ($LASTEXITCODE -eq 0)
}

if (-not $installOk) {
  Write-Host 'error: uv tool install failed. See the output above.' -ForegroundColor Red
  if ($env:MSAGENT_NO_FALLBACK) {
    Die "MSAGENT_NO_FALLBACK is set; skipping the venv fallback. Retry after fixing the issue, or install manually into a venv."
  }
  $useFallback = if ($env:MSAGENT_YES -eq '1') { $true } else { $false }
  if (-not $useFallback) {
    $answer = Read-Host 'Try the isolated venv fallback instead? [y/N]'
    $useFallback = ($answer -match '^(y|yes)$')
  }
  if ($useFallback) {
    if (-not (Invoke-VenvFallback)) {
      Die 'The venv fallback also failed. Please retry later or open an issue at https://gitcode.com/Ascend/msagent/issues'
    }
  } else {
    Die 'Install aborted. You can retry with: irm https://raw.gitcode.com/Ascend/msagent/raw/master/scripts/install.ps1 | iex'
  }
} else {
  Write-Phase 'Configuring PATH'
  $script:ToolBin = Get-ToolBinDir
  Add-ToUserPath $script:ToolBin
}
if (-not $script:ToolBin) { $script:ToolBin = Get-ToolBinDir }
$script:Summary['tool bin'] = $script:ToolBin
Expose-ToolExecutables

# Optional: provision Node and pre-install ascend-doc-mcp so the docs MCP works
# from the first run (domestic npm/node mirrors; best-effort unless required).
Write-Phase 'Preparing the Ascend docs MCP (Node.js + ascend-doc-mcp)'
if (-not (Prepare-AscendDocMcp)) {
  if ($env:MSAGENT_ASCEND_DOC_MCP_REQUIRE) {
    Die 'ascend-doc-mcp preparation failed and MSAGENT_ASCEND_DOC_MCP_REQUIRE is set.'
  }
  Write-Warn 'ascend-doc-mcp preparation did not complete (non-fatal).'
  Write-Warn 'msagent itself is installed and usable; only the Ascend docs query'
  Write-Warn '(the ascend-knowledge subagent) stays unavailable until this is fixed.'
  Write-AscendDocMcpGuidance
  $script:Summary['ascend-doc-mcp'] = 'unavailable (optional feature)'
  $script:SummaryColours['ascend-doc-mcp'] = 'Yellow'
}

# Make msagent available in the current session immediately (the user-level
# PATH change only affects newly started processes).
if ($env:PATH -notlike "*$script:ToolBin*") {
  $env:PATH = $script:ToolBin + [IO.Path]::PathSeparator + $env:PATH
  Write-Step "Updated PATH in this session; 'msagent' is ready now."
} else {
  Write-Step "'msagent' is already resolvable in this session."
}

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
Write-Phase 'Verifying the installation'
$msagentExe = Join-Path $script:ToolBin 'msagent.exe'
$versionToken = $null
$probe = $null
if (Test-Path $msagentExe) {
  $probe = Invoke-CaptureOutput $msagentExe @('--version')
} elseif (Get-Command msagent -ErrorAction SilentlyContinue) {
  $probe = Invoke-CaptureOutput 'msagent' @('--version')
} else {
  Write-Warn "msagent was installed but is not on PATH in this shell. Open a new terminal and run 'msagent --version'."
}
if ($probe) {
  if ($probe.ExitCode -ne 0) {
    $probe.Output | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    Die "msagent installed, but '--version' failed. Open a new terminal and re-run, or check your PATH."
  }
  $versionToken = Get-MsagentVersionToken $probe.Output
}
if ($versionToken) {
  Write-Key "Verified: msagent $versionToken"
  $script:Summary['msagent'] = if ($script:PreviousVersion -and $script:PreviousVersion -ne $versionToken) {
    "$versionToken (was $script:PreviousVersion)"
  } else {
    $versionToken
  }
  $script:SummaryColours['msagent'] = 'Green'
} else {
  $script:Summary['msagent'] = if ($script:PreviousVersion) {
    "installed (was $script:PreviousVersion; not on PATH in this shell)"
  } else {
    'installed (not on PATH in this shell)'
  }
  $script:SummaryColours['msagent'] = 'Yellow'
}

# Warn when an older pip-installed msagent shadows the uv tool on PATH.
$shadowMsagent = Get-Command msagent -ErrorAction SilentlyContinue
if ($shadowMsagent -and $shadowMsagent.Source -and $shadowMsagent.Source -ne $msagentExe) {
  Write-Warn "Note: 'msagent' on PATH resolves to $($shadowMsagent.Source) (possibly an older pip install)."
  Write-Warn "The uv tool install is at $msagentExe. Consider: pip uninstall mindstudio-agent"
}

$mcpExe = Join-Path $script:ToolBin 'msprof-mcp.exe'
$mcpShim = Join-Path $script:ToolBin 'msprof-mcp.cmd'
if ((Test-Path $mcpExe) -or (Test-Path $mcpShim)) {
  Write-Success 'msprof-mcp executable is available.'
  $script:Summary['msprof-mcp'] = 'OK'
  $script:SummaryColours['msprof-mcp'] = 'Green'
} else {
  Write-Warn 'msprof-mcp is not on PATH in this shell yet; restart your shell if msagent reports it missing.'
  $script:Summary['msprof-mcp'] = 'not on PATH yet'
  $script:SummaryColours['msprof-mcp'] = 'Yellow'
}

Write-Key 'msagent installation complete.'
Write-Summary
Write-NextSteps
if (-not $versionToken) {
  Write-Step "If 'msagent' is still not found (e.g. in cmd), open a new window or run: set PATH=$script:ToolBin;%PATH%"
}
