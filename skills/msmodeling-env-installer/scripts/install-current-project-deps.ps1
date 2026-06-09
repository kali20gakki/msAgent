param(
    [string]$EnvName = "myenv",
    [string]$PythonVersion = "",
    [switch]$UseExistingEnv,
    [switch]$SetProjectEnv,
    [switch]$UseHFMirror,
    [switch]$UseProjectUvCache = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PypiMirror = "https://mirrors.ustc.edu.cn/pypi/web/simple"
$MsmodelingRepoUrl = "https://gitcode.com/Ascend/msmodeling.git"

function Test-CommandExists {
    param([string]$Name)
    try {
        Get-Command $Name -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Resolve-PythonLauncher {
    if (Test-CommandExists "python") {
        return @("python")
    }
    if (Test-CommandExists "py") {
        return @("py", "-3")
    }
    throw "No Python launcher found. Install Python 3.10+ first."
}

function Invoke-Python {
    param(
        [string[]]$Launcher,
        [string[]]$PythonArgs
    )
    if ($Launcher.Count -eq 1) {
        & $Launcher[0] @PythonArgs
    } else {
        & $Launcher[0] $Launcher[1] @PythonArgs
    }
}

function Get-PythonScriptsPath {
    param([string[]]$Launcher)
    $scriptsPath = (Invoke-Python -Launcher $Launcher -PythonArgs @("-c", "import sysconfig; print(sysconfig.get_path('scripts'))")) | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($scriptsPath)) {
        return $null
    }
    return $scriptsPath.Trim()
}

function Get-PythonVersion {
    param([string[]]$Launcher)
    $versionText = (Invoke-Python -Launcher $Launcher -PythonArgs @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")) | Select-Object -First 1
    if (-not $versionText) {
        throw "Unable to detect Python version."
    }
    return [Version]($versionText.Trim())
}

function Resolve-UvCommand {
    param([string[]]$Launcher)

    $uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
    if ($uvCommand) {
        return $uvCommand.Source
    }

    $scriptsPath = Get-PythonScriptsPath -Launcher $Launcher
    if ($scriptsPath) {
        foreach ($fileName in @("uv.exe", "uv")) {
            $candidate = Join-Path $scriptsPath $fileName
            if (Test-Path $candidate) {
                return $candidate
            }
        }
    }

    throw "uv executable not found after installation. Ensure Python Scripts directory is on PATH or reinstall uv."
}

function Enable-ProjectUvCache {
    if (-not $UseProjectUvCache) {
        return
    }

    if (-not [string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) {
        Write-Host "Using existing UV_CACHE_DIR: $env:UV_CACHE_DIR"
        return
    }

    $cachePath = Join-Path (Resolve-Path ".").Path ".uv-cache"
    New-Item -ItemType Directory -Force -Path $cachePath | Out-Null
    $env:UV_CACHE_DIR = $cachePath
    Write-Host "UV_CACHE_DIR set for current session: $env:UV_CACHE_DIR"
}

function Confirm-MsmodelingRepoRoot {
    if ((Test-Path "README.md") -and (Test-Path "requirements.txt")) {
        return
    }

    if ((Test-Path "msmodeling\README.md") -and (Test-Path "msmodeling\requirements.txt")) {
        Write-Host "msmodeling repository found under .\msmodeling. Entering it..."
        Set-Location "msmodeling"
        return
    }

    if (Test-Path "msmodeling") {
        throw "Path .\msmodeling exists but does not look like the msmodeling repository root. Move it aside or run this script from a valid msmodeling repository root."
    }

    Write-Host "msmodeling repository root not found. Cloning from $MsmodelingRepoUrl ..."
    git clone $MsmodelingRepoUrl
    Set-Location "msmodeling"

    if ((-not (Test-Path "README.md")) -or (-not (Test-Path "requirements.txt"))) {
        throw "Clone finished, but README.md or requirements.txt is missing. Check repository contents."
    }
}

function Test-PythonModuleAvailable {
    param(
        [string[]]$Launcher,
        [string]$ModuleName
    )
    Invoke-Python -Launcher $Launcher -PythonArgs @("-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)") | Out-Null
    return $LASTEXITCODE -eq 0
}

function Test-PythonPackageInstalled {
    param(
        [string[]]$Launcher,
        [string]$PackageName
    )
    Invoke-Python -Launcher $Launcher -PythonArgs @("-m", "pip", "show", $PackageName) | Out-Null
    return $LASTEXITCODE -eq 0
}

function Assert-ExistingEnvironmentClean {
    param([string[]]$Launcher)

    $blockedPackages = @()
    if (Test-PythonModuleAvailable -Launcher $Launcher -ModuleName "torch_npu") {
        $blockedPackages += "torch_npu"
    }

    foreach ($packageName in @("torch-npu", "torch_npu", "cudatoolkit")) {
        if (Test-PythonPackageInstalled -Launcher $Launcher -PackageName $packageName) {
            $blockedPackages += $packageName
        }
    }

    $blockedPackages = @($blockedPackages | Select-Object -Unique)
    if ($blockedPackages.Count -gt 0) {
        $packageList = $blockedPackages -join ", "
        throw "Existing environment contains $packageList. README fallback requires an environment without torch_npu or cudatoolkit. Create a fresh environment by rerunning without -UseExistingEnv."
    }

    Write-Host "Existing environment check passed: torch_npu and cudatoolkit are absent."
}

Confirm-MsmodelingRepoRoot

$launcher = @(Resolve-PythonLauncher)
$detectedPython = Get-PythonVersion -Launcher $launcher
if ($detectedPython -lt [Version]"3.10.0") {
    throw "Detected Python $detectedPython. Python 3.10+ is required."
}
Write-Host "Detected Python version: $detectedPython"

if (-not (Test-CommandExists "uv")) {
    Write-Host "uv not found. Installing uv with pip..."
    Invoke-Python -Launcher $launcher -PythonArgs @("-m", "pip", "install", "uv", "-i", $PypiMirror)
}

$uv = Resolve-UvCommand -Launcher $launcher
Write-Host "Using uv executable: $uv"
Enable-ProjectUvCache

if ([string]::IsNullOrWhiteSpace($PythonVersion)) {
    $PythonVersion = "$($detectedPython.Major).$($detectedPython.Minor)"
    Write-Host "PythonVersion not specified. Using detected Python version for venv: $PythonVersion"
}

$venvPython = Join-Path (Get-Location) "$EnvName\Scripts\python.exe"

if (-not $UseExistingEnv) {
    if (Test-Path $EnvName) {
        throw "Environment path already exists: $EnvName. Rerun with -UseExistingEnv to reuse it, or remove the directory after confirming it can be rebuilt."
    }

    Write-Host "Creating virtual environment: $EnvName (Python $PythonVersion)"
    & $uv venv --python $PythonVersion $EnvName

    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment python not found: $venvPython"
    }

    Write-Host "Installing dependencies with uv pip..."
    & $uv pip install --python $venvPython -r requirements.txt -i $PypiMirror
} else {
    Write-Host "Using existing environment fallback: pip install -r requirements.txt"
    if (Test-Path $venvPython) {
        Assert-ExistingEnvironmentClean -Launcher @($venvPython)
        & $venvPython -m pip install -r requirements.txt
    } else {
        Assert-ExistingEnvironmentClean -Launcher $launcher
        Invoke-Python -Launcher $launcher -PythonArgs @("-m", "pip", "install", "-r", "requirements.txt")
    }
}

if (Test-Path $venvPython) {
    & $uv pip check --python $venvPython
} else {
    Invoke-Python -Launcher $launcher -PythonArgs @("-m", "pip", "check")
}

if ($SetProjectEnv) {
    $repoRoot = (Resolve-Path ".").Path
    if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
        $env:PYTHONPATH = $repoRoot
    } else {
        $env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
    }
    Write-Host "PYTHONPATH set for current session: $env:PYTHONPATH"
}

if ($UseHFMirror) {
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    Write-Host "HF_ENDPOINT set for current session: $env:HF_ENDPOINT"
}

Write-Host "Done. Activation command (Windows): $EnvName\Scripts\activate"
