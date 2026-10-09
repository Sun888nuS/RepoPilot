param(
    [switch]$SeedDemo,
    [switch]$RebuildFrontend,
    [ValidateRange(1, 65535)]
    [int]$BackendPort = 8000,
    [ValidateRange(1, 65535)]
    [int]$FrontendPort = 3000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDirectory = Join-Path $projectRoot "backend"
$frontendDirectory = Join-Path $projectRoot "frontend"
$pythonPath = Join-Path $backendDirectory ".venv\Scripts\python.exe"
$envPath = Join-Path $projectRoot ".env"
$dataDirectory = Join-Path $projectRoot ".data"
$logDirectory = Join-Path $dataDirectory "logs"
$statePath = Join-Path $dataDirectory "devflow-processes.json"
$backendOutputLog = Join-Path $logDirectory "backend.out.log"
$backendErrorLog = Join-Path $logDirectory "backend.err.log"
$frontendOutputLog = Join-Path $logDirectory "frontend.out.log"
$frontendErrorLog = Join-Path $logDirectory "frontend.err.log"

function Test-DockerReady {
    & docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}

function Wait-HttpReady {
    param(
        [Parameter(Mandatory)] [string]$Uri,
        [Parameter(Mandatory)] [int]$Attempts
    )
    for ($attempt = 0; $attempt -lt $Attempts; $attempt += 1) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2 | Out-Null
            return $true
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

function Wait-TcpPort {
    param(
        [Parameter(Mandatory)] [int]$Port,
        [int]$Attempts = 120
    )
    for ($attempt = 0; $attempt -lt $Attempts; $attempt += 1) {
        $client = [System.Net.Sockets.TcpClient]::new()
        try {
            $connect = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
            if ($connect.AsyncWaitHandle.WaitOne(1000) -and $client.Connected) {
                $client.EndConnect($connect)
                return $true
            }
        }
        catch {
        }
        finally {
            $client.Dispose()
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
    throw "未找到 .env。请先复制 .env.example，并配置本地模型路径。"
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "未找到 backend/.venv。请使用 Python 3.12 创建环境并安装后端依赖。"
}
if (-not (Test-Path -LiteralPath (Join-Path $frontendDirectory "node_modules") -PathType Container)) {
    throw "未找到 frontend/node_modules。请先在 frontend 目录执行 npm install。"
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "未安装 Docker CLI。"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

if (-not (Test-DockerReady)) {
    $dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path -LiteralPath $dockerDesktop -PathType Leaf)) {
        throw "Docker 引擎未运行，并且未找到 Docker Desktop。"
    }
    Start-Process -FilePath $dockerDesktop -WindowStyle Hidden | Out-Null
    $dockerReady = $false
    for ($attempt = 0; $attempt -lt 120; $attempt += 1) {
        Start-Sleep -Seconds 1
        if (Test-DockerReady) {
            $dockerReady = $true
            break
        }
    }
    if (-not $dockerReady) {
        throw "Docker Desktop 在 120 秒内未就绪。"
    }
}

Push-Location $projectRoot
try {
    & docker compose up -d postgres etcd minio milvus
    if ($LASTEXITCODE -ne 0) {
        throw "基础设施容器启动失败。"
    }
}
finally {
    Pop-Location
}

if (-not (Wait-TcpPort -Port 5432)) {
    throw "PostgreSQL 端口 5432 未就绪。"
}
if (-not (Wait-TcpPort -Port 19530)) {
    throw "Milvus 端口 19530 未就绪。"
}

$existingState = $null
if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    try {
        $existingState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    }
    catch {
        $existingState = $null
    }
}

$backendUrl = "http://127.0.0.1:$BackendPort"
$frontendUrl = "http://127.0.0.1:$FrontendPort"
$backendProcessId = $null
$frontendProcessId = $null

if (Wait-HttpReady -Uri "$backendUrl/health" -Attempts 1) {
    if ($existingState -and $existingState.backend_process_id) {
        $backendProcessId = [int]$existingState.backend_process_id
    }
    Write-Host "后端已经运行：$backendUrl"
}
else {
    $backendProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "$BackendPort") `
        -WorkingDirectory $backendDirectory `
        -RedirectStandardOutput $backendOutputLog `
        -RedirectStandardError $backendErrorLog `
        -WindowStyle Hidden `
        -PassThru
    $backendProcessId = $backendProcess.Id
    if (-not (Wait-HttpReady -Uri "$backendUrl/health" -Attempts 120)) {
        throw "后端在 60 秒内未就绪，请检查 $backendErrorLog。"
    }
}

if ($SeedDemo) {
    & $pythonPath (Join-Path $projectRoot "scripts\seed_demo_data.py")
    if ($LASTEXITCODE -ne 0) {
        throw "演示数据初始化失败。"
    }
}

$buildIdPath = Join-Path $frontendDirectory ".next\BUILD_ID"
if ($RebuildFrontend -or -not (Test-Path -LiteralPath $buildIdPath -PathType Leaf)) {
    Push-Location $frontendDirectory
    try {
        $env:NEXT_PUBLIC_API_BASE_URL = $backendUrl
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) {
            throw "前端生产构建失败。"
        }
    }
    finally {
        Pop-Location
    }
}

if (Wait-HttpReady -Uri $frontendUrl -Attempts 1) {
    if ($existingState -and $existingState.frontend_process_id) {
        $frontendProcessId = [int]$existingState.frontend_process_id
    }
    Write-Host "前端已经运行：$frontendUrl"
}
else {
    $frontendProcess = Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList @("run", "start", "--", "-p", "$FrontendPort") `
        -WorkingDirectory $frontendDirectory `
        -RedirectStandardOutput $frontendOutputLog `
        -RedirectStandardError $frontendErrorLog `
        -WindowStyle Hidden `
        -PassThru
    $frontendProcessId = $frontendProcess.Id
    if (-not (Wait-HttpReady -Uri $frontendUrl -Attempts 120)) {
        throw "前端在 60 秒内未就绪，请检查 $frontendErrorLog。"
    }
}

[ordered]@{
    backend_process_id = $backendProcessId
    frontend_process_id = $frontendProcessId
    backend_port = $BackendPort
    frontend_port = $FrontendPort
    started_at = [DateTimeOffset]::Now.ToString("o")
} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8

Write-Host "DevFlow AI 完整本地模式已启动："
Write-Host "  前端：$frontendUrl"
Write-Host "  后端：$backendUrl"
Write-Host "  API 文档：$backendUrl/docs"
Write-Host "  日志目录：$logDirectory"
Write-Host "  停止命令：.\scripts\stop_local.ps1"
