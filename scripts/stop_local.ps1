param(
    [switch]$KeepInfrastructure
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $projectRoot ".data\devflow-processes.json"

function Stop-ProcessTree {
    param([Parameter(Mandatory)] [int]$ProcessId)

    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId ([int]$child.ProcessId)
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

if (Test-Path -LiteralPath $statePath -PathType Leaf) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    foreach ($processId in @($state.frontend_process_id, $state.backend_process_id)) {
        if ($processId) {
            Stop-ProcessTree -ProcessId ([int]$processId)
        }
    }
    Remove-Item -LiteralPath $statePath -Force
}
else {
    Write-Host "未找到前后端进程记录；不会停止未知进程。"
}

if (-not $KeepInfrastructure) {
    Push-Location $projectRoot
    try {
        & docker compose stop postgres etcd minio milvus
        if ($LASTEXITCODE -ne 0) {
            throw "停止基础设施容器失败。"
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host "DevFlow AI 已停止。"
if ($KeepInfrastructure) {
    Write-Host "基础设施容器保持运行。"
}
