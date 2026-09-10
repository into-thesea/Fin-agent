# Fin-Agent 4.0 启动脚本 (PowerShell) — 安静模式
# 用法: PowerShell → .\start.ps1
# 特性: 零弹窗, PID 文件管理, 一键停止用 python run.py --stop

Write-Host "=== Fin-Agent 4.0 启动脚本 ===" -ForegroundColor Cyan

# Step 1: 用 run.py 的 stop 清理旧实例 (比杀所有 python 进程安全)
Write-Host "[1/4] 清理旧实例..." -ForegroundColor Yellow
python run.py --stop 2>$null
Start-Sleep 2

# Step 2: 启动 Redis (WSL)
Write-Host "[2/4] 启动 Redis..." -ForegroundColor Yellow
wsl -- redis-server --daemonize yes --port 6379
Start-Sleep 1
$redis = wsl -- redis-cli ping 2>$null
if ($redis -eq "PONG") {
    Write-Host "  Redis OK (PONG)" -ForegroundColor Green
} else {
    Write-Host "  Redis 启动失败, 功能降级" -ForegroundColor Red
}

# Step 3: 启动 Celery Worker (后台, 无窗口)
Write-Host "[3/4] 启动 Celery Worker..." -ForegroundColor Yellow
$null = New-Item -Force -Path "logs/.celery_worker_heartbeat" -ItemType File
$celery = Start-Process -NoNewWindow -PassThru python -ArgumentList "-m celery -A src.celery_app worker --pool=solo --without-mingle --without-gossip"
Start-Sleep 2
if (-not $celery.HasExited) {
    Write-Host "  Celery Worker 已启动 (PID: $($celery.Id))" -ForegroundColor Green
} else {
    Write-Host "  Celery Worker 启动失败" -ForegroundColor Red
}

# Step 4: 启动 API 服务 (安静模式, 无命令行窗口)
Write-Host "[4/4] 启动 API 服务 --daemon..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  API文档: http://localhost:8001/docs" -ForegroundColor Cyan
Write-Host "  前端界面: http://localhost:8001" -ForegroundColor Cyan
Write-Host "  健康检查: http://localhost:8001/health" -ForegroundColor Cyan
Write-Host ""

python run.py --daemon
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] 服务已在后台运行" -ForegroundColor Green
    Write-Host "  停止: python run.py --stop" -ForegroundColor Gray
} else {
    Write-Host "  [FAIL] 启动失败 (exit code: $LASTEXITCODE)" -ForegroundColor Red
}
