# scripts/diagnostics/collect_2h.ps1
# ==============================================================================
# Coleta Oficial de 2 Horas — Binance Futures (USD-M perp)
# Alvo Planejado: Terça-feira 2026-09-08, 13:30–15:30 UTC (após Labor Day)
#
# Procedimento de Execução:
#   1. Inicia `main.py` com Start-Process -PassThru e redirecionamento de streams.
#   2. Loop de liveness a cada 30s até completar 7200s (2h).
#   3. Encerra o processo de forma limpa.
#   4. Executa scripts/diagnostics/accept_futures_migration.py estendido para
#      avaliação dos 5 critérios + métricas de calibração/orderbook.
# ==============================================================================

[CmdletBinding()]
param(
    [int]$DurationSeconds = 7200,
    [int]$CheckIntervalSeconds = 30,
    [string]$DbPath = "dados/trading_bot.db"
)

$ErrorActionPreference = "Stop"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$StdoutLog = "$LogDir/collect_2h_$Timestamp.log"
$StderrLog = "$LogDir/collect_2h_${Timestamp}_stderr.log"

Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " [COLETA 2H] Iniciando monitoramento de 2 horas (Binance Futures BTCUSDT)" -ForegroundColor Cyan
Write-Host " Alvo planejado: Terça 2026-09-08 13:30-15:30 UTC" -ForegroundColor Yellow
Write-Host " Duração: $DurationSeconds s | Intervalo de verificação: $CheckIntervalSeconds s" -ForegroundColor White
Write-Host " Logs: $StdoutLog / $StderrLog" -ForegroundColor White
Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# 1. Iniciar main.py
$proc = Start-Process -FilePath "python" `
    -ArgumentList "main.py" `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru

Write-Host "Processo iniciado com PID: $($proc.Id)" -ForegroundColor Green

$elapsed = 0
$healthy = $true

try {
    while ($elapsed -lt $DurationSeconds) {
        Start-Sleep -Seconds $CheckIntervalSeconds
        $elapsed += $CheckIntervalSeconds

        if ($proc.HasExited) {
            Write-Host "❌ [ALERTA] Processo finalizou inesperadamente com código $($proc.ExitCode) aos $elapsed s!" -ForegroundColor Red
            $healthy = $false
            break
        }

        # Checar tamanho do log
        $logSizeKb = 0
        if (Test-Path $StdoutLog) {
            $logSizeKb = [math]::Round((Get-Item $StdoutLog).Length / 1KB, 1)
        }

        # Checar se DB está crescendo
        $dbSizeKb = 0
        if (Test-Path $DbPath) {
            $dbSizeKb = [math]::Round((Get-Item $DbPath).Length / 1KB, 1)
        }

        $remaining = $DurationSeconds - $elapsed
        Write-Host "⏱️ [$elapsed s / $DurationSeconds s | Restam $remaining s] PID $($proc.Id) ATIVO | Log: $logSizeKb KB | DB: $dbSizeKb KB" -ForegroundColor Gray
    }
}
finally {
    if (-not $proc.HasExited) {
        Write-Host "🛑 Tempo de coleta esgotado ($elapsed s). Encerrando processo PID $($proc.Id)..." -ForegroundColor Yellow
        $proc.Kill()
        $proc.WaitForExit(5000)
    }
}

Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host " [FIM DA SESSÃO DE 2H] Processando resultados com accept_futures_migration.py..." -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

if (Test-Path $DbPath) {
    python scripts/diagnostics/accept_futures_migration.py
} else {
    Write-Host "❌ Banco de dados $DbPath não encontrado para validação." -ForegroundColor Red
}
