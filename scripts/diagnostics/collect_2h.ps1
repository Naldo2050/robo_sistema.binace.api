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
    [string]$DbPath = "dados/trading_bot.db",
    [string]$RawTradesDumpPath = "dados/trades_collect_2h.jsonl"
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
Write-Host " Dump de trades brutos: $RawTradesDumpPath" -ForegroundColor White
Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# 1. Iniciar main.py com dump de trades brutos e temporizador gracioso de 7200s
$proc = Start-Process -FilePath "python" `
    -ArgumentList "main.py --dump-raw-trades $RawTradesDumpPath --duration-seconds $DurationSeconds" `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru

Write-Host "Processo iniciado com PID: $($proc.Id)" -ForegroundColor Green

$elapsed = 0
$healthy = $true

try {
    while ($elapsed -lt ($DurationSeconds + 60)) {
        Start-Sleep -Seconds $CheckIntervalSeconds
        $elapsed += $CheckIntervalSeconds

        if ($proc.HasExited) {
            if ($proc.ExitCode -eq 0) {
                Write-Host "✅ Processo completou a execução de $DurationSeconds s graciosamente com código 0!" -ForegroundColor Green
            } else {
                Write-Host "❌ [ALERTA] Processo finalizou com código $($proc.ExitCode) aos $elapsed s!" -ForegroundColor Red
                $healthy = $false
            }
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
Write-Host " [FIM DA SESSÃO DE 2H] Processando resultados com suite de validação estatística..." -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

if (Test-Path $DbPath) {
    Write-Host "`n>>> [1/4] Análise de Sincronização do OrderBook (Item 3)" -ForegroundColor Yellow
    python scripts/diagnostics/analyze_orderbook_sync_session.py --db $DbPath

    Write-Host "`n>>> [2/4] Validação de Threshold de Whale e Estatística de Trades (Item 1)" -ForegroundColor Yellow
    python scripts/diagnostics/validate_whale_threshold.py --dump-path $RawTradesDumpPath --db $DbPath

    Write-Host "`n>>> [3/4] Validação de Paridade de Schema em Sinais (Item 4)" -ForegroundColor Yellow
    python scripts/diagnostics/validate_signals_orderbook_schema.py --db $DbPath

    Write-Host "`n>>> [4/4] Critérios de Aceite de Migração Futures (Item 5)" -ForegroundColor Yellow
    python scripts/diagnostics/accept_futures_migration.py
} else {
    Write-Host "❌ Banco de dados $DbPath não encontrado para validação." -ForegroundColor Red
}
