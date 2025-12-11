# 🔧 Guia de Troubleshooting

> Identificação e resolução de problemas em produção

---

## Tipos de Erro Comuns

| Tipo | Sintoma | Gravidade |
|------|---------|-----------|
| **Conexão** | Bot para de receber dados | 🔴 Alta |
| **Dados Inválidos** | Alertas de correção nos logs | 🟡 Média |
| **IA Fora** | Análises não são geradas | 🟡 Média |
| **Health Check** | Timeouts frequentes | 🟠 Alta |

---

## 1. Problemas de Conexão

### Sintomas
```
❌ Erro de conexão/rede: ...
⏳ Reconectando em 2.0s (Tentativa 5/15)...
⛔ Máximo de tentativas de reconexão atingido
```

### Investigação
1. Verificar conectividade com Binance:
   ```bash
   curl https://api.binance.com/api/v3/ping
   ```

2. Checar logs de reconexão:
   ```bash
   Select-String "Reconectando" logs.txt
   ```

3. Verificar `config.py`:
   ```python
   WS_MAX_RECONNECT_ATTEMPTS = 15  # Tentativas
   WS_MAX_DELAY = 30.0             # Delay máximo
   ```

### Solução
- Se Binance online → Reiniciar bot
- Se Binance offline → Aguardar e monitorar status.binance.com

---

## 2. Dados Inválidos

### Sintomas
```
⚠️ [DATA_QUALITY_ALERT] Taxa de correção: 7.5% (limite: 5%)
📊 corrections_by_type: {"recalculated_delta": 40, "timestamp": 25}
```

### Investigação
1. Buscar alertas de qualidade:
   ```bash
   Select-String "DATA_QUALITY_ALERT" logs.txt
   ```

2. Verificar taxa de correção:
   ```bash
   Select-String "correction_rate_pct" logs.txt
   ```

3. Identificar tipo de correção mais frequente

### Problemas Específicos

#### whale_volume > total_volume
```
Causa: Dados recebidos com whale volume maior que volume total
Log: "Whale buy volume excede volume total"
Ação: Verificar se está usando fluxo_continuo (acumulado vs janela)
```

#### Timestamps Inválidos
```
Causa: Timestamps fora do range (< 2021 ou > 2038)
Log: "timestamp_validation_failed"
Ação: Normal se esporádico; investigar se frequente
```

#### Delta Inconsistente
```
Causa: delta ≠ volume_compra - volume_venda
Log: "recalculated_delta"
Ação: Correção automática; monitorar frequência
```

---

## 3. IA Fora do Ar

### Sintomas
```
❌ Erro na análise IA: Connection timeout
⚠️ Fallback para análise mock
```

### Investigação
1. Verificar chaves de API:
   ```bash
   echo $env:GROQ_API_KEY
   echo $env:DASHSCOPE_API_KEY
   ```

2. Testar API manualmente:
   ```bash
   curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/health
   ```

3. Checar rate limits nos logs

### Solução
- Verificar saldo/quota na dashboard do provedor
- Sistema usa fallback automático (Groq → DashScope → OpenAI)

---

## 4. Health Check Timeouts

### Sintomas
```
⚠️ Health check timeout: 60s sem dados
🔄 Forçando reconexão por inatividade
```

### Investigação
1. Verificar último heartbeat:
   ```bash
   Select-String "heartbeat" logs.txt | Select-Object -Last 10
   ```

2. Checar configuração:
   ```python
   HEALTH_CHECK_INTERVAL = 30  # Intervalo (segundos)
   HEALTH_CHECK_TIMEOUT = 60   # Timeout (segundos)
   ```

### Solução
- Aumentar `HEALTH_CHECK_TIMEOUT` se rede lenta
- Verificar `WS_PING_INTERVAL` vs `HEALTH_CHECK_INTERVAL`

---

## 5. Janelas com Dados Inconsistentes

### Sintomas
```
❌ volume_consistency_failed
⚠️ Janela descartada: volume_compra + volume_venda ≠ volume_total
```

### Investigação
1. Verificar taxa de descarte:
   ```bash
   Select-String "discarded_events" logs.txt
   ```

2. Checar métricas de qualidade:
   ```python
   from data_pipeline.metrics import get_quality_metrics
   print(get_quality_metrics().get_stats())
   ```

---

## Comandos Úteis (PowerShell)

```powershell
# Últimos erros
Select-String "ERROR|CRITICAL" logs.txt | Select-Object -Last 20

# Reconexões
Select-String "Reconectando|reconnect" logs.txt

# Alertas de qualidade
Select-String "DATA_QUALITY_ALERT" logs.txt

# Status da IA
Select-String "AI|Groq|DashScope" logs.txt | Select-Object -Last 10

# Health check
Select-String "health|heartbeat" logs.txt | Select-Object -Last 10
```

---

## Métricas para Monitorar

| Métrica | Normal | Atenção | Crítico |
|---------|--------|---------|---------|
| Taxa correção | < 5% | 5-10% | > 10% |
| Taxa descarte | < 2% | 2-5% | > 5% |
| Latência P95 | < 5ms | 5-10ms | > 10ms |
| Reconexões/hora | < 3 | 3-10 | > 10 |

---

## Arquivos para Investigação

| Problema | Arquivo |
|----------|---------|
| Conexão | `robust_connection.py` |
| Validação | `data_validator.py` |
| Pipeline | `data_pipeline/pipeline.py` |
| IA | `ai_analyzer_qwen.py` |
| Configuração | `config.py` |
| Métricas | `data_pipeline/metrics/` |

---

## Contato / Escalação

Antes de escalar, colete:
1. Logs dos últimos 10 minutos
2. `config.py` atual
3. Output de `python -c "from data_pipeline.metrics import get_quality_metrics; print(get_quality_metrics().get_stats())"`
