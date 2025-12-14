# OPERATIONAL RUNBOOK - Market Bot v2

**Serviço:** Market Bot (Oracle Cloud)
**SLA:** 99.9% uptime (Best Effort)

---

## 🚨 1. Triage de Incidentes

### Cenário A: Bot caiu (Alerta OCI "Heartbeat Missing")
1.  Acesse via SSH: `ssh opc@<ip-instancia>`
2.  Verifique status do Docker: `docker compose ps`
3.  Verifique logs recentes: `docker compose logs --tail=100 market-bot`
4.  Se container estiver "Exited":
    *   Tente reiniciar: `docker compose up -d`
    *   Se falhar loop, verifique espaço em disco: `df -h`

### Cenário B: Latência Alta (Alerta "TradeLag > 5000ms")
1.  Verifique carga da CPU: `htop`
2.  Verifique memória: `free -m` (Se swap estiver alto, pode ser leak)
3.  Verifique conexões de rede: `netstat -an | grep ES | wc -l`

---

## 🛠️ 2. Procedimentos Comuns

### Restart Limpo
Para aplicar novas configurações ou limpar estado de memória:
```bash
cd /opt/market-bot
docker compose down
# (Opcional) Limpar logs antigos se disco cheio
# rm logs/*.log
docker compose up -d
```

### Visualizar Logs em Tempo Real
```bash
docker compose logs -f market-bot
```
*(Use `Ctrl+C` para sair)*

### Atualizar Versão (Manual)
```bash
git pull
docker compose build
docker compose up -d
```

---

## 💾 3. Backup e Restore

### Backup Manual Imediato
```bash
docker compose exec market-bot python scripts/backup_to_oci.py
```

### Restore (Disaster Recovery)
⚠️ **PERIGO:** Isso sobrescreve os dados locais atuais.
```bash
sudo ./scripts/disaster_recovery.sh
```

---

## 📞 4. Contatos
*   **Dev Lead:** (Seu Nome/Email)
*   **Cloud Admin:** (Painel OCI Tenancy)
