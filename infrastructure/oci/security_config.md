# OCI Security Configuration for Market Bot

Este documento detalha as configurações de segurança exigidas no Oracle Cloud para operação segura 24/7.

## 1. Network Security Groups (NSG)

Recomendamos criar um NSG específico `nsg-market-bot` aplicado à VNIC da instância.

| Direção | Protocolo | Porta | Source/Dest | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| **Ingress** | TCP | 22 | Seu IP Residencial/VPN | Acesso SSH (Evite 0.0.0.0/0!) |
| **Egress** | TCP | 443 | 0.0.0.0/0 | Acesso HTTPS (Binance API, OCI API) |
| **Egress** | TCP | 80 | 0.0.0.0/0 | Updates do sistema (yum/apt) |
| **Egress** | UDP | 123 | 0.0.0.0/0 | NTP (Clock Sync) |

> ⚠️ **Nota:** Evite usar Security Lists da VCN padrão se possível, NSGs são mais flexíveis.

## 2. IAM Policies (Dynamic Groups)

Para que a autenticação "Instance Principal" funcione, você deve:

1.  Criar um **Dynamic Group** chamado `MarketBotInstances`.
    *   Rule: `ALL {instance.compartment.id = 'ocid1.compartment.oc1..aaaa...'}` (ou match por tag).

2.  Criar uma **Policy** na Root (ou Compartment) chamada `MarketBotPolicy`.

### Regras da Policy:

**Permitir upload de métricas:**
```text
Allow dynamic-group MarketBotInstances to use metrics in compartment id <OCID_DO_COMPARTMENT>
```

**Permitir upload de backups:**
```text
Allow dynamic-group MarketBotInstances to manage objects in compartment id <OCID_DO_COMPARTMENT> where target.bucket.name='<NOME_DO_BUCKET_BACKUP>'
```

**Permitir leitura de segredos (Vault):**
```text
Allow dynamic-group MarketBotInstances to read secret-bundles in compartment id <OCID_DO_COMPARTMENT>
```
*(Opcional: restrinja a segredos específicos se desejar)*

## 3. Vault (Secrets Management)

1.  Crie um **Vault** no OCI.
2.  Crie uma **Key** de criptografia (Master Key).
3.  Adicione **Secrets** para cada variável sensível que removemos do `.env`:
    *   `BINANCE_API_KEY`
    *   `BINANCE_SECRET_KEY`
    *   `GROQ_API_KEY` (se usada)
4.  Anote o OCID de cada secret e use no código/env var referenciando-os.

## 4. Hardening do OS (Oracle Linux)

*   [ ] Desabilitar Login Root via SSH (`PermitRootLogin no`).
*   [ ] Configurar Firewall local (`firewalld` ou `iptables`) para aceitar apenas established connections + SSH.
*   [ ] Habilitar Oracle OS Management para patches automáticos de segurança.
