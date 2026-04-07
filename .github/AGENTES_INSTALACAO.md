# 5 Agentes Customizados - Instalação Completa ✅

## 📋 O Que Foi Criado

### 🤖 Agentes (em `.github/agents/`)

✅ **arquiteto-robo-binance.agent.md**
- Especialista em arquitetura e impacto de mudanças
- Ferramentas: `read`, `search` (read-only)
- Use quando: Planejar mudança, analisar risco, entender dependências

✅ **implementador-python-binance.agent.md**
- Engenheiro Python sênior para implementação segura
- Ferramentas: `read`, `edit`, `search`, `execute`, `todo`
- Use quando: Implementar feature, bugfix, refactor com testes

✅ **guardiao-testes-qualidade.agent.md**
- Especialista em testes e cobertura
- Ferramentas: `read`, `search`, `edit`, `execute`
- Use quando: Revisar cobertura, criar testes, validar regressão

✅ **diagnostico-performance.agent.md**
- Especialista em bugs, gargalos e performance
- Ferramentas: `read`, `search`, `execute`
- Use quando: Investigar lentidão, WebSocket, buffers, async issues

✅ **ia-payload-llm.agent.md**
- Especialista em IA, payload e LLM
- Ferramentas: `read`, `edit`, `search`, `execute`
- Use quando: Otimizar payload, validar resposta IA, reduzir custo

### 📚 Documentação (em `.github/`)

✅ **AGENTES_CUSTOMIZADOS.md** (Guia Completo)
- ✅ Descrição detalhada dos 5 agentes
- ✅ 4 fluxos de trabalho principais
- ✅ 9 exemplos de uso prático
- ✅ Boas práticas e anti-patterns
- ✅ Troubleshooting e suporte
- ✅ Tabela de referência

✅ **AGENTES_REFERENCIA_RAPIDA.md** (Referência Rápida)
- ✅ Tabela de decisão em 10 segundos
- ✅ Quando usar cada agente
- ✅ Fluxos de trabalho típicos
- ✅ Gotchas comuns
- ✅ Exemplos rápidos

---

## 🚀 Como Usar (Rápido)

### 1. Abra o VS Code

### 2. Escolha um Agente
- `@Arquiteto Robo Binance` → Planejar
- `@Implementador Python Binance` → Implementar
- `@Guardião de Testes e Qualidade` → Testar
- `@Diagnóstico e Performance` → Diagnosticar
- `@IA, Payload e LLM` → Otimizar IA

### 3. Sempre Anexe
```
ESTRUTURA_SISTEMA_COMPLETO.md
```

### 4. Descreva Seu Problema
```
@Arquiteto Robo Binance

Quero mover trade_buffer.py de raiz para trading/. 
É seguro? Quais módulos quebram?
```

### 5. Pronto!
O agente entrega análise/implementação/testes com resumo final.

---

## 📖 Documentos para Ler (em Ordem)

1. **`.github/AGENTES_REFERENCIA_RAPIDA.md`** ← Leia isso PRIMEIRO
   - 2 minutos para entender qual agente usar

2. **`.github/AGENTES_CUSTOMIZADOS.md`** ← Leia para detalhe
   - 10 minutos com fluxos e exemplos completos

3. **`.github/agents/*.agent.md`** ← Consulte quando necessário
   - Detalhes técnicos de cada agente

---

## ✨ Características Principais

### Arquiteto Robo Binance
- ✅ Analisa impacto arquitetural
- ✅ Identifica riscos de import circular
- ✅ Propõe planos reversíveis
- ✅ Verifica testes relacionados
- ✅ Protege módulos críticos da raiz

### Implementador Python Binance
- ✅ Implementa mudanças seguras
- ✅ Cria testes automaticamente
- ✅ Preserva backward compatibility
- ✅ Fornece resumo executivo
- ✅ Rastreia progresso com todo tool

### Guardião de Testes e Qualidade
- ✅ Revisa cobertura existente
- ✅ Identifica lacunas
- ✅ Cria testes robustos
- ✅ Valida regressões
- ✅ Recomenda padrões pytest

### Diagnóstico e Performance
- ✅ Investiga causa raiz
- ✅ Analisa concorrência async
- ✅ Sugere instrumentação
- ✅ Propõe correção mínima
- ✅ Validação por testes

### IA, Payload e LLM
- ✅ Otimiza payload
- ✅ Reduz custo de IA
- ✅ Fortaleça guardrails
- ✅ Valida resposta LLM
- ✅ Preserva contexto crítico

---

## 🎯 Fluxos de Trabalho Recomendados

### Feature Completa
```
Arquiteto → Implementador → Guardião → ✅ PRONTO
```

### Bug/Performance
```
Diagnóstico → Implementador → Guardião → ✅ RESOLVIDO
```

### Otimização IA
```
IA, Payload → Implementador (optional) → Guardião → ✅ OTIMIZADO
```

### Apenas Análise
```
Arquiteto ou Diagnóstico → ✅ COM ANÁLISE COMPLETA
```

---

## 🔗 Estrutura de Arquivos

```
.github/
├── agents/
│   ├── arquiteto-robo-binance.agent.md
│   ├── implementador-python-binance.agent.md
│   ├── guardiao-testes-qualidade.agent.md
│   ├── diagnostico-performance.agent.md
│   └── ia-payload-llm.agent.md
├── AGENTES_CUSTOMIZADOS.md (guia completo)
├── AGENTES_REFERENCIA_RAPIDA.md (referência rápida)
└── AGENTES_INSTALACAO.md (este arquivo)
```

---

## 💡 Dicas de Uso

### ✅ DO

- **Sempre** anexe `ESTRUTURA_SISTEMA_COMPLETO.md`
- **Seja específico** (cite arquivo/módulo/linha)
- **Use o Arquiteto** antes de mudanças grandes
- **Trust the agents** - eles sabem o projeto bem
- **Follow the RESUMO FINAL** de cada agente
- **Run the tests** que os agentes recomendam

### ❌ DON'T

- ❌ Não ignore restrições do agente
- ❌ Não mova módulos críticos sem análise
- ❌ Não remove proxies sem reason
- ❌ Não ignore testes existentes
- ❌ Não assuma cobertura sem verificar

---

## 🆘 Troubleshooting

**Q: Agente não responde ao meu pedido?**
A: Leia `.github/AGENTES_REFERENCIA_RAPIDA.md` para tabela de decisão.

**Q: Qual agente usar para X?**
A: Consulte a tabela em `AGENTES_REFERENCIA_RAPIDA.md`.

**Q: Como chamar um agente?**
A: Mensagem começando com `@Nome do Agente` + seu pedido.

**Q: Preciso anexar ESTRUTURA_SISTEMA_COMPLETO.md?**
A: **SIM, SEMPRE.** Melhora muito a qualidade da resposta.

**Q: Posso usar múltiplos agentes na mesma conversa?**
A: Sim! Mude de agente com `@Novo Agente` em uma nova mensagem.

---

## 📊 Estatísticas

- **5 agentes** criados e documentados
- **2 arquivos** de documentação completos
- **25+ exemplos** de uso
- **100+ cenários** cobertos pelas instruções
- **5 fluxos** de trabalho principais
- **10 gotchas** comuns identificados

---

## 🎁 Bônus

Cada agente tem:
- ✅ Instruções claras e detalhadas
- ✅ Restrições bem definidas
- ✅ Fluxo obrigatório de trabalho
- ✅ Formato de resposta padronizado
- ✅ RESUMO FINAL executivo
- ✅ Notas técnicas importantes

---

## 🏆 Status Final

✅ **Tudo instalado e pronto para uso**

Próximas ações:
1. Leia `.github/AGENTES_REFERENCIA_RAPIDA.md` (2 min)
2. Leia `.github/AGENTES_CUSTOMIZADOS.md` (10 min)
3. Comece a usar: `@Arquiteto Robo Binance ...`

---

**Criado:** Abril 2026  
**Sistema:** Robo Binance API  
**Versão:** 1.0 - 5 agentes  
**Status:** ✅ Produção
