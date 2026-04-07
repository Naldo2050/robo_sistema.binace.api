# Referência Rápida dos 5 Agentes

**Use este arquivo para decidir qual agente chamar em 10 segundos.**

---

## 🎯 Tabela de Decisão Rápida

| Seu problema | Agente | Comando |
|---|---|---|
| **Planejar movimento de arquivo** | Arquiteto | `@Arquiteto Robo Binance Análise impacto de mover X para Y` |
| **Implementar feature/bug** | Implementador | `@Implementador Python Binance Implementa a feature X` |
| **Revisar cobertura de testes** | Guardião | `@Guardião de Testes Verifica cobertura de X` |
| **Sistema está lento** | Diagnóstico | `@Diagnóstico e Performance Investiga lentidão em X` |
| **WebSocket reconectando** | Diagnóstico | `@Diagnóstico e Performance Investiga reconnect WebSocket` |
| **Fila/Buffer crescendo** | Diagnóstico | `@Diagnóstico e Performance Analisa buffer acumulando em X` |
| **Async event loop bloqueado** | Diagnóstico | `@Diagnóstico e Performance Investiga bloqueio de event loop` |
| **Reduzir custo de IA** | IA, Payload | `@IA, Payload e LLM Reduza tamanho/custo de payload` |
| **Validar resposta de LLM** | IA, Payload | `@IA, Payload e LLM Fortaleça validação de resposta` |
| **Payload inválido** | IA, Payload | `@IA, Payload e LLM Diagnóstico de payload malformado` |

---

## 📊 Quando Usar Cada Agente

### 🏗️ Arquiteto Robo Binance (`@Arquiteto`)
**Situações:**
- ✅ Planejar mudança antes de implementar
- ✅ Entender impacto de alteração
- ✅ Verificar se é seguro mover arquivo
- ✅ Analisar dependências e riscos
- ✅ Avaliar compatibilidade

**NÃO USE PARA:**
- ❌ Implementar código (use Implementador)
- ❌ Criar testes (use Guardião)
- ❌ Investigar performance (use Diagnóstico)

---

### 💻 Implementador Python Binance (`@Implementador`)
**Situações:**
- ✅ Implementar feature
- ✅ Corrigir bug
- ✅ Refatorar código
- ✅ Criar/atualizar testes
- ✅ Executar mudança planejada

**NÃO USE PARA:**
- ❌ Analisar impacto (use Arquiteto)
- ❌ Investigar problema (use Diagnóstico)
- ❌ Apenas revisar cobertura (use Guardião)

---

### 🛡️ Guardião de Testes e Qualidade (`@Guardião`)
**Situações:**
- ✅ Revisar cobertura de arquivo
- ✅ Criar testes faltantes
- ✅ Validar regressão
- ✅ Checar se código está protegido
- ✅ Auditar testes existentes

**NÃO USE PARA:**
- ❌ Implementar código (use Implementador)
- ❌ Analisar arquitetura (use Arquiteto)
- ❌ Investigar performance (use Diagnóstico)

---

### 🔍 Diagnóstico e Performance (`@Diagnóstico`)
**Situações:**
- ✅ Sistema está lento
- ✅ WebSocket reconectando
- ✅ Fila crescendo
- ✅ Memória crescendo
- ✅ Travamento/deadlock
- ✅ Errores intermitentes
- ✅ Event loop bloqueado
- ✅ Buffer acumulando

**NÃO USE PARA:**
- ❌ Validação de payload IA (use IA, Payload)
- ❌ Apenas revisar testes (use Guardião)
- ❌ Planejar mudança (use Arquiteto)

---

### 🤖 IA, Payload e LLM (`@IA, Payload`)
**Situações:**
- ✅ Reduzir custo de IA
- ✅ Reduzir tamanho de payload
- ✅ Fortalecer validação de resposta
- ✅ Melhorar guardrails
- ✅ Otimizar throttling/cache
- ✅ Diagnosticar problema de payload
- ✅ Validar compatibilidade de formato

**NÃO USE PARA:**
- ❌ Problema de performance geral (use Diagnóstico)
- ❌ Apenas revisar arquitetura (use Arquiteto)
- ❌ Apenas criar testes genéricos (use Guardião)

---

## 🔄 Fluxos de Trabalho Típicos

### Fluxo 1: Feature Completa (Ideal)
```
1. @Arquiteto → Planejar impacto
2. @Implementador → Implementar mudança
3. @Guardião → Validar testes
                ↓
            ✅ PRONTO
```

### Fluxo 2: Bug/Performance
```
1. @Diagnóstico → Investigar causa
2. @Implementador → Corrigir (optional)
3. @Guardião → Validar cobertura
                ↓
            ✅ RESOLVIDO
```

### Fluxo 3: IA/Payload
```
1. @IA, Payload → Analisar/Otimizar
2. @Implementador → Implementar (optional)
3. @Guardião → Validar testes
                ↓
            ✅ OTIMIZADO
```

### Fluxo 4: Apenas Análise
```
1. @Arquiteto ou @Diagnóstico → Investigar
                ↓
            ✅ COM ANÁLISE COMPLETA
```

---

## 🚨 Gotchas Comuns

| Erro | Correção |
|------|----------|
| Chamando Implementador para analisar | Use Arquiteto ou Diagnóstico primeiro |
| Esquecendo de anexar ESTRUTURA_SISTEMA_COMPLETO.md | Sempre anexe este arquivo |
| Esperando Guardião consertar código | Guardião só revisa testes, use Implementador |
| Usando Diagnóstico para validar payload IA | Use IA, Payload e LLM para payload |
| Ignorando restrições do agente | Leia a seção "Restrições" no arquivo .agent.md |

---

## 💡 Exemplo Rápido

**Você:** "Tenho um bug no trade_buffer.py"

**Análise:**
1. O bug é intermitente/concorrência? → Diagnóstico
2. Ou é logic errada? → Implementador direto
3. Depois sempre → Guardião

**Você:** "Quero mover ai_analyzer_qwen.py"

**Análise:**
1. Primeiro → Arquiteto (muito arriscado)
2. Se seguro → Implementador
3. Finalmente → Guardião

**Você:** "Sistema muito lento"

**Análise:**
1. Primeiro → Diagnóstico (encontrar cause raiz)
2. Depois → Implementador (corrigir)
3. Finalmente → Guardião (validar)

---

## 📞 Ajuda Rápida

**Qual agente para...?**
- Architecture / Planning → **Arquiteto**
- Code / Testing → **Implementador** e **Guardião**
- Performance / Bugs → **Diagnóstico**
- IA / Payload / LLM → **IA, Payload e LLM**

**Sempre lembre:**
- 📎 Anexe `ESTRUTURA_SISTEMA_COMPLETO.md`
- 🎯 Seja específico (cite arquivos/módulos)
- 📋 Siga o fluxo correto
- ✅ Verifique testes depois

---

**Última atualização:** Abril 2026
