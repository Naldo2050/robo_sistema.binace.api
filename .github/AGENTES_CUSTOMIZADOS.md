# Agentes Customizados - Robo Binance API

Este projeto utiliza 5 agentes especializados no Copilot para planejar, implementar, testar, diagnosticar e otimizar o sistema.

## 📋 Os 5 Agentes

### 1️⃣ **Arquiteto Robo Binance**
**Arquivo:** `.github/agents/arquiteto-robo-binance.agent.md`

**Propósito:** Analisar impacto arquitetural antes de qualquer mudança.

**Quando usar:**
- Planejar uma mudança ou refatoração
- Avaliar se é seguro mover um módulo
- Entender impacto de remover proxies
- Identificar riscos de import circular
- Verificar compatibilidade de uma alteração proposta
- Analisar dependências antes de implementar

**Ferramentas:** `read`, `search` (read-only)

**Exemplo de uso:**
```
@Arquiteto Robo Binance Quero mover o arquivo ai_analyzer_qwen.py de raiz para market_orchestrator/ai/. 
Qual é o impacto? Quais módulos vão quebrar?
```

**Saída esperada:**
- Objetivo claro
- Arquivos afetados
- Impacto arquitetural
- Riscos identificados
- Testes existentes relacionados
- Testes que precisam ser criados
- Plano recomendado em etapas pequenas

---

### 2️⃣ **Implementador Python Binance**
**Arquivo:** `.github/agents/implementador-python-binance.agent.md`

**Propósito:** Implementar mudanças com segurança, preservando compatibilidade e adicionando testes.

**Quando usar:**
- Implementar uma feature nova
- Corrigir um bug
- Refatorar código
- Atualizar comportamento existente
- Realizar mudança arquitetural previamente planejada

**Ferramentas:** `read`, `edit`, `search`, `execute`, `todo`

**Exemplo de uso:**
```
@Implementador Python Binance Vou mover ai_analyzer_qwen.py para market_orchestrator/ai/. 
Implementa a mudança, atualiza os imports, cria testes necessários e faz o rollback se quebrar algo.
```

**Saída esperada:**
- Objetivo executado
- Arquivos alterados/criados
- Testes criados/alterados
- Comandos de teste recomendados
- Riscos ou limitações
- Próximo passo

---

### 3️⃣ **Guardião de Testes e Qualidade**
**Arquivo:** `.github/agents/guardiao-testes-qualidade.agent.md`

**Propósito:** Validar cobertura teste, criar testes, revisar regressões.

**Quando usar:**
- Revisar cobertura de um arquivo
- Criar testes para uma mudança
- Validar que uma mudança está protegida por testes
- Identificar lacunas de cobertura
- Auditar risco de regressão
- Verificar se testes de integração cobrem a mudança

**Ferramentas:** `read`, `search`, `edit`, `execute`

**Exemplo de uso:**
```
@Guardião de Testes e Qualidade Qual é a cobertura de testes para ai_analyzer_qwen.py? 
Há lacunas? Que testes devo criar para cobrir a integração com market_orchestrator?
```

**Saída esperada:**
- Cobertura existente
- Lacunas de cobertura
- Testes recomendados
- Testes criados/alterados
- Comandos para rodar testes
- Riscos de regressão

---

## 🔄 Fluxo Ideal de Trabalho

### Cenário 1: Planejar + Implementar + Testar (Padrão)
```
1. @Arquiteto Robo Binance → Planejar a mudança
   "Quero fazer X. Qual é o impacto?"
   ↓
2. @Implementador Python Binance → Executar a mudança
   "Implementa conforme o plano do arquiteto"
   ↓
3. @Guardião de Testes → Validar cobertura
   "Verifica se a mudança está bem coberta"
```

### Cenário 2: Investigar + Corrigir (Diagnóstico)
```
1. @Diagnóstico e Performance → Investigar sintoma
   "Por que o sistema está lento? Identify causa raiz"
   ↓
2. @Implementador Python Binance → Corrigir
   "Implementa a correção com testes"
   ↓
3. @Guardião de Testes → Validar
   "Verifica se a correção está bem protegida"
```

### Cenário 3: Otimizar IA (Payload/Custo)
```
1. @IA, Payload e LLM → Analisar e otimizar
   "Reduza custo/tamanho de payload"
   ↓
2. @Implementador Python Binance → Executar (opcional)
   "Se necessário implementar mudanças complexas"
   ↓
3. @Guardião de Testes → Validar cobertura
   "Confirma que testes cobrem mudanças"
```

### Cenário 4: Performance + Diagnóstico
```
1. @Diagnóstico e Performance → Investigar gargalo
   "Por que está lento? Onde está o problema?"
   ↓
2. @Implementador Python Binance → Corrigir
   "Implementa a solução"
   ↓
3. @Guardião de Testes → Validar
   "Testes cobrem a correção?"
```

### Cenário 5: Apenas Análise
```
@Arquiteto Robo Binance → Avaliar viabilidade
"Esta mudança é segura? Quais são os riscos?"
```

### Cenário 6: Apenas Implementação
```
@Implementador Python Binance → Implementar direto
"Implementa a feature/bugfix X com testes"
```

### Cenário 7: Apenas Testes
```
@Guardião de Testes → Revisar cobertura
"Cobre o arquivo X? Que testes faltam?"
```

### Cenário 8: Apenas Diagnóstico
```
@Diagnóstico e Performance → Investigar problema
"Por que está acontecendo X? Qual é a causa?"
```

### Cenário 9: Apenas IA/Payload
```
@IA, Payload e LLM → Otimizar ou diagnosticar
"Reduza custo de payload" ou "Por que custo está alto?"
```

---

### 4️⃣ **Diagnóstico e Performance**
**Arquivo:** `.github/agents/diagnostico-performance.agent.md`

**Propósito:** Investigar bugs, gargalos, falhas intermitentes e problemas de performance.

**Quando usar:**
- Sistema está lento ou virou gargalo
- Traços de travamento ou deadlock
- WebSocket reconectando frequentemente
- Filas crescendo sem controle
- Buffer acumulando dados
- Memória crescendo sem parar
- Errores intermitentes difíceis de reproduzir
- Async event loop bloqueado
- Circuit breaker disparando

**Ferramentas:** `read`, `search`, `execute`

**Exemplo de uso:**
```
@Diagnóstico e Performance O market_orchestrator está muito lento. 
Investigue possíveis gargalos, verifique concorrência async, 
identifique testes e proponha correção com validação.
```

**Saída esperada:**
- Sintoma observado
- Hipóteses de causa raiz
- Arquivos/módulos analisados
- Evidências observáveis
- Impacto operacional
- Testes existentes
- Correção proposta
- Métricas/logs recomendados
- Riscos remanescentes

---

### 5️⃣ **IA, Payload e LLM**
**Arquivo:** `.github/agents/ia-payload-llm.agent.md`

**Propósito:** Otimizar camada de IA, reduzir custo/tamanho de payload, fortalecer validação de respostas.

**Quando usar:**
- Reduzir tamanho de payload IA
- Reduzir custo de chamadas LLM
- Fortalecer validação de respostas
- Melhorar guardrails
- Otimizar cache/throttling
- Diagnosticar problemas de payload
- Melhorar fallback de IA
- Validar compatibilidade de formato

**Ferramentas:** `read`, `edit`, `search`, `execute`

**Exemplo de uso:**
```
@IA, Payload e LLM Otimize o payload em build_compact_payload.py 
para reduzir tamanho sem perder contexto, verifique testes e 
entregue resumo com impacto em custo/tamanho.
```

**Saída esperada:**
- Objetivo e arquivos envolvidos
- Impacto na camada de IA
- Riscos de payload/resposta
- Testes existentes
- Mudanças implementadas
- Impacto esperado em custo/tamanho/robustez
- Próximo passo

---

## 📎 Sempre Anexe Este Arquivo

Ao usar **qualquer um dos 5 agentes**, **sempre anexe** `ESTRUTURA_SISTEMA_COMPLETO.md` porque:

✅ Ajuda o agente a entender a arquitetura completa  
✅ Localizar arquivos e módulos com precisão  
✅ Evitar mover módulos críticos  
✅ Respeitar proxies da raiz  
✅ Encontrar testes relacionados  

**Como anexar:**
1. Abra o arquivo `ESTRUTURA_SISTEMA_COMPLETO.md`
2. Mencione o agente: `@Arquiteto Robo Binance ...`
3. O arquivo será automaticamente anexado ao contexto

---

## 🎯 Boas Práticas

### ✅ FAÇA

- **Use o Arquiteto primeiro** para avaliar impacto antes de implementar alterações grandes
- **Use o Diagnóstico** quando enfrentar problemas obscuros ou intermitentes
- **Use o IA, Payload e LLM** para qualquer mudança na camada de IA
- **Seja específico** nas suas perguntas (qual arquivo? qual comportamento?)
- **Anexe ESTRUTURA_SISTEMA_COMPLETO.md** sempre que for usar qualquer agente
- **Siga o fluxo**: Análise → Implementação → Testes
- **Respeite as restrições** de módulos críticos na raiz
- **Verifique testes** antes de assumir falta de cobertura

### ❌ NÃO FAÇA

- Não tente mover `ai_analyzer_qwen.py`, `orderbook_analyzer.py` sem análise
- Não remova proxies de compatibilidade sem decisão do Arquiteto
- Não ignore testes existentes nunca
- Não assuma que um arquivo está coberto sem verificação real
- Não faça mudanças agressivas sem plano incremental
- Não ignore código assíncrono, WebSocket, logging e resiliência
- Não optimize para speed sem verificar estabilidade primeiro

---

### Exemplo 1: Investigar se é seguro mover um módulo

```
@Arquiteto Robo Binance

Quero mover todos os módulos de "trading" que estão na raiz 
(trade_buffer.py, alert_engine.py, etc) para a subpasta trading/.

É seguro? Quais importadores vão quebrar? 
Preciso criar proxies na raiz para backward compatibility?
```

### Exemplo 2: Implementar uma feature com testes

```
@Implementador Python Binance

Quero adicionar um novo validador de payload em common/ai_payload_validator_v2.py.
Este validador deve:
1. Validar que payload não excede 50KB
2. Validar que campos obrigatórios existem
3. Logar warning se payload está >40KB

Implementa o código, cria testes unitários e de integração.
```

### Exemplo 3: Revisar cobertura antes de mudança

```
@Guardião de Testes e Qualidade

Estou prestes a modificar flow_analyzer/core.py. 
Qual é a cobertura atual? Há testes de integração?
Que testes devo criar/atualizar antes de fazer a mudança?
```

### Exemplo 4: Investigar Lentidão (Diagnóstico)

```
@Diagnóstico e Performance

Analise por que o processamento do market_orchestrator está lento, 
verifique possíveis gargalos, identifique testes existentes 
relacionados e proponha correção com validação. (veja tabela acima)
2. Anexe ESTRUTURA_SISTEMA_COMPLETO.md se ainda não estava anexado
3. Seja mais específico e mencione arquivos/módulos concretos
4. Se necessário, chame o agente explicitamente com `@nome-do-agente`

**Se os agentes parecem não seguir as restrições:**
1. Verifique se você comentou sobre a restrição explicitamente em seu pedido
2. Releia o arquivo do agente em `.github/agents/` (seção "Restrições")
3. A restrição deve estar documentada no arquivo `.agent.md`

**Se testes não rodam:**
1. Use `@Guardião de Testes` para diagnosticar cobertura
2. Use `@Implementador Python` para ajustar testes
3. Certifique-se de ter `.venv` ativado e dependências instaladas

**Se performance está ruim:**
1. Use `@Diagnóstico e Performance` para investigar causa raiz
2. Depois use `@Implementador Python` para corrigir
3. Finalize com `@Guardião de Testes` para validar

**Se custo de IA está alto:**
1. Use `@IA, Payload e LLM` para analisar e otimizar
2. Revise payload builder, compressor, throttler, cache
3. Valide impacto com tests/payload/WebSocket, 
diga quais módulos e testes estão envolvidos, 
proponha correção segura e finalize com resumo completo.
```

### Exemplo 6: Otimizar Payload (IA, Payload e LLM)

```
@IA, Payload e LLM

Analise o build_compact_payload.py e o market_orchestrator/ai/ 
para reduzir o tamanho do payload sem perder contexto crítico. 
Verifique os testes existentes, crie os faltantes 
e finalize com resumo completo do impacto em custo/tamanho.
```

### Exemplo 7: Validar Resposta de IA (IA, Payload e LLM)

```
@IA, Payload e LLM

Revise o ai_response_validator.py e o fluxo de resposta da IA, 
fortaleça a validação contra respostas malformadas, 
confira os testes existentes e ajuste a cobertura necessária.
```

### Exemplo 8: Investigar Buffer Crescendo (Diagnóstico)

```
@Diagnóstico e Performance

Verifique se o trade_buffer pode estar acumulando dados sem controle, 
analise risco de backpressure, confira testes existentes 
e sugira ajustes com testes.
```

### Exemplo 9: Analisar Custo de IA (IA, Payload e LLM)

```
@IA, Payload e LLM

Investigue por que o custo de chamadas de IA está alto, 
analise payload, compressão, cache e throttling, 
verifique os testes existentes e sugira correções com validação.
```

---

## 📞 Suporte e Troubleshooting

**Se um agente não responde ao seu pedido:**
1. Verifique se a descrição do agente menciona seu use case
2. Anexe ESTRUTURA_SISTEMA_COMPLETO.md se ainda não estava anexado
3. Seja mais específico na sua pergunta
4. Se necessário, chame o agente explicitamente com `@nome-do-agente`

**Se os agentes parecem não seguir as restrições:**
1. Verifique se você comentou sobre a restrição explicitamente
2. Releia o arquivo do agente em `.github/agents/` e ajuste se necessário
3. A restrição deve estar na seção "Restrições" do arquivo

---

## 🚀 Próximas Etapas

1. **Use o Arquiteto** para planejar próximas mudanças arquiteturais
2. **Use o Implementador** para executar mudanças com confiança
3. **Use o Guardião** para garantir cobertura de testes
4. **Use o Diagnóstico** quando enfrentar problemas de performance/estabilidade
5. **Use o IA, Payload e LLM** para qualquer mudança na camada de IA
6. **Documente decisões** no arquivo de estrutura quando fizer mudanças importantes

---

## 📊 Quadro de Referência Rápida

| Situação | Agente | Quando |
|----------|--------|--------|
| Planejar mudança | Arquiteto | Antes de implementar |
| Implementar código | Implementador | Quando pronto para código |
| Verificar testes | Guardião | Após implementação |
| Sistema lento | Diagnóstico | Investigar causa raiz |
| WebSocket cai | Diagnóstico | Falhas intermitentes |
| Fila crescendo | Diagnóstico | Acumular dados |
| Custo IA alto | IA, Payload e LLM | Reduzir custo/tamanho |
| Payload inválido | IA, Payload e LLM | Resposta quebrada |
| Race condition | Diagnóstico | Problema concorrência |
| Bug em arquivo crítico | Implementador | Com testes |

---

## 🎯 Os 5 Agentes - Resumo Final

✅ **Arquiteto Robo Binance** - Analisa impacto antes de mudanças  
✅ **Implementador Python Binance** - Implementa código com segurança  
✅ **Guardião de Testes e Qualidade** - Valida cobertura de testes  
✅ **Diagnóstico e Performance** - Investiga bugs e gargalos  
✅ **IA, Payload e LLM** - Otimiza camada de IA

---

**Última atualização:** Abril 2026  
**Status:** 5 agentes customizados criados e documentados

