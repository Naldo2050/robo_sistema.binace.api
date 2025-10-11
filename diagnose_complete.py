# diagnose_complete.py
"""Diagnóstico completo do fluxo de orderbook até a IA"""

def diagnose_flow():
    """Analisa todo o fluxo de dados do orderbook até a IA."""
    
    print("="*80)
    print("🔍 DIAGNÓSTICO COMPLETO DO FLUXO DE DADOS")
    print("="*80 + "\n")
    
    # 1. Verifica onde ob_event é usado
    print("📋 PASSO 1: Onde ob_event é obtido\n")
    
    with open("main.py", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    ob_event_creation = []
    for i, line in enumerate(lines, 1):
        if 'ob_event = self.orderbook_analyzer.analyze' in line:
            ob_event_creation.append((i, line.strip()))
    
    if ob_event_creation:
        for line_num, content in ob_event_creation:
            print(f"  Linha {line_num}: {content}")
    else:
        print("  ❌ Não encontrado!")
    
    # 2. Verifica onde ob_event é adicionado ao pipeline
    print("\n📋 PASSO 2: Onde ob_event é passado para pipeline/context\n")
    
    pipeline_add = []
    for i, line in enumerate(lines, 1):
        if 'orderbook_data=' in line and 'ob_event' in line:
            pipeline_add.append((i, line.strip()))
    
    if pipeline_add:
        for line_num, content in pipeline_add:
            print(f"  Linha {line_num}: {content}")
    else:
        print("  ❌ Não encontrado!")
    
    # 3. Verifica onde signals são criados
    print("\n📋 PASSO 3: Onde signals são detectados\n")
    
    signal_creation = []
    for i, line in enumerate(lines, 1):
        if 'signals = pipeline.detect_signals' in line:
            signal_creation.append((i, line.strip()))
    
    if signal_creation:
        for line_num, content in signal_creation:
            print(f"  Linha {line_num}: {content}")
    else:
        print("  ❌ Não encontrado!")
    
    # 4. Verifica onde orderbook_data é adicionado ao signal
    print("\n📋 PASSO 4: Onde orderbook_data deveria ser adicionado ao signal\n")
    
    orderbook_add = []
    for i, line in enumerate(lines, 1):
        if 'signal["orderbook_data"]' in line or "signal['orderbook_data']" in line:
            orderbook_add.append((i, line.strip()))
    
    if orderbook_add:
        for line_num, content in orderbook_add:
            print(f"  Linha {line_num}: {content}")
    else:
        print("  ❌ NÃO ENCONTRADO! ← ESTE É O PROBLEMA!")
    
    # 5. Verifica onde signal é passado para IA
    print("\n📋 PASSO 5: Onde signal é enviado para IA\n")
    
    ai_call = []
    for i, line in enumerate(lines, 1):
        if 'self.event_bus.publish("signal"' in line:
            ai_call.append((i, line.strip()))
    
    if ai_call:
        for line_num, content in ai_call:
            print(f"  Linha {line_num}: {content}")
    else:
        print("  ❌ Não encontrado!")
    
    # 6. ANÁLISE DO PROBLEMA
    print("\n" + "="*80)
    print("💡 ANÁLISE DO PROBLEMA")
    print("="*80 + "\n")
    
    if not orderbook_add:
        print("🔴 PROBLEMA CRÍTICO IDENTIFICADO:")
        print("   O código NUNCA adiciona 'orderbook_data' diretamente ao signal!")
        print("\n   O fluxo atual é:")
        print("   1. ob_event é obtido ✅")
        print("   2. ob_event é passado para pipeline.add_context() ✅")
        print("   3. pipeline.detect_signals() cria signals ✅")
        print("   4. orderbook_data deveria ser adicionado ao signal ❌ FALTANDO!")
        print("   5. signal é enviado para IA com dados incompletos ❌")
        
        print("\n🔧 SOLUÇÃO NECESSÁRIA:")
        print("   Adicionar ANTES de 'self.event_bus.publish(\"signal\", signal)':")
        print("   signal['orderbook_data'] = ob_event.get('orderbook_data', {})")
        
    else:
        print("✅ Código para adicionar orderbook_data EXISTE")
        print("   Mas pode não estar sendo executado corretamente.")
        print("\n   Verificando se está no lugar certo...")
        
        # Verifica se está antes do event_bus.publish
        publish_lines = [i for i, line in enumerate(lines, 1) 
                        if 'self.event_bus.publish("signal"' in line]
        
        if publish_lines:
            first_publish = publish_lines[0]
            last_orderbook_add = orderbook_add[-1][0]
            
            if last_orderbook_add < first_publish:
                print(f"\n   ✅ orderbook_data é adicionado ANTES de publish (linha {last_orderbook_add} < {first_publish})")
            else:
                print(f"\n   ⚠️ orderbook_data pode estar sendo adicionado DEPOIS de publish!")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    diagnose_flow()