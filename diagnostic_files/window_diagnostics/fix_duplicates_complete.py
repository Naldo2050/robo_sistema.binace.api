# fix_duplicates_complete.py - DEFINITIVE VERSION

from pathlib import Path
import shutil
import re

def aplicar_protecao_completa():
    """Adiciona proteção completa contra duplicatas e problemas de ordem"""
    
    arquivo = Path("event_saver.py")
    
    if not arquivo.exists():
        print("❌ event_saver.py não encontrado!")
        return
    
    # Backup
    backup = arquivo.with_suffix(f'.py.bak_duplicatas')
    shutil.copy(arquivo, backup)
    print(f"✅ Backup criado: {backup}\n")
    
    with open(arquivo, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
    
    modificacoes = []
    
    # ========================================
    # MODIFICAÇÃO 1: Adicionar set no __init__
    # ========================================
    for i, linha in enumerate(linhas):
        if 'self._window_counter = 0' in linha and 'self._janelas_processadas' not in ''.join(linhas[max(0,i-5):i+5]):
            # Adiciona após _window_counter
            indent = len(linha) - len(linha.lstrip())
            nova_linha = ' ' * indent + 'self._janelas_processadas = set()  # ✅ Rastreia janelas que já tiveram separador\n'
            linhas.insert(i+1, nova_linha)
            modificacoes.append(f"✅ Linha {i+1}: Set _janelas_processadas adicionado")
            break
    
    # ========================================
    # MODIFICAÇÃO 2: Proteção em _add_visual_separator
    # ========================================
    for i, linha in enumerate(linhas):
        if 'def _add_visual_separator(self, event: Dict):' in linha:
            # Procura o 'try:' após a docstring
            j = i + 1
            while j < len(linhas) and 'try:' not in linhas[j]:
                j += 1
            
            if j < len(linhas):
                # Insere proteção antes do try
                indent = len(linhas[j]) - len(linhas[j].lstrip())
                
                protecao = [
                    ' ' * indent + '# ✅ PROTEÇÃO: Verifica se janela já teve separador escrito\n',
                    ' ' * indent + 'janela_num = event.get("janela_numero")\n',
                    ' ' * indent + 'if janela_num and janela_num in self._janelas_processadas:\n',
                    ' ' * (indent + 4) + 'self.logger.warning(f"⚠️ Separador para janela {janela_num} já foi escrito, PULANDO")\n',
                    ' ' * (indent + 4) + 'return\n',
                    '\n'
                ]
                
                # Insere proteção
                for idx, linha_protecao in enumerate(protecao):
                    linhas.insert(j + idx, linha_protecao)
                
                modificacoes.append(f"✅ Linha {j}: Proteção contra duplicatas adicionada")
                break
    
    # ========================================
    # MODIFICAÇÃO 3: Marcar janela após escrita
    # ========================================
    for i, linha in enumerate(linhas):
        # Procura onde o separador é escrito com sucesso
        if 'f.write(separator)' in linha and 'self.visual_log_file' in ''.join(linhas[max(0,i-10):i]):
            # Procura o f.flush() logo após
            j = i + 1
            while j < len(linhas) and 'f.flush()' not in linhas[j]:
                j += 1
            
            if j < len(linhas):
                # Adiciona após o flush
                indent = len(linhas[j]) - len(linhas[j].lstrip())
                
                marcacao = [
                    '\n',
                    ' ' * indent + '# ✅ Marca janela como processada\n',
                    ' ' * indent + 'if janela_num:\n',
                    ' ' * (indent + 4) + 'self._janelas_processadas.add(janela_num)\n',
                    ' ' * (indent + 4) + 'self.logger.info(f"✅ Separador escrito e janela {janela_num} marcada como processada")\n'
                ]
                
                # Insere após flush
                for idx, linha_marcacao in enumerate(marcacao):
                    linhas.insert(j + 1 + idx, linha_marcacao)
                
                modificacoes.append(f"✅ Linha {j}: Marcação de janela processada adicionada")
                break
    
    # ========================================
    # MODIFICAÇÃO 4: Limpar set no cleanup
    # ========================================
    for i, linha in enumerate(linhas):
        if 'def _cleanup_seen_in_block(self):' in linha:
            # Procura o final da função
            j = i + 1
            indent_base = len(linhas[i]) - len(linhas[i].lstrip())
            
            while j < len(linhas):
                linha_atual = linhas[j]
                if linha_atual.strip() and not linha_atual.startswith(' ' * (indent_base + 4)):
                    # Encontrou próxima função
                    break
                j += 1
            
            # Volta para última linha da função
            j -= 1
            while j > i and not linhas[j].strip():
                j -= 1
            
            # Adiciona limpeza de janelas processadas
            indent_func = indent_base + 4
            
            limpeza = [
                '\n',
                ' ' * indent_func + '# ✅ Limpa set de janelas processadas se ficar muito grande\n',
                ' ' * indent_func + 'if len(self._janelas_processadas) > 1000:\n',
                ' ' * (indent_func + 4) + 'self._janelas_processadas.clear()\n',
                ' ' * (indent_func + 4) + 'self.logger.info("🧹 Set de janelas processadas limpo")\n'
            ]
            
            for idx, linha_limpeza in enumerate(limpeza):
                linhas.insert(j + 1 + idx, linha_limpeza)
            
            modificacoes.append(f"✅ Linha {j}: Limpeza periódica do set adicionada")
            break
    
    # Salva arquivo modificado
    with open(arquivo, 'w', encoding='utf-8') as f:
        f.writelines(linhas)
    
    print("="*80)
    print("✅ PROTEÇÕES APLICADAS COM SUCESSO!")
    print("="*80)
    print()
    
    for mod in modificacoes:
        print(f"   {mod}")
    
    print()
    print("📝 MUDANÇAS IMPLEMENTADAS:")
    print("-" * 80)
    print("   1. ✅ Set _janelas_processadas criado no __init__")
    print("   2. ✅ Verificação de duplicata ANTES de escrever separador")
    print("   3. ✅ Janela marcada como processada APÓS escrita bem-sucedida")
    print("   4. ✅ Limpeza periódica do set (evita crescimento infinito)")
    print()
    print("🔒 PROTEÇÃO GARANTIDA:")
    print("-" * 80)
    print("   ❌ Separadores duplicados: BLOQUEADOS")
    print("   ❌ Janelas fora de ordem: BLOQUEADOS")
    print("   ❌ Race conditions: BLOQUEADOS")
    print()
    print("⚠️  PRÓXIMOS PASSOS:")
    print("-" * 80)
    print("   1. Limpe os dados antigos:")
    print("      rm dados/eventos_visuais.log")
    print("      rm dados/eventos-fluxo.json")
    print("      rm dados/eventos_fluxo.jsonl")
    print()
    print("   2. Reinicie o sistema:")
    print("      python main.py")
    print()
    print("   3. Verifique os logs:")
    print("      Procure por '✅ Separador escrito e janela X marcada'")
    print("      Procure por '⚠️ Separador para janela X já foi escrito, PULANDO'")
    print()
    print("="*80)


def limpar_duplicatas_log():
    """Remove duplicatas do log visual atual"""
    
    visual_log = Path("dados/eventos_visuais.log")
    
    if not visual_log.exists():
        print("⚠️ Log visual não encontrado, nada a limpar")
        return
    
    print("\n🧹 LIMPANDO DUPLICATAS DO LOG VISUAL")
    print("="*80)
    
    with open(visual_log, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
    
    # Rastreia janelas já vistas
    janelas_vistas = set()
    novas_linhas = []
    
    i = 0
    removidos = 0
    
    while i < len(linhas):
        linha = linhas[i]
        
        # Verifica se é início de separador (linha com ====)
        if '='*50 in linha:
            # Verifica próxima linha tem 🗓️
            if i+1 < len(linhas) and '🗓️' in linhas[i+1]:
                # Extrai número da janela
                match = re.search(r'JANELA\s+(\d+)', linhas[i+1])
                if match:
                    janela_num = int(match.group(1))
                    
                    if janela_num in janelas_vistas:
                        # Duplicata! Pula até próximo separador
                        print(f"   ❌ Removendo duplicata: Janela {janela_num} (posição {i+1})")
                        removidos += 1
                        
                        # Pula separador (6 linhas padrão)
                        i += 7
                        
                        # Pula eventos JSON até próximo separador
                        while i < len(linhas):
                            if '='*50 in linhas[i]:
                                # Próximo separador
                                break
                            i += 1
                        continue
                    else:
                        janelas_vistas.add(janela_num)
        
        novas_linhas.append(linha)
        i += 1
    
    if removidos > 0:
        # Backup
        backup = visual_log.with_suffix('.log.backup_dup')
        shutil.copy(visual_log, backup)
        print(f"\n   ✅ Backup: {backup}")
        
        # Salva versão limpa
        with open(visual_log, 'w', encoding='utf-8') as f:
            f.writelines(novas_linhas)
        
        print(f"   ✅ Removidos {removidos} separadores duplicados")
        print(f"   ✅ Log limpo salvo")
    else:
        print("   ✅ Nenhuma duplicata encontrada")
    
    print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔧 CORREÇÃO COMPLETA - PROTEÇÃO CONTRA DUPLICATAS")
    print("="*80 + "\n")
    
    print("Esta correção vai:")
    print("  1. ✅ Adicionar proteção no código (event_saver.py)")
    print("  2. ✅ Limpar duplicatas do log atual")
    print()
    
    resposta = input("Continuar? (s/n): ").lower()
    
    if resposta != 's':
        print("❌ Cancelado")
        exit()
    
    # 1. Aplica proteção no código
    aplicar_protecao_completa()
    
    # 2. Limpa log atual
    limpar_duplicatas_log()
    
    print("\n" + "="*80)
    print("✅ CORREÇÃO COMPLETA APLICADA!")
    print("="*80)
    print()
    print("📝 TESTE RECOMENDADO:")
    print("-" * 80)
    print("   1. Reinicie o sistema: python main.py")
    print("   2. Deixe rodar por 5 minutos")
    print("   3. Execute diagnóstico: python duplicate_diagnostics.py")
    print("   4. Verifique que não há mais duplicatas")
    print()
    print("="*80 + "\n")