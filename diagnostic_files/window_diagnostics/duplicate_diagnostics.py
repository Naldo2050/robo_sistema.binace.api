# duplicate_diagnostics.py - v1.0

import re
from pathlib import Path
from collections import Counter

class DiagnosticoDuplicatas:
    """Encontra separadores duplicados no log visual"""
    
    def __init__(self):
        self.visual_log = Path("dados/eventos_visuais.log")
    
    def executar(self):
        print("\n" + "="*100)
        print("🔍 DIAGNÓSTICO DE SEPARADORES DUPLICADOS")
        print("="*100 + "\n")
        
        if not self.visual_log.exists():
            print("❌ Log visual não encontrado\n")
            return
        
        with open(self.visual_log, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        
        # Regex para capturar número da janela
        pattern = re.compile(r'🗓️\s+JANELA\s+(\d+)')
        matches = pattern.findall(conteudo)
        
        print(f"📊 ANÁLISE DE SEPARADORES:")
        print("-" * 100)
        print(f"   Total de separadores encontrados: {len(matches)}")
        print(f"   Janelas únicas: {len(set(matches))}")
        print(f"   Diferença (duplicatas): {len(matches) - len(set(matches))}\n")
        
        # Conta ocorrências
        contagem = Counter(matches)
        
        # Mostra duplicatas
        duplicatas = {k: v for k, v in contagem.items() if v > 1}
        
        if duplicatas:
            print(f"🔴 JANELAS COM SEPARADORES DUPLICADOS:")
            print("-" * 100)
            for janela, count in sorted(duplicatas.items(), key=lambda x: int(x[0])):
                print(f"   Janela {janela}: {count} separadores (duplicada {count-1}x)")
            print()
        else:
            print("✅ Nenhuma duplicata encontrada\n")
        
        # Mostra todas as janelas em ordem
        print(f"📋 LISTA COMPLETA (em ordem de aparição):")
        print("-" * 100)
        
        linha_atual = 1
        for i, janela in enumerate(matches, 1):
            # Encontra linha do separador no arquivo
            simbolo = "🔴" if contagem[janela] > 1 else "✅"
            print(f"   {i:2}. {simbolo} Janela {janela}")
        
        print()
        
        # Análise temporal
        self.analisar_ordem_temporal(matches)
        
        # Gera correção
        self.gerar_correcao(duplicatas)
    
    def analisar_ordem_temporal(self, janelas):
        """Verifica se janelas estão em ordem temporal"""
        print(f"⏰ ANÁLISE DE ORDEM TEMPORAL:")
        print("-" * 100)
        
        fora_de_ordem = []
        
        for i in range(len(janelas) - 1):
            atual = int(janelas[i])
            proximo = int(janelas[i+1])
            
            if proximo < atual:
                fora_de_ordem.append({
                    'posicao': i+1,
                    'janela_atual': atual,
                    'janela_proxima': proximo
                })
        
        if fora_de_ordem:
            print(f"   🔴 Separadores fora de ordem temporal: {len(fora_de_ordem)}\n")
            for item in fora_de_ordem[:10]:
                print(f"      Posição {item['posicao']}: Janela {item['janela_atual']} → {item['janela_proxima']}")
            print()
        else:
            print(f"   ✅ Todos os separadores estão em ordem cronológica\n")
    
    def gerar_correcao(self, duplicatas):
        """Gera script para remover duplicatas"""
        print(f"🔧 CORREÇÃO:")
        print("-" * 100)
        
        if not duplicatas:
            print("   ✅ Nenhuma correção necessária\n")
            return
        
        print(f"   Opções:")
        print(f"   1. Limpar duplicatas automaticamente")
        print(f"   2. Adicionar proteção contra duplicatas no código")
        print()
        
        resposta = input("   Escolha (1/2): ").strip()
        
        if resposta == '1':
            self.limpar_duplicatas()
        elif resposta == '2':
            self.adicionar_protecao()
    
    def limpar_duplicatas(self):
        """Remove separadores duplicados mantendo apenas o primeiro"""
        print("\n   🧹 Limpando duplicatas...")
        
        with open(self.visual_log, 'r', encoding='utf-8') as f:
            linhas = f.readlines()
        
        # Rastreia janelas já vistas
        janelas_vistas = set()
        novas_linhas = []
        
        i = 0
        removidos = 0
        
        while i < len(linhas):
            linha = linhas[i]
            
            # Verifica se é início de separador
            if '='*50 in linha and i+1 < len(linhas) and '🗓️' in linhas[i+1]:
                # Extrai número da janela
                match = re.search(r'JANELA\s+(\d+)', linhas[i+1])
                if match:
                    janela_num = match.group(1)
                    
                    if janela_num in janelas_vistas:
                        # Pula este separador e conteúdo até próximo separador
                        print(f"      ❌ Removendo separador duplicado: Janela {janela_num}")
                        removidos += 1
                        
                        # Pula até próximo separador ou próximo evento JSON
                        i += 1
                        while i < len(linhas):
                            if ('='*50 in linhas[i] and i+1 < len(linhas) and '🗓️' in linhas[i+1]):
                                # Próximo separador
                                break
                            elif linhas[i].strip().startswith('{'):
                                # Início de evento JSON
                                break
                            i += 1
                        continue
                    else:
                        janelas_vistas.add(janela_num)
            
            novas_linhas.append(linha)
            i += 1
        
        # Backup
        backup = self.visual_log.with_suffix('.log.backup')
        import shutil
        shutil.copy(self.visual_log, backup)
        print(f"\n   ✅ Backup criado: {backup}")
        
        # Salva versão limpa
        with open(self.visual_log, 'w', encoding='utf-8') as f:
            f.writelines(novas_linhas)
        
        print(f"   ✅ Removidos {removidos} separadores duplicados")
        print(f"   ✅ Arquivo limpo salvo\n")
    
    def adicionar_protecao(self):
        """Adiciona proteção contra duplicatas no event_saver.py"""
        codigo = '''# fix_duplicatas.py - Proteção contra separadores duplicados

from pathlib import Path
import shutil

def aplicar_protecao():
    """Adiciona set() para rastrear janelas já processadas"""
    
    arquivo = Path("event_saver.py")
    
    if not arquivo.exists():
        print("❌ event_saver.py não encontrado!")
        return
    
    # Backup
    backup = arquivo.with_suffix('.py.bak2')
    shutil.copy(arquivo, backup)
    print(f"✅ Backup: {backup}")
    
    with open(arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    
    # 1. Adiciona set no __init__
    if 'self._window_counter = 0' in conteudo and 'self._janelas_processadas' not in conteudo:
        conteudo = conteudo.replace(
            'self._window_counter = 0',
            'self._window_counter = 0\\n        self._janelas_processadas = set()  # Rastreia janelas que já tiveram separador'
        )
        print("✅ Set _janelas_processadas adicionado")
    
    # 2. Modifica _add_visual_separator para verificar duplicatas
    if 'def _add_visual_separator(self, event: Dict):' in conteudo:
        # Procura o início da função
        inicio = conteudo.find('def _add_visual_separator(self, event: Dict):')
        if inicio != -1:
            # Encontra o 'try:'
            try_pos = conteudo.find('try:', inicio)
            if try_pos != -1:
                # Insere verificação antes do try
                verificacao = """
        # ✅ PROTEÇÃO: Verifica se janela já teve separador escrito
        janela_num = event.get('janela_numero')
        if janela_num in self._janelas_processadas:
            self.logger.warning(f"⚠️ Separador para janela {janela_num} já foi escrito, pulando")
            return
        
        """
                conteudo = conteudo[:try_pos] + verificacao + conteudo[try_pos:]
                print("✅ Proteção contra duplicatas adicionada")
    
    # 3. Adiciona janela ao set após escrita bem-sucedida
    if 'self.logger.info(f"✅ Separador escrito com sucesso")' not in conteudo:
        # Procura onde o separador é escrito
        if 'f.write(separator)' in conteudo:
            conteudo = conteudo.replace(
                'f.write(separator)\\n                        f.flush()',
                'f.write(separator)\\n                        f.flush()\\n                    \\n                    # Marca janela como processada\\n                    if janela_num:\\n                        self._janelas_processadas.add(janela_num)\\n                        self.logger.debug(f"✅ Janela {janela_num} marcada como processada")'
            )
            print("✅ Marcação de janela processada adicionada")
    
    # Salva
    with open(arquivo, 'w', encoding='utf-8') as f:
        f.write(conteudo)
    
    print("\\n✅ Proteções aplicadas!")
    print("\\n📝 Mudanças:")
    print("   1. Set _janelas_processadas criado")
    print("   2. Verificação de duplicata antes de escrever separador")
    print("   3. Janela marcada como processada após escrita")
    print("\\n⚠️ REINICIE o sistema para aplicar")

if __name__ == "__main__":
    print("\\n" + "="*80)
    print("🔧 PROTEÇÃO CONTRA SEPARADORES DUPLICADOS")
    print("="*80 + "\\n")
    
    aplicar_protecao()
    
    print("\\n" + "="*80)
'''
        
        with open('fix_duplicates.py', 'w', encoding='utf-8') as f:
            f.write(codigo)

        print(f"\n   ✅ Script criado: fix_duplicates.py")
        print(f"\n   📝 Execute:")
        print(f"      python fix_duplicates.py")
        print()


if __name__ == "__main__":
    diag = DiagnosticoDuplicatas()
    diag.executar()
    
    print("="*100)