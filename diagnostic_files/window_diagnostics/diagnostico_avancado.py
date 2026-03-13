# diagnostico_avancado.py - v2.0

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class DiagnosticoAvancado:
    """Diagnostica timestamps negativos e eventos sem separador"""
    
    def __init__(self):
        self.dados_dir = Path("dados")
        self.visual_log = self.dados_dir / "eventos_visuais.log"
        self.json_file = self.dados_dir / "eventos-fluxo.json"
        
    def executar(self):
        print("\n" + "="*100)
        print("🔬 DIAGNÓSTICO AVANÇADO - TIMESTAMPS E SEPARADORES")
        print("="*100 + "\n")
        
        # 1. Analisa timestamps negativos
        self.analisar_timestamps_negativos()
        
        # 2. Verifica eventos sem separador
        self.verificar_eventos_sem_separador()
        
        # 3. Analisa flags de separador
        self.analisar_flags_separador()
        
        # 4. Testa escrita de separador
        self.testar_escrita_separador()
    
    def analisar_timestamps_negativos(self):
        """Encontra eventos com timestamps negativos"""
        print("⏰ ANÁLISE DE TIMESTAMPS NEGATIVOS")
        print("-" * 100)
        
        if not self.json_file.exists():
            print("❌ JSON não encontrado\n")
            return
        
        with open(self.json_file, 'r', encoding='utf-8') as f:
            eventos = json.load(f)
        
        # Agrupa por janela
        por_janela = defaultdict(list)
        for evento in eventos:
            janela = evento.get('janela_numero')
            if janela:
                por_janela[janela].append(evento)
        
        # Analisa ordem temporal
        print(f"📊 Eventos por janela:\n")
        
        problemas = []
        
        for janela in sorted(por_janela.keys()):
            eventos_janela = por_janela[janela]
            
            # Pega epochs
            epochs = []
            for e in eventos_janela:
                epoch = e.get('epoch_ms') or e.get('window_id')
                if epoch:
                    epochs.append({
                        'epoch': int(epoch),
                        'tipo': e.get('tipo_evento'),
                        'timestamp': e.get('timestamp_utc', 'N/A')
                    })
            
            if len(epochs) > 1:
                # Ordena por ordem no arquivo
                # Verifica se estão em ordem temporal
                for i in range(len(epochs) - 1):
                    diff_ms = epochs[i+1]['epoch'] - epochs[i]['epoch']
                    diff_seg = diff_ms / 1000
                    
                    if diff_seg < 0:
                        problemas.append({
                            'janela': janela,
                            'evento1': epochs[i],
                            'evento2': epochs[i+1],
                            'diff_seg': diff_seg
                        })
                        
                        print(f"   🔴 Janela {janela}: TIMESTAMPS FORA DE ORDEM")
                        print(f"      Evento 1: {epochs[i]['tipo']} | {epochs[i]['timestamp']} | Epoch: {epochs[i]['epoch']}")
                        print(f"      Evento 2: {epochs[i+1]['tipo']} | {epochs[i+1]['timestamp']} | Epoch: {epochs[i+1]['epoch']}")
                        print(f"      Diferença: {diff_seg:.1f}s (NEGATIVO!)\n")
            
            # Mostra janelas com múltiplos eventos
            if len(eventos_janela) > 1:
                tipos = [e.get('tipo_evento', 'N/A') for e in eventos_janela]
                print(f"   ⚠️ Janela {janela}: {len(eventos_janela)} eventos ({', '.join(tipos)})")
        
        if not problemas:
            print("\n   ✅ Nenhum timestamp negativo detectado")
        else:
            print(f"\n🔴 Total de problemas: {len(problemas)}")
            
            print("\n💡 CAUSA:")
            print("   Eventos sendo adicionados ANTES de eventos mais antigos")
            print("   → Buffer processando fora de ordem")
            print("   → Ou eventos chegando com epoch_ms do passado")
        
        print()
    
    def verificar_eventos_sem_separador(self):
        """Verifica se há eventos no JSON sem separador correspondente"""
        print("📝 VERIFICAÇÃO DE SEPARADORES")
        print("-" * 100)
        
        # Conta separadores no log visual
        separadores_count = 0
        if self.visual_log.exists():
            with open(self.visual_log, 'r', encoding='utf-8') as f:
                conteudo = f.read()
                separadores_count = conteudo.count('🗓️  JANELA')
        
        # Conta janelas únicas no JSON
        with open(self.json_file, 'r', encoding='utf-8') as f:
            eventos = json.load(f)
        
        janelas_json = set(e.get('janela_numero') for e in eventos if e.get('janela_numero'))
        
        print(f"   Separadores no log visual: {separadores_count}")
        print(f"   Janelas únicas no JSON:    {len(janelas_json)}")
        
        if separadores_count == len(janelas_json):
            print(f"   ✅ Números batem!\n")
        else:
            diff = len(janelas_json) - separadores_count
            print(f"   ❌ Diferença: {diff} janelas sem separador\n")
            
            print(f"💡 PROBLEMA:")
            print(f"   Janelas estão sendo criadas no JSON mas separador não está sendo escrito")
            print(f"   → Verificar _add_visual_separator()")
            print(f"   → Verificar lock timeout")
        
        print()
    
    def analisar_flags_separador(self):
        """Verifica se eventos têm a flag _needs_separator"""
        print("🚩 ANÁLISE DE FLAGS _needs_separator")
        print("-" * 100)
        
        with open(self.json_file, 'r', encoding='utf-8') as f:
            eventos = json.load(f)
        
        # Verifica flags
        com_flag = []
        sem_flag = []
        
        for i, evento in enumerate(eventos):
            if evento.get('_needs_separator'):
                com_flag.append({
                    'index': i,
                    'janela': evento.get('janela_numero'),
                    'tipo': evento.get('tipo_evento')
                })
            elif evento.get('janela_numero'):
                # Verifica se é o primeiro evento da janela
                janela = evento.get('janela_numero')
                
                # Conta eventos anteriores na mesma janela
                anteriores = sum(1 for e in eventos[:i] if e.get('janela_numero') == janela)
                
                if anteriores == 0:  # É o primeiro da janela
                    sem_flag.append({
                        'index': i,
                        'janela': janela,
                        'tipo': evento.get('tipo_evento')
                    })
        
        print(f"   Eventos com _needs_separator: {len(com_flag)}")
        print(f"   Primeiros eventos sem flag:   {len(sem_flag)}")
        
        if com_flag:
            print(f"\n   🔴 PROBLEMA: Eventos ainda têm a flag _needs_separator!")
            print(f"      → Flag deveria ser removida em _flush_buffer")
            print(f"      → Separador NÃO foi escrito\n")
            
            for e in com_flag[:5]:
                print(f"      Janela {e['janela']}: {e['tipo']} (index {e['index']})")
        
        if sem_flag:
            print(f"\n   ⚠️ Primeiros eventos sem flag (pode ser normal se já foi processada):")
            for e in sem_flag[:5]:
                print(f"      Janela {e['janela']}: {e['tipo']} (index {e['index']})")
        
        print()
    
    def testar_escrita_separador(self):
        """Testa se consegue escrever separador"""
        print("🧪 TESTE DE ESCRITA DE SEPARADOR")
        print("-" * 100)
        
        try:
            # Tenta escrever um separador de teste
            test_file = self.dados_dir / "test_separator.log"
            
            separador_teste = "\n" + "="*100 + "\n"
            separador_teste += "🧪 TESTE DE SEPARADOR\n"
            separador_teste += "⏰ " + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") + "\n"
            separador_teste += "="*100 + "\n"
            
            # Teste 1: Escrita simples
            print("   Teste 1: Escrita simples...")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(separador_teste)
                f.flush()
            print("   ✅ Sucesso\n")
            
            # Teste 2: Escrita com lock
            print("   Teste 2: Escrita com file lock...")
            from event_saver import acquire_file_lock, release_file_lock
            
            lock_file_path = test_file.with_suffix('.lock')
            lock_file = open(lock_file_path, 'w')
            
            lock_acquired = acquire_file_lock(lock_file, blocking=True, timeout=3.0)
            
            if lock_acquired:
                print("   ✅ Lock adquirido")
                
                with open(test_file, 'a', encoding='utf-8') as f:
                    f.write("\n✅ Escrita com lock OK\n")
                    f.flush()
                
                release_file_lock(lock_file)
                print("   ✅ Lock liberado\n")
            else:
                print("   ❌ Timeout ao adquirir lock!\n")
            
            lock_file.close()
            lock_file_path.unlink(missing_ok=True)
            
            # Teste 3: Verifica se EventSaver consegue escrever
            print("   Teste 3: Escrita via EventSaver...")
            from event_saver import EventSaver
            
            saver = EventSaver(sound_alert=False)
            
            # Força criação de evento com nova janela
            import time
            epoch_now = int(time.time() * 1000)
            epoch_minuto = (epoch_now // 60000) * 60000
            epoch_minuto += 60000 * 999  # Janela bem no futuro
            
            evento_teste = {
                'tipo_evento': 'TESTE_SEPARADOR_DIRETO',
                'epoch_ms': epoch_minuto,
                'window_id': epoch_minuto,
                'is_signal': False
            }
            
            # Chama _add_visual_separator diretamente
            evento_teste['janela_numero'] = 9999
            saver._add_visual_separator(evento_teste)
            
            # Verifica se foi escrito
            time.sleep(0.5)
            
            if self.visual_log.exists():
                with open(self.visual_log, 'r', encoding='utf-8') as f:
                    conteudo = f.read()
                
                if 'JANELA 9999' in conteudo:
                    print("   ✅ Separador escrito com sucesso!\n")
                    
                    # Remove teste
                    linhas = conteudo.split('\n')
                    novas_linhas = []
                    skip = False
                    
                    for linha in linhas:
                        if 'JANELA 9999' in linha:
                            skip = True
                        elif skip and '='*50 in linha:
                            skip = False
                            continue
                        
                        if not skip:
                            novas_linhas.append(linha)
                    
                    with open(self.visual_log, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(novas_linhas))
                    
                    print("   🧹 Teste removido do log\n")
                else:
                    print("   ❌ Separador NÃO foi escrito!\n")
                    print("   💡 PROBLEMA CONFIRMADO: _add_visual_separator() não está funcionando")
            
            saver.stop()
            
            # Limpa arquivo de teste
            test_file.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"   ❌ Erro no teste: {e}\n")
            import traceback
            traceback.print_exc()
        
        print()


def gerar_correcao_final():
    """Gera correção definitiva baseada nos achados"""
    codigo = '''# fix_separador_final.py - Correção definitiva

from pathlib import Path
import shutil

def aplicar_fix():
    """Aplica correção no _flush_buffer"""
    
    arquivo = Path("event_saver.py")
    
    if not arquivo.exists():
        print("❌ event_saver.py não encontrado!")
        return
    
    # Backup
    backup = arquivo.with_suffix('.py.bak')
    shutil.copy(arquivo, backup)
    print(f"✅ Backup: {backup}")
    
    with open(arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    
    # Adiciona log em _add_visual_separator
    if 'def _add_visual_separator(self, event: Dict):' in conteudo:
        # Adiciona log no início da função
        conteudo = conteudo.replace(
            '    def _add_visual_separator(self, event: Dict):\\n'
            '        """Adiciona separador visual para nova janela de tempo."""\\n'
            '        try:',
            '    def _add_visual_separator(self, event: Dict):\\n'
            '        """Adiciona separador visual para nova janela de tempo."""\\n'
            '        self.logger.info(f"📝 Escrevendo separador para janela {event.get(\\'janela_numero\\', \\'N/A\\')}")\\n'
            '        try:'
        )
        
        # Adiciona log após escrita
        conteudo = conteudo.replace(
            '                    with open(self.visual_log_file, "a", encoding="utf-8") as f:\\n'
            '                        f.write(separator)\\n'
            '                        f.flush()',
            '                    with open(self.visual_log_file, "a", encoding="utf-8") as f:\\n'
            '                        f.write(separator)\\n'
            '                        f.flush()\\n'
            '                    self.logger.info(f"✅ Separador escrito com sucesso")'
        )
        
        # Adiciona log em timeout
        conteudo = conteudo.replace(
            '                else:\\n'
            '                    self.logger.warning("Timeout ao adquirir lock do visual log")',
            '                else:\\n'
            '                    self.logger.error(f"❌ TIMEOUT ao adquirir lock - Separador NÃO foi escrito para janela {event.get(\\'janela_numero\\')}")'
        )
    
    # Adiciona log em _flush_buffer
    if 'def _flush_buffer(self, events: List[Dict]):' in conteudo:
        conteudo = conteudo.replace(
            '                if event.get("_needs_separator"):\\n'
            '                    self._add_visual_separator(event)\\n'
            '                    event.pop("_needs_separator", None)',
            '                if event.get("_needs_separator"):\\n'
            '                    self.logger.debug(f"🚩 Processando flag _needs_separator para janela {event.get(\\'janela_numero\\')}")\\n'
            '                    self._add_visual_separator(event)\\n'
            '                    event.pop("_needs_separator", None)\\n'
            '                    self.logger.debug(f"✅ Flag removida")'
        )
    
    # Salva
    with open(arquivo, 'w', encoding='utf-8') as f:
        f.write(conteudo)
    
    print("✅ Logs de debug adicionados!")
    print("\\n📝 Logs adicionados:")
    print("   - Início de _add_visual_separator")
    print("   - Após escrita bem-sucedida")
    print("   - Em caso de timeout")
    print("   - Processamento de flag em _flush_buffer")
    print("\\n🔍 Execute o sistema e verifique os logs!")

if __name__ == "__main__":
    print("\\n" + "="*80)
    print("🔧 ADIÇÃO DE LOGS DE DEBUG")
    print("="*80 + "\\n")
    
    aplicar_fix()
    
    print("\\n" + "="*80)
'''
    
    with open('fix_separador_final.py', 'w', encoding='utf-8') as f:
        f.write(codigo)
    
    print("✅ Arquivo 'fix_separador_final.py' criado!")


if __name__ == "__main__":
    diag = DiagnosticoAvancado()
    diag.executar()
    
    print("\n" + "="*100)
    resposta = input("🤔 Deseja gerar correção com logs de debug? (s/n): ").lower()
    
    if resposta == 's':
        gerar_correcao_final()
        print("\n📝 Execute:")
        print("   python fix_separador_final.py")
        print("   python main.py  # e observe os logs")
    
    print("\n" + "="*100)