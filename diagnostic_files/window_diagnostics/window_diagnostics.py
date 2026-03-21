# window_diagnostics.py - v1.0


import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

class DiagnosticoJanelas:
    """Diagnostica discrepância entre janelas no terminal e JSON"""
    
    def __init__(self):
        self.dados_dir = Path("dados")
        self.visual_log = self.dados_dir / "eventos_visuais.log"
        self.json_file = self.dados_dir / "eventos-fluxo.json"
        self.jsonl_file = self.dados_dir / "eventos_fluxo.jsonl"
        
    def executar(self):
        """Executa diagnóstico completo"""
        print("\n" + "="*100)
        print("🔍 DIAGNÓSTICO DE JANELAS - COMPARAÇÃO TERMINAL vs JSON")
        print("="*100 + "\n")
        
        # 1. Analisa log visual (terminal)
        janelas_terminal = self.analisar_log_visual()
        
        # 2. Analisa JSON
        janelas_json = self.analisar_json()
        
        # 3. Analisa JSONL (histórico)
        janelas_jsonl = self.analisar_jsonl()
        
        # 4. Compara
        self.comparar_janelas(janelas_terminal, janelas_json, janelas_jsonl)
        
        # 5. Detecta padrões
        self.detectar_padroes(janelas_terminal, janelas_json)
        
        # 6. Recomendações
        self.gerar_recomendacoes(janelas_terminal, janelas_json)
        
    def analisar_log_visual(self):
        """Analisa eventos_visuais.log"""
        print("📄 Analisando eventos_visuais.log (TERMINAL)...")
        print("-" * 100)
        
        if not self.visual_log.exists():
            print("❌ Arquivo não encontrado!\n")
            return {}
        
        with open(self.visual_log, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        
        # Regex para separadores
            separador_pattern = re.compile(
            r'={70,}\n'
            r'🗓️\s+JANELA\s+(\d+)\n'
            r'🕒\s+UTC:\s+(.+?)\n'
            r'🗽\s+New York:\s+(.+?)\n'
            r'📍\s+São Paulo:\s+(.+?)\n'
            r'📊\s+Contexto:\s+(.+?)\n'
            r'={70,}'
        )
        
        janelas = {}
        matches = list(separador_pattern.finditer(conteudo))
        
        for i, match in enumerate(matches):
            janela_num = int(match.group(1))
            timestamp_utc = match.group(2).strip()
            timestamp_ny = match.group(3).strip()
            timestamp_sp = match.group(4).strip()
            contexto = match.group(5).strip()
            
            # Pega conteúdo até próximo separador
            inicio = match.end()
            if i < len(matches) - 1:
                fim = matches[i + 1].start()
            else:
                fim = len(conteudo)
            
            conteudo_janela = conteudo[inicio:fim].strip()
            
            # Conta eventos JSON na janela
            eventos_json = conteudo_janela.count('"tipo_evento"')
            tem_conteudo = len(conteudo_janela) > 50
            
            janelas[janela_num] = {
                'numero': janela_num,
                'timestamp_utc': timestamp_utc,
                'timestamp_ny': timestamp_ny,
                'timestamp_sp': timestamp_sp,
                'contexto': contexto,
                'tamanho_bytes': len(conteudo_janela),
                'tem_conteudo': tem_conteudo,
                'eventos_count': eventos_json,
                'pos_arquivo': match.start(),
                'origem': 'terminal'
            }
        
        print(f"✅ Total de janelas no terminal: {len(janelas)}")
        
        vazias = sum(1 for j in janelas.values() if not j['tem_conteudo'])
        print(f"   - Com conteúdo: {len(janelas) - vazias}")
        print(f"   - Vazias: {vazias}")
        
        if janelas:
            print(f"   - Primeira janela: {min(janelas.keys())}")
            print(f"   - Última janela: {max(janelas.keys())}")
        
        print()
        return janelas
    
    def analisar_json(self):
        """Analisa eventos-fluxo.json"""
        print("📄 Analisando eventos-fluxo.json (SNAPSHOT)...")
        print("-" * 100)
        
        if not self.json_file.exists():
            print("❌ Arquivo não encontrado!\n")
            return {}
        
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                eventos = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Erro ao ler JSON: {e}\n")
            return {}
        
        if not isinstance(eventos, list):
            print(f"⚠️ Formato inesperado: {type(eventos)}\n")
            return {}
        
        janelas = {}
        por_janela = defaultdict(list)
        
        for i, evento in enumerate(eventos):
            janela_num = evento.get('janela_numero')
            if janela_num:
                por_janela[janela_num].append({
                    'index': i,
                    'tipo_evento': evento.get('tipo_evento', 'N/A'),
                    'timestamp': evento.get('timestamp', evento.get('timestamp_utc', 'N/A')),
                    'epoch_ms': evento.get('epoch_ms'),
                    'is_signal': evento.get('is_signal', False),
                    'evento': evento
                })
        
        for janela_num, eventos_janela in por_janela.items():
            primeiro = eventos_janela[0]
            ultimo = eventos_janela[-1]
            
            janelas[janela_num] = {
                'numero': janela_num,
                'eventos_count': len(eventos_janela),
                'primeiro_timestamp': primeiro['timestamp'],
                'ultimo_timestamp': ultimo['timestamp'],
                'primeiro_epoch': primeiro['epoch_ms'],
                'ultimo_epoch': ultimo['epoch_ms'],
                'tipos_eventos': [e['tipo_evento'] for e in eventos_janela],
                'tem_signals': any(e['is_signal'] for e in eventos_janela),
                'eventos': eventos_janela,
                'origem': 'json'
            }
        
        print(f"✅ Total de eventos no JSON: {len(eventos)}")
        print(f"✅ Total de janelas no JSON: {len(janelas)}")
        
        if janelas:
            print(f"   - Primeira janela: {min(janelas.keys())}")
            print(f"   - Última janela: {max(janelas.keys())}")
            
            # Estatísticas por janela
            eventos_por_janela = [j['eventos_count'] for j in janelas.values()]
            print(f"   - Média de eventos/janela: {sum(eventos_por_janela)/len(eventos_por_janela):.1f}")
            print(f"   - Mín eventos/janela: {min(eventos_por_janela)}")
            print(f"   - Máx eventos/janela: {max(eventos_por_janela)}")
        
        print()
        return janelas
    
    def analisar_jsonl(self):
        """Analisa eventos_fluxo.jsonl (histórico)"""
        print("📄 Analisando eventos_fluxo.jsonl (HISTÓRICO)...")
        print("-" * 100)
        
        if not self.jsonl_file.exists():
            print("❌ Arquivo não encontrado!\n")
            return {}
        
        janelas = defaultdict(list)
        total_linhas = 0
        
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for linha in f:
                total_linhas += 1
                try:
                    evento = json.loads(linha.strip())
                    janela_num = evento.get('janela_numero')
                    if janela_num:
                        janelas[janela_num].append(evento)
                except json.JSONDecodeError:
                    continue
        
        print(f"✅ Total de linhas no JSONL: {total_linhas}")
        print(f"✅ Total de janelas no JSONL: {len(janelas)}")
        
        if janelas:
            print(f"   - Primeira janela: {min(janelas.keys())}")
            print(f"   - Última janela: {max(janelas.keys())}")
        
        print()
        return dict(janelas)
    
    def comparar_janelas(self, terminal, json_data, jsonl_data):
        """Compara janelas entre terminal e JSON"""
        print("🔄 COMPARAÇÃO ENTRE FONTES")
        print("="*100)
        
        nums_terminal = set(terminal.keys())
        nums_json = set(json_data.keys())
        nums_jsonl = set(jsonl_data.keys())
        
        # Estatísticas
        print(f"\n📊 CONTADORES:")
        print(f"   Terminal (log visual):  {len(nums_terminal)} janelas")
        print(f"   JSON (snapshot):        {len(nums_json)} janelas")
        print(f"   JSONL (histórico):      {len(nums_jsonl)} janelas")
        
        # Diferenças
        print(f"\n🔍 DIFERENÇAS:")
        
        apenas_terminal = nums_terminal - nums_json
        apenas_json = nums_json - nums_terminal
        em_ambos = nums_terminal & nums_json
        
        if apenas_json:
            print(f"\n❌ Janelas APENAS no JSON (não aparecem no terminal): {sorted(apenas_json)}")
            print(f"   Total: {len(apenas_json)} janelas")
            
            # Detalhes das janelas que faltam
            print(f"\n   📋 Detalhes:")
            for num in sorted(apenas_json)[:10]:  # Limita a 10
                j = json_data[num]
                print(f"      Janela {num}: {j['eventos_count']} eventos | "
                      f"Timestamp: {j['primeiro_timestamp']}")
                print(f"         Tipos: {Counter(j['tipos_eventos']).most_common(3)}")
        
        if apenas_terminal:
            print(f"\n⚠️ Janelas APENAS no terminal (não estão no JSON): {sorted(apenas_terminal)}")
            print(f"   Total: {len(apenas_terminal)} janelas")
        
        print(f"\n✅ Janelas em AMBOS: {len(em_ambos)}")
        
        # Verifica sequência
        print(f"\n📈 ANÁLISE DE SEQUÊNCIA:")
        
        todos_nums = sorted(nums_terminal | nums_json)
        if todos_nums:
            esperado = list(range(min(todos_nums), max(todos_nums) + 1))
            faltando = set(esperado) - set(todos_nums)
            
            if faltando:
                print(f"   ❌ Janelas faltando na sequência: {sorted(faltando)}")
            else:
                print(f"   ✅ Sequência contínua de {min(todos_nums)} até {max(todos_nums)}")
            
            # Verifica duplicatas
            all_nums = list(terminal.keys()) + list(json_data.keys())
            duplicatas = [n for n, count in Counter(all_nums).items() if count > 1]
            
            if duplicatas:
                print(f"   ⚠️ Números duplicados: {duplicatas}")
        
        print()
    
    def detectar_padroes(self, terminal, json_data):
        """Detecta padrões temporais"""
        print("🕐 ANÁLISE TEMPORAL")
        print("="*100)
        
        if not json_data:
            print("❌ Sem dados JSON para análise\n")
            return
        
        # Analisa epochs
        epochs = []
        for janela_num, dados in sorted(json_data.items()):
            if dados.get('primeiro_epoch'):
                epochs.append({
                    'janela': janela_num,
                    'epoch': dados['primeiro_epoch'],
                    'timestamp': dados['primeiro_timestamp']
                })
        
        if len(epochs) < 2:
            print("⚠️ Dados insuficientes\n")
            return
        
        # Calcula intervalos
        intervalos = []
        for i in range(len(epochs) - 1):
            diff_ms = epochs[i+1]['epoch'] - epochs[i]['epoch']
            diff_seg = diff_ms / 1000
            intervalos.append({
                'janela_de': epochs[i]['janela'],
                'janela_ate': epochs[i+1]['janela'],
                'intervalo_ms': diff_ms,
                'intervalo_seg': diff_seg
            })
        
        if intervalos:
            int_segs = [i['intervalo_seg'] for i in intervalos]
            
            print(f"📊 Intervalos entre janelas:")
            print(f"   Média: {sum(int_segs)/len(int_segs):.1f}s")
            print(f"   Mínimo: {min(int_segs):.1f}s")
            print(f"   Máximo: {max(int_segs):.1f}s")
            
            # Detecta anomalias (intervalos muito pequenos ou grandes)
            media = sum(int_segs) / len(int_segs)
            anomalias = [i for i in intervalos if abs(i['intervalo_seg'] - media) > media * 0.5]
            
            if anomalias:
                print(f"\n   ⚠️ Anomalias detectadas ({len(anomalias)}):")
                for a in anomalias[:5]:
                    print(f"      Janela {a['janela_de']} → {a['janela_ate']}: "
                          f"{a['intervalo_seg']:.1f}s")
            
            # Verifica criação simultânea (mesmo segundo)
            mesmo_segundo = defaultdict(list)
            for e in epochs:
                segundo = int(e['epoch'] / 1000)
                mesmo_segundo[segundo].append(e['janela'])
            
            simultaneas = {k: v for k, v in mesmo_segundo.items() if len(v) > 1}
            
            if simultaneas:
                print(f"\n   🔴 Janelas criadas no mesmo segundo ({len(simultaneas)} ocorrências):")
                for segundo, janelas in list(simultaneas.items())[:5]:
                    dt = datetime.fromtimestamp(segundo)
                    print(f"      {dt.strftime('%H:%M:%S')}: Janelas {janelas}")
                
                print(f"\n   💡 CAUSA PROVÁVEL: Múltiplos eventos processados simultaneamente")
                print(f"      → Verificar lógica de detecção de nova janela")
        
        print()
    
    def gerar_recomendacoes(self, terminal, json_data):
        """Gera recomendações de correção"""
        print("💡 RECOMENDAÇÕES")
        print("="*100 + "\n")
        
        apenas_json = set(json_data.keys()) - set(terminal.keys())
        
        if len(apenas_json) > 0:
            print("🔴 PROBLEMA: Janelas no JSON mas não no terminal")
            print("-" * 100)
            print(f"   Quantidade: {len(apenas_json)} janelas")
            print(f"   Janelas: {sorted(apenas_json)}\n")
            
            print("📋 POSSÍVEIS CAUSAS:")
            print("   1. ❌ Separador não sendo escrito (lock timeout)")
            print("   2. ❌ Buffer processando fora de ordem")
            print("   3. ❌ Flag '_needs_separator' sendo removida antes do flush")
            print("   4. ❌ Múltiplas threads criando janelas simultâneas\n")
            
            print("🔧 SOLUÇÕES:")
            print("   1. Verificar logs de 'Timeout ao adquirir lock do visual log'")
            print("   2. Verificar se _flush_buffer está processando a flag corretamente")
            print("   3. Adicionar log em _add_visual_separator para debug")
            print("   4. Aumentar timeout do lock (linha ~870 event_saver.py)\n")
            
            print("🧪 TESTE RECOMENDADO:")
            print("   python teste_separador.py  # (script abaixo)\n")
        
        # Verifica se há janelas simultâneas
        if json_data:
            epochs_por_segundo = defaultdict(list)
            for num, dados in json_data.items():
                if dados.get('primeiro_epoch'):
                    segundo = int(dados['primeiro_epoch'] / 1000)
                    epochs_por_segundo[segundo].append(num)
            
            simultaneas = {k: v for k, v in epochs_por_segundo.items() if len(v) > 1}
            
            if simultaneas:
                print("🔴 PROBLEMA: Múltiplas janelas no mesmo segundo")
                print("-" * 100)
                print(f"   Ocorrências: {len(simultaneas)}")
                print(f"\n📋 CAUSA:")
                print(f"   Lógica de detecção está criando nova janela para cada evento")
                print(f"   em vez de agrupar por minuto\n")
                
                print("🔧 SOLUÇÃO:")
                print("   Linha ~775 event_saver.py:")
                print("   window_time = dt.replace(second=0, microsecond=0)  # ✅ Já correto")
                print("   window_key = window_time.strftime('%Y%m%d_%H%M')  # ✅ Agrupa por minuto")
                print("\n   ⚠️ MAS: Verificar se 'window_id' está mudando a cada segundo")
                print("   Solução: Usar epoch do MINUTO, não do segundo:\n")
                print("   # ADICIONAR após linha 774:")
                print("   window_id = int(window_time.timestamp() * 1000)  # epoch do minuto")
                print()


def gerar_teste_separador():
    """Gera script de teste para debug"""
    codigo = '''# teste_separador.py - Debug de separadores

import logging
import time
from events.event_saver import EventSaver
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("🧪 TESTE DE SEPARADORES\\n")
print("="*80)

saver = EventSaver(sound_alert=False)

# Força criação de 5 janelas diferentes
for i in range(5):
    epoch_base = int(datetime.now().timestamp() * 1000)
    epoch_minuto = (epoch_base // 60000) * 60000  # Arredonda para minuto
    epoch_minuto += i * 60000  # Adiciona 1 minuto por iteração
    
    evento = {
        "tipo_evento": f"TESTE_JANELA_{i+1}",
        "is_signal": True,
        "epoch_ms": epoch_minuto,
        "window_id": epoch_minuto,
        "price_data": {
            "current": {
                "last": 110000 + i * 100,
                "volume": 100 + i
            }
        }
    }
    
    print(f"\\n📌 Salvando evento {i+1} (epoch: {epoch_minuto})...")
    saver.save_event(evento)
    time.sleep(0.5)

print("\\n⏳ Aguardando flush...")
time.sleep(7)


        if event.get("tipo_evento") == "ANALYSIS_TRIGGER":




print("\\n📊 Estatísticas:")
stats = saver.get_stats()
for k, v in stats.items():
    if isinstance(v, dict):
        print(f"  {k}:")
        for k2, v2 in v.items():
            print(f"    {k2}: {v2}")
    else:
        print(f"  {k}: {v}")

saver.stop()

print("\\n✅ Teste concluído!")
print("\\n🔍 Agora execute:")
print("   python window_diagnostics.py")
print("="*80)
'''

    with open('teste_separador.py', 'w', encoding='utf-8') as f:
        f.write(codigo)

    print("✅ Arquivo 'teste_separador.py' criado!")


if __name__ == "__main__":
    diag = DiagnosticoJanelas()
    diag.executar()

    print("\n" + "="*100)
    resposta = input("🤔 Deseja gerar script de teste? (s/n): ").lower()

    if resposta == 's':
        gerar_teste_separador()
        print("\n📝 Execute:")
        print("   python teste_separador.py")
        print("   python window_diagnostics.py  # para verificar resultado")
    
    print("\n" + "="*100)