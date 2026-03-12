# fix_duplicatas.py - Proteção contra separadores duplicados

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
            'self._window_counter = 0\n        self._janelas_processadas = set()  # Rastreia janelas que já tiveram separador'
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
                'f.write(separator)\n                        f.flush()',
                'f.write(separator)\n                        f.flush()\n                    \n                    # Marca janela como processada\n                    if janela_num:\n                        self._janelas_processadas.add(janela_num)\n                        self.logger.debug(f"✅ Janela {janela_num} marcada como processada")'
            )
            print("✅ Marcação de janela processada adicionada")
    
    # Salva
    with open(arquivo, 'w', encoding='utf-8') as f:
        f.write(conteudo)
    
    print("\n✅ Proteções aplicadas!")
    print("\n📝 Mudanças:")
    print("   1. Set _janelas_processadas criado")
    print("   2. Verificação de duplicata antes de escrever separador")
    print("   3. Janela marcada como processada após escrita")
    print("\n⚠️ REINICIE o sistema para aplicar")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔧 PROTEÇÃO CONTRA SEPARADORES DUPLICADOS")
    print("="*80 + "\n")
    
    aplicar_protecao()
    
    print("\n" + "="*80)
