# fix_separador_final.py - Correção definitiva

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
            '    def _add_visual_separator(self, event: Dict):\n'
            '        """Adiciona separador visual para nova janela de tempo."""\n'
            '        try:',
            '    def _add_visual_separator(self, event: Dict):\n'
            '        """Adiciona separador visual para nova janela de tempo."""\n'
            '        self.logger.info(f"📝 Escrevendo separador para janela {event.get(\'janela_numero\', \'N/A\')}")\n'
            '        try:'
        )
        
        # Adiciona log após escrita
        conteudo = conteudo.replace(
            '                    with open(self.visual_log_file, "a", encoding="utf-8") as f:\n'
            '                        f.write(separator)\n'
            '                        f.flush()',
            '                    with open(self.visual_log_file, "a", encoding="utf-8") as f:\n'
            '                        f.write(separator)\n'
            '                        f.flush()\n'
            '                    self.logger.info(f"✅ Separador escrito com sucesso")'
        )
        
        # Adiciona log em timeout
        conteudo = conteudo.replace(
            '                else:\n'
            '                    self.logger.warning("Timeout ao adquirir lock do visual log")',
            '                else:\n'
            '                    self.logger.error(f"❌ TIMEOUT ao adquirir lock - Separador NÃO foi escrito para janela {event.get(\'janela_numero\')}")'
        )
    
    # Adiciona log em _flush_buffer
    if 'def _flush_buffer(self, events: List[Dict]):' in conteudo:
        conteudo = conteudo.replace(
            '                if event.get("_needs_separator"):\n'
            '                    self._add_visual_separator(event)\n'
            '                    event.pop("_needs_separator", None)',
            '                if event.get("_needs_separator"):\n'
            '                    self.logger.debug(f"🚩 Processando flag _needs_separator para janela {event.get(\'janela_numero\')}")\n'
            '                    self._add_visual_separator(event)\n'
            '                    event.pop("_needs_separator", None)\n'
            '                    self.logger.debug(f"✅ Flag removida")'
        )
    
    # Salva
    with open(arquivo, 'w', encoding='utf-8') as f:
        f.write(conteudo)
    
    print("✅ Logs de debug adicionados!")
    print("\n📝 Logs adicionados:")
    print("   - Início de _add_visual_separator")
    print("   - Após escrita bem-sucedida")
    print("   - Em caso de timeout")
    print("   - Processamento de flag em _flush_buffer")
    print("\n🔍 Execute o sistema e verifique os logs!")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔧 ADIÇÃO DE LOGS DE DEBUG")
    print("="*80 + "\n")
    
    aplicar_fix()
    
    print("\n" + "="*80)
