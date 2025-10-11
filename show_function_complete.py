# show_function_complete.py
"""Mostra a função _extract_orderbook_data completa."""

def show_function():
    """Mostra conteúdo completo da função."""
    
    filepath = "ai_analyzer_qwen.py"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print("="*80)
        print("🔍 FUNÇÃO _extract_orderbook_data COMPLETA")
        print("="*80 + "\n")
        
        # Encontra linha 202
        start_line = 202
        indent_level = None
        
        for i in range(start_line-1, len(lines)):
            line = lines[i]
            
            # Primeira linha define indentação
            if indent_level is None:
                indent_level = len(line) - len(line.lstrip())
            
            current_indent = len(line) - len(line.lstrip())
            
            # Se voltou ao nível original E não é linha vazia/comentário, acabou
            if i > start_line and current_indent <= indent_level and line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                print(f"\n{'='*80}")
                print(f"📍 Função vai da linha {start_line} até {i}")
                print(f"{'='*80}\n")
                break
            
            # Imprime linha
            print(f"{i+1:4d} | {line.rstrip()}")
        
        return start_line, i
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    show_function()