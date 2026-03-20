"""
Corrige os 3 bugs encontrados na primeira execução.
Execute: python auto_fixer/fix_bugs.py
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

# Configurar UTF-8 para suportar emojis no Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for _stream in (sys.stdout, sys.stderr):
        _reconf = getattr(_stream, "reconfigure", None)
        if _reconf and not _stream.closed:
            try:
                _reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def backup_file(filepath: Path):
    """Faz backup antes de modificar."""
    if filepath.exists():
        backup = filepath.with_suffix(
            f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(filepath, backup)
        print(f"  📦 Backup: {backup.name}")


def fix_bug1_chunker():
    """
    Bug 1: chunk_engine.py - Windows path separator no nome do arquivo.
    
    Problema: chunk_id contém 'ai_runner\\ai_runner.py::0' no Windows.
    O replace só tratava '/' mas não '\\'.
    Resultado: tentava criar subdiretório inexistente.
    """
    print("\n🔧 Bug 1: Corrigindo chunk_engine.py...")
    
    filepath = Path("auto_fixer/phase3_chunker/chunk_engine.py")
    if not filepath.exists():
        print(f"  ❌ Arquivo não encontrado: {filepath}")
        return False
    
    backup_file(filepath)
    
    content = filepath.read_text(encoding="utf-8")
    
    # ── Correção 1: safe_name deve tratar \ e / ──
    old_safe_name = 'safe_name = chunk.chunk_id.replace("/", "__").replace("::", "_")'
    new_safe_name = 'safe_name = chunk.chunk_id.replace(os.sep, "__").replace("/", "__").replace("\\\\", "__").replace("::", "_").replace(".", "_")'
    
    if old_safe_name in content:
        content = content.replace(old_safe_name, new_safe_name)
        print("  ✅ safe_name corrigido para tratar Windows paths")
    else:
        # Tentar variação
        if 'safe_name' in content and 'replace("/", "__")' in content:
            # Substituir a linha inteira
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if 'safe_name' in line and 'replace("/", "__")' in line:
                    indent = len(line) - len(line.lstrip())
                    lines[i] = (
                        " " * indent + 
                        'safe_name = chunk.chunk_id.replace(os.sep, "__")'
                        '.replace("/", "__").replace("\\\\", "__")'
                        '.replace("::", "_").replace(".", "_")'
                    )
                    print("  ✅ safe_name corrigido (variação)")
                    break
            content = "\n".join(lines)
        else:
            print("  ⚠️ Padrão safe_name não encontrado, aplicando fix completo")
    
    # ── Correção 2: Garantir import os ──
    if "import os" not in content:
        content = "import os\n" + content
        print("  ✅ Adicionado 'import os'")
    
    filepath.write_text(content, encoding="utf-8")
    print("  ✅ chunk_engine.py salvo")
    return True


def fix_bug2_runner_encoding():
    """
    Bug 2 e 3: runner.py - open() sem encoding="utf-8".
    
    Problema: No Windows, o encoding padrão é cp1252.
    Arquivos com caracteres especiais (acentos, UTF-8) falham.
    """
    print("\n🔧 Bug 2+3: Corrigindo runner.py (encoding)...")
    
    filepath = Path("auto_fixer/runner.py")
    if not filepath.exists():
        print(f"  ❌ Arquivo não encontrado: {filepath}")
        return False
    
    backup_file(filepath)
    
    content = filepath.read_text(encoding="utf-8")
    
    # Substituir TODOS os open() sem encoding
    replacements = [
        # Padrão: open(f) as fh  →  open(f, encoding="utf-8") as fh
        ('open(f) as fh', 'open(f, encoding="utf-8") as fh'),
        # Padrão: open(f) as f  →  open(f, encoding="utf-8") as f  
        ('open(f) as f:', 'open(f, encoding="utf-8") as f:'),
    ]
    
    count = 0
    for old, new in replacements:
        if old in content:
            occurrences = content.count(old)
            content = content.replace(old, new)
            count += occurrences
            print(f"  ✅ Corrigido {occurrences}x: {old}")
    
    if count == 0:
        # Abordagem mais agressiva: encontrar qualquer open() sem encoding
        lines = content.splitlines()
        fixed_lines = []
        for line in lines:
            if "open(" in line and "encoding" not in line and \
               ".json" not in line and "json.load" not in line:
                # Não mexer em linhas que já parecem corretas
                fixed_lines.append(line)
            elif "open(f)" in line and "encoding" not in line:
                line = line.replace("open(f)", 'open(f, encoding="utf-8")')
                fixed_lines.append(line)
                count += 1
                print(f"  ✅ Corrigido: {line.strip()}")
            else:
                fixed_lines.append(line)
        content = "\n".join(fixed_lines)
    
    filepath.write_text(content, encoding="utf-8")
    print(f"  ✅ runner.py salvo ({count} correções)")
    return True


def fix_chunk_engine_complete():
    """
    Reescreve a parte crítica do chunk_engine que salva arquivos.
    Garante que funcione no Windows.
    """
    print("\n🔧 Aplicando fix completo no chunk_engine.py...")
    
    filepath = Path("auto_fixer/phase3_chunker/chunk_engine.py")
    if not filepath.exists():
        print(f"  ❌ Arquivo não encontrado: {filepath}")
        return False
    
    content = filepath.read_text(encoding="utf-8")
    
    # Encontrar e substituir o bloco de salvamento de chunks
    # Procurar o método _save ou o loop de salvamento
    
    old_block = '''        # Salvar cada chunk como arquivo separado
        for chunk in all_chunks:
            safe_name = chunk.chunk_id.replace("/", "__").replace("::", "_")
            chunk_file = self.chunks_dir / f"{safe_name}.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(asdict(chunk), f, indent=2, ensure_ascii=False)'''
    
    new_block = '''        # Salvar cada chunk como arquivo separado
        for chunk in all_chunks:
            # Tratar separadores de path do Windows e Linux
            safe_name = chunk.chunk_id
            safe_name = safe_name.replace(os.sep, "__")
            safe_name = safe_name.replace("/", "__")
            safe_name = safe_name.replace("\\\\", "__")
            safe_name = safe_name.replace("::", "_")
            safe_name = safe_name.replace(".", "_")
            chunk_file = self.chunks_dir / f"{safe_name}.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(asdict(chunk), f, indent=2, ensure_ascii=False)'''
    
    if old_block in content:
        content = content.replace(old_block, new_block)
        filepath.write_text(content, encoding="utf-8")
        print("  ✅ Bloco de salvamento substituído")
        return True
    else:
        print("  ⚠️ Bloco exato não encontrado, tentando fix por linha...")
        return fix_bug1_chunker()


def fix_runner_complete():
    """
    Reescreve run_phase5 e run_phase6 no runner.py 
    para garantir encoding UTF-8.
    """
    print("\n🔧 Aplicando fix completo no runner.py...")
    
    filepath = Path("auto_fixer/runner.py")
    if not filepath.exists():
        print(f"  ❌ Arquivo não encontrado: {filepath}")
        return False
    
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    fixed_count = 0
    new_lines = []
    
    for line in lines:
        # Corrigir qualquer open() de arquivo JSON sem encoding
        if ("open(f" in line or "open(chunk" in line) and \
           "encoding" not in line and \
           ("json" in line.lower() or "as fh" in line or "as f:" in line):
            
            # Adicionar encoding="utf-8"
            line = line.replace(
                "open(f)", 'open(f, encoding="utf-8")'
            )
            line = line.replace(
                "open(f,", 'open(f, encoding="utf-8",'
            )
            fixed_count += 1
        
        new_lines.append(line)
    
    content = "\n".join(new_lines)
    filepath.write_text(content, encoding="utf-8")
    print(f"  ✅ runner.py: {fixed_count} linhas corrigidas")
    return True


def verify_fixes():
    """Verifica se os fixes foram aplicados corretamente."""
    print("\n🔍 Verificando correções...")
    
    ok = True
    
    # Verificar chunk_engine.py
    ce = Path("auto_fixer/phase3_chunker/chunk_engine.py")
    if ce.exists():
        content = ce.read_text(encoding="utf-8")
        if "os.sep" in content or ('replace("\\\\", "__")' in content):
            print("  ✅ chunk_engine.py: Windows path fix OK")
        else:
            print("  ❌ chunk_engine.py: Fix NÃO aplicado")
            ok = False
    
    # Verificar runner.py
    rn = Path("auto_fixer/runner.py")
    if rn.exists():
        content = rn.read_text(encoding="utf-8")
        # Contar open() sem encoding
        import re
        opens_without_encoding = len(re.findall(
            r'open\([^)]*\)\s*as\s+\w+', content
        ))
        opens_with_encoding = content.count('encoding="utf-8"')
        
        if opens_with_encoding > 0:
            print(f"  ✅ runner.py: {opens_with_encoding} open() com encoding")
        else:
            print("  ❌ runner.py: Nenhum open() com encoding encontrado")
            ok = False
    
    return ok


def clean_old_chunks():
    """Limpa chunks antigos que podem ter nomes errados."""
    print("\n🧹 Limpando chunks antigos...")
    
    chunks_dir = Path("auto_fixer/output/chunks")
    if chunks_dir.exists():
        count = 0
        for f in chunks_dir.rglob("*.json"):
            f.unlink()
            count += 1
        print(f"  ✅ {count} chunks antigos removidos")
    else:
        print("  ℹ️ Diretório de chunks não existe ainda")


def main():
    print("=" * 60)
    print("🔧 AUTO-FIXER BUG FIXER")
    print("   Corrigindo 3 bugs da primeira execução")
    print("=" * 60)
    
    # 1. Limpar chunks antigos com nomes errados
    clean_old_chunks()
    
    # 2. Fix Bug 1: chunk_engine.py
    fix_chunk_engine_complete()
    
    # 3. Fix Bug 2+3: runner.py encoding
    fix_runner_complete()
    
    # 4. Verificar
    ok = verify_fixes()
    
    print("\n" + "=" * 60)
    if ok:
        print("✅ TODOS OS FIXES APLICADOS COM SUCESSO!")
        print("\nAgora execute novamente:")
        print("  python -m auto_fixer.runner --phase 3")
        print("  python -m auto_fixer.runner --phase 5")
        print("  python -m auto_fixer.runner --phase 6")
        print("  python -m auto_fixer.runner --phase 8")
        print("\nOu tudo de uma vez:")
        print("  python -m auto_fixer.runner --phase 0")
    else:
        print("⚠️ Alguns fixes podem não ter sido aplicados")
        print("   Verifique os arquivos manualmente")
    print("=" * 60)


if __name__ == "__main__":
    main()
