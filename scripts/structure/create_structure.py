#!/usr/bin/env python3
"""
Script para gerar uma visualizacao em arvore da estrutura de diretorios e arquivos.
"""

import os
from pathlib import Path


def get_file_info(filepath: Path) -> str:
    """Retorna informacoes sobre o arquivo (tamanho, tipo, etc)."""
    try:
        size = filepath.stat().st_size
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f}KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f}MB"
        return size_str
    except:
        return "N/A"


def get_file_type(filepath: Path) -> str:
    """Retorna o tipo de arquivo baseado na extensao."""
    ext = filepath.suffix.lower()
    type_map = {
        '.py': 'Python',
        '.md': 'Markdown',
        '.json': 'JSON',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.txt': 'Texto',
        '.csv': 'CSV',
        '.sh': 'Shell',
        '.bat': 'Batch',
        '.ps1': 'PowerShell',
        '.toml': 'TOML',
        '.ini': 'Config',
        '.dockerfile': 'Docker',
        '.sql': 'SQL',
        '.html': 'HTML',
        '.css': 'CSS',
        '.js': 'JavaScript',
    }
    return type_map.get(ext, 'Arquivo')


def should_ignore(name: str, path: Path) -> bool:
    """Verifica se um arquivo/pasta deve ser ignorado."""
    ignore_patterns = {
        '__pycache__',
        '.git',
        '.pytest_cache',
        '.mypy_cache',
        'node_modules',
        '.coverage',
        'htmlcov',
        'coverage_html',
        '.agent',
        '-p/',
        '.github',
    }
    
    # Ignorar diretorios/padroes especificos
    if name in ignore_patterns:
        return True
    if name.startswith('.') and name not in ['.dockerignore', '.gitignore']:
        return True
    return False


def build_tree(directory: Path, prefix: str = "", is_last: bool = True, 
               max_depth: int = 10, current_depth: int = 0) -> list:
    """Constroi a arvore de diretorios recursivamente."""
    
    if current_depth > max_depth:
        return []
    
    result = []
    
    try:
        items = []
        for item in directory.iterdir():
            if should_ignore(item.name, item):
                continue
            items.append(item)
        
        # Ordenar: pastas primeiro, depois arquivos, ambos em ordem alfabetica
        items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
        
        for i, item in enumerate(items):
            is_last_item = i == len(items) - 1
            connector = "+-- " if is_last_item else "+-- "
            
            if item.is_dir():
                # Contar arquivos na pasta
                try:
                    file_count = sum(1 for _ in item.rglob('*') if _.is_file() and not should_ignore(_.name, _))
                    dir_info = f" [{file_count} arquivos]"
                except:
                    dir_info = ""
                
                result.append(f"{prefix}{connector}[DIR] {item.name}{dir_info}")
                
                # Recursao para subdiretorios
                extension = "    " if is_last_item else "|   "
                subtree = build_tree(item, prefix + extension, is_last_item, 
                                   max_depth, current_depth + 1)
                result.extend(subtree)
            else:
                size = get_file_info(item)
                file_type = get_file_type(item)
                result.append(f"{prefix}{connector}[{file_type}] {item.name} ({size})")
    
    except PermissionError:
        result.append(f"{prefix}[Sem permissao de acesso]")
    except Exception as e:
        result.append(f"{prefix}[Erro: {e}]")
    
    return result


def generate_structure():
    """Gera e salva a estrutura completa do sistema."""
    
    root = Path('.')
    
    print("=" * 80)
    print("ESTRUTURA COMPLETA DO SISTEMA - ROBO_SISTEMA.BINACE.API")
    print("=" * 80)
    print()
    
    # Cabecalho com informacoes gerais
    print("RESUMO DO SISTEMA")
    print("-" * 80)
    
    # Contar arquivos por tipo
    file_types = {}
    total_files = 0
    total_dirs = 0
    
    for item in root.rglob('*'):
        if should_ignore(item.name, item):
            continue
        if item.is_file():
            total_files += 1
            ext = item.suffix.lower() or 'sem_extensao'
            file_types[ext] = file_types.get(ext, 0) + 1
        elif item.is_dir():
            total_dirs += 1
    
    print(f"Total de diretorios: {total_dirs}")
    print(f"Total de arquivos: {total_files}")
    print()
    
    print("ARQUIVOS POR TIPO:")
    sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)
    for ext, count in sorted_types[:10]:
        ext_display = ext if ext != 'sem_extensao' else '(sem extensao)'
        print(f"   {ext_display}: {count}")
    print()
    
    print("=" * 80)
    print("ARVORE DE DIRETORIOS")
    print("=" * 80)
    print()
    
    # Gerar arvore
    tree_lines = build_tree(root, max_depth=15)
    tree_output = "\n".join(tree_lines)
    print(tree_output)
    
    # Salvar em arquivo
    output_file = Path("ESTRUTURA_SISTEMA_VISUAL.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Estrutura Completa do Sistema\n\n")
        f.write("## Resumo\n\n")
        f.write(f"- **Total de diretorios:** {total_dirs}\n")
        f.write(f"- **Total de arquivos:** {total_files}\n\n")
        
        f.write("### Arquivos por Tipo\n\n")
        f.write("| Extensao | Quantidade |\n")
        f.write("|----------|------------|\n")
        for ext, count in sorted_types[:15]:
            ext_display = ext if ext != 'sem_extensao' else '(sem extensao)'
            f.write(f"| {ext_display} | {count} |\n")
        f.write("\n")
        
        f.write("## Arvore de Diretorios\n\n")
        f.write("```\n")
        f.write(f"robo_sistema.binace.api/\n")
        f.write(tree_output)
        f.write("\n```\n\n")
        
        f.write("## Legenda\n\n")
        f.write("- [DIR] = Diretorio\n")
        f.write("- Python = Arquivo Python (.py)\n")
        f.write("- Markdown = Arquivo Markdown (.md)\n")
        f.write("- JSON = Arquivo JSON (.json)\n")
        f.write("- YAML = Arquivo YAML (.yaml, .yml)\n")
        f.write("- Config = Arquivo de configuracao (.toml, .ini)\n")
        f.write("- Texto = Arquivo de texto (.txt)\n")
        f.write("- CSV = Arquivo CSV (.csv)\n")
        f.write("- Shell/Script = Scripts (.sh, .bat, .ps1)\n")
        f.write("- Docker = Arquivos Docker\n")
        f.write("- Web = Arquivos web (.html, .css, .js)\n")
    
    print()
    print("=" * 80)
    print(f"Estrutura salva em: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    generate_structure()
