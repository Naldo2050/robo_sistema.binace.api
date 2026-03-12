import os
import re

def read_manifest_from_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    start = content.find("```")
    end = content.find("```", start + 3)
    if start == -1 or end == -1:
        return set()
    
    structure_text = content[start+3:end]
    files = set()
    
    for line in structure_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('|') or line.startswith('---') or line.startswith('robo_sistema'):
            continue
        
        match = re.match(r'^(?:├─┬|├─|└─|│ )+(.*)$', line.strip())
        if match:
            path = match.group(1).strip()
            if path and not path.startswith('[') and not path.endswith(']') and '[' not in path and ']' not in path:
                path = path.split('#')[0].strip()
                if path and len(path.strip()) > 0:
                    # Convert Windows backslashes to forward slashes
                    files.add(path.replace('\\', '/'))
    
    return files

def get_project_structure(root_dir):
    project_files = set()
    excluded_dirs = {
        '.git', '.mypy_cache', '__pycache__', '.pytest_cache',
        'coverage_html', '.agent', '.claude', '-p', '.venv', 'playwright_user_data',
        'logs', 'dados', 'features', '.vscode', '.github', 'Regras', 'MQL5', 'tests', 'coverage_html'
    }
    
    excluded_files = {
        '.DS_Store', '*.pyc', '*.bak', '*.backup', '*~', '*.swp'
    }
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path == '.':
            for filename in filenames:
                if any(filename.endswith(ext) for ext in ['.pyc', '.bak', '.backup']) or filename == '.DS_Store' or filename == 'compare_structure.py' or filename == 'compare_structure_filtered.py' or filename == 'find_missing_files.py':
                    continue
                project_files.add(filename)
        else:
            for filename in filenames:
                if any(filename.endswith(ext) for ext in ['.pyc', '.bak', '.backup']) or filename == '.DS_Store':
                    continue
                project_files.add(os.path.join(rel_path, filename).replace('\\', '/'))
    
    return project_files

def main():
    md_file = "ESTRUTURA_VISUAL_SISTEMA.md"
    root_dir = "."
    
    md_files = read_manifest_from_md(md_file)
    project_files = get_project_structure(root_dir)
    
    missing_in_md = project_files - md_files
    
    print("Arquivos do projeto presentes mas NAO documentados (projetado para inclusao):")
    print("=" * 60)
    for file in sorted(missing_in_md):
        print(f"- {file}")
    
    print("\n" + "=" * 60)
    print(f"Total de arquivos do projeto (filtrado para documentacao): {len(project_files)}")
    print(f"Total de arquivos na documentação: {len(md_files)}")
    print(f"Total de arquivos não documentados: {len(missing_in_md)}")

if __name__ == "__main__":
    main()
