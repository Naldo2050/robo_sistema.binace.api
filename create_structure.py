"""
Script para criar estrutura de pastas - VS Code Terminal
Funciona em qualquer sistema operacional
"""

import os
from pathlib import Path

def create_structure():
    """Cria toda a estrutura de pastas e arquivos"""
    
    print("🚀 Iniciando criação da estrutura...")
    print(f"📁 Pasta atual: {Path.cwd()}\n")
    
    # Criar pasta principal
    main_dir = Path('institutional_analysis')
    main_dir.mkdir(exist_ok=True)
    print(f"✅ Criada: institutional_analysis/")
    
    # Criar arquivos principais
    files_main = ['__init__.py', 'system.py', 'legacy.py']
    for file in files_main:
        (main_dir / file).touch()
        print(f"   ✅ {file}")
    
    # Estrutura de subpastas
    structure = {
        'core': [
            '__init__.py',
            'types.py',
            'exceptions.py',
            'constants.py',
            'config.py'
        ],
        'utils': [
            '__init__.py',
            'validation.py',
            'performance.py',
            'logging.py',
            'serialization.py',
            'statistics.py'
        ],
        'analyzers': [
            '__init__.py',
            'base.py',
            'pivot_points.py',
            'volume_profile.py',
            'support_resistance.py',
            'confluence.py'
        ],
        'monitoring': [
            '__init__.py',
            'market_monitor.py',
            'health.py'
        ]
    }
    
    # Criar subpastas e arquivos
    for folder, files in structure.items():
        folder_path = main_dir / folder
        folder_path.mkdir(exist_ok=True)
        print(f"\n✅ Criada: institutional_analysis/{folder}/")
        
        for file in files:
            file_path = folder_path / file
            file_path.touch()
            print(f"   ✅ {file}")
    
    print("\n" + "="*60)
    print("🎉 ESTRUTURA CRIADA COM SUCESSO!")
    print("="*60)
    
    # Mostrar estrutura criada
    print("\n📁 Estrutura final:\n")
    show_tree(main_dir)
    
    # Próximos passos
    print("\n" + "="*60)
    print("📋 PRÓXIMOS PASSOS:")
    print("="*60)
    print("1. Mova seu arquivo 'support_resistance.py' para:")
    print("   institutional_analysis/legacy.py")
    print("\n2. Verifique a estrutura no explorador do VS Code")
    print("\n3. Avise que concluiu: 'Fase 1 concluída'")

def show_tree(directory, prefix='', is_last=True):
    """Imprime árvore de diretórios"""
    print(f"{directory.name}/")
    
    items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))
    
    for i, item in enumerate(items):
        is_last_item = i == len(items) - 1
        current_prefix = '└── ' if is_last_item else '├── '
        
        if item.is_dir():
            print(f"{prefix}{current_prefix}{item.name}/")
            next_prefix = prefix + ('    ' if is_last_item else '│   ')
            
            # Mostrar conteúdo da subpasta
            sub_items = sorted(item.iterdir(), key=lambda x: x.name)
            for j, sub_item in enumerate(sub_items):
                is_last_sub = j == len(sub_items) - 1
                sub_current_prefix = '└── ' if is_last_sub else '├── '
                print(f"{next_prefix}{sub_current_prefix}{sub_item.name}")
        else:
            print(f"{prefix}{current_prefix}{item.name}")

if __name__ == '__main__':
    try:
        create_structure()
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()