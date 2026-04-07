import os
import json

file_path = "build_compact_payload.py"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'import os' not in content[:500]:
    content = content.replace('import logging\nimport time', 'import logging\nimport os\nimport time')

debug_code = '''
    # ═══════════════════════════════════════════════════════════
    # DEBUG: Instrumentação de tamanho por seção (TEMPORÁRIA)
    # ═══════════════════════════════════════════════════════════
    if os.environ.get("PAYLOAD_DEBUG_SIZES") == "1":
        section_sizes = {}
        for key, value in payload.items():
            section_json = json.dumps({key: value}, ensure_ascii=False, separators=(",", ":"))
            section_sizes[key] = len(section_json)
        
        sorted_sections = sorted(section_sizes.items(), key=lambda x: x[1], reverse=True)
        
        print("\\n" + "="*80)
        print(f"PAYLOAD_DEBUG_SIZES: total={payload_size} bytes")
        print(f"Sections by size (top 15 of {len(section_sizes)}):")
        for i, (section, size) in enumerate(sorted_sections[:15], 1):
            pct = (size / payload_size * 100) if payload_size > 0 else 0
            print(f"  {i:2d}. {section:15s} {size:6d} bytes ({pct:5.1f}%)")
        print("="*80)
'''

# Encontrar e substituir - procurar antes do último return payload
idx = content.rfind('\n    return payload')
if idx != -1:
    content = content[:idx] + debug_code + content[idx:]
    print("Instrumentação adicionada")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Arquivo salvo")
else:
    print("return payload não encontrado")
