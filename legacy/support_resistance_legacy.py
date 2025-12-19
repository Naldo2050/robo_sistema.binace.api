"""
Arquivo de compatibilidade para manter importações existentes
"""

import sys
import os

# Adicionar o diretório support_resistance ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'support_resistance'))

# Re-exportar tudo do novo sistema
from support_resistance import *

print("NOTA: O sistema de suporte/resistência foi modularizado.")
print("Considere atualizar suas importações para:")
print("  from support_resistance import InstitutionalSupportResistanceSystem")