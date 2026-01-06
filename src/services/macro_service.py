#!/usr/bin/env python3
"""
MacroService - Serviço para gerenciamento de dados macroeconômicos
"""

from typing import Dict, Any, Optional
from .macro_update_service import MacroUpdateService


class MacroService:
    """Serviço para gerenciamento de dados macroeconômicos"""
    
    def __init__(self):
        """Inicializa o serviço de macro dados"""
        self._update_service = MacroUpdateService()
    
    def get_all(self) -> Dict[str, Any]:
        """Obtém todos os dados macroeconômicos disponíveis"""
        try:
            return self._update_service.get_all_data()
        except Exception as e:
            print(f"Erro ao obter dados macro: {e}")
            return {}
    
    def update_data(self) -> bool:
        """Atualiza os dados macroeconômicos"""
        try:
            self._update_service.update_all()
            return True
        except Exception as e:
            print(f"Erro ao atualizar dados macro: {e}")
            return False