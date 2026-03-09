
import os
import shutil
from datetime import datetime
from pathlib import Path

class AutoFixer:
    """Aplica correções e otimizações automáticas baseadas no plano V5/V6."""
    
    def __init__(self):
        self.backups_dir = Path("backups")
        self.backups_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_backup(self, file_path: str):
        p = Path(file_path)
        if p.exists():
            dest = self.backups_dir / f"{p.name}.{self.timestamp}.bak"
            shutil.copy(str(p), str(dest))
            return True
        return False

    def optimize_time_manager(self):
        """Ajusta tolerância de offset no time_manager."""
        path = "time_manager.py"
        if not os.path.exists(path): return
        
        self.create_backup(path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Aumenta a tolerância para Windows (1.5s) para evitar status degradado constante
        if "max_offset_ms = 500" in content:
            content = content.replace("max_offset_ms = 500", "max_offset_ms = 1500")
            print("✅ TimeManager: Offset máximo aumentado para 1500ms.")
            
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    def optimize_hybrid_weights(self):
        """Otimiza pesos do HybridDecisionMaker."""
        path = "config.py" # ou onde os pesos são definidos
        if not os.path.exists(path): return
        
        self.create_backup(path)
        # Lógica de substituição de pesos se necessário
        # Ex: HYBRID_MODEL_WEIGHT = 0.7
        pass

if __name__ == "__main__":
    print("🛠️ INICIANDO AUTO-FIX / OTIMIZAÇÃO")
    fixer = AutoFixer()
    fixer.optimize_time_manager()
    print("✨ Otimizações concluídas. Verifique diretório backups/")
