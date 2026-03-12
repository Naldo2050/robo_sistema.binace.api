# fix_playwright.py
import os
import shutil
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO)

def install_playwright():
    """Instala/atualiza o Playwright e browsers."""
    try:
        logging.info("📦 Atualizando Playwright...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "playwright"], check=True)
        
        logging.info("🌐 Instalando browsers...")
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        
        logging.info("✅ Playwright atualizado com sucesso!")
        return True
    except Exception as e:
        logging.error(f"❌ Erro ao atualizar Playwright: {e}")
        return False

def fix_playwright_extensions():
    """Remove arquivos problemáticos do diretório do Playwright."""
    
    playwright_dirs = [
        "./playwright_user_data",
        "./gemini_data"
    ]
    
    for dir_path in playwright_dirs:
        if os.path.exists(dir_path):
            logging.info(f"🔍 Verificando diretório: {dir_path}")
            
            # Arquivos/pastas problemáticos mais abrangente
            problematic_paths = [
                "Default/Extensions",
                "Default/Local Extension Settings",
                "Default/Network Persistent State",
                "Default/TransportSecurity",
                "Default/Web Data",
                "Default/Web Data-journal",
                "CrashpadMetrics-active.pma",
                "chrome_debug.log",
                "Default/Preferences",
                "Default/Secure Preferences",
                "Default/Login Data",
                "Default/Login Data-journal"
            ]
            
            for prob_path in problematic_paths:
                full_path = os.path.join(dir_path, prob_path)
                try:
                    if os.path.exists(full_path):
                        if os.path.isfile(full_path):
                            os.remove(full_path)
                            logging.info(f"✅ Arquivo removido: {prob_path}")
                        elif os.path.isdir(full_path):
                            shutil.rmtree(full_path)
                            logging.info(f"✅ Diretório removido: {prob_path}")
                except Exception as e:
                    logging.warning(f"⚠️ Não foi possível remover {prob_path}: {e}")
        else:
            # Cria o diretório limpo
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"📁 Diretório criado: {dir_path}")

def clean_all_playwright_data():
    """Remove completamente todos os dados do Playwright."""
    print("⚠️  ATENÇÃO: Isso vai remover TODOS os dados salvos do browser!")
    print("   Você precisará fazer login novamente no Google Gemini.")
    
    choice = input("Deseja continuar? (s/N): ").lower().strip()
    
    if choice == 's':
        dirs_to_clean = ["./playwright_user_data", "./gemini_data"]
        
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    logging.info(f"🧹 Diretório removido: {dir_path}")
                except Exception as e:
                    logging.error(f"❌ Erro ao remover {dir_path}: {e}")
        
        logging.info("✅ Limpeza completa realizada!")
    else:
        logging.info("❌ Limpeza cancelada.")

def setup_first_time():
    """Configuração para primeira execução."""
    print("🚀 Configuração inicial do sistema...")
    
    # Atualiza Playwright
    if not install_playwright():
        print("❌ Falha na instalação do Playwright")
        return False
    
    # Remove arquivos problemáticos
    fix_playwright_extensions()
    
    print("\n✅ Configuração inicial concluída!")
    print("💡 IMPORTANTE: Na primeira execução do bot, uma janela do browser será aberta.")
    print("   Faça login no Google/Gemini e depois você pode fechar a janela.")
    print("   Nas próximas execuções, o login será lembrado.")
    
    return True

if __name__ == "__main__":
    print("=== CORRETOR DO PLAYWRIGHT ===")
    print("1. Configuração inicial (recomendado para primeira vez)")
    print("2. Corrigir problemas de extensão")
    print("3. Limpeza completa (remove login salvo)")
    print("4. Apenas atualizar Playwright")
    print("5. Sair")
    
    choice = input("Escolha uma opção (1/2/3/4/5): ").strip()
    
    if choice == "1":
        setup_first_time()
        
    elif choice == "2":
        fix_playwright_extensions()
        print("\n✅ Correção aplicada! Tente executar o bot novamente.")
        
    elif choice == "3":
        clean_all_playwright_data()
        
    elif choice == "4":
        install_playwright()
        
    else:
        print("👋 Saindo...")