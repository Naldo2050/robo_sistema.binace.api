# setup_google_login.py
import os
import subprocess
import sys
import logging
from playwright.sync_api import sync_playwright
import time

logging.basicConfig(level=logging.INFO)

def setup_google_login():
    """Configura login no Google de forma interativa."""
    
    print("🔐 CONFIGURAÇÃO DE LOGIN NO GOOGLE")
    print("="*50)
    print("Este script vai abrir um browser para você fazer login no Google.")
    print("Siga estas etapas:")
    print("1. Faça login na sua conta Google")
    print("2. Acesse https://gemini.google.com")
    print("3. Aceite os termos se necessário")
    print("4. Volte aqui e confirme")
    print("="*50)
    
    input("Pressione Enter para continuar...")
    
    try:
        with sync_playwright() as p:
            # Configurações otimizadas para Google
            user_data_dir = "./gemini_data"
            
            # Remove diretório antigo se existir
            if os.path.exists(user_data_dir):
                import shutil
                shutil.rmtree(user_data_dir)
                
            # Cria novo diretório
            os.makedirs(user_data_dir, exist_ok=True)
            
            # Configurações stealth
            browser_args = [
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-default-apps",
                "--disable-popup-blocking",
                "--disable-translate"
            ]
            
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            
            # Abre browser persistente
            context = p.chromium.launch_persistent_context(
                user_data_dir=user_data_dir,
                headless=False,  # Sempre visível para login
                args=browser_args,
                user_agent=user_agent,
                viewport={"width": 1920, "height": 1080},
                ignore_default_args=["--enable-automation"],
                locale="pt-BR",
                timezone_id="America/Sao_Paulo"
            )
            
            # Adiciona script stealth
            context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                window.chrome = { runtime: {} };
            """)
            
            page = context.new_page()
            
            print("\n🌐 Abrindo Google...")
            page.goto("https://accounts.google.com", wait_until="domcontentloaded")
            
            print("\n✅ Browser aberto!")
            print("🔸 Faça login na sua conta Google")
            print("🔸 Depois vá para https://gemini.google.com")
            print("🔸 Aceite os termos do Gemini se necessário")
            
            # Aguarda confirmação do usuário
            input("\nPressione Enter quando terminar o login...")
            
            # Testa o Gemini
            print("\n🧪 Testando acesso ao Gemini...")
            page.goto("https://gemini.google.com/app", wait_until="domcontentloaded")
            time.sleep(5)
            
            # Verifica se conseguiu acessar
            if "gemini.google.com" in page.url:
                print("✅ Acesso ao Gemini confirmado!")
                print("✅ Login configurado com sucesso!")
                
                # Aguarda mais um pouco para garantir que salvou
                print("💾 Salvando configurações...")
                time.sleep(5)
                
            else:
                print("❌ Problema ao acessar o Gemini. Tente novamente.")
                
            context.close()
            
    except Exception as e:
        logging.error(f"❌ Erro durante configuração: {e}")
        return False
    
    print("\n🎉 Configuração concluída!")
    print("💡 Agora você pode executar o bot normalmente.")
    print("💡 O login será lembrado nas próximas execuções.")
    
    return True

if __name__ == "__main__":
    setup_google_login()