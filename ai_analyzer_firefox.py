# ai_analyzer_firefox.py

import json
from playwright.sync_api import sync_playwright, TimeoutError
import logging
import time
import os
import shutil
import random

# URL do Google Gemini
GEMINI_URL = "https://gemini.google.com/app"

class AIAnalyzer:
    def __init__(self, headless=False, user_data_dir="./firefox_data"):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.browser = None
        self.context = None
        self.page = None
        self.playwright = None
        self.is_logged_in = False
        self._ensure_user_data_dir()
    
    def _ensure_user_data_dir(self):
        """Garante que o diretório de dados do usuário existe."""
        if not os.path.exists(self.user_data_dir):
            os.makedirs(self.user_data_dir, exist_ok=True)
            logging.info(f"📁 Diretório Firefox criado: {self.user_data_dir}")

    def clean_user_data_dir(self):
        """Limpa completamente o diretório de dados do usuário."""
        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
                
            time.sleep(2)
            
            if os.path.exists(self.user_data_dir):
                shutil.rmtree(self.user_data_dir)
                logging.info(f"🧹 Diretório Firefox limpo: {self.user_data_dir}")
            self._ensure_user_data_dir()
            
            # Reinicializa variáveis
            self.browser = None
            self.context = None
            self.page = None
            self.playwright = None
            self.is_logged_in = False
            
        except Exception as e:
            logging.error(f"❌ Erro ao limpar diretório: {e}")

    def _create_prompt(self, event_data: dict) -> str:
        """Cria um prompt otimizado para análise de eventos de mercado."""
        tipo_evento = event_data.get("tipo_evento", "N/A")
        ativo = event_data.get("ativo", "N/A")
        descricao = event_data.get("descricao", "Sem descrição.")
        delta = event_data.get("delta", 0.0)
        volume_total = event_data.get("volume_total", 0.0)
        preco_fechamento = event_data.get("preco_fechamento", 0.0)
        volume_compra = event_data.get("volume_compra", 0.0)
        volume_venda = event_data.get("volume_venda", 0.0)
        
        contexto_extra = ""
        if event_data.get("contexto_sma"):
            contexto_extra += f"- **Contexto SMA:** {event_data.get('contexto_sma')}\n"
        if event_data.get("indice_absorcao"):
            contexto_extra += f"- **Índice de Absorção:** {event_data.get('indice_absorcao'):.2f}\n"

        prompt = f"""Analise este evento de mercado como um especialista em order flow:

**DADOS DO EVENTO:**
- Ativo: {ativo}
- Tipo: {tipo_evento}
- Descrição: {descricao}
- Delta: {delta:.2f}
- Volume Total: {volume_total:.0f}
- Volume Compra: {volume_compra:.0f}
- Volume Venda: {volume_venda:.0f}
- Preço Fechamento: ${preco_fechamento:.2f}
{contexto_extra}

**ANÁLISE SOLICITADA:**
Forneça uma análise concisa (máximo 150 palavras) respondendo:

1. **Interpretação:** O que este evento revela sobre o fluxo de ordens?
2. **Força Dominante:** Compradores ou vendedores estão no controle?
3. **Expectativa:** Qual movimento é mais provável no curto prazo?
4. **Ação:** Recomendação prática para este cenário.

Seja direto e objetivo."""
        
        return prompt

    def _wait_for_element_safely(self, page, selector, timeout=15000):
        """Espera por um elemento com tratamento de erro melhorado."""
        try:
            element = page.wait_for_selector(selector, timeout=timeout, state="visible")
            return element
        except TimeoutError:
            logging.warning(f"⏰ Timeout aguardando elemento: {selector}")
            return None
        except Exception as e:
            logging.warning(f"⚠️ Erro aguardando elemento {selector}: {e}")
            return None

    def _initialize_browser(self):
        """Inicializa o Firefox com configurações otimizadas."""
        try:
            if self.playwright is None:
                logging.info("🦊 Inicializando Playwright com Firefox...")
                self.playwright = sync_playwright().start()
                
            if self.browser is None:
                logging.info("🌐 Abrindo Firefox...")
                
                # Firefox tem menos detecção de automação que Chrome
                firefox_args = [
                    "--width=1920",
                    "--height=1080"
                ]
                
                self.browser = self.playwright.firefox.launch(
                    headless=self.headless,
                    args=firefox_args
                )
                
                # Cria contexto com configurações realistas
                self.context = self.browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
                    locale="pt-BR",
                    timezone_id="America/Sao_Paulo",
                    extra_http_headers={
                        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Encoding": "gzip, deflate, br",
                        "DNT": "1",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1"
                    }
                )
                
            if self.page is None:
                logging.info("📄 Criando nova página...")
                self.page = self.context.new_page()
                
                # Bloqueia apenas recursos desnecessários
                self.page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2}", lambda route: route.abort())
                
        except Exception as e:
            logging.error(f"❌ Erro ao inicializar Firefox: {e}")
            raise e

    def _handle_initial_popups(self, page):
        """Lida com popups iniciais que podem aparecer no Gemini."""
        logging.info("🔍 Verificando popups iniciais...")
        
        popup_selectors = [
            # Seletores em português
            "button:has-text('Entendi')",
            "button:has-text('OK')",
            "button:has-text('Aceitar')", 
            "button:has-text('Continuar')",
            "button:has-text('Começar')",
            "button:has-text('Fechar')",
            # Seletores em inglês
            "button:has-text('Got it')",
            "button:has-text('Accept')",
            "button:has-text('Continue')",
            "button:has-text('Start')",
            "button:has-text('Close')",
            "button:has-text('Dismiss')",
            # Seletores por atributos
            "[data-testid='dismiss-button']",
            "[data-testid='close-button']",
            "[data-testid='continue-button']",
            "[aria-label*='dismiss']",
            "[aria-label*='close']",
            "[aria-label*='fechar']",
            # Seletores genéricos
            ".dismiss-button",
            ".close-button",
            "button[class*='dismiss']",
            "button[class*='close']"
        ]
        
        for selector in popup_selectors:
            try:
                elements = page.locator(selector)
                count = elements.count()
                if count > 0:
                    for i in range(count):
                        element = elements.nth(i)
                        if element.is_visible(timeout=1000):
                            element.click(timeout=3000)
                            logging.info(f"✅ Popup fechado: {selector}")
                            time.sleep(1)
                            break
            except Exception as e:
                continue
        
        time.sleep(2)

    def _check_login_status(self, page):
        """Verifica se o usuário está logado no Google."""
        try:
            # Verifica indicadores de login
            login_indicators = [
                "[data-testid='user-avatar']",
                ".gb_d",  # Avatar do Google
                "[aria-label*='Account']",
                "[aria-label*='Conta']",
                "img[alt*='profile']",
                "[data-feature-request-trigger-label='avatar']",
                "img[src*='googleusercontent']"
            ]
            
            for indicator in login_indicators:
                try:
                    if page.locator(indicator).first.is_visible(timeout=3000):
                        logging.info("✅ Usuário parece estar logado")
                        self.is_logged_in = True
                        return True
                except:
                    continue
            
            # Verifica se há botões de login
            login_buttons = [
                "button:has-text('Sign in')",
                "button:has-text('Entrar')",
                "a:has-text('Sign in')",
                "a:has-text('Entrar')",
                "button:has-text('Fazer login')"
            ]
            
            for button in login_buttons:
                try:
                    if page.locator(button).first.is_visible(timeout=2000):
                        logging.warning("⚠️ Usuário não está logado - botão de login encontrado")
                        self.is_logged_in = False
                        return False
                except:
                    continue
                    
            # Se não encontrou nem login nem logout, assume que está logado
            self.is_logged_in = True
            return True
            
        except Exception as e:
            logging.error(f"❌ Erro ao verificar status de login: {e}")
            return False

    def _find_text_input(self, page):
        """Encontra a área de input de texto do Gemini."""
        text_input_selectors = [
            # Seletores mais específicos primeiro
            "div[contenteditable='true'][role='textbox']",
            "div[contenteditable='true'][data-testid*='input']",
            "div[contenteditable='true'][aria-label*='message']",
            "div[contenteditable='true'][aria-label*='pergunta']",
            "div[contenteditable='true'][aria-label*='question']",
            # Seletores por classe
            ".ql-editor[contenteditable='true']",
            "div[class*='input'][contenteditable='true']",
            "div[class*='editor'][contenteditable='true']",
            "div[class*='composer'][contenteditable='true']",
            # Seletores mais genéricos
            "[contenteditable='true']",
            "textarea[placeholder*='message']",
            "textarea[placeholder*='pergunta']",
            "input[type='text'][placeholder*='message']",
            # Fallbacks
            "div[role='textbox']",
            "textarea",
            "input[type='text']"
        ]
        
        for selector in text_input_selectors:
            try:
                elements = page.locator(selector)
                count = elements.count()
                
                if count > 0:
                    for i in range(count):
                        element = elements.nth(i)
                        try:
                            if element.is_visible(timeout=2000) and element.is_enabled(timeout=1000):
                                logging.info(f"✅ Campo de texto encontrado: {selector} (elemento {i+1}/{count})")
                                return element
                        except:
                            continue
            except Exception as e:
                continue
        
        logging.error("❌ Nenhum campo de texto encontrado")
        return None

    def _find_send_button(self, page):
        """Encontra o botão de envio com múltiplas estratégias."""
        send_button_selectors = [
            # Seletores específicos por texto
            "button:has-text('Enviar')",
            "button:has-text('Send')", 
            "button:has-text('Submit')",
            "button[title='Enviar']",
            "button[title='Send']",
            "button[aria-label*='Enviar']",
            "button[aria-label*='Send']",
            "button[aria-label*='Submit']",
            # Seletores por ícones/símbolos
            "button:has([data-testid*='send'])",
            "button:has([aria-label*='send'])",
            "button[data-testid*='send']",
            "button[class*='send']",
            # Seletores por posição/contexto
            "form button[type='submit']",
            "div[class*='input'] button",
            "div[class*='composer'] button",
            # Seletores mais genéricos
            "button[type='submit']",
            "button:last-child"
        ]
        
        for selector in send_button_selectors:
            try:
                elements = page.locator(selector)
                count = elements.count()
                
                if count > 0:
                    for i in range(count):
                        element = elements.nth(i)
                        try:
                            if element.is_visible(timeout=2000) and element.is_enabled(timeout=1000):
                                logging.info(f"✅ Botão de envio encontrado: {selector} (elemento {i+1}/{count})")
                                return element
                        except:
                            continue
            except Exception as e:
                continue
        
        logging.warning("⚠️ Botão de envio não encontrado")
        return None

    def _clear_text_input(self, text_input):
        """Limpa o campo de texto de forma robusta."""
        try:
            # Método 1: Click e select all + delete
            text_input.click()
            time.sleep(0.5)
            text_input.press("Control+A")
            time.sleep(0.2)
            text_input.press("Delete")
            time.sleep(0.3)
            return True
        except Exception as e:
            logging.warning(f"⚠️ Método 1 de limpeza falhou: {e}")
            
        try:
            # Método 2: Usar fill para limpar
            text_input.fill("")
            time.sleep(0.3)
            return True
        except Exception as e:
            logging.warning(f"⚠️ Método 2 de limpeza falhou: {e}")
            
        return False

    def _type_message(self, text_input, message):
        """Digita a mensagem de forma robusta."""
        try:
            # Garante foco no campo
            text_input.click()
            time.sleep(0.5)
            
            # Digita com delay humano
            delay = random.randint(30, 80)
            text_input.type(message, delay=delay)
            
            time.sleep(1)
            return True
            
        except Exception as e:
            logging.error(f"❌ Erro ao digitar mensagem: {e}")
            return False

    def _send_message_to_gemini(self, prompt):
        """Envia mensagem para o Gemini usando Firefox."""
        try:
            # Navega para o Gemini se necessário
            current_url = self.page.url
            if not current_url.startswith("https://gemini.google.com"):
                logging.info("🌐 Navegando para o Google Gemini...")
                self.page.goto(GEMINI_URL, wait_until="domcontentloaded", timeout=30000)
                time.sleep(5)
            
            # Verifica e lida com popups
            self._handle_initial_popups(self.page)
            
            # Verifica status de login (versão simplificada para Firefox)
            if not self._check_login_status(self.page):
                logging.error("❌ Usuário não está logado no Google.")
                if not self.headless:
                    print("\n" + "="*60)
                    print("🔐 AÇÃO NECESSÁRIA: LOGIN MANUAL NO FIREFOX")
                    print("="*60)
                    print("1. Na janela do Firefox que foi aberta:")
                    print("2. Vá para https://accounts.google.com")
                    print("3. Faça login com sua conta Google")
                    print("4. Depois vá para https://gemini.google.com")
                    print("5. Aceite os termos se solicitado")
                    print("6. Volte aqui e pressione Enter para continuar")
                    print("="*60)
                    input("Pressione Enter após fazer login...")
                    
                    # Recarrega e verifica novamente
                    self.page.reload()
                    time.sleep(3)
                    if not self._check_login_status(self.page):
                        return "Erro: Login ainda não detectado. Verifique se fez login corretamente."
                else:
                    return "Erro: Usuário não logado. Execute com headless=False para fazer login manual."
            
            # Aguarda a página carregar completamente
            time.sleep(3)
            
            # Encontra o campo de texto
            text_input = self._find_text_input(self.page)
            if not text_input:
                raise Exception("Campo de input de texto não encontrado no Gemini")
            
            # Limpa o campo de texto
            logging.info("🧹 Limpando campo de texto...")
            if not self._clear_text_input(text_input):
                logging.warning("⚠️ Não foi possível limpar completamente o campo")
            
            # Digita a mensagem
            logging.info("⌨️ Digitando prompt...")
            if not self._type_message(text_input, prompt):
                raise Exception("Falha ao digitar mensagem no campo de texto")
            
            # Encontra e clica no botão de envio
            send_button = self._find_send_button(self.page)
            sent_successfully = False
            
            if send_button:
                try:
                    logging.info("📤 Enviando mensagem via botão...")
                    send_button.click(timeout=10000)
                    sent_successfully = True
                except Exception as e:
                    logging.warning(f"⚠️ Falha ao clicar no botão: {e}")
            
            # Fallbacks para envio
            if not sent_successfully:
                try:
                    logging.info("📤 Tentando enviar com Enter...")
                    text_input.press("Enter")
                    sent_successfully = True
                except Exception as e:
                    logging.warning(f"⚠️ Falha ao enviar com Enter: {e}")
            
            if not sent_successfully:
                try:
                    logging.info("📤 Tentativa com Ctrl+Enter...")
                    self.page.keyboard.press("Control+Enter")
                    sent_successfully = True
                except Exception as e:
                    logging.warning(f"⚠️ Falha ao enviar com Ctrl+Enter: {e}")
            
            if not sent_successfully:
                raise Exception("Não foi possível enviar a mensagem por nenhum método")

            # Aguarda resposta
            logging.info("⏳ Aguardando resposta da IA...")
            
            # Aguarda indicadores de que a resposta está sendo gerada
            response_indicators = [
                "[aria-label*='generating']",
                "[aria-label*='gerando']",
                "div:has-text('Thinking')",
                "div:has-text('Pensando')",
                ".generating",
                "[data-testid*='generating']"
            ]
            
            # Aguarda início da resposta
            for indicator in response_indicators:
                try:
                    self.page.wait_for_selector(indicator, timeout=10000)
                    logging.info("✅ IA começou a responder")
                    break
                except:
                    continue
            
            # Aguarda um tempo mínimo para resposta
            time.sleep(10)
            
            # Aguarda indicadores de conclusão
            completion_indicators = [
                "button:has-text('Regenerar')",
                "button:has-text('Regenerate')",
                "button[aria-label*='regenerate']",
                "button[aria-label*='regenerar']",
                "[data-testid*='regenerate']"
            ]
            
            completed = False
            for indicator in completion_indicators:
                try:
                    self.page.wait_for_selector(indicator, timeout=90000)
                    completed = True
                    logging.info("✅ Resposta concluída")
                    break
                except:
                    continue
            
            if not completed:
                logging.warning("⚠️ Indicadores de conclusão não encontrados, aguardando tempo fixo...")
                time.sleep(30)
            
            # Aguarda adicional para garantir
            time.sleep(2)

            # Captura a resposta
            logging.info("📖 Capturando resposta...")
            
            response_selectors = [
                # Seletores mais específicos primeiro
                "div[data-message-author-role='model'] div[class*='markdown']",
                "div[data-message-author-role='model'] div",
                "[data-testid='response-text']",
                ".model-response .markdown-content",
                ".model-response div",
                ".response-container-content",
                ".markdown-content",
                "div[class*='response'] div[class*='content']"
            ]
            
            response_text = ""
            for selector in response_selectors:
                try:
                    elements = self.page.locator(selector)
                    count = elements.count()
                    
                    if count > 0:
                        # Tenta pegar o último elemento (resposta mais recente)
                        last_element = elements.last
                        if last_element.is_visible():
                            response_text = last_element.inner_text()
                            if response_text.strip() and len(response_text.strip()) > 10:
                                logging.info(f"✅ Resposta capturada: {selector} ({len(response_text)} chars)")
                                break
                except Exception as e:
                    continue
            
            # Fallback: JavaScript para capturar
            if not response_text.strip():
                try:
                    response_text = self.page.evaluate("""
                        () => {
                            const selectors = [
                                '[data-message-author-role="model"] div',
                                '.model-response div',
                                '.response div',
                                'div[class*="response"] div'
                            ];
                            
                            for (const selector of selectors) {
                                const elements = document.querySelectorAll(selector);
                                if (elements.length > 0) {
                                    const lastElement = elements[elements.length - 1];
                                    if (lastElement.innerText && lastElement.innerText.trim().length > 10) {
                                        return lastElement.innerText.trim();
                                    }
                                }
                            }
                            return '';
                        }
                    """)
                except Exception as e:
                    logging.error(f"❌ Erro no fallback JavaScript: {e}")

            if response_text.strip():
                logging.info(f"✅ Resposta capturada com sucesso ({len(response_text)} caracteres)")
                return response_text.strip()
            else:
                return "A IA retornou uma resposta vazia ou não foi possível capturá-la."

        except Exception as e:
            logging.error(f"❌ Erro ao enviar mensagem para o Gemini: {e}")
            return f"Erro na comunicação com a IA: {str(e)}"

    def analyze_event(self, event_data: dict) -> str:
        """Analisa um evento de mercado usando IA."""
        try:
            self._initialize_browser()
            prompt = self._create_prompt(event_data)
            
            logging.info("🤖 Iniciando análise com IA (Firefox)...")
            response = self._send_message_to_gemini(prompt)
            
            if response and len(response.strip()) > 10:
                logging.info("✅ Análise da IA concluída com sucesso!")
                return response
            else:
                return "A IA retornou uma resposta vazia ou muito curta."

        except Exception as e:
            logging.error(f"❌ Erro durante análise da IA: {e}")
            return f"Erro na análise da IA: {str(e)}"

    def close(self):
        """Fecha o browser e limpa recursos."""
        try:
            if self.page:
                self.page.close()
                self.page = None
            if self.context:
                self.context.close()
                self.context = None
            if self.browser:
                self.browser.close()
                self.browser = None
            if self.playwright:
                self.playwright.stop()
                self.playwright = None
            logging.info("🔒 Firefox fechado e recursos liberados.")
        except Exception as e:
            logging.error(f"Erro ao fechar Firefox: {e}")

    def __del__(self):
        """Destructor para garantir limpeza de recursos."""
        self.close()