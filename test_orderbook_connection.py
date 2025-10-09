#!/usr/bin/env python3
"""
Teste de Conexão com OrderBook da Binance

Este script testa:
1. Conexão com a API de orderbook da Binance
2. Extração de dados brutos
3. Validação dos dados extraídos
4. Processamento através do OrderBookAnalyzer
"""

import json
import logging
import time
from typing import Dict, Any, Optional

import requests

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurações
SYMBOL = "BTCUSDT"
API_ENDPOINTS = {
    "futures": "https://fapi.binance.com/fapi/v1/depth",
    "spot": "https://api.binance.com/api/v3/depth",
}
TIMEOUT_SECONDS = 10


def test_raw_api_connection():
    """Testa conexão direta com a API da Binance."""
    logger.info("=" * 60)
    logger.info("TESTE 1: Conexão Direta com a API da Binance")
    logger.info("=" * 60)
    
    for market_type, endpoint in API_ENDPOINTS.items():
        logger.info(f"\nTestando conexão com {market_type.upper()}: {endpoint}")
        
        try:
            # Parâmetros da requisição
            params = {
                "symbol": SYMBOL,
                "limit": 20  # Obtém 20 níveis de profundidade
            }
            
            # Faz a requisição
            start_time = time.time()
            response = requests.get(endpoint, params=params, timeout=TIMEOUT_SECONDS)
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Verifica o status
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Tempo de resposta: {elapsed_ms:.2f}ms")
            
            if response.status_code == 200:
                # Processa a resposta
                data = response.json()
                
                # Verifica estrutura básica
                if "bids" in data and "asks" in data:
                    bids_count = len(data["bids"])
                    asks_count = len(data["asks"])
                    
                    logger.info(f"Estrutura válida: bids={bids_count}, asks={asks_count}")
                    
                    # Extrai alguns dados para verificação
                    if bids_count > 0 and asks_count > 0:
                        best_bid = data["bids"][0]
                        best_ask = data["asks"][0]
                        
                        bid_price = float(best_bid[0])
                        bid_qty = float(best_bid[1])
                        ask_price = float(best_ask[0])
                        ask_qty = float(best_ask[1])
                        
                        logger.info(f"Melhor BID: ${bid_price:.2f} (qty: {bid_qty:.6f})")
                        logger.info(f"Melhor ASK: ${ask_price:.2f} (qty: {ask_qty:.6f})")
                        
                        # Calcula spread
                        spread = ask_price - bid_price
                        spread_pct = (spread / bid_price) * 100
                        logger.info(f"Spread: ${spread:.2f} ({spread_pct:.4f}%)")
                        
                        # Calcula profundidade
                        bid_depth = sum(float(bid[0]) * float(bid[1]) for bid in data["bids"][:5])
                        ask_depth = sum(float(ask[0]) * float(ask[1]) for ask in data["asks"][:5])
                        logger.info(f"Profundidade (top 5): BID=${bid_depth:,.0f}, ASK=${ask_depth:,.0f}")
                        
                        # Salva dados brutos para análise posterior
                        with open(f"orderbook_raw_{market_type}.json", "w") as f:
                            json.dump(data, f, indent=2)
                        logger.info(f"Dados brutos salvos em orderbook_raw_{market_type}.json")
                        
                        logger.info(f"✅ Conexão com {market_type.upper()} bem-sucedida!")
                    else:
                        logger.error(f"❌ Dados de orderbook vazios para {market_type.upper()}")
                else:
                    logger.error(f"❌ Estrutura inválida na resposta de {market_type.upper()}")
                    logger.debug(f"Resposta: {data}")
            else:
                logger.error(f"❌ Falha na requisição: {response.status_code}")
                logger.error(f"Resposta: {response.text}")
        
        except requests.exceptions.Timeout:
            logger.error(f"❌ Timeout na conexão com {market_type.upper()}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro na requisição com {market_type.upper()}: {e}")
        except Exception as e:
            logger.error(f"❌ Erro inesperado com {market_type.upper()}: {e}")


def test_orderbook_analyzer():
    """Testa o processamento através do OrderBookAnalyzer."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTE 2: Processamento com OrderBookAnalyzer")
    logger.info("=" * 60)
    
    try:
        # Importa o OrderBookAnalyzer
        from orderbook_analyzer import OrderBookAnalyzer
        from time_manager import TimeManager
        
        # Inicializa o TimeManager
        tm = TimeManager()
        
        # Inicializa o OrderBookAnalyzer
        analyzer = OrderBookAnalyzer(
            symbol=SYMBOL,
            market_type="futures",
            time_manager=tm,
            cache_ttl_seconds=5.0,
            max_stale_seconds=30.0
        )
        
        logger.info("OrderBookAnalyzer inicializado com sucesso")
        
        # Faz a análise
        logger.info("Executando análise...")
        event = analyzer.analyze()
        
        # Verifica o resultado
        if event:
            logger.info("✅ Análise concluída com sucesso!")
            
            # Extrai informações principais
            is_valid = event.get("is_valid", False)
            is_data_valid = event.get("is_data_valid", False)
            bid_depth = event.get("orderbook_data", {}).get("bid_depth_usd", 0)
            ask_depth = event.get("orderbook_data", {}).get("ask_depth_usd", 0)
            imbalance = event.get("orderbook_data", {}).get("imbalance", 0)
            
            logger.info(f"Evento válido: {is_valid}")
            logger.info(f"Dados válidos: {is_data_valid}")
            logger.info(f"Profundidade BID: ${bid_depth:,.2f}")
            logger.info(f"Profundidade ASK: ${ask_depth:,.2f}")
            logger.info(f"Imbalance: {imbalance:.4f}")
            
            # Salva o evento completo para análise
            with open("orderbook_analyzed.json", "w") as f:
                json.dump(event, f, indent=2, default=str)
            logger.info("Evento analisado salvo em orderbook_analyzed.json")
            
            # Obtém estatísticas
            stats = analyzer.get_stats()
            logger.info(f"Estatísticas: {json.dumps(stats, indent=2)}")
            
            return True
        else:
            logger.error("❌ Falha na análise - evento retornado é None")
            return False
    
    except ImportError as e:
        logger.error(f"❌ Erro ao importar OrderBookAnalyzer: {e}")
        logger.error("Verifique se o arquivo orderbook_analyzer.py está no mesmo diretório")
        return False
    except Exception as e:
        logger.error(f"❌ Erro ao processar com OrderBookAnalyzer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_data_extraction():
    """Testa extração de dados do evento para o AIAnalyzer."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTE 3: Extração de Dados para AIAnalyzer")
    logger.info("=" * 60)
    
    try:
        # Carrega o evento analisado
        with open("orderbook_analyzed.json", "r") as f:
            event = json.load(f)
        
        # Importa o AIAnalyzer
        from ai_analyzer_qwen import AIAnalyzer
        
        # Inicializa o AIAnalyzer
        ai = AIAnalyzer()
        
        logger.info("AIAnalyzer inicializado com sucesso")
        
        # Testa extração de orderbook
        ob_data = ai._extract_orderbook_data(event)
        
        if ob_data:
            logger.info("✅ Extração de orderbook bem-sucedida!")
            logger.info(f"Dados extraídos: {json.dumps(ob_data, indent=2)}")
            
            # Testa criação de prompt
            prompt = ai._create_prompt(event)
            
            if prompt:
                logger.info("✅ Criação de prompt bem-sucedida!")
                logger.info(f"Prompt (primeiros 500 chars): {prompt[:500]}...")
                
                # 🔧 CORREÇÃO: Salva o prompt com codificação UTF-8
                try:
                    with open("orderbook_prompt.txt", "w", encoding="utf-8") as f:
                        f.write(prompt)
                    logger.info("Prompt salvo em orderbook_prompt.txt")
                except UnicodeEncodeError:
                    # Se ainda falhar, remove os emojis e salva novamente
                    logger.warning("⚠️ Problema de codificação detectado, removendo emojis...")
                    prompt_clean = prompt.encode('ascii', 'ignore').decode('ascii')
                    with open("orderbook_prompt.txt", "w") as f:
                        f.write(prompt_clean)
                    logger.info("Prompt (sem emojis) salvo em orderbook_prompt.txt")
                
                return True
            else:
                logger.error("❌ Falha na criação do prompt")
                return False
        else:
            logger.error("❌ Falha na extração de orderbook")
            return False
    
    except FileNotFoundError:
        logger.error("❌ Arquivo orderbook_analyzed.json não encontrado")
        logger.error("Execute o TESTE 2 primeiro para gerar este arquivo")
        return False
    except ImportError as e:
        logger.error(f"❌ Erro ao importar AIAnalyzer: {e}")
        logger.error("Verifique se o arquivo ai_analyzer_qwen.py está no mesmo diretório")
        return False
    except Exception as e:
        logger.error(f"❌ Erro na extração de dados: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Função principal que executa todos os testes."""
    logger.info("Iniciando testes de conexão com OrderBook da Binance")
    logger.info(f"Símbolo: {SYMBOL}")
    
    # TESTE 1: Conexão direta com a API
    test_raw_api_connection()
    
    # Aguarda um pouco entre os testes
    time.sleep(2)
    
    # TESTE 2: Processamento com OrderBookAnalyzer
    if test_orderbook_analyzer():
        # Aguarda um pouco entre os testes
        time.sleep(2)
        
        # TESTE 3: Extração de dados para AIAnalyzer
        test_data_extraction()
    
    logger.info("\n" + "=" * 60)
    logger.info("Testes concluídos!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()