# utils/price_fetcher.py
# -*- coding: utf-8 -*-

"""
Função utilitária para buscar preço atual via REST.
Extraída integralmente do market_orchestrator.py sem mudanças de lógica.
"""

import logging
import time
import requests
from typing import Optional


def get_current_price(symbol: str) -> Optional[float]:
    """
    Obtém preço atual via REST API com retry.

    Levanta RuntimeError se não for possível obter o preço após todas as tentativas.
    Código 100% idêntico ao original.
    """
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/price"
            params = {"symbol": symbol}

            res = requests.get(url, params=params, timeout=5)
            res.raise_for_status()

            data = res.json()
            return float(data["price"])

        except requests.exceptions.RequestException as e:
            logging.error(
                f"Erro ao buscar preço via REST "
                f"(tentativa {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))

        except Exception as e:
            logging.error(
                f"Erro inesperado ao buscar preço via REST "
                f"(tentativa {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))

    message = (
        "💀 FALHA CRÍTICA: Não foi possível obter preço via REST "
        "após todas as tentativas"
    )
    logging.critical(message)
    raise RuntimeError(message)
