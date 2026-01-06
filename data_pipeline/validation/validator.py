# data_pipeline/validation/validator.py
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xxhash
except ImportError:
    xxhash = None  # type: ignore

from ..cache.lru_cache import LRUCache
from ..logging_utils import PipelineLogger


class TradeValidator:
    """
    Validador de trades com dois modos: vetorizado (rápido) e loop (fallback).

    Modo vetorizado:
    - Usa operações pandas nativas
    - 10-18x mais rápido que loop
    - Preferido quando disponível

    Modo loop:
    - Fallback para compatibilidade
    - Mais lento mas sempre funciona

    Características:
    - Cache de validações
    - Logging especializado
    - Estatísticas detalhadas
    """

    def __init__(
        self,
        enable_vectorized: bool = True,
        logger: Optional[PipelineLogger] = None
    ) -> None:
        """
        Inicializa validador.

        Args:
            enable_vectorized: Se True, usa validação vetorizada
            logger: Logger especializado (opcional)
        """
        self.enable_vectorized = enable_vectorized
        self.logger = logger
        self._validation_cache = LRUCache(max_items=100, ttl_seconds=60)
        self._stats: Dict[str, Any] = {
            'total_validations': 0,
            'vectorized_validations': 0,
            'loop_validations': 0,
            'cache_hits': 0,
            'total_time_ms': 0.0
        }

    def _validate_vectorized(
        self,
        trades: List[Dict[str, Any]],
        min_trades: int = 3
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validação vetorizada usando pandas.

        ⚡ OTIMIZAÇÃO: 10-18x mais rápido que loop

        Args:
            trades: Lista de trades
            min_trades: Mínimo de trades válidos necessário

        Returns:
            Tupla (DataFrame validado, estatísticas)
        """
        start_time = time.perf_counter()

        if not trades:
            raise ValueError("Lista de trades vazia")

        # Criar DataFrame direto
        df = pd.DataFrame(trades)

        # Verificar colunas obrigatórias
        required_cols = ["p", "q", "T"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas ausentes: {missing_cols}")

        # Adicionar coluna 'm' se ausente
        if "m" not in df.columns:
            df["m"] = False

        total_received = len(df)

        # ⚡ CONVERSÃO VETORIZADA
        df["p"] = pd.to_numeric(df["p"], errors="coerce")
        df["q"] = pd.to_numeric(df["q"], errors="coerce")
        df["T"] = pd.to_numeric(df["T"], errors="coerce").astype('Int64')

        # ⚡ FILTRAGEM VETORIZADA
        valid_mask = df["p"].notna() & df["q"].notna() & df["T"].notna()
        df = df[valid_mask].copy()
        after_nan_removal = len(df)

        positive_mask = (df["p"] > 0) & (df["q"] > 0) & (df["T"] > 0)
        df = df[positive_mask].copy()
        after_positive_filter = len(df)

        # Ordenar por timestamp
        df = df.sort_values("T", kind="mergesort").reset_index(drop=True)

        # Validar quantidade mínima
        if len(df) < min_trades:
            raise ValueError(
                f"Dados insuficientes: {len(df)} trades válidos "
                f"(mínimo: {min_trades}, recebidos: {total_received})"
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Estatísticas
        stats: Dict[str, Any] = {
            'total_received': total_received,
            'total_validated': len(df),
            'invalid_trades': total_received - len(df),
            'removed_nan': total_received - after_nan_removal,
            'removed_negative': after_nan_removal - after_positive_filter,
            'validation_time_ms': round(elapsed_ms, 2),
            'method': 'vectorized',
            'trades_per_ms': round(len(df) / max(elapsed_ms, 0.001), 2)
        }

        # Calcular range de preços
        if len(df) > 0:
            price_range = float(df["p"].max() - df["p"].min())
            avg_price = float(df["p"].mean())
            price_variance_pct = (price_range / avg_price * 100) if avg_price > 0 else 0

            stats['price_variance_pct'] = round(price_variance_pct, 2)
            stats['price_range'] = (float(df["p"].min()), float(df["p"].max()))
            stats['volume_total'] = float(df["q"].sum())

        self._stats['vectorized_validations'] += 1
        self._stats['total_time_ms'] += elapsed_ms

        # Logging
        if self.logger:
            self.logger.validation_debug(
                f"✅ Validação vetorizada",
                trades=len(df),
                time_ms=round(elapsed_ms, 2),
                rate=f"{stats['trades_per_ms']:.0f}/ms"
            )

        return df, stats

    def _validate_loop(
        self,
        trades: List[Dict[str, Any]],
        min_trades: int = 3
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validação com loop Python (fallback).

        Args:
            trades: Lista de trades
            min_trades: Mínimo de trades válidos necessário

        Returns:
            Tupla (DataFrame validado, estatísticas)
        """
        start_time = time.perf_counter()

        if not trades:
            raise ValueError("Lista de trades vazia")

        validated: List[Dict[str, Any]] = []

        for trade in trades:
            if not isinstance(trade, dict):
                continue

            try:
                price = float(trade.get("p", 0))
                quantity = float(trade.get("q", 0))
                timestamp = int(trade.get("T", 0))
                is_maker = trade.get("m", False)

                if price <= 0 or quantity <= 0 or timestamp <= 0:
                    continue

                validated.append({
                    "p": price,
                    "q": quantity,
                    "T": timestamp,
                    "m": is_maker
                })
            except (ValueError, TypeError):
                continue

        if len(validated) < min_trades:
            raise ValueError(
                f"Dados insuficientes: {len(validated)} trades válidos "
                f"(mínimo: {min_trades}, recebidos: {len(trades)})"
            )

        df = pd.DataFrame(validated)
        df = df.sort_values("T").reset_index(drop=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        stats: Dict[str, Any] = {
            'total_received': len(trades),
            'total_validated': len(validated),
            'invalid_trades': len(trades) - len(validated),
            'validation_time_ms': round(elapsed_ms, 2),
            'method': 'loop',
            'trades_per_ms': round(len(validated) / max(elapsed_ms, 0.001), 2)
        }

        if len(df) > 0:
            stats['price_range'] = (float(df["p"].min()), float(df["p"].max()))
            stats['volume_total'] = float(df["q"].sum())

        self._stats['loop_validations'] += 1
        self._stats['total_time_ms'] += elapsed_ms

        # Logging
        if self.logger:
            self.logger.validation_warning(
                f"⚠️ Usando validação loop (fallback)",
                trades=len(df),
                time_ms=round(elapsed_ms, 2)
            )

        return df, stats

    def _make_cache_key(self, trades: List[Dict[str, Any]]) -> str:
        """
        Gera chave de cache leve para um lote de trades.

        Usa (len, primeiro T, último T) e xxhash/md5,
        ao invés de serializar a lista completa.
        """
        if not trades:
            key_tuple = (0, 0, 0)
        else:
            try:
                first_T = int(trades[0].get("T", 0) or 0)
                last_T = int(trades[-1].get("T", 0) or 0)
            except Exception:
                first_T = last_T = 0
            key_tuple = (len(trades), first_T, last_T)

        key_bytes = repr(key_tuple).encode("utf-8")
        if xxhash is not None:
            return xxhash.xxh64_hexdigest(key_bytes)[:16]
        return hashlib.md5(key_bytes).hexdigest()[:16]

    def validate_batch(
        self,
        trades: List[Dict[str, Any]],
        min_trades: int = 3,
        max_price_variance_pct: float = 10.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Valida lote de trades escolhendo método automaticamente.

        Args:
            trades: Lista de trades
            min_trades: Mínimo de trades válidos
            max_price_variance_pct: Máxima variação de preço permitida

        Returns:
            Tupla (DataFrame validado, estatísticas)
        """
        self._stats['total_validations'] += 1

        # Verificar cache com chave leve
        cache_key = self._make_cache_key(trades)
        cached = self._validation_cache.get(cache_key)
        if cached:
            self._stats['cache_hits'] += 1
            if self.logger:
                self.logger.validation_debug("✨ Cache hit", key=cache_key[:8])
            return cached['df'].copy(), cached['stats']

        # Escolher método de validação
        if self.enable_vectorized:
            try:
                df, stats = self._validate_vectorized(trades, min_trades)
            except Exception as e:
                if self.logger:
                    self.logger.validation_warning(
                        f"⚠️ Validação vetorizada falhou: {e}",
                        fallback="loop"
                    )
                df, stats = self._validate_loop(trades, min_trades)
        else:
            df, stats = self._validate_loop(trades, min_trades)

        # Validar variância de preço
        if 'price_variance_pct' in stats:
            if stats['price_variance_pct'] > max_price_variance_pct:
                if self.logger:
                    self.logger.validation_warning(
                        f"⚠️ Variação de preço alta",
                        variance=f"{stats['price_variance_pct']:.2f}%",
                        limit=f"{max_price_variance_pct}%"
                    )

        # Cachear resultado
        self._validation_cache.set(cache_key, {'df': df.copy(), 'stats': stats})

        return df, stats

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do validador."""
        total = self._stats['total_validations']

        if total == 0:
            return self._stats

        return {
            **self._stats,
            'avg_time_ms': round(self._stats['total_time_ms'] / total, 2),
            'vectorized_pct': round(
                self._stats['vectorized_validations'] / total * 100, 2
            ),
            'cache_hit_rate': round(
                self._stats['cache_hits'] / total * 100, 2
            )
        }