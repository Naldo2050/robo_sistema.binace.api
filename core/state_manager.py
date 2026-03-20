"""
StateManager — Gerencia WindowState entre janelas.
Mantém histórico para cálculos que precisam de janelas anteriores.
"""

from collections import deque
from typing import Optional, List
import logging

from core.window_state import WindowState

logger = logging.getLogger(__name__)


class StateManager:
    """
    Singleton que gerencia o WindowState atual e histórico.

    Uso:
        mgr = StateManager.instance()

        # Início de nova janela:
        state = mgr.new_window(window_number=3, symbol="BTCUSDT")

        # Módulos preenchem:
        state.price.close = 70544.7
        state.volume.buy = 3.225
        state.mark_written('pipeline')

        # Fim da janela — validar:
        errors = mgr.finalize_window()
        if not errors:
            # Enviar para IA
            pass

        # Acessar histórico:
        prev = mgr.get_previous_state(n=1)  # Janela anterior
        history = mgr.get_returns_history(n=10)
    """

    _instance = None

    @classmethod
    def instance(cls) -> 'StateManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, max_history: int = 50):
        self._current: Optional[WindowState] = None
        self._history: deque = deque(maxlen=max_history)
        self._window_count: int = 0
        self._validation_failures: int = 0

    @property
    def current(self) -> Optional[WindowState]:
        return self._current

    def new_window(
        self, window_number: int, symbol: str = "BTCUSDT"
    ) -> WindowState:
        """Cria novo WindowState para a próxima janela"""
        # Salvar anterior no histórico
        if self._current is not None:
            self._history.append(self._current)

        self._current = WindowState(
            symbol=symbol,
            window_number=window_number,
        )
        self._window_count += 1

        logger.info(
            f"WindowState #{window_number} criado "
            f"(historico: {len(self._history)} janelas)"
        )
        return self._current

    def finalize_window(self) -> List[str]:
        """
        Valida o WindowState atual.
        Retorna erros. Lista vazia = OK para enviar para IA.
        """
        if self._current is None:
            return ["NO_ACTIVE_WINDOW"]

        errors = self._current.validate_all()

        if errors:
            self._validation_failures += 1
            logger.error(
                f"Janela {self._current.window_number} "
                f"FALHOU validacao ({len(errors)} erros). "
                f"Total falhas: {self._validation_failures}"
            )
        else:
            logger.info(
                f"Janela {self._current.window_number} "
                f"validada com sucesso"
            )

        return errors

    def get_previous_state(self, n: int = 1) -> Optional[WindowState]:
        """Retorna o state de N janelas atrás"""
        if len(self._history) >= n:
            return self._history[-n]
        return None

    def get_returns_history(self, n: int = 10) -> List[float]:
        """Retorna os últimos N returns para cálculos ML"""
        returns = []
        states = list(self._history)[-n:]
        for i in range(1, len(states)):
            prev_close = states[i - 1].price.close
            curr_close = states[i].price.close
            if prev_close > 0:
                returns.append(
                    (curr_close - prev_close) / prev_close
                )
        return returns

    def get_rsi_history(self, n: int = 14) -> List[float]:
        """Retorna os últimos N valores de RSI"""
        return [
            s.indicators.rsi
            for s in list(self._history)[-n:]
        ]

    @property
    def history_length(self) -> int:
        return len(self._history)

    @property
    def stats(self) -> dict:
        return {
            'total_windows': self._window_count,
            'history_size': len(self._history),
            'validation_failures': self._validation_failures,
            'failure_rate': (
                self._validation_failures / max(self._window_count, 1)
            ),
        }
