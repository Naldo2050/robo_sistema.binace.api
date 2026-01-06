# market_orchestrator/flow/signal_processor.py
from typing import Dict, Any, Optional, List
from datetime import datetime


class SignalProcessor:
    """Signal processing component."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.signal_history = []
        self.processed_signals_count = 0

    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Process a trading signal."""
        signal_type = signal.get('tipo_evento', 'UNKNOWN')
        processed_signal = {
            'original_signal': signal,
            'processed_at': datetime.now().isoformat(),
            'signal_type': signal_type,
            'symbol': self.symbol,
            'processing_id': self.processed_signals_count + 1,
            'status': 'processed'
        }

        # Basic signal processing logic
        if signal_type in ['BUY', 'SELL']:
            processed_signal['action'] = signal_type
            processed_signal['confidence'] = signal.get('confidence', 0.5)
        else:
            processed_signal['action'] = 'HOLD'
            processed_signal['confidence'] = 0.0

        self.signal_history.append(processed_signal)
        self.processed_signals_count += 1

        return processed_signal

    def batch_process_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple signals in batch."""
        results = []
        for signal in signals:
            result = self.process_signal(signal)
            results.append(result)
        return results

    def get_signal_metrics(self) -> Dict[str, Any]:
        """Get signal processing metrics."""
        return {
            'total_signals_processed': self.processed_signals_count,
            'symbol': self.symbol,
            'recent_signals': self.signal_history[-10:] if self.signal_history else [],
            'signal_types': list(set(s.get('signal_type', 'UNKNOWN') for s in self.signal_history))
        }

    def clear_history(self) -> None:
        """Clear signal processing history."""
        self.signal_history.clear()
        self.processed_signals_count = 0