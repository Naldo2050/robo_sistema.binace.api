# market_orchestrator/orchestrator.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class OrchestratorConfig:
    """Configuration for market orchestrator."""
    symbol: str
    stream_url: str
    window_size_minutes: int = 5
    vol_factor_exh: float = 1.5
    history_size: int = 100
    delta_std_dev_factor: float = 2.0
    context_sma_period: int = 20
    liquidity_flow_alert_percentage: float = 0.4
    wall_std_dev_factor: float = 3.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.window_size_minutes <= 0:
            raise ValueError("window_size_minutes must be positive")
        if self.vol_factor_exh <= 0:
            raise ValueError("vol_factor_exh must be positive")
        if self.history_size <= 0:
            raise ValueError("history_size must be positive")


class MarketOrchestrator:
    """Market orchestrator for managing trading operations."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.symbol = config.symbol
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.last_update: Optional[datetime] = None
        self.state: Dict[str, Any] = {}

    def start(self) -> None:
        """Start the market orchestrator."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.state = {
            'status': 'running',
            'symbol': self.symbol,
            'start_time': self.start_time.isoformat(),
            'config': self.config.__dict__.copy()
        }

    def stop(self) -> None:
        """Stop the market orchestrator."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.state['status'] = 'stopped'
        self.state['stop_time'] = datetime.now().isoformat()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the orchestrator."""
        return {
            'is_running': self.is_running,
            'symbol': self.symbol,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'state': self.state.copy()
        }

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update orchestrator state."""
        self.state.update(updates)
        self.last_update = datetime.now()

    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a market event."""
        if not self.is_running:
            return {'status': 'error', 'message': 'Orchestrator is not running'}

        event_type = event.get('tipo_evento', 'UNKNOWN')
        self.update_state({'last_event': event_type, 'last_event_time': datetime.now().isoformat()})

        # Basic event processing logic
        result = {
            'status': 'processed',
            'event_type': event_type,
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat()
        }

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()

        return {
            'uptime_seconds': uptime_seconds,
            'symbol': self.symbol,
            'is_running': self.is_running,
            'config': self.config.__dict__.copy(),
            'state': self.state.copy()
        }