# market_analyzer.py (stub de desligamento)
import warnings

warnings.warn(
    "market_analyzer.EnhancedMarketAnalyzer foi removido da arquitetura atual. "
    "Use DataPipeline + EnhancedMarketBot em market_orchestrator.py.",
    DeprecationWarning,
    stacklevel=2,
)

raise RuntimeError(
    "market_analyzer.py é LEGADO e não faz mais parte do sistema atual.\n"
    "Use DataPipeline (data_pipeline.DataPipeline) + EnhancedMarketBot.\n"
    "Se você ainda depende disso, migre o código para a arquitetura nova."
)