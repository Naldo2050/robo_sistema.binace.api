# data_pipeline.py - Compatibility layer for modular data_pipeline package
# -*- coding: utf-8 -*-
"""
Data Pipeline v3.2.1 - Modular Package Compatibility Layer

This file maintains 100% backward compatibility by importing all classes
and functions from the new modular data_pipeline package.

🔄 REFACTORED TO MODULAR PACKAGE:
  ✅ PipelineConfig -> data_pipeline.config
  ✅ PipelineLogger, setup_pipeline_logging -> data_pipeline.logging_utils
  ✅ CacheEntry, LRUCache -> data_pipeline.cache.lru_cache
  ✅ EventBatch, EventBuffer -> data_pipeline.cache.buffer
  ✅ AdaptiveThresholds -> data_pipeline.validation.adaptive
  ✅ TradeValidator -> data_pipeline.validation.validator
  ✅ MetricsProcessor -> data_pipeline.metrics.processor
  ✅ FallbackRegistry -> data_pipeline.fallback.registry
  ✅ DataPipeline -> data_pipeline.pipeline

All functionality preserved with improved modularity and maintainability.
"""

# Import all classes and functions from the modular package for backward compatibility
from data_pipeline import (
    DataPipeline,
    PipelineConfig,
    PipelineLogger,
    setup_pipeline_logging,
)

# Also import individual modules for direct access if needed
from data_pipeline.cache import CacheEntry, LRUCache, EventBatch, EventBuffer
from data_pipeline.validation import AdaptiveThresholds, TradeValidator
from data_pipeline.metrics import MetricsProcessor
from data_pipeline.fallback import FallbackRegistry

# Re-export everything for backward compatibility
__all__ = [
    # Main classes
    "DataPipeline",
    "PipelineConfig",
    "PipelineLogger",

    # Cache components
    "CacheEntry",
    "LRUCache",
    "EventBatch",
    "EventBuffer",

    # Validation components
    "AdaptiveThresholds",
    "TradeValidator",

    # Metrics components
    "MetricsProcessor",

    # Fallback components
    "FallbackRegistry",

    # Functions
    "setup_pipeline_logging",
]