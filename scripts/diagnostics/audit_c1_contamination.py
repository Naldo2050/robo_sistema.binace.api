#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/audit_c1_contamination.py
Examina cada arquivo listado em C1 para encontrar a linha exata e campos cruzados.
"""

import os

modules = [
    "flow_analyzer/absorption.py",
    "institutional/absorption_detector.py",
    "institutional/iceberg_detector.py",
    "institutional/enricher.py",
    "support_resistance/defense_zones.py",
    "orderbook_analyzer/core.py",
    "data_processing/data_handler.py",
    "market_orchestrator/signals/signal_processor.py",
    "trading/alert_engine.py",
]

for m in modules:
    exists = os.path.exists(m)
    print(f"=== {m} (Existe: {exists}) ===")
    if not exists:
        continue
    with open(m, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    print(f"Total linhas: {len(lines)}")
    # Buscar termos de cruzamento
    flow_terms = ["delta", "cvd", "aggress", "flow", "trade", "volume_compra", "volume_venda"]
    book_terms = ["wall", "depth", "orderbook", "order_book", "bid_depth", "ask_depth", "imbalance"]
    
    # Procurar funções ou trechos que usem ambos
    matches = []
    for idx, line in enumerate(lines):
        has_f = any(ft in line.lower() for ft in flow_terms)
        has_b = any(bt in line.lower() for bt in book_terms)
        if has_f and has_b:
            matches.append((idx + 1, line.strip()))
            
    print(f"Linhas contendo termos de ambos: {len(matches)}")
    for l_num, l_txt in matches[:5]:
        print(f"  L{l_num}: {l_txt[:100]}")
    print()
