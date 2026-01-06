import sys
import os
import json
import sqlite3
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from datetime import datetime

# Adiciona diretório raiz ao path para importar módulos do sistema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importações dos módulos do sistema
try:
    from flow_analyzer import FlowAnalyzer
    from data_pipeline import DataPipeline
    from orderbook_analyzer import OrderBookAnalyzer
    from time_manager import TimeManager
    from format_utils import format_price, format_quantity
except ImportError as e:
    print(f"❌ Erro ao importar módulos do sistema: {e}")
    print("Execute este script a partir da raiz do projeto ou verifique o PYTHONPATH.")
    sys.exit(1)

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("diagnostics_replay.log"),
        logging.StreamHandler()
    ]
)

class ReplayValidator:
    """
    Validador offline que reprocessa eventos históricos para detectar divergências
    de lógica, bugs ou inconsistências matemáticas.
    """
    
    def __init__(self):
        self.time_manager = TimeManager()
        
        # Inicializa módulos para replay
        self.flow_analyzer = FlowAnalyzer(time_manager=self.time_manager)

        # OrderBookAnalyzer poderia ser usado futuramente para reprocessar snapshots,
        # mas atualmente não é necessário neste script.
        self.ob_analyzer = None
        
        # Estatísticas
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "divergences": 0,
            "consistent": 0
        }
        
        self.discrepancies = []

    def load_events(self, source: str) -> Generator[Dict[str, Any], None, None]:
        """
        Gera eventos a partir de um arquivo JSONL ou banco SQLite.
        """
        path = Path(source)
        
        if not path.exists():
            logging.error(f"Fonte de dados não encontrada: {source}")
            return

        if path.suffix == '.db':
            logging.info(f"Lendo do banco de dados SQLite: {source}")
            try:
                conn = sqlite3.connect(source)
                cursor = conn.cursor()
                # Tenta ler da tabela events
                cursor.execute("SELECT payload FROM events ORDER BY timestamp_ms ASC")
                for row in cursor:
                    try:
                        yield json.loads(row[0])
                    except json.JSONDecodeError:
                        continue
                conn.close()
            except Exception as e:
                logging.error(f"Erro ao ler SQLite: {e}")
        
        elif path.suffix == '.jsonl' or path.suffix == '.json':
            logging.info(f"Lendo do arquivo JSON/JSONL: {source}")
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        else:
            logging.error("Formato não suportado. Use .db, .json ou .jsonl")

    def validate_calculations(self, event: Dict[str, Any]):
        """
        Verifica consistência matemática interna dos campos já processados.
        Útil quando os dados brutos (trades) não estão disponíveis para reprocessamento total.
        """
        issues = []
        
        # 1. Validação de Volumes (Delta = Buy - Sell)
        if 'volume_compra' in event and 'volume_venda' in event:
            vol_buy = float(event.get('volume_compra', 0))
            vol_sell = float(event.get('volume_venda', 0))
            vol_total = float(event.get('volume_total', 0))
            delta = float(event.get('delta', 0))
            
            # Check Total
            if abs((vol_buy + vol_sell) - vol_total) > 0.0001:
                issues.append(f"Vol Total Inconsistente: (B:{vol_buy} + S:{vol_sell}) != {vol_total}")
            
            # Check Delta
            if abs((vol_buy - vol_sell) - delta) > 0.0001:
                issues.append(f"Delta Inconsistente: (B:{vol_buy} - S:{vol_sell}) != {delta}")

        # 2. Validação de OrderBook (Imbalance)
        if 'orderbook_data' in event:
            ob = event['orderbook_data']
            bid_depth = float(ob.get('bid_depth_usd', 0))
            ask_depth = float(ob.get('ask_depth_usd', 0))
            imbalance = float(ob.get('imbalance', 0))
            
            if (bid_depth + ask_depth) > 0:
                calc_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
                if abs(calc_imbalance - imbalance) > 0.01:
                    issues.append(f"OB Imbalance Inconsistente: Calc {calc_imbalance:.4f} != Stored {imbalance:.4f}")

        return issues

    def replay_pipeline(self, event: Dict[str, Any]) -> List[str]:
        """
        Tenta reprocessar os dados brutos se disponíveis no evento (ex: 'raw_trades' ou 'window_data').
        """
        divergences = []
        
        # Verifica se temos dados brutos para reprocessar fluxo/pipeline
        raw_trades = event.get('window_data') or event.get('raw_trades')
        
        if raw_trades and isinstance(raw_trades, list) and len(raw_trades) > 0:
            try:
                # 1. Reprocessa DataPipeline
                # Usa configuração padrão para teste
                pipeline = DataPipeline(
                    raw_trades=raw_trades,
                    symbol=event.get('ativo', 'BTCUSDT'),
                    time_manager=self.time_manager
                )
                
                enriched = pipeline.enrich()
                
                # Compara métricas calculadas agora vs armazenadas
                stored_delta = float(event.get('delta', 0))
                recalc_delta = float(enriched.get('delta_fechamento', 0)) # ou equivalente no pipeline
                
                # Nota: DataPipeline pode usar nomes diferentes, ajustar conforme implementação exata
                # Se DataPipeline calcula 'delta_fechamento', comparamos com isso.
                # Se usa 'VolumeBuy' - 'VolumeSell', calculamos.
                
                if abs(recalc_delta - stored_delta) > 0.0001:
                    divergences.append(
                        f"Pipeline Delta Divergente: Stored {stored_delta} vs Replay {recalc_delta}"
                    )
                
            except Exception as e:
                logging.warning(f"Erro ao reprocessar DataPipeline para evento {event.get('event_id')}: {e}")

        return divergences

    def run(self, file_path: str):
        logging.info(f"Iniciando validação de replay para: {file_path}")
        
        results = []
        
        for event in self.load_events(file_path):
            self.stats["processed"] += 1
            event_id = event.get("event_id") or event.get("timestamp") or "unknown"
            
            # 1. Validação Matemática (Sanity Check)
            math_issues = self.validate_calculations(event)
            
            # 2. Replay Lógico (se dados brutos existirem)
            replay_issues = self.replay_pipeline(event)
            
            all_issues = math_issues + replay_issues
            
            if all_issues:
                self.stats["divergences"] += 1
                for issue in all_issues:
                    self.discrepancies.append({
                        "timestamp": event.get("timestamp"),
                        "event_type": event.get("tipo_evento"),
                        "issue": issue,
                        "event_id": event_id
                    })
                    logging.warning(f"⚠️ Divergência em {event_id}: {issue}")
            else:
                self.stats["consistent"] += 1

            if self.stats["processed"] % 100 == 0:
                logging.info(f"Processados: {self.stats['processed']}...")

        self.generate_report()

    def generate_report(self):
        logging.info("="*60)
        logging.info("📊 RELATÓRIO DE DIAGNÓSTICO (REPLAY)")
        logging.info("="*60)
        logging.info(f"Total Processado:   {self.stats['processed']}")
        logging.info(f"Consistentes:       {self.stats['consistent']}")
        logging.info(f"Com Divergências:   {self.stats['divergences']}")
        logging.info(f"Erros de Leitura:   {self.stats['errors']}")
        
        if self.stats['processed'] > 0:
            pct_divergence = (self.stats['divergences'] / self.stats['processed']) * 100
            logging.info(f"Taxa de Divergência: {pct_divergence:.2f}%")
        
        if self.discrepancies:
            # Salva CSV detalhado
            df = pd.DataFrame(self.discrepancies)
            output_file = "diagnostics_report.csv"
            df.to_csv(output_file, index=False)
            logging.info(f"\n📁 Detalhes salvos em: {output_file}")
            
            # Mostra top 5 problemas
            logging.info("\n🔝 Top 5 Divergências Recentes:")
            for d in self.discrepancies[-5:]:
                logging.info(f"   - {d['timestamp']} [{d['event_type']}]: {d['issue']}")
        else:
            logging.info("\n✅ Nenhuma divergência encontrada! O sistema está matematicamente consistente.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Replay Validator - Diagnóstico Offline")
    parser.add_argument("file", help="Caminho para o arquivo de eventos (.db, .jsonl, .json)")
    args = parser.parse_args()
    
    validator = ReplayValidator()
    validator.run(args.file)