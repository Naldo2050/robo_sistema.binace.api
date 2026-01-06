# test_feature_store.py
"""
Teste completo e aprimorado para o FeatureStore.
Inclui validação, dados realistas e verificação de integridade.
"""

import os
import sys
import time
import tempfile
import argparse
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import glob

import numpy as np
import pandas as pd
from feature_store import FeatureStore

# Suprimir warnings do pandas
warnings.filterwarnings('ignore', message="errors='ignore' is deprecated")

# Configuração de logging (sem emojis para compatibilidade Windows)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class FeatureStoreTester:
    """Classe para testes abrangentes do FeatureStore"""
    
    def __init__(self, base_dir: str = None, cleanup: bool = False):
        """
        Inicializa o tester.
        
        Args:
            base_dir: Diretório base para os testes (None = cria temp dir)
            cleanup: Se True, limpa diretório após teste
        """
        self.cleanup = cleanup
        
        if base_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="feature_store_test_")
            self.base_dir = self.temp_dir
            logger.info(f"Diretório temporário criado: {self.temp_dir}")
        else:
            self.base_dir = base_dir
            self.temp_dir = None
            Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        
        self.fs = None
        
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit com limpeza"""
        self.cleanup_resources()
        
    def cleanup_resources(self):
        """Limpa recursos após teste"""
        if self.fs:
            try:
                self.fs.close()
            except Exception as e:
                logger.warning(f"Erro ao fechar FeatureStore: {e}")
        
        if self.cleanup and self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Diretório temporário removido: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Erro ao remover diretório: {e}")
    
    def generate_market_data(self, num_samples: int, base_price: float = 100.0):
        """
        Gera dados de mercado sintéticos mais realistas.
        
        Args:
            num_samples: Número de amostras
            base_price: Preço base inicial
            
        Returns:
            Lista de dicionários com features
        """
        np.random.seed(42)  # Para reprodução
        
        # Gera séries temporais com características realistas
        timestamps = [
            datetime.now() - timedelta(minutes=i) 
            for i in reversed(range(num_samples))
        ]
        
        # Preços com tendência, volatilidade e ruído
        returns = np.random.normal(0.0005, 0.02, num_samples)  # Retornos diários ~2%
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Volumes com padrão de mercado
        base_volume = 1000
        volume_pattern = np.sin(np.linspace(0, 4*np.pi, num_samples)) * 0.5 + 1
        volumes = base_volume * volume_pattern * np.exp(np.random.normal(0, 0.3, num_samples))
        volumes = np.abs(volumes)  # Garante positivo
        
        # Indicadores técnicos simples
        price_series = pd.Series(prices)
        sma_10 = price_series.rolling(window=min(10, num_samples), min_periods=1).mean()
        
        data_points = []
        for i in range(num_samples):
            features = {
                "timestamp": timestamps[i].isoformat(),
                "window_id": f"test_{timestamps[i].strftime('%Y%m%d_%H%M%S')}_{i:04d}",
                "price_close": float(prices[i]),
                "price_open": float(prices[i] * (1 + np.random.normal(0, 0.01))),
                "price_high": float(prices[i] * (1 + abs(np.random.normal(0.005, 0.015)))),
                "price_low": float(prices[i] * (1 - abs(np.random.normal(0.005, 0.015)))),
                "volume": float(volumes[i]),
                "sma_10": float(sma_10[i]) if i < len(sma_10) else float(prices[i]),
                "returns_1d": float(returns[i]) if i > 0 else 0.0,
                "volatility": float(np.std(returns[max(0, i-20):i+1])) if i > 0 else 0.02,
            }
            data_points.append(features)
        
        return data_points
    
    def validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Valida estrutura e tipos das features.
        
        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["price_close", "volume"]
        
        # Verifica campos obrigatórios
        for field in required_fields:
            if field not in features:
                logger.error(f"Campo obrigatório faltando: {field}")
                return False
        
        # Valida tipos numéricos
        numeric_fields = ["price_close", "volume", "sma_10", "returns_1d", "volatility"]
        for field in numeric_fields:
            if field in features:
                value = features[field]
                if not isinstance(value, (int, float, np.number)):
                    logger.warning(f"Campo {field} não é numérico: {type(value)}")
                    return False
                
                # Validações específicas
                if field == "price_close" and value <= 0:
                    logger.warning(f"Preço inválido: {value}")
                    return False
                if field == "volume" and value < 0:
                    logger.warning(f"Volume negativo: {value}")
                    return False
        
        return True
    
    def run_performance_test(self, num_samples: int = 1000):
        """
        Teste de performance do FeatureStore.
        
        Returns:
            dict com métricas de performance
        """
        logger.info(f"Iniciando teste de performance com {num_samples} amostras")
        
        # Inicializa FeatureStore
        self.fs = FeatureStore(base_dir=self.base_dir)
        
        # Gera dados
        data_points = self.generate_market_data(num_samples)
        
        # Teste de escrita
        write_times = []
        for i, features in enumerate(data_points):
            start_time = time.perf_counter()
            
            self.fs.save_features(
                window_id=features["window_id"],
                features={k: v for k, v in features.items() if k not in ["window_id", "timestamp"]}
            )
            
            write_times.append(time.perf_counter() - start_time)
            
            # Log de progresso
            if (i + 1) % max(1, num_samples // 10) == 0:
                logger.info(f"  Progresso: {i + 1}/{num_samples} ({((i+1)/num_samples)*100:.1f}%)")
        
        # Fecha e flush
        close_start = time.perf_counter()
        self.fs.close()
        close_time = time.perf_counter() - close_start
        
        # Estatísticas
        stats = {
            "total_samples": num_samples,
            "total_write_time": sum(write_times),
            "avg_write_time": np.mean(write_times) if write_times else 0,
            "p95_write_time": np.percentile(write_times, 95) if write_times else 0,
            "close_time": close_time,
            "samples_per_second": num_samples / sum(write_times) if sum(write_times) > 0 else 0,
            "file_size_bytes": self._get_total_file_size(),
        }
        
        logger.info("Teste de performance concluído:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return stats
    
    def _get_total_file_size(self) -> int:
        """Obtém tamanho total de todos os arquivos Parquet gerados"""
        total_size = 0
        parquet_files = self._find_parquet_files()
        
        for file_path in parquet_files:
            try:
                total_size += os.path.getsize(file_path)
            except OSError:
                pass
        
        return total_size
    
    def _find_parquet_files(self) -> List[str]:
        """Encontra todos os arquivos Parquet no diretório base"""
        parquet_files = []
        
        # Procura recursivamente por arquivos .parquet
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        
        # Também procura usando glob pattern
        if not parquet_files:
            pattern = os.path.join(self.base_dir, "**", "*.parquet")
            parquet_files = glob.glob(pattern, recursive=True)
        
        return parquet_files
    
    def verify_data_integrity(self):
        """
        Verifica integridade dos dados salvos.
        
        Returns:
            dict com resultados da verificação
        """
        parquet_files = self._find_parquet_files()
        
        if not parquet_files:
            logger.error("Nenhum arquivo Parquet encontrado")
            logger.error(f"Diretório: {self.base_dir}")
            logger.error(f"Conteúdo do diretório: {os.listdir(self.base_dir)}")
            return {"success": False, "error": "No Parquet files found"}
        
        try:
            # Lê todos os arquivos Parquet
            dfs = []
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                    logger.info(f"Arquivo lido: {file_path} ({len(df)} registros)")
                except Exception as e:
                    logger.error(f"Erro ao ler {file_path}: {e}")
            
            if not dfs:
                return {"success": False, "error": "Could not read any Parquet files"}
            
            # Combina todos os DataFrames
            combined_df = pd.concat(dfs, ignore_index=True)
            
            if combined_df.empty:
                return {"success": False, "error": "DataFrame vazio após combinação"}
            
            # Verificações
            checks = {
                "total_records": len(combined_df),
                "has_required_columns": all(col in combined_df.columns for col in ["price_close", "volume"]),
                "no_null_price": combined_df["price_close"].notnull().all() if "price_close" in combined_df.columns else False,
                "positive_price": (combined_df["price_close"] > 0).all() if "price_close" in combined_df.columns else False,
                "non_negative_volume": (combined_df["volume"] >= 0).all() if "volume" in combined_df.columns else False,
                "data_types_ok": True,
            }
            
            # Verifica tipos de dados
            if not combined_df.empty:
                numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
                checks["data_types_ok"] = len(numeric_cols) > 0
            
            # Estatísticas
            stats = {}
            if "price_close" in combined_df.columns:
                stats["price_stats"] = {
                    "mean": combined_df["price_close"].mean(),
                    "std": combined_df["price_close"].std(),
                    "min": combined_df["price_close"].min(),
                    "max": combined_df["price_close"].max(),
                }
            
            if "volume" in combined_df.columns:
                stats["volume_stats"] = {
                    "mean": combined_df["volume"].mean(),
                    "total": combined_df["volume"].sum(),
                }
            
            logger.info("Verificação de integridade:")
            logger.info(f"  Arquivos encontrados: {len(parquet_files)}")
            logger.info(f"  Total de registros: {checks['total_records']}")
            logger.info(f"  Colunas disponíveis: {list(combined_df.columns)}")
            
            for check, result in checks.items():
                status = "OK" if result else "FAIL"
                logger.info(f"  {status} {check}: {result}")
            
            if stats:
                logger.info("\nEstatísticas:")
                for category, values in stats.items():
                    logger.info(f"  {category}:")
                    for key, value in values.items():
                        logger.info(f"    {key}: {value:.2f}")
            
            return {
                "success": all(checks.values()),
                "checks": checks,
                "stats": stats,
                "dataframe_shape": combined_df.shape,
                "columns": combined_df.columns.tolist(),
                "files_found": len(parquet_files),
            }
            
        except Exception as e:
            logger.error(f"Erro na verificação: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def run_edge_cases_test(self):
        """Testa casos extremos e especiais"""
        test_cases = [
            {
                "name": "dados_minimos",
                "window_id": "edge_minimal",
                "features": {"price_close": 1.0, "volume": 0}
            },
            {
                "name": "valores_extremos",
                "window_id": "edge_extreme",
                "features": {"price_close": 1e6, "volume": 1e9}
            },
            {
                "name": "decimais_longo",
                "window_id": "edge_decimals",
                "features": {"price_close": 123.456789, "volume": 987654.321}
            },
            {
                "name": "caracteres_especiais_id",
                "window_id": "test_id_with_special-chars_123",
                "features": {"price_close": 100.0, "volume": 1000}
            },
        ]
        
        results = []
        self.fs = FeatureStore(base_dir=self.base_dir)
        
        for test_case in test_cases:
            try:
                self.fs.save_features(
                    window_id=test_case["window_id"],
                    features=test_case["features"]
                )
                results.append({"test": test_case["name"], "status": "PASS"})
                logger.info(f"OK - Caso extremo '{test_case['name']}' passou")
            except Exception as e:
                results.append({
                    "test": test_case["name"], 
                    "status": "FAIL", 
                    "error": str(e)
                })
                logger.error(f"FAIL - Caso extremo '{test_case['name']}' falhou: {e}")
        
        self.fs.close()
        return results


def main():
    """Função principal com argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description="Teste completo do FeatureStore",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python test_feature_store.py --samples 500
  python test_feature_store.py --samples 1000 --performance
  python test_feature_store.py --output-dir ./test_data --keep-files
  python test_feature_store.py --edge-cases-only
        """
    )
    
    parser.add_argument(
        "--samples", 
        type=int, 
        default=200,
        help="Número de amostras para gerar (padrão: 200)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Diretório de saída (None = diretório temporário)"
    )
    
    parser.add_argument(
        "--keep-files", 
        action="store_true",
        help="Manter arquivos após teste (ignorado se --output-dir for especificado)"
    )
    
    parser.add_argument(
        "--performance", 
        action="store_true",
        help="Executar teste de performance detalhado"
    )
    
    parser.add_argument(
        "--edge-cases", 
        action="store_true",
        help="Testar casos extremos"
    )
    
    parser.add_argument(
        "--verify", 
        action="store_true",
        default=True,
        help="Verificar integridade dos dados após teste"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de logging"
    )
    
    parser.add_argument(
        "--simple-data",
        action="store_true",
        help="Usar dados simples (apenas price_close e volume) para teste rápido"
    )
    
    args = parser.parse_args()
    
    # Configura nível de logging
    logger.setLevel(getattr(logging, args.log_level))
    
    # Determina se deve limpar
    cleanup = not args.keep_files and args.output_dir is None
    
    # Executa testes
    tester = FeatureStoreTester(base_dir=args.output_dir, cleanup=cleanup)
    
    try:
        results = {}
        
        # Teste de performance ou básico
        if args.performance:
            logger.info("=" * 60)
            logger.info("EXECUTANDO TESTE DE PERFORMANCE")
            logger.info("=" * 60)
            results["performance"] = tester.run_performance_test(args.samples)
        else:
            logger.info("=" * 60)
            logger.info(f"EXECUTANDO TESTE COM {args.samples} AMOSTRAS")
            logger.info("=" * 60)
            
            # Teste básico
            start_time = time.time()
            tester.fs = FeatureStore(base_dir=tester.base_dir)
            
            if args.simple_data:
                # Dados simples para teste rápido
                saved_count = 0
                for i in range(args.samples):
                    try:
                        features = {
                            "price_close": 100 + i + np.random.randn() * 2,
                            "volume": max(0, i * 10 + np.random.randn() * 50),
                        }
                        
                        tester.fs.save_features(
                            window_id=f"test_{i:04d}",
                            features=features
                        )
                        saved_count += 1
                        
                        if (i + 1) % max(1, args.samples // 10) == 0:
                            logger.info(f"Progresso: {i + 1}/{args.samples} janelas")
                            
                    except Exception as e:
                        logger.error(f"Erro ao salvar janela {i}: {e}")
            else:
                # Dados completos
                data_points = tester.generate_market_data(args.samples)
                saved_count = 0
                
                for i, features in enumerate(data_points):
                    try:
                        if tester.validate_features(features):
                            # Remove campos que não são features
                            features_to_save = {k: v for k, v in features.items() 
                                             if k not in ["window_id", "timestamp"]}
                            
                            tester.fs.save_features(
                                window_id=features["window_id"],
                                features=features_to_save
                            )
                            saved_count += 1
                        
                        if (i + 1) % max(1, args.samples // 10) == 0:
                            logger.info(f"Progresso: {i + 1}/{args.samples} janelas")
                            
                    except Exception as e:
                        logger.error(f"Erro ao salvar janela {i}: {e}")
            
            tester.fs.close()
            elapsed = time.time() - start_time
            
            results["basic_test"] = {
                "samples_generated": args.samples,
                "samples_saved": saved_count,
                "elapsed_time": elapsed,
                "samples_per_second": saved_count / elapsed if elapsed > 0 else 0,
            }
        
        # Testes de casos extremos
        if args.edge_cases:
            logger.info("\n" + "=" * 60)
            logger.info("TESTANDO CASOS EXTREMOS")
            logger.info("=" * 60)
            results["edge_cases"] = tester.run_edge_cases_test()
        
        # Verificação de integridade
        if args.verify:
            logger.info("\n" + "=" * 60)
            logger.info("VERIFICANDO INTEGRIDADE DOS DADOS")
            logger.info("=" * 60)
            results["integrity"] = tester.verify_data_integrity()
        
        # Relatório final
        logger.info("\n" + "=" * 60)
        logger.info("RELATÓRIO FINAL DE TESTES")
        logger.info("=" * 60)
        
        all_passed = True
        failure_details = []
        
        for test_name, result in results.items():
            if isinstance(result, dict):
                if "success" in result and not result["success"]:
                    all_passed = False
                    failure_details.append(f"{test_name}: {result.get('error', 'Unknown error')}")
                elif test_name == "edge_cases":
                    failures = [r for r in result if r["status"] == "FAIL"]
                    if failures:
                        all_passed = False
                        for fail in failures:
                            failure_details.append(f"{test_name}.{fail['test']}: {fail.get('error', 'Unknown error')}")
        
        if all_passed:
            logger.info("SUCESSO - TODOS OS TESTES PASSARAM")
            print("\n" + "=" * 60)
            print("  TODOS OS TESTES PASSARAM COM SUCESSO!")
            print("=" * 60)
            return 0
        else:
            logger.error("FALHA - ALGUNS TESTES FALHARAM")
            print("\n" + "=" * 60)
            print("  ALGUNS TESTES FALHARAM!")
            print("=" * 60)
            
            # Imprime detalhes dos erros
            for detail in failure_details:
                logger.error(f"Falha: {detail}")
            
            return 1
            
    finally:
        tester.cleanup_resources()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nTeste interrompido pelo usuário")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Erro não tratado: {e}", exc_info=True)
        sys.exit(1)