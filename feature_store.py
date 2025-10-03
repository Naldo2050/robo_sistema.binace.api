import os
import json
import time
import logging
from typing import Any, Dict, Optional, Iterable, Tuple

import pandas as pd


class FeatureStore:
    """
    Armazena features por janela em CSV (features/features.csv por padrão).
    - Evita FutureWarning do pandas ao concatenar com DF vazio/todos-NA.
    - Faz flatten/serialização segura para campos aninhados (dict/list).
    - Garante união de colunas entre o arquivo existente e a nova linha.
    - Escrita atômica + lock simples (Windows-friendly) com retry/backoff.
    - Dedup por window_id (mantém a última ocorrência).
    - Rotação opcional por tamanho/linhas.
    """

    def __init__(
        self,
        base_dir: str = "features",
        filename: str = "features.csv",
        rotate_max_bytes: Optional[int] = 20_000_000,  # ~20MB; None para desativar
        rotate_max_rows: Optional[int] = 200_000,      # None para desativar
        lock_filename: str = ".features.lock",
        backup_suffix: str = ".bak",
    ):
        self.base_dir = base_dir
        self.filename = filename
        self.file_path = os.path.join(base_dir, filename)
        self.lock_path = os.path.join(base_dir, lock_filename)
        self.backup_suffix = backup_suffix
        self.rotate_max_bytes = rotate_max_bytes
        self.rotate_max_rows = rotate_max_rows
        os.makedirs(self.base_dir, exist_ok=True)
        logging.info(f"✅ FeatureStore inicializado. Dados salvos em: {self.base_dir}")

    # -------------------- utils de locking/atomicidade --------------------

    def _acquire_lock(self, timeout: float = 5.0, interval: float = 0.05) -> bool:
        """Lock por arquivo usando O_CREAT|O_EXCL via open exclusivo (cross-platform simples)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                # modo exclusivo: falha se já existir
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                time.sleep(interval)
            except Exception as e:
                logging.warning(f"FeatureStore: erro inesperado ao adquirir lock: {e}")
                time.sleep(interval)
        return False

    def _release_lock(self) -> None:
        try:
            if os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        except Exception as e:
            logging.warning(f"FeatureStore: falha ao liberar lock: {e}")

    def _atomic_write_csv(self, df: pd.DataFrame, target: str) -> None:
        """Escreve CSV em arquivo temporário e faz os.replace (atômico)."""
        tmp_path = target + ".tmp"
        # newline='' para não duplicar linhas no Windows; utf-8-sig evita problemas em Excel
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, target)

    # -------------------- flatten/serialização/IO --------------------

    @staticmethod
    def _flatten(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (d or {}).items():
            key = f"{parent}{sep}{k}" if parent else k
            if isinstance(v, dict):
                out.update(FeatureStore._flatten(v, key, sep))
            else:
                out[key] = v
        return out

    @staticmethod
    def _to_serializable(v: Any) -> Any:
        # Serializa estruturas complexas; normaliza NaN/Inf
        try:
            if isinstance(v, (dict, list, tuple)):
                return json.dumps(v, ensure_ascii=False)
            if isinstance(v, float):
                if v != v or v in (float("inf"), float("-inf")):
                    return None
            return v
        except Exception:
            return str(v)

    def _read_existing(self) -> Optional[pd.DataFrame]:
        if not os.path.exists(self.file_path):
            return None
        try:
            # dtype='string' para colunas texto; deixa numéricos serem inferidos
            df = pd.read_csv(self.file_path)
            if df is None or df.empty:
                return None
            return df
        except Exception as e:
            logging.warning(f"FeatureStore: falha ao ler arquivo existente ({self.file_path}): {e}")
            return None

    # -------------------- rotação --------------------

    def _should_rotate(self, df_new_rows: int = 1) -> Tuple[bool, str]:
        """
        Decide por rotação baseada em bytes ou número de linhas.
        Retorna (bool_should_rotate, new_path_for_rotation).
        """
        if not os.path.exists(self.file_path):
            return (False, "")

        try:
            by_size = False
            if self.rotate_max_bytes is not None:
                size = os.path.getsize(self.file_path)
                by_size = size >= self.rotate_max_bytes

            by_rows = False
            if self.rotate_max_rows is not None:
                # Leitura só do header + tail rápido seria ideal; como é CSV simples,
                # optamos por uma leitura leve de shape quando necessário.
                # Custo é amortizado se rotate estiver desativado.
                df = self._read_existing()
                if df is not None and not df.empty:
                    by_rows = (len(df) + df_new_rows) > self.rotate_max_rows

            if by_size or by_rows:
                # gera nome com timestamp para arquivo antigo
                ts = time.strftime("%Y%m%d-%H%M%S")
                new_path = self.file_path.replace(".csv", f".{ts}.csv")
                return (True, new_path)
        except Exception as e:
            logging.warning(f"FeatureStore: erro ao avaliar rotação: {e}")

        return (False, "")

    def _rotate_now(self, rotated_path: str) -> None:
        try:
            # faz um backup simples e limpa arquivo atual
            os.replace(self.file_path, rotated_path)
            logging.info(f"🌀 FeatureStore: arquivo rotacionado para {rotated_path}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logging.warning(f"FeatureStore: falha na rotação ({e}). Prosseguindo sem rotacionar.")

    # -------------------- API pública --------------------

    def save_features(self, window_id: str, features: Dict[str, Any]) -> None:
        """
        Salva uma linha de features referente à window_id.
        - Não concatena DataFrames vazios/todos-NA.
        - Une colunas com o arquivo existente para evitar dtype/colunas inconsistentes.
        - Escrita atômica + lock + retry/backoff.
        - Dedup por window_id.
        """
        if not isinstance(features, dict):
            logging.warning("FeatureStore: features não é dict; ignorado.")
            return

        # Flatten + serialização segura
        flat = self._flatten(features)
        flat = {k: self._to_serializable(v) for k, v in flat.items()}
        flat["window_id"] = window_id

        row_df = pd.DataFrame([flat])

        # Se a linha nova é toda NA (após serialização), não escreve
        if row_df.drop(columns=["window_id"], errors="ignore").dropna(how="all").empty:
            logging.debug("FeatureStore: linha de features toda NA; não será gravada.")
            return

        # Retry curto em caso de sharing violation no Windows
        for attempt in range(6):  # ~ 0.8s total (0.05 * (2**5))
            got_lock = self._acquire_lock(timeout=0.5)
            if not got_lock:
                # não conseguiu lock; backoff e tenta de novo
                time.sleep(0.05 * (2 ** attempt))
                continue

            try:
                existing_df = self._read_existing()

                if existing_df is None or existing_df.empty:
                    # arquivo não existe ou está vazio → escreve do zero (atômico)
                    # Garantimos que não existem colunas todos-NA:
                    cleaned = row_df.dropna(axis=1, how="all")
                    self._atomic_write_csv(cleaned, self.file_path)
                    return

                # Unifica colunas entre existente e nova linha, preservando ordem
                union_cols = list(dict.fromkeys(list(existing_df.columns) + list(row_df.columns)))
                existing_df = existing_df.reindex(columns=union_cols)
                row_df = row_df.reindex(columns=union_cols)

                # Concat final
                final_df = pd.concat([existing_df, row_df], ignore_index=True, sort=False, copy=False)
                # Dedup por window_id (mantém a última ocorrência)
                if "window_id" in final_df.columns:
                    final_df = final_df.drop_duplicates(subset=["window_id"], keep="last", ignore_index=True)

                # Opcional: dropar colunas 100% NA (mantém as relevantes)
                # (ativa por padrão para blindar contra FutureWarning de all-NA)
                final_df = final_df.dropna(axis=1, how="all")

                # Rotação se necessário
                rotate, rotated_path = self._should_rotate(df_new_rows=0)
                if rotate:
                    self._rotate_now(rotated_path)

                # Escrita atômica
                self._atomic_write_csv(final_df, self.file_path)
                return

            except Exception as e:
                logging.error(f"FeatureStore: erro ao salvar features (tentativa {attempt+1}): {e}")
                # pequeno backoff antes de tentar novamente
                time.sleep(0.05 * (2 ** attempt))

            finally:
                self._release_lock()

        # Se chegou aqui, todas as tentativas falharam
        logging.critical("FeatureStore: falha repetida ao salvar features após múltiplas tentativas.")
