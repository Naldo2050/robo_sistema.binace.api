import csv
import os
import logging
from datetime import datetime
from typing import Any  # para anotações de tipo (corrige aviso Pylance)
import json  # para serializar campos complexos

# 🔹 IMPORTA TIME MANAGER
from time_manager import TimeManager

class ReportGenerator:
    def __init__(self, output_dir: str = "./reports", mode: str = "csv"):
        """
        Gera relatórios consolidados de sinais + análise da IA
        - output_dir: pasta principal onde salvar relatórios
        - mode: 'csv' para planilhas de análise, 'md' para relatórios legíveis
        """
        self.output_dir = output_dir
        self.mode = mode.lower()
        os.makedirs(output_dir, exist_ok=True)

        # 🔹 Inicializa TimeManager
        self.time_manager = TimeManager()

    def _get_daily_path(self, symbol: str, date: str):
        """Retorna caminho do arquivo diário para o ativo"""
        if self.mode == "csv":
            return os.path.join(self.output_dir, f"{symbol}_{date}.csv")
        elif self.mode == "md":
            daily_dir = os.path.join(self.output_dir, date)
            os.makedirs(daily_dir, exist_ok=True)
            return daily_dir
        else:
            return None

    def save_report(self, event: dict, ai_analysis: str = ""):
        """Salva evento + análise IA em CSV diário ou Markdown"""
        try:
            # 🔹 USA TIME MANAGER PARA TIMESTAMP
            timestamp = event.get("timestamp", self.time_manager.now_iso())
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
            ativo = event.get("ativo", "N/A")

            # Junta campos básicos e campos avançados (contexto, ambiente, order book, fluxo, participantes)
            row = {
                "timestamp": timestamp,
                "ativo": ativo,
                "tipo_evento": event.get("tipo_evento", "N/A"),
                "resultado_da_batalha": event.get("resultado_da_batalha", "N/A"),
                "descricao": event.get("descricao", ""),
                "delta": event.get("delta", 0),
                "volume_total": event.get("volume_total", 0),
                # Multi-timeframe (fallback: procura em contextual se não estiver diretamente no evento)
                "multi_tf": event.get("multi_tf") or (event.get("contextual", {}) if isinstance(event.get("contextual"), dict) else {}).get("multi_tf", {}),
                "historical_confidence": event.get("historical_confidence", {}),
                # Novos campos: contexto de mercado, ambiente, profundidade, spread, fluxo e participantes
                "market_context": event.get("market_context") or (event.get("contextual", {}) if isinstance(event.get("contextual"), dict) else {}).get("market_context", {}),
                "market_environment": event.get("market_environment") or (event.get("contextual", {}) if isinstance(event.get("contextual"), dict) else {}).get("market_environment", {}),
                "order_book_depth": event.get("order_book_depth", {}),
                "spread_analysis": event.get("spread_analysis", {}),
                # Fluxo contínuo contém order_flow e participant_analysis
                "order_flow": None,
                "participant_analysis": None,
                "derivatives": event.get("derivatives", {}),
                "ai_analysis": (ai_analysis.strip() if ai_analysis else "N/A")
            }

            # Extrai order_flow e participant_analysis de fluxo_continuo (ou flow_metrics)
            fluxo = event.get("fluxo_continuo") or event.get("flow_metrics") or {}  # type: ignore
            try:
                if isinstance(fluxo, dict):
                    if "order_flow" in fluxo:
                        row["order_flow"] = fluxo.get("order_flow")
                    else:
                        row["order_flow"] = None
                    if "participant_analysis" in fluxo:
                        row["participant_analysis"] = fluxo.get("participant_analysis")
                    else:
                        row["participant_analysis"] = None
            except Exception:
                row["order_flow"] = None
                row["participant_analysis"] = None

            if self.mode == "csv":
                self._save_csv(row, date_str)
            elif self.mode == "md":
                self._save_md(row, date_str)
            else:
                logging.warning(f"Formato não suportado: {self.mode}")

        except Exception as e:
            logging.error(f"Erro ao salvar relatório: {e}", exc_info=True)

    # ====================================================
    # CSV Export (um arquivo por dia e por ativo)
    # ====================================================
    def _save_csv(self, row: dict, date_str: str):
        try:
            csv_path = self._get_daily_path(row["ativo"], date_str)
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp",
                        "ativo",
                        "tipo_evento",
                        "resultado_da_batalha",
                        "descricao",
                        "delta",
                        "volume_total",
                        "multi_tf",
                        "historical_confidence",
                        "market_context",
                        "market_environment",
                        "order_book_depth",
                        "spread_analysis",
                        "order_flow",
                        "participant_analysis",
                        "derivatives",
                        "ai_analysis"
                    ])
                # Serializa campos complexos em JSON para CSV
                def serialize(val):
                    try:
                        if isinstance(val, (dict, list)):
                            return json.dumps(val, ensure_ascii=False)
                        return val
                    except Exception:
                        return str(val)
                writer.writerow([
                    row["timestamp"], row["ativo"], row["tipo_evento"],
                    row["resultado_da_batalha"], row["descricao"],
                    row["delta"], row["volume_total"],
                    serialize(row["multi_tf"]), serialize(row["historical_confidence"]),
                    serialize(row["market_context"]), serialize(row["market_environment"]),
                    serialize(row["order_book_depth"]), serialize(row["spread_analysis"]),
                    serialize(row["order_flow"]), serialize(row["participant_analysis"]),
                    serialize(row["derivatives"]),
                    serialize(row["ai_analysis"])
                ])
        except Exception as e:
            logging.error(f"Erro ao gravar CSV: {e}")

    # ====================================================
    # Markdown Export (um diretório por dia, arquivos por evento)
    # ====================================================
    def _save_md(self, row: dict, date_str: str):
        try:
            md_dir = self._get_daily_path(row["ativo"], date_str)
            os.makedirs(md_dir, exist_ok=True)
            # 🔹 USA TIMESTAMP SINCRONIZADO NO NOME DO ARQUIVO
            filename = f"{row['timestamp'].replace(':','-').replace('.','_')}_{row['ativo']}.md"
            md_path = os.path.join(md_dir, filename)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# 📊 Relatório de Sinal - {row['ativo']} ({row['tipo_evento']})\n\n")
                f.write(f"- **Timestamp:** {row['timestamp']}\n")
                f.write(f"- **Resultado:** {row['resultado_da_batalha']}\n")
                f.write(f"- **Descrição:** {row['descricao']}\n")
                f.write(f"- **Delta:** {row['delta']}\n")
                f.write(f"- **Volume Total:** {row['volume_total']}\n\n")

                def write_section(title: str, data: Any):
                    f.write(f"## {title}\n")
                    # Usa json para formatar dicts de forma legível
                    try:
                        if isinstance(data, (dict, list)):
                            formatted = json.dumps(data, indent=2, ensure_ascii=False)
                            f.write(f"```\n{formatted}\n```\n\n")
                        else:
                            f.write(f"{data}\n\n")
                    except Exception:
                        f.write(f"{data}\n\n")

                write_section("🔎 Multi-Timeframes", row.get("multi_tf", {}))
                write_section("📉 Probabilidade Histórica", row.get("historical_confidence", {}))
                write_section("🌍 Contexto de Mercado", row.get("market_context", {}))
                write_section("🌡 Ambiente de Mercado", row.get("market_environment", {}))
                write_section("📑 Profundidade do Livro", row.get("order_book_depth", {}))
                write_section("📏 Análise de Spread", row.get("spread_analysis", {}))
                write_section("🚰 Fluxo de Ordens", row.get("order_flow", {}))
                write_section("👥 Participantes", row.get("participant_analysis", {}))
                write_section("🏦 Derivativos", row.get("derivatives", {}))
                write_section("🧠 Análise da IA", row.get("ai_analysis", "N/A"))
        except Exception as e:
            logging.error(f"Erro ao gravar Markdown: {e}")