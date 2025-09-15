import csv
import os
import logging
from datetime import datetime

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

            row = {
                "timestamp": timestamp,
                "ativo": ativo,
                "tipo_evento": event.get("tipo_evento", "N/A"),
                "resultado_da_batalha": event.get("resultado_da_batalha", "N/A"),
                "descricao": event.get("descricao", ""),
                "delta": event.get("delta", 0),
                "volume_total": event.get("volume_total", 0),
                "multi_tf": event.get("multi_tf", {}),
                "historical_confidence": event.get("historical_confidence", {}),
                "ai_analysis": (ai_analysis.strip() if ai_analysis else "N/A")
            }

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
                        "ai_analysis"
                    ])
                writer.writerow([
                    row["timestamp"], row["ativo"], row["tipo_evento"],
                    row["resultado_da_batalha"], row["descricao"],
                    row["delta"], row["volume_total"],
                    row["multi_tf"], row["historical_confidence"],
                    row["ai_analysis"]
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

                f.write("## 🔎 Multi-Timeframes\n")
                f.write(f"```\n{row['multi_tf']}\n```\n\n")

                f.write("## 📉 Probabilidade Histórica\n")
                f.write(f"```\n{row['historical_confidence']}\n```\n\n")

                f.write("## 🧠 Análise da IA\n")
                f.write(row["ai_analysis"] or "N/A")
                f.write("\n")
        except Exception as e:
            logging.error(f"Erro ao gravar Markdown: {e}")