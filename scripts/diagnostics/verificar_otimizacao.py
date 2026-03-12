import json
import os
from pathlib import Path


def verificar_arquivos_json() -> None:
    """Analisa arquivos JSON gerados para verificar otimização."""
    # Locais comuns onde JSONs são salvos
    locais = [
        "dados/",
        "exports_jsonl/",
        "logs/",
        ".",
    ]

    print("=" * 60)
    print("VERIFICAÇÃO DE OTIMIZAÇÃO DE PAYLOAD")
    print("=" * 60)

    arquivos_encontrados = []

    for local in locais:
        path = Path(local)
        if path.exists():
            for arquivo in path.glob("*.json"):
                arquivos_encontrados.append(arquivo)
            for arquivo in path.glob("*.jsonl"):
                arquivos_encontrados.append(arquivo)

    if not arquivos_encontrados:
        print("❌ Nenhum arquivo JSON/JSONL encontrado")
        return

    print(f"\nEncontrados {len(arquivos_encontrados)} arquivos\n")

    for arquivo in arquivos_encontrados[:10]:  # Primeiros 10
        try:
            tamanho_kb = arquivo.stat().st_size / 1024

            with open(arquivo, "r", encoding="utf-8") as f:
                conteudo = f.read(5000)  # Primeiros 5KB

            # Verificar se é formato otimizado
            indicadores_otimizado = [
                '"price":',  # Novo formato
                '"vp":',  # Volume profile compacto
                '"flow":',  # Fluxo compacto
                '"ob":',  # Orderbook compacto
                '"tf":',  # Timeframes compacto
                '"_w":',  # Window ID compacto
            ]

            indicadores_antigo = [
                '"raw_event":',
                '"contextual_snapshot":',
                '"enriched_snapshot":',
                '"observability":',
                '"circuit_breaker":',
            ]

            otimizado_count = sum(1 for i in indicadores_otimizado if i in conteudo)
            antigo_count = sum(1 for i in indicadores_antigo if i in conteudo)

            if otimizado_count >= 3:
                status = "OK OTIMIZADO"
            elif antigo_count >= 2:
                status = "ATENCAO FORMATO ANTIGO"
            else:
                status = "INDETERMINADO"

            print(f"{status} | {tamanho_kb:6.1f} KB | {arquivo.name}")

        except Exception as e:
            print(f"❌ Erro ao ler {arquivo.name}: {e}")

    print("\n" + "=" * 60)
    print("LEGENDA:")
    print("  OK OTIMIZADO = Payload compacto (~2KB por evento)")
    print("  ATENCAO FORMATO ANTIGO = Payload completo (~18KB por evento)")
    print("  INDETERMINADO = Nao foi possivel determinar")
    print("=" * 60)


if __name__ == "__main__":
    verificar_arquivos_json()
