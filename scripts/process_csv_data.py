#!/usr/bin/env python3
"""
Script para processar dados CSV de eventos de mercado.
"""

import csv
from typing import List, Dict


def read_csv_data(file_path: str) -> List[Dict[str, str]]:
    """
    Lê dados de um arquivo CSV e retorna uma lista de dicionários.
    
    Args:
        file_path (str): Caminho para o arquivo CSV.
    
    Returns:
        List[Dict[str, str]]: Lista de dicionários representando os dados.
    """
    data = []
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data


def display_data(data: List[Dict[str, str]]) -> None:
    """
    Exibe os dados formatados.
    
    Args:
        data (List[Dict[str, str]]): Lista de dicionários com os dados.
    """
    if not data:
        print("Nenhum dado para exibir.")
        return
    
    print("\nDados do CSV:")
    print("-" * 80)
    for idx, row in enumerate(data, start=1):
        print(f"Registro {idx}:")
        for key, value in row.items():
            print(f"  {key}: {value}")
        print("-" * 80)


def filter_data_by_event(data: List[Dict[str, str]], event_type: str) -> List[Dict[str, str]]:
    """
    Filtra dados por tipo de evento.
    
    Args:
        data (List[Dict[str, str]]): Lista de dicionários com os dados.
        event_type (str): Tipo de evento para filtrar.
    
    Returns:
        List[Dict[str, str]]: Lista filtrada de dados.
    """
    return [row for row in data if row.get('event_type') == event_type]


def calculate_statistics(data: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Calcula estatísticas básicas dos dados.
    
    Args:
        data (List[Dict[str, str]]): Lista de dicionários com os dados.
    
    Returns:
        Dict[str, float]: Dicionário com estatísticas calculadas.
    """
    if not data:
        return {}
    
    prices = []
    volumes = []
    
    for row in data:
        try:
            price = float(row.get('price', 0))
            volume = float(row.get('volume', 0))
            prices.append(price)
            volumes.append(volume)
        except ValueError:
            continue
    
    stats = {
        'preco_medio': sum(prices) / len(prices) if prices else 0,
        'volume_total': sum(volumes),
        'preco_maximo': max(prices) if prices else 0,
        'preco_minimo': min(prices) if prices else 0,
    }
    
    return stats


def main() -> None:
    """
    Função principal para executar o script.
    """
    # Caminho para o arquivo CSV (substitua pelo caminho real)
    csv_file_path = 'dados_mercado.csv'
    
    # Ler dados do CSV
    data = read_csv_data(csv_file_path)
    
    # Exibir dados
    display_data(data)
    
    # Filtrar dados por tipo de evento
    event_type = 'Absorcao de Venda Detectada'
    filtered_data = filter_data_by_event(data, event_type)
    print(f"\nDados filtrados por evento: {event_type}")
    display_data(filtered_data)
    
    # Calcular e exibir estatísticas
    stats = calculate_statistics(data)
    print("\nEstatísticas:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")


if __name__ == '__main__':
    main()