# Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from pathlib import Path
import logging

# 🔹 IMPORTAÇÃO DO EVENT STORE (SQLITE)
try:
    from database.event_store import EventStore
    HAS_DB = True
except ImportError:
    HAS_DB = False
    logging.warning("⚠️ Módulo database.event_store não encontrado.")

# Configuração do logging (opcional, para debug)
logging.basicConfig(level=logging.WARNING)

# Configuração da página Streamlit
st.set_page_config(
    page_title="📊 Dashboard Trader AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e introdução
st.title("📊 Dashboard Trader AI")
st.markdown("""
    ### Sistema de Trading Algorítmico Inteligente
    Visualização em tempo real de sinais, zonas de liquidez, volume profile e eventos históricos.
    
    🔍 **Funcionalidades:**
    - Sinais de Absorção/Exaustão
    - Zonas de Suporte/Resistência
    - Mapa de Calor de Liquidez
    - Histórico de eventos por hora
    - Filtros por tipo e ativo
""")

# Caminhos dos dados (ajuste conforme seu diretório)
DATA_DIR = Path("./dados")
EVENTS_FILE = DATA_DIR / "eventos_fluxo.jsonl"

def convert_to_sao_paulo_tz(timestamp_str):
    """
    Converte timestamp string para timezone de São Paulo de forma segura.
    """
    try:
        # Primeiro, converte para datetime
        dt = pd.to_datetime(timestamp_str, errors='coerce')
        
        if pd.isna(dt):
            return pd.NaT
        
        # Se já tem timezone, converte diretamente
        if dt.tz is not None:
            return dt.tz_convert('America/Sao_Paulo')
        else:
            # Se não tem timezone, assume UTC e converte
            return dt.tz_localize('UTC').tz_convert('America/Sao_Paulo')
            
    except Exception as e:
        # st.warning(f"⚠️ Erro ao converter timestamp: {e}") # Silenciado para evitar poluição visual
        return pd.NaT

# --- FUNÇÃO LEGADA (BACKUP) ---
def load_events_from_file_legacy():
    """(LEGADO) Carrega todos os eventos do arquivo eventos_fluxo.jsonl."""
    if not EVENTS_FILE.exists():
        st.warning(f"⚠️ Arquivo de eventos não encontrado: `{EVENTS_FILE}`")
        return pd.DataFrame()
    
    events = []
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                # 🔹 Conversão segura do timestamp
                if "timestamp" in event and isinstance(event["timestamp"], str):
                    event["timestamp"] = convert_to_sao_paulo_tz(event["timestamp"])
                else:
                    event["timestamp"] = pd.NaT
                events.append(event)
            except json.JSONDecodeError:
                st.warning(f"⚠️ Linha inválida no JSONL (linha {line_num}): {line[:50]}...")
            except Exception as e:
                st.warning(f"⚠️ Erro ao processar evento na linha {line_num}: {e}")
    
    df = pd.DataFrame(events)
    if not df.empty:
        # Remove eventos com timestamp inválido
        df = df.dropna(subset=["timestamp"])
        # Ordena por timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df

# --- NOVA FUNÇÃO PRINCIPAL (SQLITE) ---
@st.cache_data(ttl=30)  # Cache reduzido para 30s para maior agilidade
def load_events():
    """Carrega os eventos mais recentes diretamente do SQLite."""
    if not HAS_DB:
        st.error("❌ Módulo de Banco de Dados não disponível. Tentando fallback para arquivo...")
        return load_events_from_file_legacy()

    try:
        # Instancia o store e busca os últimos 2000 eventos
        db = EventStore()
        events = db.get_recent_events(limit=2000)
        
        if not events:
            st.info("ℹ️ Nenhum evento encontrado no banco de dados.")
            # Tenta fallback se banco estiver vazio (durante migração)
            if EVENTS_FILE.exists():
                return load_events_from_file_legacy()
            return pd.DataFrame()

        # Processamento similar ao original para garantir compatibilidade
        processed_events = []
        for event in events:
            # Garante que temos um timestamp utilizável
            if "timestamp" in event and isinstance(event["timestamp"], str):
                event["timestamp"] = convert_to_sao_paulo_tz(event["timestamp"])
            elif "timestamp_utc" in event:
                 event["timestamp"] = convert_to_sao_paulo_tz(event["timestamp_utc"])
            elif "epoch_ms" in event:
                # Fallback para epoch se string não existir
                try:
                    ts = pd.to_datetime(event["epoch_ms"], unit='ms', utc=True).tz_convert('America/Sao_Paulo')
                    event["timestamp"] = ts
                except:
                    event["timestamp"] = pd.NaT
            else:
                event["timestamp"] = pd.NaT
            
            processed_events.append(event)

        df = pd.DataFrame(processed_events)

        # Normalização de colunas para compatibilidade
        if not df.empty:
            # Garante coluna 'ativo'
            if "ativo" not in df.columns:
                if "symbol" in df.columns:
                    df["ativo"] = df["symbol"]
                elif "par" in df.columns:
                    df["ativo"] = df["par"]
                else:
                    df["ativo"] = None

        if not df.empty:
            # Limpeza e ordenação padrão
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            st.success(f"✅ Carregados {len(df)} eventos do Banco de Dados.")
        
        return df

    except Exception as e:
        st.error(f"❌ Erro ao ler do SQLite: {e}")
        # Em caso de erro grave no banco, tenta o arquivo como último recurso
        return load_events_from_file_legacy()

def safe_timezone_convert(series):
    """
    Converte uma série de timestamps para timezone de São Paulo de forma segura.
    """
    converted_series = []
    for timestamp in series:
        if pd.isna(timestamp):
            converted_series.append(pd.NaT)
            continue
            
        try:
            # Se já tem timezone, converte diretamente
            if hasattr(timestamp, 'tz') and timestamp.tz is not None:
                converted_series.append(timestamp.tz_convert('America/Sao_Paulo'))
            else:
                # Se não tem timezone, assume UTC e converte
                converted_series.append(timestamp.tz_localize('UTC').tz_convert('America/Sao_Paulo'))
        except Exception:
            # Se der erro, mantém o timestamp original
            converted_series.append(timestamp)
    
    return pd.Series(converted_series)

# Carregar eventos (AGORA VIA SQLITE)
df_events = load_events()

# Sidebar de filtros
with st.sidebar:
    st.header("🔍 Filtros")
    
    # Seletor de ativo
    ativos = ["Todos"] + sorted(df_events["ativo"].dropna().unique().tolist()) if not df_events.empty else ["Todos"]
    selected_asset = st.selectbox("Ativo", ativos)
    
    # Seletor de tipo de evento
    tipos_evento = ["Todos"] + sorted(df_events["tipo_evento"].dropna().unique().tolist()) if not df_events.empty else ["Todos"]
    selected_type = st.selectbox("Tipo de Evento", tipos_evento)
    
    # Filtro por resultado da batalha
    resultados = ["Todos"]
    if not df_events.empty and "resultado_da_batalha" in df_events.columns:
        resultados += sorted(df_events["resultado_da_batalha"].dropna().unique().tolist())
    selected_result = st.selectbox("Resultado da Batalha", resultados)
    
    # Slider de quantidade máxima de eventos a mostrar
    max_events = st.slider("Máximo de eventos para exibir", min_value=10, max_value=200, value=50)
    
    # Botão de atualização manual
    if st.button("🔄 Atualizar Dados"):
        st.cache_data.clear()
        st.rerun()

# Aplicar filtros
if not df_events.empty:
    filtered_df = df_events.copy()
    
    if selected_asset != "Todos":
        filtered_df = filtered_df[filtered_df["ativo"] == selected_asset]
    if selected_type != "Todos":
        filtered_df = filtered_df[filtered_df["tipo_evento"] == selected_type]
    if selected_result != "Todos":
        filtered_df = filtered_df[filtered_df["resultado_da_batalha"] == selected_result]
    
    # Ordena novamente após filtragem (garantia adicional)
    # Garante que timestamp existe antes de ordenar
    if "timestamp" in filtered_df.columns:
        filtered_df = filtered_df.sort_values("timestamp", ascending=False).head(max_events).reset_index(drop=True)
    else:
        st.warning("Coluna 'timestamp' não encontrada nos dados filtrados.")
else:
    filtered_df = pd.DataFrame()

# --- GRÁFICO 1: EVENTOS POR HORA ---
st.subheader("📅 Evolução de Eventos por Hora")

if not filtered_df.empty and "timestamp" in filtered_df.columns:
    # Remove eventos com timestamp inválido
    valid_df = filtered_df.dropna(subset=["timestamp"]).copy()
    
    if not valid_df.empty:
        # 🔹 Conversão segura para timezone
        valid_df["timestamp"] = safe_timezone_convert(valid_df["timestamp"])
        
        # Remove novamente qualquer timestamp que ainda seja inválido
        valid_df = valid_df.dropna(subset=["timestamp"])
        
        if not valid_df.empty:
            # 🔹 Cria coluna de hora arredondada para o início da hora
            valid_df["timestamp_hour"] = valid_df["timestamp"].dt.floor("H")
            
            # 🔹 Agrupa por hora e conta eventos
            hourly_counts = valid_df.groupby("timestamp_hour").size().reset_index(name='count')
            
            # 🔹 Cria gráfico de linha
            fig_hourly = px.line(
                hourly_counts,
                x="timestamp_hour",
                y="count",
                title="Quantidade de Eventos por Hora",
                labels={"count": "Número de Eventos", "timestamp_hour": "Hora"},
                color_discrete_sequence=["#2E86AB"]
            )
            fig_hourly.update_layout(
                hovermode="x unified",
                xaxis_tickformat="%H:%M",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            st.info("⏳ Nenhum evento com timestamp válido para gerar gráfico.")
    else:
        st.info("⏳ Nenhum evento válido com timestamp para gerar gráfico.")
else:
    st.info("⏳ Nenhum dado disponível para gerar gráfico de eventos por hora.")

# --- GRÁFICO 2: TIPOS DE EVENTO ---
st.subheader("🎯 Distribuição de Tipos de Evento")

if not filtered_df.empty and "tipo_evento" in filtered_df.columns:
    type_counts = filtered_df["tipo_evento"].value_counts()
    fig_type = px.pie(
        names=type_counts.index,
        values=type_counts.values,
        title="Proporção de Tipos de Evento",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_type, use_container_width=True)
else:
    st.info("📭 Nenhum evento filtrado para análise.")

# --- MAPA DE CALOR DE LIQUIDEZ ---
st.subheader("🔥 Mapa de Calor de Liquidez (Clusters)")

if not filtered_df.empty:
    # Verifica se o campo existe
    if "liquidity_heatmap" not in filtered_df.columns:
        st.warning("⚠️ Campo 'liquidity_heatmap' não encontrado. Verifique se o FlowAnalyzer está ativado.")
    else:
        # Converte para string se necessário para garantir que não quebre
        filtered_df["liquidity_heatmap_str"] = filtered_df["liquidity_heatmap"].astype(str)
        
        clusters_list = []
        for idx, row in filtered_df.iterrows():
            heatmap_raw = row.get("liquidity_heatmap")
            
            # Tenta fazer parse se for string, ou usa direto se for dict
            heatmap_data = {}
            if isinstance(heatmap_raw, dict):
                heatmap_data = heatmap_raw
            elif isinstance(heatmap_raw, str):
                try:
                    heatmap_data = json.loads(heatmap_raw)
                except:
                    continue
            
            if not heatmap_data:
                continue

            try:
                clusters = heatmap_data.get("clusters", [])
                for cluster in clusters:
                    clusters_list.append({
                        "center": cluster.get("center", 0),
                        "total_volume": cluster.get("total_volume", 0),
                        "imbalance_ratio": cluster.get("imbalance_ratio", 0),
                        "trades_count": cluster.get("trades_count", 0),
                        "age_ms": cluster.get("age_ms", 0),
                        "timestamp": row["timestamp"],
                        "symbol": row.get("ativo", "Unknown")
                    })
            except Exception as e:
                # st.warning(f"⚠️ Erro ao processar cluster: {e}")
                continue
        
        if clusters_list:
            df_clusters = pd.DataFrame(clusters_list)
            
            # Padroniza casas decimais para exibição no gráfico
            if "center" in df_clusters.columns:
                df_clusters["center"] = pd.to_numeric(df_clusters["center"], errors="coerce").round(2)
            if "total_volume" in df_clusters.columns:
                df_clusters["total_volume"] = pd.to_numeric(df_clusters["total_volume"], errors="coerce").round(3)
            
            # Conversão segura dos timestamps dos clusters
            if "timestamp" in df_clusters.columns:
                df_clusters["timestamp"] = safe_timezone_convert(df_clusters["timestamp"])
            
            # Criar gráfico de dispersão
            fig_cluster = px.scatter(
                df_clusters,
                x="center",
                y="total_volume",
                size="trades_count",
                color="imbalance_ratio",
                color_continuous_scale="RdBu_r",
                hover_data=["trades_count", "age_ms", "symbol"],
                title="Clusters de Liquidez: Preço × Volume × Imbalance",
                labels={
                    "center": "Preço Central ($)",
                    "total_volume": "Volume Total (BTC)",
                    "imbalance_ratio": "Imbalance Ratio (+Compra / -Venda)",
                    "trades_count": "Número de Trades"
                }
            )
            
            fig_cluster.update_layout(
                xaxis_title="Preço Central ($)",
                yaxis_title="Volume Total (BTC)",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.info("☁️ Nenhum cluster de liquidez detectado nos eventos.")
else:
    st.info("📊 Nenhum evento disponível para análise.")

# --- TABELA DE EVENTOS RECENTES ---
st.subheader("📋 Últimos Eventos Detectados")

if not filtered_df.empty:
    # Selecionar colunas relevantes
    display_cols = [
        "timestamp", "ativo", "tipo_evento", "resultado_da_batalha", 
        "delta", "volume_total", "descricao"
    ]
    
    # Filtrar apenas colunas que existem
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    
    if available_cols:
        # Formatar colunas
        df_display = filtered_df[available_cols].copy()
        
        # Formatação segura do timestamp
        if "timestamp" in df_display.columns:
            df_display["timestamp"] = df_display["timestamp"].apply(
                lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(x) else "N/A"
            )
        
        # Formatação segura de colunas numéricas
        if "delta" in df_display.columns:
            df_display["delta"] = pd.to_numeric(df_display["delta"], errors='coerce').round(2)
        if "volume_total" in df_display.columns:
            df_display["volume_total"] = pd.to_numeric(df_display["volume_total"], errors='coerce').round(3)
        
        # Renomear colunas para legibilidade
        column_mapping = {
            "ativo": "Ativo",
            "tipo_evento": "Tipo",
            "resultado_da_batalha": "Resultado",
            "delta": "Δ (Delta)",
            "volume_total": "Vol (BTC)",
            "descricao": "Descrição"
        }
        
        # Aplicar apenas renomeações para colunas que existem
        rename_dict = {k: v for k, v in column_mapping.items() if k in df_display.columns}
        df_display.rename(columns=rename_dict, inplace=True)
        
        # Exibir tabela interativa
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            height=min(400, len(df_display) * 35)
        )
    else:
        st.warning("⚠️ Nenhuma coluna disponível para exibição.")
else:
    st.info("🔎 Nenhum evento correspondente aos filtros selecionados.")

# --- ESTATÍSTICAS RESUMIDAS ---
if not filtered_df.empty:
    st.subheader("📊 Estatísticas Resumidas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Eventos", len(filtered_df))
    
    with col2:
        if "ativo" in filtered_df.columns:
            unique_assets = filtered_df["ativo"].nunique()
            st.metric("Ativos Únicos", unique_assets)
    
    with col3:
        if "tipo_evento" in filtered_df.columns:
            unique_types = filtered_df["tipo_evento"].nunique()
            st.metric("Tipos de Evento", unique_types)
    
    with col4:
        if "volume_total" in filtered_df.columns:
            total_volume = pd.to_numeric(filtered_df["volume_total"], errors='coerce').sum()
            if not pd.isna(total_volume):
                st.metric("Volume Total (BTC)", f"{total_volume:.3f}")

# --- RODAPÉ ---
st.divider()
st.caption("""
    💡 Este dashboard lê os eventos salvos no **SQLite (trading_bot.db)**.  
    Recarregue esta página para ver novos eventos em tempo real.
""")

# Nota de rodapé sobre desempenho
st.sidebar.divider()
st.sidebar.caption("🛠️ Dashboard construído com Streamlit | Atualizado em: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))