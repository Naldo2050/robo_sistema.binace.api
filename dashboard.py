# dashboard.py - v3.1.0 (SQLite + Legacy Features)
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuração de logging
logging.basicConfig(level=logging.WARNING)

# Configuração da página Streamlit
st.set_page_config(
    page_title="🚀 Trader AI - Institutional Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Caminho do banco SQLite
DB_PATH = Path("dados/trading_bot.db")

# ---------------- FUNÇÕES DE DADOS ---------------- #

def get_connection():
    """Cria conexão somente leitura com o banco SQLite."""
    if not DB_PATH.exists():
        msg = f"❌ Banco de dados não encontrado: `{DB_PATH}`"
        st.error(msg)
        # Em ambiente Streamlit, st.stop() interrompe a execução.
        # Em modo 'bare' (python dashboard.py), levantamos uma exceção normal.
        try:
            st.stop()
        except Exception:
            pass
        raise FileNotFoundError(msg)

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        return conn
    except Exception as e:
        msg = f"❌ Erro ao conectar ao banco SQLite: {e}"
        st.error(msg)
        try:
            st.stop()
        except Exception:
            pass
        raise


def _parse_payload(raw_payload: str) -> pd.Series:
    """
    Faz parse do payload JSON e extrai campos usados pelo dashboard.
    Sempre retorna todas as chaves esperadas, mesmo em caso de erro.
    """
    base = {
        "resultado_da_batalha": None,
        "delta": None,
        "volume_total": None,
        "preco_fechamento": None,
        "descricao": None,
        "liquidity_heatmap": None,
        "ai_analysis": None,
    }

    if not isinstance(raw_payload, str) or not raw_payload.strip():
        return pd.Series(base)

    try:
        data = json.loads(raw_payload)
    except Exception as e:
        logging.debug("Falha ao fazer json.loads do payload: %s", e)
        return pd.Series(base)

    try:
        # --- CAMPOS DIRETOS (compatível com formato antigo) ---
        base["resultado_da_batalha"] = data.get("resultado_da_batalha")

        # Derivar resultado_da_batalha de absorption_type se necessário
        ai_payload = data.get("ai_payload") or {}
        if isinstance(ai_payload, dict):
            flow_ctx = ai_payload.get("flow_context") or {}
            if isinstance(flow_ctx, dict):
                if base["resultado_da_batalha"] is None:
                    abs_type = flow_ctx.get("absorption_type")
                    if isinstance(abs_type, str) and abs_type.strip():
                        base["resultado_da_batalha"] = abs_type

        val = data.get("delta")
        base["delta"] = float(val) if val not in (None, "") else None

        val = data.get("volume_total")
        base["volume_total"] = float(val) if val not in (None, "") else None

        val = data.get("preco_fechamento")
        base["preco_fechamento"] = float(val) if val not in (None, "") else None

        base["descricao"] = data.get("descricao")

        # Heatmap pode vir dentro de fluxo_continuo ou direto
        hm = None
        fluxo = data.get("fluxo_continuo")
        if isinstance(fluxo, dict):
            hm = fluxo.get("liquidity_heatmap")
        if hm is None:
            hm = data.get("liquidity_heatmap")
        # Check in ai_payload for AI_ANALYSIS events
        if hm is None:
            ai_payload = data.get("ai_payload", {})
            if isinstance(ai_payload, dict):
                hm = ai_payload.get("liquidity_heatmap")
        base["liquidity_heatmap"] = hm

        base["ai_analysis"] = data.get("ai_analysis")

        # --- CAMPOS ESPECÍFICOS DO NOVO FORMATO (AI_ANALYSIS, etc.) ---

        # 1) Preço: usar anchor_price se não houver preco_fechamento
        if base["preco_fechamento"] is None:
            val = data.get("anchor_price")
            if val not in (None, ""):
                try:
                    base["preco_fechamento"] = float(val)
                except Exception:
                    pass

        # 2) Delta e volume a partir de ai_payload.flow_context
        ai_payload = data.get("ai_payload") or {}
        if isinstance(ai_payload, dict):
            flow_ctx = ai_payload.get("flow_context") or {}
            if isinstance(flow_ctx, dict):
                # Delta: usar cvd_accumulated se delta ainda não estiver preenchido
                if base["delta"] is None:
                    val = flow_ctx.get("cvd_accumulated")
                    if val not in (None, ""):
                        try:
                            base["delta"] = float(val)
                        except Exception:
                            pass

                # Volume: somar whale_buy_vol + whale_sell_vol se volume_total vazio
                if base["volume_total"] is None:
                    whale = flow_ctx.get("whale_activity") or {}
                    if isinstance(whale, dict):
                        buy = whale.get("whale_buy_vol")
                        sell = whale.get("whale_sell_vol")
                        if buy not in (None, "") or sell not in (None, ""):
                            try:
                                buy_f = float(buy) if buy not in (None, "") else 0.0
                                sell_f = float(sell) if sell not in (None, "") else 0.0
                                base["volume_total"] = buy_f + sell_f
                            except Exception:
                                pass

    except Exception as e:
        logging.warning("Erro ao extrair campos do payload: %s", e)

    return pd.Series(base)


@st.cache_data(ttl=10)
def load_data(hours_back: int = 24, limit: int = 5000) -> pd.DataFrame:
    """
    Carrega eventos do SQLite com filtro de tempo e limite máximo.
    Retorna DataFrame pandas já com colunas derivadas do payload.
    """
    conn = get_connection()

    cutoff_ms = int((datetime.utcnow() - timedelta(hours=hours_back)).timestamp() * 1000)

    query = """
        SELECT timestamp_ms, event_type, symbol, is_signal, payload
        FROM events
        WHERE timestamp_ms >= ?
        ORDER BY timestamp_ms DESC
        LIMIT ?
    """

    try:
        df = pd.read_sql_query(query, conn, params=(cutoff_ms, limit))
    except Exception as e:
        st.error(f"❌ Erro ao ler dados da tabela 'events': {e}")
        return pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    # Verificações de schema com mensagens amigáveis
    expected_columns = {
        'timestamp_ms': 'integer',
        'event_type': 'text',
        'symbol': 'text',
        'is_signal': 'integer',
        'payload': 'text'
    }
    for col, expected_type in expected_columns.items():
        if col not in df.columns:
            st.warning(f"⚠️ Coluna '{col}' não encontrada na tabela 'events'. Alguns recursos podem não funcionar corretamente.")
        elif expected_type == 'integer' and not pd.api.types.is_integer_dtype(df[col]):
            st.warning(f"⚠️ Coluna '{col}' não é do tipo inteiro esperado. Valores podem estar incorretos.")
        elif expected_type == 'text' and not pd.api.types.is_object_dtype(df[col]):
            st.warning(f"⚠️ Coluna '{col}' não é do tipo texto esperado. Valores podem estar incorretos.")

    # Conversão de timestamp
    if "timestamp_ms" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df["datetime_sp"] = df["datetime_utc"].dt.tz_convert("America/Sao_Paulo")
        # Compatibilidade com dashboard antigo
        df["timestamp"] = df["datetime_sp"]
    else:
        st.warning("Coluna 'timestamp_ms' não encontrada na tabela 'events'.")
        df["timestamp"] = pd.NaT

    # Parse do payload JSON
    if "payload" in df.columns:
        payload_df = df["payload"].apply(_parse_payload)
        df = pd.concat([df, payload_df], axis=1)
    else:
        st.warning("Coluna 'payload' não encontrada na tabela 'events'.")

    # Alias / conversões numéricas
    if "preco_fechamento" in df.columns:
        df["preco"] = pd.to_numeric(df["preco_fechamento"], errors="coerce")
    if "volume_total" in df.columns:
        df["volume_total"] = pd.to_numeric(df["volume_total"], errors="coerce")
        df["volume"] = df["volume_total"]
    if "delta" in df.columns:
        df["delta"] = pd.to_numeric(df["delta"], errors="coerce")

    # Alias para facilitar filtros e gráficos
    if "symbol" in df.columns:
        df["ativo"] = df["symbol"]
    if "event_type" in df.columns:
        df["tipo_evento"] = df["event_type"]
    if "resultado_da_batalha" in df.columns:
        df["resultado"] = df["resultado_da_batalha"]

    return df

# ---------------- INTERFACE ---------------- #

st.title("📊 Trader AI - Dashboard Institucional")
st.markdown(
    """
### Sistema de Trading Algorítmico Inteligente
Monitoramento em tempo real via **SQLite Engine** com filtros avançados.

🔍 **Funcionalidades combinadas:**
- Sinais de Absorção / Exaustão (compra / venda)
- Zonas de liquidez e clusters (heatmap)
- Linha de preço com marcação de sinais
- Evolução de eventos por hora
- Distribuição por tipo de evento
- Filtros por ativo, tipo, resultado e sinais
"""
)

# -------- Sidebar: Controles Globais -------- #
with st.sidebar:
    st.header("🎛️ Janela & Limite")

    time_range = st.selectbox(
        "Janela de Tempo",
        options=[1, 4, 12, 24, 72, 168],
        format_func=lambda x: f"Últimas {x}h",
        index=3,  # 24h
    )

    max_events = st.slider(
        "Máximo de eventos a carregar",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100,
    )

    if st.button("🔄 Atualizar Agora"):
        st.cache_data.clear()
        st.rerun()

# Carregamento de dados bruto (somente filtro de tempo e limite)
with st.spinner("Conectando ao Neural Core..."):
    df = load_data(hours_back=time_range, limit=max_events)

if df.empty:
    st.warning("📭 Nenhum dado encontrado para o período selecionado.")
    st.stop()

# -------- Sidebar: Filtros de Conteúdo -------- #
with st.sidebar:
    st.header("🔍 Filtros de Conteúdo")

    ativos = (
        ["Todos"]
        + sorted(df["ativo"].dropna().unique().tolist())
        if "ativo" in df.columns and not df.empty
        else ["Todos"]
    )
    selected_asset = st.selectbox("Ativo", ativos)

    tipos_evento = (
        ["Todos"]
        + sorted(df["tipo_evento"].dropna().unique().tolist())
        if "tipo_evento" in df.columns and not df.empty
        else ["Todos"]
    )
    selected_type = st.selectbox("Tipo de Evento", tipos_evento)

    resultados = (
        ["Todos"]
        + sorted(df["resultado_da_batalha"].dropna().unique().tolist())
        if "resultado_da_batalha" in df.columns and not df.empty
        else ["Todos"]
    )
    selected_result = st.selectbox("Resultado da Batalha", resultados)

    only_signals = st.checkbox("Apenas sinais relevantes (is_signal = 1)", value=False)

# Aplicar filtros em memória
df_filtered = df.copy()

if selected_asset != "Todos" and "ativo" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["ativo"] == selected_asset]

if selected_type != "Todos" and "tipo_evento" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["tipo_evento"] == selected_type]

if selected_result != "Todos" and "resultado_da_batalha" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["resultado_da_batalha"] == selected_result]

if only_signals and "is_signal" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["is_signal"] == 1]

# Garante ordenação por tempo (mais recentes primeiro) e reaplica limite
if "timestamp_ms" in df_filtered.columns:
    df_filtered = (
        df_filtered.sort_values("timestamp_ms", ascending=False)
        .head(max_events)
        .reset_index(drop=True)
    )

if df_filtered.empty:
    st.warning("📭 Nenhum evento corresponde aos filtros selecionados.")
    st.stop()

# ---------------- MÉTRICAS DE TOPO ---------------- #
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Eventos no Período (filtrados)", len(df_filtered))

with col2:
    last_price = (
        df_filtered["preco"].iloc[0]
        if "preco" in df_filtered.columns and not df_filtered["preco"].isna().all()
        else None
    )
    if last_price is not None:
        st.metric("Último Preço", f"${last_price:,.2f}")
    else:
        st.metric("Último Preço", "N/D")

with col3:
    vol_period = (
        df_filtered["volume_total"].sum()
        if "volume_total" in df_filtered.columns
        else 0
    )
    st.metric("Volume Acumulado (BTC)", f"{vol_period:,.3f}")

with col4:
    signals_count = 0
    if "resultado" in df_filtered.columns:
        if "is_signal" in df_filtered.columns and df_filtered["is_signal"].max() == 1:
            mask_signal = df_filtered["is_signal"] == 1
        else:
            mask_signal = df_filtered["resultado"].str.contains("Absorção", case=False, na=False)

        signals_count = df_filtered[mask_signal].shape[0]
    st.metric("Sinais Detectados", signals_count)

# ---------------- GRÁFICOS PRINCIPAIS ---------------- #

# 1. Timeline de Preço + Sinais
st.subheader("📈 Timeline de Mercado (Preço + Sinais)")

if "datetime_sp" in df_filtered.columns and "preco" in df_filtered.columns:
    fig_price = go.Figure()

    fig_price.add_trace(
        go.Scatter(
            x=df_filtered["datetime_sp"],
            y=df_filtered["preco"],
            mode="lines",
            name="Preço",
            line=dict(color="#1f77b4", width=1),
        )
    )

    # Sinais de compra/venda baseados em 'resultado'
    if "resultado" in df_filtered.columns and "datetime_sp" in df_filtered.columns and "preco" in df_filtered.columns:
        if "is_signal" in df_filtered.columns and df_filtered["is_signal"].max() == 1:
            signals_df = df_filtered[df_filtered["is_signal"] == 1].copy()
        else:
            signals_df = df_filtered[df_filtered["resultado"].str.contains("Absorção", case=False, na=False)].copy()

        buy_signals = signals_df[
            signals_df["resultado"].str.contains("Absorção de Venda", case=False, na=False)
        ]
        sell_signals = signals_df[
            signals_df["resultado"].str.contains("Absorção de Compra", case=False, na=False)
        ]

        if not buy_signals.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=buy_signals["datetime_sp"],
                    y=buy_signals["preco"],
                    mode="markers",
                    name="Sinal Compra",
                    marker=dict(color="green", size=10, symbol="triangle-up"),
                )
            )

        if not sell_signals.empty:
            fig_price.add_trace(
                go.Scatter(
                    x=sell_signals["datetime_sp"],
                    y=sell_signals["preco"],
                    mode="markers",
                    name="Sinal Venda",
                    marker=dict(color="red", size=10, symbol="triangle-down"),
                )
            )

    fig_price.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.info("Não há dados suficientes para montar a timeline de preço.")

# 2. Eventos por Hora (versão adaptada do dashboard antigo)
st.subheader("⏱️ Evolução de Eventos por Hora")

if "datetime_sp" in df_filtered.columns:
    df_time = df_filtered.dropna(subset=["datetime_sp"]).copy()
    if not df_time.empty:
        df_time["timestamp_hour"] = df_time["datetime_sp"].dt.floor("H")
        hourly_counts = (
            df_time.groupby("timestamp_hour")
            .size()
            .reset_index(name="count")
            .sort_values("timestamp_hour")
        )

        fig_hourly = px.line(
            hourly_counts,
            x="timestamp_hour",
            y="count",
            title="Quantidade de Eventos por Hora",
            labels={"count": "Número de Eventos", "timestamp_hour": "Hora"},
            color_discrete_sequence=["#2E86AB"],
        )
        fig_hourly.update_layout(
            hovermode="x unified",
            xaxis_tickformat="%d/%m %H:%M",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    else:
        st.info("Nenhum evento com timestamp válido para gerar gráfico por hora.")
else:
    st.info("Coluna de tempo não disponível para o gráfico por hora.")

# 3. Delta e Distribuição de Tipos de Evento
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("📊 Volume Delta por Evento")
    if "delta" in df_filtered.columns and "datetime_sp" in df_filtered.columns:
        fig_delta = px.bar(
            df_filtered,
            x="datetime_sp",
            y="delta",
            color="delta",
            color_continuous_scale=["red", "gray", "green"],
            labels={"datetime_sp": "Horário (SP)", "delta": "Delta Líquido"},
        )
        fig_delta.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_delta, use_container_width=True)
    else:
        st.info("Coluna 'delta' não disponível para montar o gráfico.")

with col_g2:
    st.subheader("🎯 Distribuição de Tipos de Evento")
    if "tipo_evento" in df_filtered.columns:
        type_counts = df_filtered["tipo_evento"].value_counts()
        if not type_counts.empty:
            fig_type = px.pie(
                names=type_counts.index,
                values=type_counts.values,
                title="Proporção de Tipos de Evento",
                hole=0.4,
            )
            st.plotly_chart(fig_type, use_container_width=True)
        else:
            st.info("Nenhum tipo de evento para exibir.")
    else:
        st.info("Coluna 'tipo_evento' não encontrada nos dados.")

# 4. Heatmap de Liquidez (clusters acumulados) - VERSÃO PROFISSIONAL COM VOLUME PROFILE
st.subheader("🔥 Análise de Liquidez e Perfil de Volume")

clusters_list = []
if "liquidity_heatmap" in df_filtered.columns:
    for _, row in df_filtered.iterrows():
        hm_data = row["liquidity_heatmap"]
        if hm_data is None or (isinstance(hm_data, float) and pd.isna(hm_data)):
            continue

        # hm_data pode ser dict ou string JSON
        if isinstance(hm_data, str):
            try:
                hm_data = json.loads(hm_data)
            except Exception:
                continue

        if not isinstance(hm_data, dict):
            continue

        clusters = hm_data.get("clusters", [])
        for cluster in clusters:
            clusters_list.append(
                {
                    "center": cluster.get("center", 0),
                    "total_volume": cluster.get("total_volume", 0),
                    "imbalance_ratio": cluster.get("imbalance_ratio", 0),
                    "trades_count": cluster.get("trades_count", 0),
                    "age_ms": cluster.get("age_ms", 0),
                    "timestamp": row.get("datetime_sp", row.get("timestamp")),
                    "symbol": row.get("ativo") or row.get("symbol"),
                }
            )

if clusters_list:
    df_clusters = pd.DataFrame(clusters_list)

    # Garantir que df_clusters não esteja vazio antes de processar
    if df_clusters.empty:
        st.info("☁️ Dados de clusters insuficientes após processamento.")
    else:
        if "center" in df_clusters.columns:
            df_clusters["center"] = pd.to_numeric(
                df_clusters["center"], errors="coerce"
            ).round(2)
        if "total_volume" in df_clusters.columns:
            df_clusters["total_volume"] = pd.to_numeric(
                df_clusters["total_volume"], errors="coerce"
            ).round(3)
        if "imbalance_ratio" in df_clusters.columns:
            df_clusters["imbalance_ratio"] = pd.to_numeric(
                df_clusters["imbalance_ratio"], errors="coerce"
            ).round(3)

        # Remover linhas com dados inválidos
        df_clusters = df_clusters.dropna(subset=["center", "total_volume"])

        if df_clusters.empty:
            st.info("☁️ Dados de clusters insuficientes após processamento.")
        else:
            # ============= CRIAR DOIS GRÁFICOS LADO A LADO =============
            col_chart1, col_chart2 = st.columns([1.2, 1])

            with col_chart1:
                st.markdown("#### 📊 Mapa de Calor de Clusters")

                # Criar o heatmap com scatter
                fig_heatmap = go.Figure()

                # Normalizar tamanho dos marcadores - REDUZIDO
                size_min, size_max = 8, 35  # Tamanhos menores
                if df_clusters["trades_count"].max() > df_clusters["trades_count"].min():
                    size_norm = (
                        (df_clusters["trades_count"] - df_clusters["trades_count"].min()) /
                        (df_clusters["trades_count"].max() - df_clusters["trades_count"].min())
                    ) * (size_max - size_min) + size_min
                else:
                    size_norm = [size_min] * len(df_clusters)

                # Classificar cada cluster
                def get_zone_label(imb):
                    if imb >= 0.4:
                        return "🟢 Compra Forte"
                    elif imb >= 0.15:
                        return "🟢 Compra"
                    elif imb <= -0.4:
                        return "🔴 Venda Forte"
                    elif imb <= -0.15:
                        return "🔴 Venda"
                    else:
                        return "⚪ Neutro"

                df_clusters["zone_label"] = df_clusters["imbalance_ratio"].apply(get_zone_label)

                # Adicionar scatter plot
                fig_heatmap.add_trace(go.Scatter(
                    x=df_clusters["center"],
                    y=df_clusters["total_volume"],
                    mode='markers',
                    marker=dict(
                        size=size_norm,
                        color=df_clusters["imbalance_ratio"],
                        colorscale=[
                            [0.0, '#8B0000'],    # Vermelho escuro (venda forte)
                            [0.25, '#FF4444'],   # Vermelho (venda)
                            [0.5, '#A9A9A9'],    # CINZA (neutro) - ALTERADO
                            [0.75, '#90EE90'],   # Verde claro (compra)
                            [1.0, '#006400']     # Verde escuro (compra forte)
                        ],
                        cmin=-1.0,
                        cmax=1.0,
                        colorbar=dict(
                            title=dict(
                                text="<b>Pressão</b>",
                                font=dict(size=13, color='#333')
                            ),
                            thickness=18,
                            len=0.75,
                            x=1.15,
                            tickmode='array',
                            tickvals=[-0.8, -0.4, 0, 0.4, 0.8],
                            ticktext=[
                                '🔴 Venda<br>Forte',
                                '🔴 Venda',
                                '⚪ Neutro',
                                '🟢 Compra',
                                '🟢 Compra<br>Forte'
                            ],
                            tickfont=dict(size=10, color='#333'),
                            outlinewidth=1,
                            outlinecolor='#ccc'
                        ),
                        line=dict(width=2, color='white'),
                        opacity=0.9,
                        showscale=True
                    ),
                    text=[
                        f"<b>{row['zone_label']}</b><br><br>" +
                        f"<b>💰 Preço:</b> ${row['center']:,.2f}<br>" +
                        f"<b>📊 Volume:</b> {row['total_volume']:.3f} BTC<br>" +
                        f"<b>⚖️ Desequilíbrio:</b> {row['imbalance_ratio']:.1%}<br>" +
                        f"<b>🔢 Trades:</b> {int(row['trades_count'])}<br>" +
                        f"<b>🎯 Ativo:</b> {row['symbol']}"
                        for _, row in df_clusters.iterrows()
                    ],
                    hovertemplate='%{text}<extra></extra>',
                    name='',
                    showlegend=False
                ))

                # Adicionar linhas de referência
                if not df_clusters.empty:
                    vol_median = df_clusters["total_volume"].median()
                    price_median = df_clusters["center"].median()

                    fig_heatmap.add_hline(
                        y=vol_median,
                        line_dash="dash",
                        line_color="rgba(100, 100, 100, 0.4)",
                        line_width=1.5,
                        annotation_text="Volume Mediano",
                        annotation_position="right",
                        annotation_font_size=10,
                        annotation_font_color="#666"
                    )

                    fig_heatmap.add_vline(
                        x=price_median,
                        line_dash="dash",
                        line_color="rgba(100, 100, 100, 0.4)",
                        line_width=1.5,
                        annotation_text="Preço Mediano",
                        annotation_position="top",
                        annotation_font_size=10,
                        annotation_font_color="#666"
                    )

                # Layout do heatmap
                fig_heatmap.update_layout(
                    xaxis=dict(
                        title="<b>Nível de Preço (USD)</b>",
                        showgrid=True,
                        gridcolor='rgba(200, 200, 200, 0.2)',
                        zeroline=False,
                        tickformat='$,.0f',
                        title_font=dict(size=12, color='#333'),
                        tickfont=dict(size=10, color='#555')
                    ),
                    yaxis=dict(
                        title="<b>Volume Acumulado (BTC)</b>",
                        showgrid=True,
                        gridcolor='rgba(200, 200, 200, 0.2)',
                        zeroline=False,
                        tickformat=',.2f',
                        title_font=dict(size=12, color='#333'),
                        tickfont=dict(size=10, color='#555')
                    ),
                    plot_bgcolor='#f8f9fa',
                    paper_bgcolor='white',
                    hovermode='closest',
                    height=550,
                    margin=dict(l=70, r=120, t=20, b=60),
                    font=dict(family="Inter, Arial, sans-serif", size=11)
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)

            with col_chart2:
                st.markdown("#### 📈 Perfil de Volume (POC)")

                # Criar volume profile (histograma horizontal)
                # Agrupar por faixas de preço
                price_min = df_clusters["center"].min()
                price_max = df_clusters["center"].max()
                price_range = price_max - price_min

                # Proteger contra faixa de preço zero ou muito estreita
                if price_range <= 0:
                    # todos os clusters no mesmo preço → bin único
                    df_clusters["price_bin"] = df_clusters["center"].round(2)
                else:
                    # Criar bins de preço (15–30 níveis, dependendo da quantidade)
                    num_bins = min(30, max(15, int(len(df_clusters) / 2)))
                    if num_bins <= 0:
                        num_bins = 1

                    bin_size = price_range / num_bins if num_bins > 0 else price_range or 1.0

                    df_clusters["price_bin"] = (
                        ((df_clusters["center"] - price_min) // bin_size) * bin_size + price_min
                    ).round(2)

                # Agregar volume e imbalance por bin
                volume_profile = df_clusters.groupby("price_bin").agg({
                    "total_volume": "sum",
                    "imbalance_ratio": "mean",
                    "trades_count": "sum"
                }).reset_index()

                if volume_profile.empty:
                    st.info("☁️ Dados insuficientes para gerar o perfil de volume.")
                else:
                    # ORDENAR POR PREÇO CRESCENTE
                    volume_profile = volume_profile.sort_values("price_bin", ascending=True).reset_index(drop=True)

                    # Identificar POC (Point of Control - maior volume)
                    poc_idx = volume_profile["total_volume"].idxmax()
                    poc_price = float(volume_profile.loc[poc_idx, "price_bin"])
                    poc_volume = float(volume_profile.loc[poc_idx, "total_volume"])

                    # Criar figura do volume profile
                    fig_profile = go.Figure()

                    # Cores baseadas no imbalance + POC amarelo ouro
                    colors = []
                    texts = []
                    for idx, row_vp in volume_profile.iterrows():
                        imb = float(row_vp["imbalance_ratio"]) if row_vp["imbalance_ratio"] is not None else 0.0
                        vol = float(row_vp["total_volume"])

                        if idx == poc_idx:
                            # POC override: amarelo ouro
                            colors.append("#FFD700")  # POC
                            texts.append(f"🎯 POC: {vol:.2f} BTC | {imb:+.1%}")
                        elif imb >= 0.4:
                            colors.append("#006400")  # Verde escuro - Compra Forte
                            texts.append(f"🟢 {vol:.2f} BTC | {imb:+.1%}")
                        elif imb >= 0.15:
                            colors.append("#90EE90")  # Verde claro - Compra
                            texts.append(f"🟢 {vol:.2f} BTC | {imb:+.1%}")
                        elif imb <= -0.4:
                            colors.append("#8B0000")  # Vermelho escuro - Venda Forte
                            texts.append(f"🔴 {vol:.2f} BTC | {imb:+.1%}")
                        elif imb <= -0.15:
                            colors.append("#FFB6C1")  # Rosa - Venda
                            texts.append(f"🔴 {vol:.2f} BTC | {imb:+.1%}")
                        else:
                            colors.append("#A9A9A9")  # Cinza - Neutro
                            texts.append(f"⚪ {vol:.2f} BTC | {imb:+.1%}")

                    # Adicionar barras horizontais
                    fig_profile.add_trace(go.Bar(
                        x=volume_profile["total_volume"],
                        y=volume_profile["price_bin"],
                        orientation='h',
                        marker=dict(
                            color=colors,
                            line=dict(color='white', width=1.5)
                        ),
                        text=texts,
                        textposition='inside',
                        textfont=dict(size=9, color='white', family='monospace'),
                        hovertemplate=(
                            "<b>Preço:</b> $%{y:,.2f}<br>"
                            "<b>Volume:</b> %{x:.3f} BTC<br>"
                            "<extra></extra>"
                        ),
                        showlegend=False
                    ))

                    # Destacar POC com linha laranja tracejada
                    fig_profile.add_hline(
                        y=poc_price,
                        line_color='#FF6600',  # Laranja forte
                        line_width=2.5,
                        line_dash="dot",
                        annotation_text="← POC",
                        annotation_position="right",
                        annotation_font=dict(size=10, color='#FF6600', family='monospace')
                    )

                    # Layout do profile
                    fig_profile.update_layout(
                        xaxis=dict(
                            title="<b>Volume Total (BTC)</b>",
                            showgrid=True,
                            gridcolor='rgba(200, 200, 200, 0.2)',
                            title_font=dict(size=11, color='#333'),
                            tickfont=dict(size=9, color='#555')
                        ),
                        yaxis=dict(
                            title="<b>Nível de Preço ($)</b>",
                            showgrid=True,
                            gridcolor='rgba(200, 200, 200, 0.2)',
                            tickformat='$,.0f',
                            title_font=dict(size=11, color='#333'),
                            tickfont=dict(size=9, color='#555')
                        ),
                        plot_bgcolor='#f8f9fa',
                        paper_bgcolor='white',
                        height=550,
                        margin=dict(l=70, r=20, t=20, b=60),
                        font=dict(family="Inter, Arial, sans-serif")
                    )

                    # Legenda do profile - 6 cores
                    fig_profile.add_annotation(
                        text=(
                            "<b>🎯 POC</b> (amarelo): Maior volume<br>"
                            "<b>🟢 Verde escuro</b>: Compra forte (≥ 0.4)<br>"
                            "<b>🟢 Verde claro</b>: Compra (≥ 0.15)<br>"
                            "<b>🔴 Vermelho escuro</b>: Venda forte (≤ -0.4)<br>"
                            "<b>🩷 Rosa</b>: Venda (≤ -0.15)<br>"
                            "<b>⚪ Cinza</b>: Neutro (-0.15 a 0.15)"
                        ),
                        xref="paper", yref="paper",
                        x=0.98, y=0.98,
                        showarrow=False,
                        font=dict(size=9, color='#444'),
                        bgcolor='rgba(255, 255, 255, 0.95)',
                        bordercolor='#999',
                        borderwidth=1,
                        borderpad=10,
                        align='left',
                        xanchor='right',
                        yanchor='top'
                    )

                    st.plotly_chart(fig_profile, use_container_width=True)

        # ============= ESTATÍSTICAS DETALHADAS =============
        st.markdown("---")
        st.markdown("### 📊 Análise de Zonas de Liquidez")

        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)

        with col_s1:
            strong_buy = len(df_clusters[df_clusters["imbalance_ratio"] >= 0.4])
            buy_vol = df_clusters[df_clusters["imbalance_ratio"] >= 0.4]["total_volume"].sum()
            st.metric(
                "🟢 Compra Forte",
                f"{strong_buy} zonas",
                f"{buy_vol:.2f} BTC"
            )

        with col_s2:
            strong_sell = len(df_clusters[df_clusters["imbalance_ratio"] <= -0.4])
            sell_vol = df_clusters[df_clusters["imbalance_ratio"] <= -0.4]["total_volume"].sum()
            st.metric(
                "🔴 Venda Forte",
                f"{strong_sell} zonas",
                f"{sell_vol:.2f} BTC"
            )

        with col_s3:
            neutral = len(df_clusters[
                (df_clusters["imbalance_ratio"] > -0.15) &
                (df_clusters["imbalance_ratio"] < 0.15)
            ])
            neutral_vol = df_clusters[
                (df_clusters["imbalance_ratio"] > -0.15) &
                (df_clusters["imbalance_ratio"] < 0.15)
            ]["total_volume"].sum()
            st.metric(
                "⚪ Zonas Neutras",
                f"{neutral} zonas",
                f"{neutral_vol:.2f} BTC"
            )

        with col_s4:
            if 'poc_price' in locals() and 'poc_volume' in locals():
                st.metric(
                    "🎯 POC (Point of Control)",
                    f"${poc_price:,.2f}",
                    f"{poc_volume:.3f} BTC",
                    help="Nível de preço com maior volume negociado"
                )
            else:
                st.metric("🎯 POC (Point of Control)", "N/D")

        with col_s5:
            total_trades = int(df_clusters["trades_count"].sum())
            avg_cluster_size = df_clusters["total_volume"].mean()
            st.metric(
                "📈 Atividade Total",
                f"{total_trades:,} trades",  # Formatação com vírgula
                f"Avg: {avg_cluster_size:.2f} BTC/cluster"
            )

        # Tabela de principais clusters - ORDENADA POR PREÇO CRESCENTE
        st.markdown("### 🔝 Top 10 Clusters por Volume")

        top_clusters = df_clusters.nlargest(10, "total_volume")[
            ["center", "total_volume", "imbalance_ratio", "trades_count", "zone_label"]
        ].copy()

        # ORDENAR POR PREÇO CRESCENTE
        top_clusters = top_clusters.sort_values("center", ascending=True)

        top_clusters.columns = ["Preço ($)", "Volume (BTC)", "Imbalance", "Trades", "Zona"]
        top_clusters["Preço ($)"] = top_clusters["Preço ($)"].apply(lambda x: f"${x:,.2f}")
        top_clusters["Volume (BTC)"] = top_clusters["Volume (BTC)"].apply(lambda x: f"{x:.3f}")
        top_clusters["Imbalance"] = top_clusters["Imbalance"].apply(lambda x: f"{x:+.1%}")
        top_clusters["Trades"] = top_clusters["Trades"].apply(lambda x: f"{int(x):,}")

        # Destacar POC na tabela
        top_clusters_display = top_clusters.reset_index(drop=True).copy()

        # Verificar se POC está no top 10
        poc_in_top10 = False
        if 'poc_price' in locals():
            for idx, row in top_clusters_display.iterrows():
                price_str = row["Preço ($)"].replace("$", "").replace(",", "")
                try:
                    price_val = float(price_str)
                    if abs(price_val - poc_price) < 1.0:  # Tolerância de $1
                        poc_in_top10 = True
                        # Adicionar marcador POC
                        top_clusters_display.at[idx, "Zona"] = f"🎯 {row['Zona']} (POC)"
                except (ValueError, TypeError):
                    continue

        st.dataframe(
            top_clusters_display,
            use_container_width=True,
            height=min(400, len(top_clusters_display) * 35 + 50)
        )

        if not poc_in_top10 and 'poc_price' in locals() and 'poc_volume' in locals():
            st.info(f"ℹ️ **POC** está em **${poc_price:,.2f}** com **{poc_volume:.3f} BTC** de volume (fora do Top 10 por volume total)")

else:
    st.info("☁️ Nenhum cluster de liquidez detectado nos eventos filtrados.")

# 5. Tabela de eventos recentes
st.subheader("📋 Últimos Eventos Detalhados")

display_cols = [
    "datetime_sp",
    "ativo",
    "tipo_evento",
    "resultado_da_batalha",
    "preco",
    "delta",
    "volume_total",
    "descricao",
]
available_cols = [c for c in display_cols if c in df_filtered.columns]

if available_cols:
    df_display = (
        df_filtered.sort_values("timestamp_ms", ascending=False)[available_cols].copy()
        if "timestamp_ms" in df_filtered.columns
        else df_filtered[available_cols].copy()
    )

    # Formatação de timestamp
    if "datetime_sp" in df_display.columns:
        df_display["datetime_sp"] = (
            df_display["datetime_sp"]
            .dt.tz_convert("America/Sao_Paulo")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

    # Formatação numérica
    if "delta" in df_display.columns:
        df_display["delta"] = pd.to_numeric(
            df_display["delta"], errors="coerce"
        ).round(3)
    if "volume_total" in df_display.columns:
        df_display["volume_total"] = pd.to_numeric(
            df_display["volume_total"], errors="coerce"
        ).round(3)
    if "preco" in df_display.columns:
        df_display["preco"] = pd.to_numeric(
            df_display["preco"], errors="coerce"
        ).round(2)

    # Renomear colunas para legibilidade
    column_mapping = {
        "datetime_sp": "Horário (SP)",
        "ativo": "Ativo",
        "tipo_evento": "Tipo",
        "resultado_da_batalha": "Resultado",
        "preco": "Preço",
        "delta": "Δ (Delta)",
        "volume_total": "Vol (BTC)",
        "descricao": "Descrição",
    }
    df_display.rename(columns=column_mapping, inplace=True)

    st.dataframe(
        df_display,
        hide_index=True,
        use_container_width=True,
        height=min(400, len(df_display) * 35 + 40),
    )
else:
    st.warning("⚠️ Nenhuma coluna disponível para exibição na tabela.")

# 6. Estatísticas Resumidas
st.subheader("📊 Estatísticas Resumidas")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.metric("Total de Eventos (filtrados)", len(df_filtered))

with col_s2:
    if "ativo" in df_filtered.columns:
        unique_assets = df_filtered["ativo"].nunique()
        st.metric("Ativos Únicos", unique_assets)

with col_s3:
    if "tipo_evento" in df_filtered.columns:
        unique_types = df_filtered["tipo_evento"].nunique()
        st.metric("Tipos de Evento", unique_types)

with col_s4:
    if "volume_total" in df_filtered.columns:
        total_volume = pd.to_numeric(
            df_filtered["volume_total"], errors="coerce"
        ).sum()
        if not pd.isna(total_volume):
            st.metric("Volume Total (BTC)", f"{total_volume:.3f}")

# Rodapé
st.divider()
st.caption(
    f"💾 Dados carregados de: `{DB_PATH.absolute()}` | "
    f"Atualizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
st.sidebar.divider()
st.sidebar.caption("🛠️ Dashboard construído com Streamlit (SQLite + JSON payload parser)")