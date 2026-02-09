import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide",
)

# Tema y estilos personalizados
st.markdown(
    """
    <style>
        /* Fondo general */
        .stApp {
            background-color: #0f172a; /* azul/gris oscuro */
            color: #e5e7eb;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid #1f2937;
        }

        /* Títulos */
        h1, h2, h3, h4 {
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #e5e7eb;
        }

        /* Tarjetas de métricas */
        .metric-card {
            background: #020617;
            padding: 18px 20px;
            border-radius: 14px;
            border: 1px solid #1f2937;
            box-shadow: 0 10px 30px rgba(0,0,0,0.45);
        }
        .metric-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
            margin-top: 4px;
            color: #f9fafb;
        }
        .metric-sub {
            font-size: 0.9rem;
            color: #6b7280;
            margin-top: 2px;
        }

        /* Caja de factores / tabla */
        .panel-card {
            background: #020617;
            padding: 18px 20px;
            border-radius: 14px;
            border: 1px solid #1f2937;
        }

        /* DataFrame */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }

        /* Ocultar footer y menú de Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    base_path = Path(__file__).parent
    df = pd.read_csv(base_path / "data_forecast.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("### ⚙️ Configuración")
store_ids = sorted(df["store_nbr"].unique())
family_ids = sorted(df["family"].unique())

store_sel = st.sidebar.selectbox(
    "Tienda", store_ids, format_func=lambda x: f"Tienda {x}"
)
family_sel = st.sidebar.selectbox("Categoría", family_ids, format_func=str)

st.sidebar.markdown("---")
st.sidebar.markdown("**Información del modelo**")
st.sidebar.markdown("- Modelo: XGBoost GPU")
st.sidebar.markdown("- Horizonte: 15 días")
st.sidebar.markdown("- Métrica global RMSLE: **0.40**")
st.sidebar.markdown("- Métrica global WAPE: **16.9%**")

# =========================
# FILTRO PRINCIPAL
# =========================
df_sel = df[(df["store_nbr"] == store_sel) & (df["family"] == family_sel)].sort_values(
    "date"
)

if df_sel.empty:
    st.error("No hay datos para la combinación seleccionada.")
    st.stop()

total_real = df_sel["sales"].sum()
total_pred = df_sel["prediction"].sum()
wape_local = (
    abs(df_sel["sales"] - df_sel["prediction"]).sum() / (total_real + 1e-9)
) * 100
bias_local = ((total_pred - total_real) / (total_real + 1e-9)) * 100

# Rango temporal
start_date = df_sel["date"].min().date()
end_date = df_sel["date"].max().date()

# =========================
# CABECERA
# =========================
st.markdown("## 🚀 Sales Forecasting System")
st.markdown(
    f"**Tienda:** `{store_sel}` &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"**Categoría:** `{family_sel}` &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"**Período validación:** `{start_date}` → `{end_date}`"
)

# =========================
# MÉTRICAS (TARJETAS)
# =========================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Ventas reales (15 días)</div>
            <div class="metric-value">{total_real:,.0f}</div>
            <div class="metric-sub">Unidades totales</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Ventas predichas</div>
            <div class="metric-value">{total_pred:,.0f}</div>
            <div class="metric-sub">Unidades totales previstas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Precisión local (WAPE)</div>
            <div class="metric-value">{wape_local:.1f}%</div>
            <div class="metric-sub">Error absoluto sobre volumen de esta serie</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Sesgo (Bias)</div>
            <div class="metric-value">{bias_local:+.1f}%</div>
            <div class="metric-sub">Predicción vs. ventas reales</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# =========================
# GRÁFICO PRINCIPAL
# =========================
st.markdown("### 📈 Pronóstico vs. Realidad")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_sel["date"],
        y=df_sel["sales"],
        name="Venta real",
        mode="lines+markers",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=6),
    )
)
fig.add_trace(
    go.Scatter(
        x=df_sel["date"],
        y=df_sel["prediction"],
        name="Predicción (modelo)",
        mode="lines+markers",
        line=dict(color="#f97316", width=3, dash="dash"),
        marker=dict(size=6),
    )
)

fig.update_layout(
    height=420,
    margin=dict(l=0, r=10, t=10, b=0),
    hovermode="x unified",
    plot_bgcolor="rgba(15,23,42,1)",
    paper_bgcolor="rgba(15,23,42,0)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#e5e7eb"),
    ),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#9ca3af"),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(55,65,81,0.5)",
        zeroline=False,
        tickfont=dict(color="#9ca3af"),
    ),
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# BLOQUE INFERIOR
# =========================
st.markdown("")
c1, c2 = st.columns([1.2, 1.0])

with c1:
    st.markdown("#### 🔍 Factores que influyen en la serie")
    st.markdown(
        """
        <div class="panel-card">
        <ul>
            <li><b>Precio del petróleo:</b> correlacionado con el poder adquisitivo y la demanda agregada.</li>
            <li><b>Transacciones por tienda:</b> flujo de clientes, capturado mediante lags y medias móviles.</li>
            <li><b>Histórico de ventas:</b> lags y rolling windows (7–30 días) que capturan estacionalidad y tendencia.</li>
            <li><b>Feriados y fines de semana:</b> cambios de patrón en días no laborales.</li>
        </ul>
        <p style="color:#9ca3af; font-size:0.9rem;">
        El modelo está entrenado en escala logarítmica para estabilizar la varianza y optimizar la métrica RMSLE.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown("#### 📋 Detalle de inventario sugerido (15 días)")
    df_table = df_sel[["date", "prediction"]].copy()
    df_table.columns = ["Fecha", "Stock sugerido (unidades)"]
    df_table["Stock sugerido (unidades)"] = df_table["Stock sugerido (unidades)"].round(
        1
    )
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.dataframe(
        df_table.set_index("Fecha"),
        use_container_width=True,
        height=260,
    )
    st.markdown("<​/div>", unsafe_allow_html=True)
