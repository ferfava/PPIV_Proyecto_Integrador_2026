from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Decisión de negocio · Retención", page_icon="📈", layout="wide")

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

st.markdown("""
<style>
:root{
    --navy:#101828;
    --navy-2:#1d2939;
    --blue:#2457d6;
    --bg:#f5f7fb;
    --card:#ffffff;
    --border:#e5e9f2;
    --text:#172033;
    --muted:#667085;
}
.stApp{background:var(--bg);color:var(--text)}
.block-container{padding-top:1.4rem;max-width:1450px}
[data-testid="stSidebar"]{
    background:var(--navy)!important;
    border-right:1px solid #263449;
}
[data-testid="stSidebar"] *{color:#f8fafc!important}
[data-testid="stSidebarNav"] a{
    border-radius:10px!important;
    margin:.12rem .35rem!important;
    transition:background .15s ease!important;
}
[data-testid="stSidebarNav"] a:hover{background:var(--navy-2)!important}
[data-testid="stSidebarNav"] a[aria-current="page"]{
    background:var(--blue)!important;
    color:white!important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] *{color:white!important}
.hero{
    background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 100%);
    padding:1.6rem 1.8rem;border-radius:18px;color:white;margin-bottom:1rem;
    box-shadow:0 10px 30px rgba(31,41,55,.10)
}
.hero h1{margin:0 0 .35rem;color:white}
.hero p{margin:0;opacity:.88;color:white}
div[data-testid="stMetric"]{
    background:var(--card);border:1px solid var(--border);padding:1rem;border-radius:14px;
    box-shadow:0 3px 12px rgba(31,41,55,.04)
}
.note{
    background:var(--card);border:1px solid var(--border);border-left:4px solid var(--blue);
    border-radius:12px;padding:1rem;margin:.5rem 0;color:var(--text)
}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    cal = pd.read_csv(DATA / "calibration_comparison.csv")
    cap = pd.read_csv(DATA / "capacity_thresholds.csv")
    return cal, cap

cal, cap = load_data()

st.markdown('<div class="hero"><h1>Decisión de negocio basada en riesgo</h1><p>Calibración de probabilidades y política de intervención según capacidad operativa.</p></div>', unsafe_allow_html=True)

st.markdown("## 1. ¿La probabilidad del modelo es confiable?")
platt = cal.loc[cal["metodo"] == "Platt"].iloc[0]
raw = cal.loc[cal["metodo"] == "Sin calibrar"].iloc[0]

c1,c2,c3,c4 = st.columns(4)
c1.metric("ROC-AUC", f"{platt['roc_auc']:.3f}", help="Capacidad del modelo para ordenar correctamente clientes con y sin abandono.")
c2.metric("PR-AUC", f"{platt['pr_auc']:.3f}", help="Área bajo la curva Precision-Recall; especialmente útil con clases desbalanceadas.")
c3.metric(
    "Brier Score",
    f"{platt['brier']:.3f}",
    delta=f"{(platt['brier']-raw['brier']):.3f}",
    delta_color="inverse",
    help="Error cuadrático medio de las probabilidades. Menor es mejor.",
)
c4.metric(
    "ECE",
    f"{platt['ece_10_bins']:.3f}",
    delta=f"{(platt['ece_10_bins']-raw['ece_10_bins']):.3f}",
    delta_color="inverse",
    help="Expected Calibration Error: diferencia promedio entre probabilidad estimada y frecuencia observada. Menor es mejor.",
)

comp = cal.rename(columns={
    "metodo":"Método",
    "brier":"Brier Score",
    "ece_10_bins":"ECE",
    "roc_auc":"ROC-AUC",
    "pr_auc":"PR-AUC",
})
comp_long = comp.melt(
    id_vars="Método",
    value_vars=["Brier Score", "ECE"],
    var_name="Métrica de calibración",
    value_name="Error",
)
fig = px.bar(
    comp_long,
    x="Método",
    y="Error",
    color="Métrica de calibración",
    barmode="group",
    title="Calibración antes y después",
)
fig.update_layout(
    height=380,
    margin=dict(l=10,r=10,t=50,b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#344054"),
    legend_title_text="Métrica",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="note"><b>Interpretación.</b> Platt scaling mejora fuertemente la calibración sin modificar la capacidad de ranking del modelo. Esto permite usar la probabilidad como una magnitud más interpretable, no solo como un score de ordenamiento.</div>', unsafe_allow_html=True)

with st.expander("¿Qué significan Brier Score y ECE?"):
    st.markdown(
        "- **Brier Score:** mide el error cuadrático entre la probabilidad predicha y el resultado real. "
        "Un valor menor indica probabilidades más precisas.\n"
        "- **ECE (Expected Calibration Error):** compara, por intervalos de probabilidad, el riesgo estimado con la frecuencia observada. "
        "Un ECE cercano a 0 indica buena calibración.\n"
        "- **Calibrar no mejora necesariamente el ranking:** busca que un riesgo estimado del 30% sea interpretable aproximadamente como 30 de cada 100 casos similares."
    )

st.markdown("## 2. Política por capacidad de intervención")
options = {f"Top {int(r.capacidad*100)}%": r.capacidad for _, r in cap.iterrows()}
label = st.select_slider("Capacidad disponible para intervenir clientes", options=list(options.keys()), value="Top 20%")
selected = cap.loc[cap["capacidad"] == options[label]].iloc[0]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Clientes a contactar", f"{int(selected['clientes_seleccionados']):,}".replace(",","."))
c2.metric("Abandonos capturados", f"{selected['captura_abandono']*100:.1f}%")
c3.metric("Tasa de abandono del grupo", f"{selected['tasa_abandono_grupo']*100:.1f}%")
c4.metric("Umbral aproximado", f"{selected['threshold_aprox']:.3f}")

chart_df = cap.copy()
chart_df["Capacidad (%)"] = (chart_df["capacidad"] * 100).astype(int)
fig = px.line(
    chart_df,
    x="Capacidad (%)",
    y="captura_abandono",
    markers=True,
    title="Cobertura de abandonos según capacidad operativa",
)
fig.update_yaxes(tickformat=".0%", title="Abandonos capturados")
fig.update_xaxes(
    title="Porción de clientes intervenida",
    tickmode="array",
    tickvals=chart_df["Capacidad (%)"].tolist(),
    ticktext=[f"{v}%" for v in chart_df["Capacidad (%)"].tolist()],
)
fig.update_layout(height=390, margin=dict(l=10,r=10,t=50,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#344054"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("## 3. Lectura operativa")
if selected["capacidad"] <= .10:
    st.info("Estrategia conservadora: concentra recursos en una fracción pequeña de clientes con alta densidad de abandono.")
elif selected["capacidad"] <= .20:
    st.success("Estrategia equilibrada: permite capturar una proporción alta de abandonos con una intervención acotada.")
elif selected["capacidad"] <= .25:
    st.warning("Estrategia intensiva: aumenta fuertemente la cobertura de abandono a costa de contactar más clientes sin churn.")
else:
    st.warning("Estrategia de máxima cobertura: en este test alcanza prácticamente todos los abandonos, pero requiere intervenir una porción mayor de la cartera.")

st.markdown('<div class="note"><b>Principio de gobierno del modelo.</b> La probabilidad no decide por sí sola. El modelo estima riesgo; la política de negocio define cuántos clientes intervenir según presupuesto, costo de contacto y capacidad operativa.</div>', unsafe_allow_html=True)

st.markdown("## 4. Trazabilidad")
table = cap.copy()
table["capacidad"] *= 100
table["captura_abandono"] *= 100
table["tasa_abandono_grupo"] *= 100
table = table.rename(columns={
    "capacidad":"Capacidad (%)",
    "threshold_aprox":"Umbral aprox.",
    "clientes_seleccionados":"Clientes",
    "captura_abandono":"Captura abandono (%)",
    "tasa_abandono_grupo":"Tasa abandono grupo (%)",
    "tp":"TP","fp":"FP","fn":"FN"
})
st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Capacidad (%)": st.column_config.NumberColumn(format="%.0f"),
        "Captura abandono (%)": st.column_config.NumberColumn(format="%.1f"),
        "Tasa abandono grupo (%)": st.column_config.NumberColumn(format="%.1f"),
        "Umbral aprox.": st.column_config.NumberColumn(format="%.3f"),
    }
)
st.caption("Resultados calculados sobre el conjunto de test final. La elección de capacidad debe definirse con costos reales del negocio; los porcentajes mostrados no sustituyen una evaluación económica específica.")

st.markdown("---")
st.caption("Proyecto Integrador · Calibración probabilística · Decisión bajo restricción de capacidad")
