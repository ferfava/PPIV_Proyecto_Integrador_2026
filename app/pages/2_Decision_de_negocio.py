from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from ui import apply_theme, primary_navigation, project_footer_sidebar

st.set_page_config(page_title="Decisión de negocio · Retención", page_icon="📈", layout="wide")
DATA = BASE / "data"

apply_theme()
primary_navigation("decision")
project_footer_sidebar()


@st.cache_data
def load_data():
    cal = pd.read_csv(DATA / "calibration_comparison.csv")
    cap = pd.read_csv(DATA / "capacity_thresholds.csv")
    up_comp = pd.read_csv(DATA / "uplift_comparacion.csv")
    up_cuad = pd.read_csv(DATA / "uplift_cuadrantes.csv")
    up_dec = pd.read_csv(DATA / "uplift_deciles.csv")
    return cal, cap, up_comp, up_cuad, up_dec


cal, cap, up_comp, up_cuad, up_dec = load_data()

st.markdown('<div class="hero"><h1>Decisión de negocio basada en riesgo y capacidad</h1><p>Calibración de probabilidades, capacidad operativa y prueba de concepto de Uplift.</p></div>', unsafe_allow_html=True)

st.markdown("## 1. ¿La probabilidad del modelo es confiable?")
platt = cal.loc[cal["metodo"] == "Platt"].iloc[0]
raw = cal.loc[cal["metodo"] == "Sin calibrar"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC", f"{platt['roc_auc']:.3f}", help="Capacidad del modelo para ordenar correctamente clientes con y sin abandono.")
c2.metric("PR-AUC", f"{platt['pr_auc']:.3f}", help="Área bajo la curva Precision-Recall; especialmente útil con clases desbalanceadas.")
c3.metric("Brier Score", f"{platt['brier']:.3f}", delta=f"{(platt['brier']-raw['brier']):.3f}", delta_color="inverse", help="Error cuadrático medio de las probabilidades. Menor es mejor.")
c4.metric("ECE", f"{platt['ece_10_bins']:.3f}", delta=f"{(platt['ece_10_bins']-raw['ece_10_bins']):.3f}", delta_color="inverse", help="Expected Calibration Error. Menor es mejor.")

comp = cal.rename(columns={"metodo":"Método", "brier":"Brier Score", "ece_10_bins":"ECE", "roc_auc":"ROC-AUC", "pr_auc":"PR-AUC"})
comp_long = comp.melt(id_vars="Método", value_vars=["Brier Score", "ECE"], var_name="Métrica de calibración", value_name="Error")
fig = px.bar(comp_long, x="Método", y="Error", color="Métrica de calibración", barmode="group", title="Calibración antes y después")
fig.update_layout(height=380, margin=dict(l=10,r=10,t=50,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="Métrica", font=dict(color="#344054"))
st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="note"><b>Interpretación.</b> Platt scaling mejora fuertemente la calibración sin modificar la capacidad de ranking del modelo. Esto permite usar la probabilidad como una magnitud más interpretable, no solo como un score de ordenamiento.</div>', unsafe_allow_html=True)

with st.expander("¿Qué significan Brier Score y ECE?"):
    st.markdown("- **Brier Score:** mide el error cuadrático entre la probabilidad predicha y el resultado real. Menor es mejor.\n- **ECE:** compara el riesgo estimado con la frecuencia observada por intervalos. Un valor cercano a 0 indica buena calibración.\n- **Calibrar no mejora necesariamente el ranking:** busca que una probabilidad estimada tenga interpretación probabilística más fiel.")

st.markdown("## 2. Política por capacidad de intervención")
options = {f"Top {int(r.capacidad*100)}%": r.capacidad for _, r in cap.iterrows()}
label = st.select_slider("Capacidad disponible para intervenir clientes", options=list(options.keys()), value="Top 20%")
selected = cap.loc[cap["capacidad"] == options[label]].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes a contactar", f"{int(selected['clientes_seleccionados']):,}".replace(",","."))
c2.metric("Abandonos capturados", f"{selected['captura_abandono']*100:.1f}%")
c3.metric("Tasa de abandono del grupo", f"{selected['tasa_abandono_grupo']*100:.1f}%")
c4.metric("Umbral aproximado", f"{selected['threshold_aprox']:.3f}")

chart_df = cap.copy()
chart_df["Capacidad (%)"] = (chart_df["capacidad"] * 100).astype(int)
fig = px.line(chart_df, x="Capacidad (%)", y="captura_abandono", markers=True, title="Cobertura de abandonos según capacidad operativa")
fig.update_yaxes(tickformat=".0%", title="Abandonos capturados")
fig.update_xaxes(title="Porción de clientes intervenida", tickmode="array", tickvals=chart_df["Capacidad (%)"].tolist(), ticktext=[f"{v}%" for v in chart_df["Capacidad (%)"].tolist()])
fig.update_layout(height=390, margin=dict(l=10,r=10,t=50,b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#344054"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("## 3. Prueba de concepto Uplift: ¿cómo priorizar según respuesta potencial?")
st.warning(
    "Esta sección usa un experimento y resultados sintéticos generados por reglas basadas en satisfacción y soporte; no son resultados de una campaña real. "
    "Las diferencias y retenciones mostradas ilustran el método, no prueban impacto causal ni ROI. "
    "Validar con una prueba A/B de campo antes de tomar decisiones operativas."
)
st.markdown(
    '<div class="insight"><b>De la analítica predictiva a una hipótesis prescriptiva.</b><br>'
    'El modelo de churn tradicional responde <i>"¿quién tiene alta probabilidad de irse?"</i>, pero en ese grupo hay clientes que abandonan hagan lo que hagan '
    '(<i>Causas perdidas</i>) y clientes para quienes el contacto podría ser adverso (<i>Perros dormidos</i>).<br>'
    '<b>Uplift Modeling</b> busca estimar la respuesta incremental. Aquí los perfiles se obtienen en una simulación sintética y requieren validación experimental antes de orientar campañas.</div>',
    unsafe_allow_html=True,
)

# Comparación directa para la capacidad seleccionada
matched_up = up_comp[up_comp["fraccion_capacidad"] == label]
if not matched_up.empty:
    up_row = matched_up.iloc[0]
    st.markdown(f"### Rendimiento comparativo para capacidad **{label}** ({int(up_row['clientes_contactados']):,} clientes)")
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Retenciones simuladas (Churn)", f"{up_row['salvados_tradicional']:.1f}")
    u2.metric("Retenciones simuladas (Uplift)", f"{up_row['salvados_uplift']:.1f}")
    u3.metric("Diferencia simulada", f"{up_row['clientes_adicionales_salvados']:+.1f}", delta=f"{up_row['mejora_porcentual']}")
    u4.metric("Cambio relativo simulado", f"{up_row['mejora_porcentual']}", help="Cambio relativo en las retenciones estimadas dentro del escenario sintético.")

l_col, r_col = st.columns([1.2, 1])
with l_col:
    # Gráfico comparativo de retenciones estimadas en la simulación
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=up_comp["fraccion_capacidad"],
        y=up_comp["salvados_tradicional"],
        name="Modelo Churn (simulado)",
        marker_color="#94A3B8",
        text=up_comp["salvados_tradicional"].apply(lambda v: f"{v:.1f}"),
        textposition="auto"
    ))
    fig_comp.add_trace(go.Bar(
        x=up_comp["fraccion_capacidad"],
        y=up_comp["salvados_uplift"],
        name="Modelo Uplift (simulado)",
        marker_color="#2563EB",
        text=up_comp["salvados_uplift"].apply(lambda v: f"{v:.1f}"),
        textposition="auto"
    ))
    fig_comp.update_layout(
        title="Retenciones estimadas por política (escenario simulado)",
        barmode="group",
        height=380,
        margin=dict(l=10, r=10, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#344054"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_comp.update_yaxes(title="Retenciones estimadas en simulación", gridcolor="#e9edf5")
    fig_comp.update_xaxes(title="Capacidad operativa")
    st.plotly_chart(fig_comp, use_container_width=True)

with r_col:
    # Gráfico de deciles de uplift
    dec_chart = up_dec.copy()
    dec_chart["Decil"] = "Decil " + dec_chart["decil"].astype(str)
    fig_dec = go.Figure()
    fig_dec.add_trace(go.Bar(
        x=dec_chart["Decil"],
        y=dec_chart["churn_control"] * 100,
        name="Control (sin contacto)",
        marker_color="#EF4444"
    ))
    fig_dec.add_trace(go.Bar(
        x=dec_chart["Decil"],
        y=dec_chart["churn_tratados"] * 100,
        name="Tratados (intervenidos)",
        marker_color="#10B981"
    ))
    fig_dec.update_layout(
        title="Tasa de abandono por Decil de Uplift",
        barmode="group",
        height=380,
        margin=dict(l=10, r=10, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#344054"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_dec.update_layout(title="Abandono por decil (escenario simulado)")
    fig_dec.update_yaxes(title="Tasa de abandono (%)", gridcolor="#e9edf5")
    st.plotly_chart(fig_dec, use_container_width=True)

# Conclusión constructiva de la comparación de modelos
st.markdown(
    '<div class="note"><b>Lectura del ejercicio sintético: complementariedad y especialización.</b><br>'
    'No se trata de que un enfoque anule al otro, sino de una <b>evolución natural del análisis</b>:<br>'
    '• <b>El Modelo de Churn</b> cumple un rol inicial indispensable: detecta dónde está concentrado el riesgo y filtra al ~70% de clientes estables que no requieren atención.<br>'
    '• <b>La prueba de concepto Uplift</b> muestra cómo un ranking de respuesta incremental podría especializar esa priorización; sus categorías y resultados son sintéticos y no están validados en clientes reales.<br>'
    '<b>Resultado simulado:</b> Para una capacidad equivalente a 600 contactos, la comparación arroja <b>209 frente a 132 retenciones simuladas (+58,2%)</b>; no es una medición de impacto de negocio.</div>',
    unsafe_allow_html=True,
)

st.markdown("### Cuadrantes estimados en la simulación")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Persuadibles (sim.)", "23,8%", "714 clientes en test", help="Perfil potencialmente sensible a la intervención según los resultados sintéticos.")
q2.metric("Seguros (sim.)", "70,3%", "2.108 clientes en test", help="Perfil con baja respuesta estimada en la simulación.")
q3.metric("Causas perdidas (sim.)", "3,4%", "101 clientes en test", help="Perfil cuyo resultado sintético predice abandono aun con intervención.")
q4.metric("Perros dormidos (sim.)", "2,6%", "77 clientes en test", help="Perfil con posible efecto adverso según las reglas de simulación.")

with st.expander("¿Qué significa cada uno de los 4 cuadrantes?", expanded=True):
    ca, cb = st.columns(2)
    with ca:
        st.markdown(
            "- 🎯 **Persuadibles (23,8%):** El modelo los estima sensibles a la intervención bajo las reglas sintéticas; no se ha verificado que una campaña los retenga.\n"
            "- 🛡️ **Seguros / Sure Things (70,3%):** El resultado sintético predice baja respuesta a la intervención; no confirma el comportamiento real de estos clientes."
        )
    with cb:
        st.markdown(
            "- ❌ **Causas Perdidas / Lost Causes (3,4%):** El modelo predice abandono aun con intervención en este escenario sintético; no demuestra que el abandono sea inevitable.\n"
            "- 🚫 **Perros Dormidos / Sleeping Dogs (2,6%):** El modelo predice posible efecto adverso según las reglas simuladas; no demuestra que contactar acelere una baja real."
        )

# Tabla de técnicas comerciales según cuadrante
cuad_view = up_cuad.rename(columns={
    "cuadrante": "Cuadrante",
    "porcentaje": "Porcentaje (%)",
    "clientes": "Clientes (Test sintético)",
    "estrategia": "Estrategia Operativa",
    "accion_clave": "Técnica / Acción Clave"
})
st.dataframe(
    cuad_view,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Porcentaje (%)": st.column_config.NumberColumn(format="%.1f%%"),
        "Clientes (Test sintético)": st.column_config.NumberColumn(format="%d"),
        "Estrategia Operativa": st.column_config.TextColumn(width="large"),
        "Técnica / Acción Clave": st.column_config.TextColumn(width="medium")
    }
)

st.markdown("---")
st.markdown("## 4. Lectura operativa y decisión de capacidad")
st.caption(f"Interpretación comparativa para la capacidad seleccionada: **{label}**")

col_trad, col_up = st.columns(2)
with col_trad:
    st.markdown("#### Enfoque Tradicional (Cobertura de riesgo)")
    if selected["capacidad"] <= .10:
        st.info("Estrategia conservadora: concentra recursos en una fracción pequeña (300 clientes), capturando el 35,7% de los abandonos.")
    elif selected["capacidad"] <= .20:
        st.success("Estrategia equilibrada: permite capturar el 67,4% de los abandonos contactando a 600 clientes de riesgo.")
    elif selected["capacidad"] <= .25:
        st.warning("Estrategia intensiva: eleva la captura al 84,8% (750 clientes), pero aumenta fuertemente los falsos positivos.")
    else:
        st.warning("Estrategia de máxima cobertura: alcanza al 100% de los abandonos (900 clientes, FN = 0), pero contacta a 440 clientes sin churn.")

with col_up:
    st.markdown("#### Uplift (resultados sintéticos)")
    if selected["capacidad"] <= .10:
        st.info("En la simulación, el Decil 1 muestra una diferencia de +38,1% entre tratamiento y control.")
    elif selected["capacidad"] <= .20:
        st.success("En la simulación, el Top 20% arroja 209 frente a 132 retenciones estimadas por el enfoque tradicional (+58,2% relativo). Validar en campo antes de inferir rentabilidad.")
    elif selected["capacidad"] <= .25:
        st.warning("En la simulación, el rendimiento incremental disminuye a medida que se amplía la capacidad.")
    else:
        st.warning("En la simulación, las retenciones incrementales se estancan cerca de 210; esto no estima costos ni beneficios reales.")

st.markdown('<div class="note"><b>Principio de gobierno del modelo.</b> La probabilidad y el score no deciden por sí solos. El modelo de churn estima riesgo; el ranking uplift de esta prueba es sintético y no está validado en campo. Cualquier política de intervención debe contrastarse con resultados de un experimento real y costos observados.</div>', unsafe_allow_html=True)

st.markdown("## 5. Trazabilidad")
tab1, tab2 = st.tabs(["Capacidad y calibración tradicional", "Trazabilidad completa de Uplift"])

with tab1:
    table = cap.copy()
    table["capacidad"] *= 100
    table["captura_abandono"] *= 100
    table["tasa_abandono_grupo"] *= 100
    table = table.rename(columns={"capacidad":"Capacidad (%)", "threshold_aprox":"Umbral aprox.", "clientes_seleccionados":"Clientes", "captura_abandono":"Captura abandono (%)", "tasa_abandono_grupo":"Tasa abandono grupo (%)", "tp":"TP", "fp":"FP", "fn":"FN"})
    st.dataframe(table, use_container_width=True, hide_index=True, column_config={"Capacidad (%)": st.column_config.NumberColumn(format="%.0f"), "Captura abandono (%)": st.column_config.NumberColumn(format="%.1f"), "Tasa abandono grupo (%)": st.column_config.NumberColumn(format="%.1f"), "Umbral aprox.": st.column_config.NumberColumn(format="%.3f")})

with tab2:
    st.markdown("**Comparativa de políticas a distintas capacidades:**")
    comp_view = up_comp.rename(columns={
        "fraccion_capacidad": "Capacidad",
        "clientes_contactados": "Clientes",
        "salvados_tradicional": "Retenciones simuladas (Tradicional)",
        "salvados_uplift": "Retenciones simuladas (Uplift)",
        "clientes_adicionales_salvados": "Diferencia simulada",
        "mejora_porcentual": "Cambio relativo simulado"
    })
    st.dataframe(comp_view, use_container_width=True, hide_index=True)

    st.markdown("**Detalle de los 10 Deciles de Uplift:**")
    dec_view = up_dec.copy()
    dec_view["churn_control"] = (dec_view["churn_control"] * 100).round(2).astype(str) + "%"
    dec_view["churn_tratados"] = (dec_view["churn_tratados"] * 100).round(2).astype(str) + "%"
    dec_view["uplift_real"] = (dec_view["uplift_real"] * 100).round(2).astype(str) + "%"
    dec_view["uplift_predicho"] = (dec_view["uplift_predicho"] * 100).round(2).astype(str) + "%"
    dec_view = dec_view.rename(columns={
        "decil": "Decil",
        "clientes": "Clientes",
        "clientes_control": "En Control",
        "clientes_tratados": "En Tratamiento",
        "churn_control": "Abandono Control",
        "churn_tratados": "Abandono Tratados",
        "uplift_real": "Diferencia observada en simulación",
        "uplift_predicho": "Uplift Predicho"
    })
    st.dataframe(dec_view, use_container_width=True, hide_index=True)

st.caption("Métricas uplift calculadas sobre un test sintético de 3.000 clientes. La simulación no sustituye un A/B test real; la capacidad y la política deben definirse con costos y evidencia de campo.")

st.markdown("---")
st.caption("Proyecto Integrador · Calibración probabilística · Decisión bajo restricción de capacidad · Uplift Modeling")
