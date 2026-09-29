from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Inteligencia de Retención de Clientes",
    page_icon="📊",
    layout="wide",
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

st.markdown("""
<style>
.stApp{background:#f5f7fb}
.block-container{padding-top:1.4rem;max-width:1500px}
[data-testid="stSidebar"]{background:#101828}
[data-testid="stSidebar"] *{color:#f8fafc}
.hero{
    background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 100%);
    padding:1.6rem 1.8rem;border-radius:18px;color:white;
    margin-bottom:1rem;box-shadow:0 10px 30px rgba(31,41,55,.10)
}
.hero h1{margin:0 0 .35rem;font-size:2rem}
.hero p{margin:0;opacity:.88}
.insight{
    background:white;border:1px solid #e5e9f2;border-left:4px solid #2457d6;
    border-radius:12px;padding:.9rem 1rem;margin:.45rem 0
}
div[data-testid="stMetric"]{
    background:white;border:1px solid #e5e9f2;padding:1rem;border-radius:14px;
    box-shadow:0 3px 12px rgba(31,41,55,.04)
}
.text-kpi{
    background:white;border:1px solid #e5e9f2;padding:1rem;border-radius:14px;
    box-shadow:0 3px 12px rgba(31,41,55,.04);min-height:105px
}
.text-kpi span{display:block;font-size:.82rem;color:#475467;margin-bottom:.5rem}
.text-kpi strong{display:block;font-size:1.45rem;line-height:1.15;color:#101828}
.segment-card{
    background:white;border:1px solid #e5e9f2;padding:1rem;border-radius:14px;
    box-shadow:0 3px 12px rgba(31,41,55,.04);min-height:125px
}
.segment-card .name{font-size:.85rem;color:#344054;min-height:38px}
.segment-card .rate{font-size:1.8rem;color:#101828;margin:.3rem 0}
.segment-card .count{font-size:.85rem;color:#667085}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def cargar():
    files = {
        "ranking": "ranking_retencion_top100.csv",
        "bandas": "bandas_prioridad.csv",
        "segmentos": "segmentos.csv",
        "evaluacion": "evaluacion_ranking.csv",
        "modelos": "modelos.csv",
        "hipotesis": "hipotesis.csv",
        "shap": "importancia_shap.csv",
        "shap_clientes": "shap_clientes_top100.csv",
        "eda_satisfaccion": "eda_satisfaccion.csv",
        "eda_tickets": "eda_tickets.csv",
        "eda_gasto": "eda_gasto.csv",
        "calidad": "calidad_datos.csv",
    }
    d = {k: pd.read_csv(DATA / v) for k, v in files.items()}
    d["meta"] = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    return d


def pct(x, d=1):
    return f"{x*100:.{d}f}%".replace(".", ",")


def num(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def hero(txt):
    st.markdown(
        f'<div class="hero"><h1>Inteligencia de Retención de Clientes</h1><p>{txt}</p></div>',
        unsafe_allow_html=True,
    )


def chart(fig, h=390):
    fig.update_layout(
        height=h,
        margin=dict(l=10, r=10, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#344054"),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e9edf5")
    return fig


def nombre_feature(x):
    x = str(x).replace("num__", "").replace("cat__", "")
    m = {
        "satisfaction_score": "Satisfacción",
        "total_spent": "Gasto total",
        "support_tickets": "Tickets de soporte",
        "spend_per_visit": "Gasto por visita",
        "support_ticket_rate": "Tasa de tickets de soporte",
        "avg_order_value": "Ticket promedio",
        "last_3_month_purchase_freq": "Frecuencia de compra (3 meses)",
        "recency_days": "Recencia de compra",
        "email_open_rate": "Apertura de email",
        "email_click_rate": "Clics de email",
        "email_engagement_ratio": "Interacción con emails",
        "avg_session_time": "Tiempo promedio de sesión",
        "total_visits": "Visitas totales",
        "pages_per_session": "Páginas por sesión",
        "customer_tenure_days": "Antigüedad del cliente",
        "nps_score": "NPS",
    }
    for k, v in m.items():
        if x == k or x.startswith(k):
            return v
    return x.replace("_", " ").capitalize()


d = cargar()
meta = d["meta"]
ranking = d["ranking"]

st.sidebar.markdown("### Navegación")
sec = st.sidebar.radio(
    "",
    [
        "Resumen ejecutivo",
        "Riesgo y priorización",
        "Desempeño de modelos",
        "IA explicable",
        "Segmentación de clientes",
        "Hipótesis e insights",
        "Calidad de datos y EDA",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**PPIV · Proyecto Integrador 2026**")
st.sidebar.caption("Ciencia de Datos e Inteligencia Artificial")
st.sidebar.caption(
    "Métricas validadas sobre 3.000 clientes de test. "
    "El ranking visible es una muestra anonimizada de clientes priorizados."
)

if sec == "Resumen ejecutivo":
    hero("Predicción de abandono, segmentación y priorización de acciones de retención")
    a, b, c, e = st.columns(4)
    a.metric("Clientes analizados", f"{meta['clientes']:,}".replace(",", "."))
    b.metric("Tasa de abandono", pct(meta["tasa_abandono"]))
    c.metric("ROC-AUC del ensemble", num(meta["roc_auc_ensemble"], 3))
    e.metric("Abandono capturado en Top 20%", pct(meta["captura_top_20"]))

    l, r = st.columns([1.2, 1])
    with l:
        st.markdown(
            f'<div class="insight"><b>Riesgo concentrado.</b><br>'
            f'El 20% de mayor prioridad concentra el <b>{pct(meta["captura_top_20"])}</b> '
            f'de los abandonos reales.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="insight"><b>Señales consistentes.</b><br>'
            'Satisfacción, gasto y tickets de soporte reaparecen en EDA, hipótesis, modelos y SHAP.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="insight"><b>Acciones diferenciadas.</b><br>'
            'La segmentación distingue insatisfacción, inactividad y fricción operativa.</div>',
            unsafe_allow_html=True,
        )
    with r:
        ev = d["evaluacion"].copy()
        ev["Top"] = (ev["fraccion_top"] * 100).astype(int).astype(str) + "%"
        f = px.line(ev, x="Top", y="captura_abandono", markers=True, title="Captura acumulada de abandono")
        f.update_yaxes(tickformat=".0%", title="Abandono capturado")
        f.update_xaxes(title="Porción priorizada")
        st.plotly_chart(chart(f), use_container_width=True)

    l, r = st.columns(2)
    with l:
        s = d["segmentos"].sort_values("tasa_abandono", ascending=False)
        f = px.bar(s, x="segmento", y="tasa_abandono", text_auto=".1%", title="Tasa de abandono por segmento")
        f.update_yaxes(tickformat=".0%", title="Tasa")
        f.update_xaxes(title="")
        st.plotly_chart(chart(f), use_container_width=True)
    with r:
        bnd = d["bandas"].rename(columns={"banda": "Banda", "clientes": "Clientes"})
        f = px.bar(bnd, x="Banda", y="Clientes", text_auto=True, title="Distribución de prioridad operativa")
        f.update_xaxes(title="")
        st.plotly_chart(chart(f), use_container_width=True)

elif sec == "Riesgo y priorización":
    hero("De la probabilidad de abandono a una lista operativa de acción")
    x, y, z = st.columns([1, 1.3, 1])
    bandas = x.multiselect("Banda de prioridad", ["Crítica", "Alta", "Media", "Baja"], default=["Crítica", "Alta"])
    segmentos = sorted(ranking["segmento"].dropna().unique())
    segs = y.multiselect("Segmento", segmentos, default=segmentos)
    n = z.slider("Cantidad de clientes", 10, 20, 20, 10)

    f = ranking[ranking["banda_prioridad"].isin(bandas) & ranking["segmento"].isin(segs)].head(n).copy()
    a, b, c, e = st.columns(4)
    a.metric("Clientes mostrados", len(f))
    b.metric("Riesgo medio", pct(f["probabilidad_abandono"].mean()) if len(f) else "—")
    c.metric("Prioridad media", num(f["puntaje_prioridad"].mean(), 1) if len(f) else "—")
    e.metric("Valor medio", num(f["score_valor"].mean(), 2) if len(f) else "—")

    t = f[["cliente_id", "banda_prioridad", "puntaje_prioridad", "probabilidad_abandono", "segmento", "gasto_total", "satisfaccion", "tickets_soporte", "accion_recomendada"]].rename(columns={"cliente_id": "Cliente", "banda_prioridad": "Prioridad", "puntaje_prioridad": "Puntaje", "probabilidad_abandono": "Prob. abandono", "segmento": "Segmento", "gasto_total": "Gasto total", "satisfaccion": "Satisfacción", "tickets_soporte": "Tickets soporte", "accion_recomendada": "Acción recomendada"})
    t["Satisfacción"] = t["Satisfacción"].apply(lambda v: "Sin dato" if pd.isna(v) else str(int(round(float(v)))))

    st.dataframe(
        t,
        use_container_width=True,
        hide_index=True,
        height=430,
        column_config={
            "Puntaje": st.column_config.NumberColumn(format="%.1f"),
            "Prob. abandono": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
            "Gasto total": st.column_config.NumberColumn(format="%.2f"),
            "Acción recomendada": st.column_config.TextColumn(width="large"),
            "Segmento": st.column_config.TextColumn(width="medium"),
        },
    )
    st.download_button("Descargar ranking visible", t.to_csv(index=False).encode("utf-8-sig"), "ranking_retencion.csv", "text/csv")
    with st.expander("¿Cómo se calcula el puntaje?"):
        st.write("65% riesgo de abandono + 25% valor comercial + 10% riesgo del segmento. Es un índice operativo de 0 a 100, no una probabilidad.")

elif sec == "Desempeño de modelos":
    hero("Comparación de modelos con métricas adecuadas para una clase desbalanceada")
    m = d["modelos"].copy()
    adv = m[m["modelo"].isin(["Logística balanceada", "Random Forest", "Gradient Boosting", "XGBoost", "XGBoost optimizado", "CatBoost", "Ensemble GB+XGB+CAT"])]

    a, b, c = st.columns(3)
    a.metric("ROC-AUC del ensemble", num(meta["roc_auc_ensemble"], 3))
    b.metric("PR-AUC del ensemble", num(meta["pr_auc_ensemble"], 3))
    c.metric("Sensibilidad (recall) del ensemble", pct(float(m.loc[m["modelo"] == "Ensemble GB+XGB+CAT", "recall"].iloc[0])))

    l, r = st.columns(2)
    with l:
        q = adv.melt(id_vars="modelo", value_vars=["roc_auc", "pr_auc"], var_name="Métrica", value_name="Valor")
        q["Métrica"] = q["Métrica"].map({"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
        f = px.bar(q, y="modelo", x="Valor", color="Métrica", barmode="group", orientation="h", title="Capacidad de discriminación")
        f.update_xaxes(range=[0, 1], title="")
        f.update_yaxes(title="")
        st.plotly_chart(chart(f, 430), use_container_width=True)

    with r:
        q = adv.melt(id_vars="modelo", value_vars=["recall", "f1"], var_name="Métrica", value_name="Valor")
        q["Métrica"] = q["Métrica"].map({"recall": "Sensibilidad (recall)", "f1": "F1"})
        f = px.bar(q, y="modelo", x="Valor", color="Métrica", barmode="group", orientation="h", title="Detección de abandono y equilibrio")
        f.update_xaxes(range=[0, 1], title="")
        f.update_yaxes(title="")
        st.plotly_chart(chart(f, 430), use_container_width=True)

    st.info("El DummyClassifier alcanza ~84,7% de exactitud (accuracy) pero detecta 0% de los abandonos. Por eso se priorizan sensibilidad (recall), F1, ROC-AUC y PR-AUC.")

    tabla_modelos = m[["modelo", "recall", "f1", "roc_auc", "pr_auc"]].rename(columns={"modelo": "Modelo", "recall": "Sensibilidad (recall)", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
    st.dataframe(
        tabla_modelos,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sensibilidad (recall)": st.column_config.NumberColumn(format="%.3f"),
            "F1": st.column_config.NumberColumn(format="%.3f"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
        },
    )

elif sec == "IA explicable":
    hero("Explicaciones globales e individuales para entender por qué el modelo asigna riesgo")
    s = d["shap"].copy()
    s["Variable"] = s["variable"].map(nombre_feature)
    f = px.bar(s.sort_values("importancia_shap"), x="importancia_shap", y="Variable", orientation="h", title="Importancia global según SHAP")
    f.update_xaxes(title="Impacto SHAP medio absoluto")
    f.update_yaxes(title="")
    st.plotly_chart(chart(f, 360), use_container_width=True)

    cli = d["shap_clientes"]
    cid = st.selectbox("Seleccionar cliente de alto riesgo", cli["cliente_id"].tolist())
    row = cli[cli["cliente_id"] == cid].iloc[0]
    det = ranking[ranking["cliente_id"] == cid].iloc[0]

    a, b, c, e = st.columns(4)
    a.metric("Probabilidad de abandono", pct(row["probabilidad_abandono"]))
    b.metric("Puntaje de prioridad", num(row["puntaje_prioridad"], 1))
    c.markdown(f'<div class="text-kpi"><span>Segmento</span><strong>{row["segmento"]}</strong></div>', unsafe_allow_html=True)
    e.metric("Satisfacción", "Sin dato" if pd.isna(det["satisfaccion"]) else num(det["satisfaccion"], 0))

    fac = pd.DataFrame([{"Variable": nombre_feature(row[f"variable_{i}"]), "Impacto SHAP": row[f"shap_{i}"]} for i in range(1, 6)])
    fac["Efecto"] = fac["Impacto SHAP"].apply(lambda v: "Aumenta el riesgo" if v > 0 else "Reduce el riesgo")
    fac = fac.sort_values("Impacto SHAP")
    f = px.bar(fac, x="Impacto SHAP", y="Variable", color="Efecto", orientation="h", title=f"Factores que explican la predicción de {cid}")
    f.add_vline(x=0, line_dash="dash", line_color="#98a2b3")
    st.plotly_chart(chart(f, 360), use_container_width=True)
    st.caption("SHAP positivo aumenta el riesgo estimado; SHAP negativo lo reduce. Para explicabilidad se usa XGBoost, con desempeño prácticamente equivalente al ensemble y TreeSHAP fiel y aditivo.")

elif sec == "Segmentación de clientes":
    hero("Perfiles no supervisados para diferenciar necesidades y acciones de retención")
    s = d["segmentos"].sort_values("tasa_abandono", ascending=False)
    cols = st.columns(4)
    for col, (_, r) in zip(cols, s.iterrows()):
        col.markdown(
            f'<div class="segment-card"><div class="name">{r["segmento"]}</div><div class="rate">{pct(r["tasa_abandono"])}</div><div class="count">{int(r["clientes"]):,} clientes</div></div>'.replace(",", "."),
            unsafe_allow_html=True,
        )

    l, r = st.columns(2)
    with l:
        f = px.bar(s, x="segmento", y="tasa_abandono", text_auto=".1%", title="Riesgo observado por segmento")
        f.update_yaxes(tickformat=".0%", title="Tasa de abandono")
        f.update_xaxes(title="")
        st.plotly_chart(chart(f), use_container_width=True)

    with r:
        f = px.scatter(
            s,
            x="satisfaccion",
            y="tickets_soporte",
            size="clientes",
            color="tasa_abandono",
            hover_name="segmento",
            labels={"tasa_abandono": "Tasa de abandono", "satisfaccion": "Satisfacción media", "tickets_soporte": "Tickets de soporte promedio"},
            title="Mapa de fricción y satisfacción",
        )
        st.plotly_chart(chart(f), use_container_width=True)

    st.caption("Los clusters se generaron sin usar la variable de abandono. La tasa de abandono se observó después para interpretar los perfiles. Estabilidad entre semillas: ARI promedio > 0,96.")

elif sec == "Hipótesis e insights":
    hero("Evidencia estadística que conecta comportamiento, experiencia y abandono")
    for _, r in d["hipotesis"].iterrows():
        comp = str(r["comparacion"])
        comp = comp.replace("5+ tickets vs 0–4", "5 o más tickets frente a 0–4")
        comp = comp.replace(" vs ", " frente a ")
        with st.container(border=True):
            st.markdown(f"### {r['hipotesis']} · {r['titulo']}")
            a, b, c, e = st.columns(4)
            a.metric("Grupo de riesgo", pct(r["tasa_grupo_riesgo"]))
            b.metric("Grupo de referencia", pct(r["tasa_referencia"]))
            c.metric("Riesgo relativo", num(r["risk_ratio"], 2) + "×")
            e.metric("V de Cramér", num(r["cramers_v"], 3))
            st.caption(f"Comparación: {comp}. Resultado: {r['resultado']}. Se interpreta como asociación estadística, no como causalidad.")

    st.success("Los resultados convergen: menor satisfacción, mayor fricción de soporte y menor gasto están asociados con mayor abandono. Las mismas señales reaparecen en EDA, pruebas estadísticas, modelos y SHAP.")

else:
    hero("Trazabilidad del análisis: desde la calidad del dato hasta los patrones exploratorios")
    a, b, c, e = st.columns(4)
    a.metric("Filas originales", "15.000")
    b.metric("Variables originales", "30")
    c.metric("Duplicados completos", "0")
    e.metric("Inconsistencias temporales", "3.762")

    with st.expander("Auditoría de calidad de datos"):
        st.dataframe(d["calidad"].rename(columns={"indicador": "Indicador", "valor": "Valor"}), use_container_width=True, hide_index=True)
        st.caption("La imputación estadística se realiza dentro del pipeline de modelado para evitar fuga de información (data leakage).")

    cols = st.columns(3)
    for col, key, x, tit in zip(
        cols,
        ["eda_satisfaccion", "eda_tickets", "eda_gasto"],
        ["satisfaccion_grupo", "tickets_grupo", "gasto_quintil"],
        ["Abandono por satisfacción", "Abandono por tickets de soporte", "Abandono por quintil de gasto"],
    ):
        with col:
            f = px.bar(d[key], x=x, y="tasa_abandono", text_auto=".1%", title=tit)
            f.update_yaxes(tickformat=".0%", title="Tasa")
            f.update_xaxes(title="")
            st.plotly_chart(chart(f, 330), use_container_width=True)

    st.write("Se detectaron edades inválidas, faltantes, inconsistencias temporales y combinaciones país/ciudad poco confiables. No se corrigieron relaciones geográficas sin una fuente autoritativa. También se excluyeron variables con potencial de fuga de información (data leakage) o valor operativo ambiguo.")

st.markdown("---")
st.caption("Proyecto Integrador · Ciencia de Datos e Inteligencia Artificial · Streamlit + Plotly")
