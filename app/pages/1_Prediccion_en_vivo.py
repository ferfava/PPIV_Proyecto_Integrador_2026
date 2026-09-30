from pathlib import Path
import json
import math
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from ui import apply_theme, primary_navigation, project_footer_sidebar

st.set_page_config(page_title="Predicción en vivo · Retención", page_icon="🎯", layout="wide")
MODEL_DIR = BASE / "model"

apply_theme()
primary_navigation("live")
project_footer_sidebar()


@st.cache_resource
def load_assets():
    model_json = json.loads((MODEL_DIR / "xgb_live_model.json").read_text(encoding="utf-8"))
    meta = json.loads((MODEL_DIR / "live_model_meta.json").read_text(encoding="utf-8"))
    learner = model_json["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    base_score_raw = learner["learner_model_param"].get("base_score", "[5E-1]")
    base_score = float(str(base_score_raw).strip("[]"))
    return trees, base_score, meta


TREES, BASE_SCORE, meta = load_assets()
FEATURES = meta["features"]
MEDIANS = meta["medians"]

LABELS = {
    "satisfaction_score": "Satisfacción (1–5)",
    "total_spent": "Gasto total",
    "support_tickets": "Tickets de soporte",
    "last_3_month_purchase_freq": "Compras en últimos 3 meses",
    "avg_order_value": "Ticket promedio",
    "total_visits": "Visitas totales",
    "email_open_rate": "Apertura de emails",
    "email_click_rate": "Clics en emails",
    "avg_session_time": "Tiempo promedio de sesión",
    "pages_per_session": "Páginas por sesión",
}


def _logit(p):
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def _sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _tree_leaf(tree, values):
    node = 0
    left = tree["left_children"]
    right = tree["right_children"]
    split_indices = tree["split_indices"]
    split_conditions = tree["split_conditions"]
    default_left = tree.get("default_left", [1] * len(left))
    while left[node] != -1:
        fidx = int(split_indices[node])
        threshold = float(split_conditions[node])
        value = values[fidx]
        if pd.isna(value):
            node = int(left[node] if default_left[node] else right[node])
        elif float(value) < threshold:
            node = int(left[node])
        else:
            node = int(right[node])
    return float(split_conditions[node])


def predict_probability_matrix(x):
    arr = x[FEATURES].to_numpy(dtype=float)
    out = []
    base_margin = _logit(BASE_SCORE)
    for row in arr:
        margin = base_margin
        for tree in TREES:
            margin += _tree_leaf(tree, row)
        out.append(_sigmoid(margin))
    return np.asarray(out, dtype=float)


def prepare(df):
    x = df.copy()
    missing = [c for c in FEATURES if c not in x.columns]
    for c in missing:
        x[c] = MEDIANS[c]
    x = x[FEATURES].apply(pd.to_numeric, errors="coerce")
    for c in FEATURES:
        x[c] = x[c].fillna(MEDIANS[c])
    for c in ["email_open_rate", "email_click_rate"]:
        if len(x) and x[c].max() > 1:
            x[c] = x[c] / 100.0
    return x, missing


def risk_band(p):
    if p >= .80:
        return "Crítico"
    if p >= .60:
        return "Alto"
    if p >= .35:
        return "Medio"
    return "Bajo"


def action(row, p):
    if row["satisfaction_score"] <= 2:
        return "Recuperación de experiencia: contacto personalizado y resolución de causa"
    if row["support_tickets"] >= 5:
        return "Escalamiento de soporte y seguimiento preventivo"
    if row["last_3_month_purchase_freq"] <= 3:
        return "Campaña de reactivación con oferta personalizada"
    if p >= .60:
        return "Seguimiento preventivo multicanal"
    return "Mantener estrategia de fidelización"


def predict(df):
    x, missing = prepare(df)
    score = predict_probability_matrix(x)
    out = df.copy()
    out["probabilidad_abandono"] = score
    out["nivel_riesgo"] = [risk_band(p) for p in score]
    out["accion_recomendada"] = [action(x.iloc[i], p) for i, p in enumerate(score)]
    return out, x, missing


def local_perturbation_explanation(x_one, p_full):
    rows = []
    for feature in FEATURES:
        alt = x_one.copy()
        alt.loc[alt.index[0], feature] = MEDIANS[feature]
        p_alt = float(predict_probability_matrix(alt)[0])
        impact = p_full - p_alt
        rows.append({"Variable": LABELS[feature], "Impacto local": impact, "Efecto": "Aumenta el riesgo" if impact >= 0 else "Reduce el riesgo"})
    exp = pd.DataFrame(rows)
    exp = exp.reindex(exp["Impacto local"].abs().sort_values(ascending=False).index).head(6)
    return exp.sort_values("Impacto local")


st.markdown('<div class="hero"><h1>Predicción de abandono en vivo</h1><p>Aplicación del modelo XGBoost sobre clientes nuevos, con explicación individual y scoring masivo por CSV.</p></div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC-AUC validado", f"{meta['metrics']['roc_auc']:.3f}")
m2.metric("PR-AUC validado", f"{meta['metrics']['pr_auc']:.3f}")
m3.metric("Sensibilidad (recall)", f"{meta['metrics']['recall']*100:.1f}%")
m4.metric("Clientes de test", f"{meta['test_rows']:,}".replace(",", "."))
st.caption("Este es el modelo operativo compacto para inferencia. El ensemble completo se mantiene como benchmark de máximo desempeño en la vista analítica.")

st.info("Gobierno del modelo: esta pantalla muestra el score del modelo operativo para explorar clientes nuevos. La calibración probabilística y la política formal de intervención se validan por separado en ‘Decisión de negocio’; por eso no se usa un corte universal de 50%.")

tab1, tab2 = st.tabs(["Predicción individual", "Predicción masiva por CSV"])

with tab1:
    st.subheader("Simular un cliente")
    c1, c2, c3, c4 = st.columns(4)
    satisf = c1.slider("Satisfacción", 1, 5, 4)
    spent = c2.number_input("Gasto total", min_value=0.0, value=500.0, step=25.0)
    tickets = c3.number_input("Tickets de soporte", min_value=0, value=2, step=1)
    freq = c4.number_input("Compras en últimos 3 meses", min_value=0, value=7, step=1)
    c1, c2, c3 = st.columns(3)
    aov = c1.number_input("Ticket promedio", min_value=0.0, value=60.0, step=5.0)
    visits = c2.number_input("Visitas totales", min_value=0, value=15, step=1)
    session = c3.number_input("Tiempo promedio de sesión", min_value=0.0, value=8.0, step=.5)
    c1, c2, c3 = st.columns(3)
    pages = c1.number_input("Páginas por sesión", min_value=0.0, value=4.0, step=.25)
    open_rate = c2.slider("Apertura de emails (%)", 0, 100, 50)
    click_rate = c3.slider("Clics en emails (%)", 0, 100, 25)

    row = pd.DataFrame([{"satisfaction_score": satisf, "total_spent": spent, "support_tickets": tickets, "last_3_month_purchase_freq": freq, "avg_order_value": aov, "total_visits": visits, "email_open_rate": open_rate / 100, "email_click_rate": click_rate / 100, "avg_session_time": session, "pages_per_session": pages}])
    pred, x, _ = predict(row)
    p = float(pred.loc[0, "probabilidad_abandono"])
    band = pred.loc[0, "nivel_riesgo"]
    rec = pred.loc[0, "accion_recomendada"]

    st.markdown("### Resultado")
    a, b, c = st.columns(3)
    a.metric("Score de riesgo del modelo", f"{p*100:.1f}%")
    b.metric("Banda orientativa", band)
    c.metric("Política operativa", "Ranking + capacidad")

    if p >= .60:
        st.error(f"Acción sugerida: {rec}")
    elif p >= .35:
        st.warning(f"Acción sugerida: {rec}")
    else:
        st.success(f"Acción sugerida: {rec}")

    st.caption("Las bandas son una ayuda de lectura para la simulación individual. La decisión operativa final se define mediante ranking, calibración y capacidad disponible.")

    exp = local_perturbation_explanation(x, p)
    fig = px.bar(exp, x="Impacto local", y="Variable", color="Efecto", orientation="h", title="Qué variables mueven este score respecto de un cliente típico")
    fig.add_vline(x=0, line_dash="dash", line_color="#98a2b3")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#344054"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Explicación local por perturbación: para cada variable se reemplaza su valor por la mediana de entrenamiento y se mide cuánto cambia el score. La sección IA explicable de la vista analítica conserva el análisis SHAP formal.")

with tab2:
    st.subheader("Scoring masivo")
    st.write("Subí un CSV con las variables del modelo. Si falta alguna columna, la app completa ese campo con la mediana del conjunto de entrenamiento y lo informa.")
    template = pd.DataFrame([{c: MEDIANS[c] for c in FEATURES}])
    st.download_button("Descargar plantilla CSV", template.to_csv(index=False).encode("utf-8-sig"), "plantilla_prediccion_churn.csv", "text/csv")
    file = st.file_uploader("Archivo CSV de clientes", type=["csv"])
    if file is not None:
        try:
            raw = pd.read_csv(file)
            if len(raw) == 0:
                st.warning("El archivo no contiene filas.")
            else:
                scored, _, missing = predict(raw)
                if missing:
                    st.warning("Columnas ausentes completadas con medianas de entrenamiento: " + ", ".join(LABELS[c] for c in missing))
                st.success(f"Se procesaron {len(scored):,} clientes.".replace(",", "."))
                a, b, c = st.columns(3)
                a.metric("Score medio", f"{scored['probabilidad_abandono'].mean()*100:.1f}%")
                b.metric("Bandas alta/crítica", int(scored["nivel_riesgo"].isin(["Alto", "Crítico"]).sum()))
                c.metric("Banda crítica", int((scored["nivel_riesgo"] == "Crítico").sum()))
                show = scored.sort_values("probabilidad_abandono", ascending=False).copy()
                st.dataframe(show.head(200), use_container_width=True, hide_index=True, column_config={"probabilidad_abandono": st.column_config.ProgressColumn("Score de riesgo", min_value=0, max_value=1, format="%.1f%%")})
                st.download_button("Descargar resultados", show.to_csv(index=False).encode("utf-8-sig"), "clientes_scoring_churn.csv", "text/csv")
        except Exception as exc:
            st.error(f"No se pudo procesar el archivo: {exc}")

with st.expander("Variables requeridas por el modelo"):
    st.dataframe(pd.DataFrame({"Columna técnica": FEATURES, "Descripción": [LABELS[f] for f in FEATURES], "Valor por defecto": [MEDIANS[f] for f in FEATURES]}), hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("Proyecto Integrador · Modelo operativo XGBoost · Inferencia sobre datos nuevos")
