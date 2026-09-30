import streamlit as st


GLOBAL_CSS = """
<style>
:root{
  --bg:#F5F7FB;
  --surface:#FFFFFF;
  --sidebar:#101828;
  --sidebar-hover:#1D2939;
  --primary:#2457D6;
  --primary-2:#1D4ED8;
  --text:#172033;
  --muted:#667085;
  --border:#E5E9F2;
}
.stApp{background:var(--bg);color:var(--text)}
.block-container{padding-top:1.4rem;max-width:1500px}
[data-testid="stSidebar"]{background:var(--sidebar)}
[data-testid="stSidebar"] *{color:#F8FAFC}
[data-testid="stSidebarNav"]{display:none}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]{
  border-radius:9px;padding:.42rem .55rem;margin:.08rem 0;
}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover{
  background:var(--sidebar-hover)
}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][aria-current="page"]{
  background:var(--primary)
}
.hero{
  background:linear-gradient(135deg,#0F172A 0%,#1D4ED8 100%);
  padding:1.6rem 1.8rem;border-radius:18px;color:white;
  margin-bottom:1rem;box-shadow:0 10px 30px rgba(31,41,55,.10)
}
.hero h1{margin:0 0 .35rem;font-size:2rem;color:white}
.hero p{margin:0;opacity:.9;color:white}
.insight,.note{
  background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--primary);
  border-radius:12px;padding:.9rem 1rem;margin:.45rem 0
}
div[data-testid="stMetric"]{
  background:var(--surface);border:1px solid var(--border);padding:1rem;border-radius:14px;
  box-shadow:0 3px 12px rgba(31,41,55,.04)
}
.text-kpi,.segment-card{
  background:var(--surface);border:1px solid var(--border);padding:1rem;border-radius:14px;
  box-shadow:0 3px 12px rgba(31,41,55,.04)
}
.text-kpi{min-height:105px}.segment-card{min-height:125px}
.text-kpi span{display:block;font-size:.82rem;color:#475467;margin-bottom:.5rem}
.text-kpi strong{display:block;font-size:1.45rem;line-height:1.15;color:#101828}
.segment-card .name{font-size:.85rem;color:#344054;min-height:38px}
.segment-card .rate{font-size:1.8rem;color:#101828;margin:.3rem 0}
.segment-card .count{font-size:.85rem;color:#667085}
[data-testid="stDataFrame"], [data-testid="stTable"]{border-radius:12px;overflow:hidden}
hr{border-color:var(--border)}
</style>
"""


def apply_theme():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def primary_navigation(active: str):
    st.sidebar.markdown("## Retention Intelligence")
    st.sidebar.caption("PPIV · Ciencia de Datos e IA")
    st.sidebar.markdown("### Navegación principal")
    st.sidebar.page_link("streamlit_app.py", label="Analítica y monitoreo", icon="📊")
    st.sidebar.page_link("pages/1_Prediccion_en_vivo.py", label="Predicción en vivo", icon="🎯")
    st.sidebar.page_link("pages/2_Decision_de_negocio.py", label="Decisión de negocio", icon="📈")
    st.sidebar.markdown("---")


def project_footer_sidebar():
    st.sidebar.markdown("**PPIV · Proyecto Integrador 2026**")
    st.sidebar.caption("Ciencia de Datos e Inteligencia Artificial")
    st.sidebar.caption("Modelo, datos y decisiones con trazabilidad metodológica.")
