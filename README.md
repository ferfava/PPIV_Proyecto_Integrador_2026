# PPIV: Proyecto Integrador 2026

## Ciencia de Datos e Inteligencia Artificial

### Customer Churn & Marketing Intelligence

Proyecto integrador orientado al análisis exploratorio de datos, calidad y preprocesamiento, formulación de hipótesis, modelado predictivo, segmentación de clientes, explicabilidad de modelos y construcción de una aplicación interactiva para apoyar decisiones de retención.

## Objetivo

Analizar el comportamiento de clientes de un negocio de comercio electrónico para identificar patrones asociados al abandono (`churn`), estimar el riesgo de baja y transformar las predicciones en acciones de negocio priorizadas.

## Alcance académico

El proyecto contempla:

- Auditoría y calidad de datos.
- Limpieza y preprocesamiento.
- Estadística descriptiva.
- Análisis exploratorio de datos (EDA).
- Visualización de patrones.
- Formulación y contraste de al menos tres hipótesis.
- Feature engineering.
- Modelos supervisados de clasificación.
- Comparación de modelos y métricas adecuadas para clases desbalanceadas.
- Interpretabilidad mediante SHAP.
- Segmentación de clientes con técnicas no supervisadas.
- Motor de priorización de retención orientado a negocio.
- Aplicación interactiva con Streamlit.
- Documento final y exposición.

## Dataset

Dataset de clientes con variables demográficas, comportamiento de compra, marketing, engagement, soporte, satisfacción y valor económico.

La versión original se conserva localmente en `data/raw/` y no se versiona en GitHub. Los datos procesados necesarios para reproducibilidad se documentarán durante el desarrollo.

## Estructura del repositorio

```text
PPIV_Proyecto_Integrador_2026/
├── app/               # Aplicación Streamlit
├── data/
│   ├── raw/           # Datos originales (no versionados)
│   └── processed/     # Datos procesados
├── docs/              # Notas y documentación del proyecto
├── models/            # Modelos serializados
├── notebooks/         # Desarrollo analítico paso a paso
├── reports/           # Figuras, tablas e informe final
├── src/               # Código reutilizable
├── .gitignore
├── README.md
└── requirements.txt
```

## Notebooks previstos

1. `01_data_audit.ipynb` — Auditoría inicial y calidad de datos.
2. `02_data_cleaning.ipynb` — Limpieza y preprocesamiento.
3. `03_eda.ipynb` — Análisis exploratorio.
4. `04_hypothesis_testing.ipynb` — Contraste de hipótesis.
5. `05_feature_engineering.ipynb` — Ingeniería y selección de variables.
6. `06_baseline_models.ipynb` — Modelos base.
7. `07_advanced_models.ipynb` — Modelos avanzados y optimización.
8. `08_explainability_shap.ipynb` — Explicabilidad global e individual.
9. `09_customer_segmentation.ipynb` — Segmentación no supervisada.
10. `10_retention_engine.ipynb` — Priorización de clientes y acciones.

## Estado

🟡 En desarrollo — PPIV 2026.
