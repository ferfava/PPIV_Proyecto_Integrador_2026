# Checklist de cumplimiento — PPIV Proyecto Integrador 2026

Este documento traduce la consigna del trabajo práctico a entregables verificables dentro del repositorio.

## Consigna obligatoria

- [x] Dataset con problemas reales de calidad para trabajar limpieza y preprocessing.
- [x] Auditoría inicial de estructura, faltantes, duplicados, rangos e inconsistencias.
- [x] Limpieza y preprocesamiento documentado y reproducible.
- [x] Estadística descriptiva e interpretación.
- [x] Visualización de patrones relevantes.
- [x] Formulación de al menos 3 hipótesis.
- [x] Contraste estadístico de las hipótesis cuando corresponde.
- [x] Extracción de insights accionables preliminares.
- [x] Recomendaciones orientadas a potenciar retención, valor y/o ventas mediante motor de priorización.
- [x] Visualización final integrada y validada en Streamlit Cloud.
- [x] Modelos predictivos de clasificación implementados y evaluados.
- [x] Comparación de algoritmos avanzados con justificación técnica y de negocio.
- [x] Técnica no supervisada de segmentación implementada y evaluada.
- [ ] Documento funcional de alto nivel explicando el proceso completo.
- [x] Entrega de archivos fuente utilizados en los desarrollos versionados en el repositorio.
- [ ] Preparación de exposición final.

## Componentes que agregamos para elevar el proyecto

- [x] Prevención explícita de data leakage mediante fecha de corte y revisión de variables.
- [x] Feature engineering reproducible con pipeline y split estratificado previo al preprocessing.
- [x] Baseline interpretable con Regresión Logística.
- [x] Comparación de Random Forest, Gradient Boosting, XGBoost y CatBoost.
- [x] Validación cruzada estratificada y optimización de hiperparámetros.
- [x] Métricas adecuadas al desbalance: ROC-AUC, PR-AUC, recall, precision, F1, balanced accuracy y matriz de confusión.
- [x] Ensemble de probabilidades entre modelos de boosting.
- [x] Calibración de probabilidades con comparación Platt vs isotónica.
- [x] Optimización de la política de decisión según costo y capacidad operativa.
- [x] Explicabilidad global e individual con SHAP.
- [x] Segmentación no supervisada de clientes con análisis de estabilidad.
- [x] Motor de priorización de retención combinando riesgo, valor y segmento, validado con lift/capture rate.
- [ ] Evaluación de explicaciones contrafactuales si los datos y el modelo lo permiten.
- [x] Aplicación Streamlit desarrollada en español con vistas de EDA, riesgo, modelos, SHAP, segmentación y priorización.
- [x] Predicción individual en vivo y scoring masivo por CSV.
- [x] Despliegue y validación visual final en Streamlit Community Cloud.

## Orden de desarrollo

1. `01_data_audit.ipynb` — auditoría y calidad.
2. `02_data_cleaning.ipynb` — limpieza y preprocessing.
3. `03_eda.ipynb` / `scripts/run_eda.py` — EDA y visualizaciones.
4. `04_hypothesis_testing.ipynb` — hipótesis y pruebas.
5. `05_feature_engineering.ipynb` / `src/features/feature_engineering.py` — variables derivadas y preparación del modelado.
6. `06_baseline_models.ipynb` / `scripts/run_baseline_models.py` — baseline y evaluación inicial.
7. `07_advanced_models.ipynb` / `scripts/run_advanced_models.py` — modelos avanzados, optimización y ensemble.
8. `08_explainability_shap.ipynb` / `scripts/run_shap_explainability.py` — interpretabilidad global e individual.
9. `09_customer_segmentation.ipynb` / `scripts/run_customer_segmentation.py` — clustering, perfiles y estabilidad.
10. `10_retention_engine.ipynb` / `scripts/run_retention_engine.py` — priorización y decisión de negocio.
11. `11_calibration_threshold.ipynb` / `scripts/run_calibration_threshold.py` — calibración de probabilidades y política de decisión.
12. `app/streamlit_app.py` + `app/pages/1_Prediccion_en_vivo.py` — integración visual, inferencia y scoring masivo.
13. `reports/final_report/` — documento final.

## Regla de calidad del proyecto

No se considera una etapa terminada hasta que su código haya sido ejecutado de punta a punta con el dataset real sin errores y sus decisiones hayan sido documentadas.
