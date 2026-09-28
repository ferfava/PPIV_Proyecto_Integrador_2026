# Checklist de cumplimiento — PPIV Proyecto Integrador 2026

Este documento traduce la consigna del trabajo práctico a entregables verificables dentro del repositorio.

## Consigna obligatoria

- [x] Dataset con problemas reales de calidad para trabajar limpieza y preprocessing.
- [x] Auditoría inicial de estructura, faltantes, duplicados, rangos e inconsistencias.
- [ ] Limpieza y preprocesamiento documentado y reproducible.
- [ ] Estadística descriptiva e interpretación.
- [ ] Visualización de patrones relevantes.
- [ ] Formulación de al menos 3 hipótesis.
- [ ] Contraste estadístico de las hipótesis cuando corresponda.
- [ ] Extracción de insights accionables.
- [ ] Recomendaciones orientadas a potenciar retención, valor y/o ventas.
- [ ] Visualización final integrada en Streamlit.
- [ ] Modelos predictivos de clasificación y/o técnicas no supervisadas.
- [ ] Comparación de algoritmos con justificación técnica y de negocio.
- [ ] Documento funcional de alto nivel explicando el proceso completo.
- [ ] Entrega de archivos fuente utilizados en los desarrollos.
- [ ] Preparación de exposición final.

## Componentes que agregamos para elevar el proyecto

- [ ] Prevención explícita de data leakage mediante fecha de corte y revisión de variables.
- [ ] Baseline interpretable con Regresión Logística.
- [ ] Comparación de Random Forest, Gradient Boosting/XGBoost y CatBoost.
- [ ] Validación cruzada y optimización de hiperparámetros.
- [ ] Métricas adecuadas al desbalance: ROC-AUC, PR-AUC, recall, precision, F1 y matriz de confusión.
- [ ] Calibración de probabilidades.
- [ ] Optimización del umbral según costo de negocio.
- [ ] Explicabilidad global e individual con SHAP.
- [ ] Segmentación no supervisada de clientes.
- [ ] Motor de priorización de retención combinando riesgo y valor del cliente.
- [ ] Evaluación de explicaciones contrafactuales si los datos y el modelo lo permiten.
- [ ] Aplicación Streamlit con vistas de EDA, riesgo, explicación y priorización.

## Orden de desarrollo

1. `01_data_audit.ipynb` — auditoría y calidad.
2. `02_data_cleaning.ipynb` — limpieza y preprocessing.
3. `03_eda.ipynb` — EDA y visualizaciones.
4. `04_hypothesis_testing.ipynb` — hipótesis y pruebas.
5. `05_feature_engineering.ipynb` — variables derivadas y preparación del modelado.
6. `06_baseline_models.ipynb` — baseline y evaluación inicial.
7. `07_advanced_models.ipynb` — modelos avanzados y optimización.
8. `08_explainability_shap.ipynb` — interpretabilidad.
9. `09_customer_segmentation.ipynb` — clustering/segmentación.
10. `10_retention_engine.ipynb` — priorización y decisión de negocio.
11. `app/streamlit_app.py` — integración visual y funcional.
12. `reports/final_report/` — documento final.

## Regla de calidad del proyecto

No se considera una etapa terminada hasta que su código haya sido ejecutado de punta a punta con el dataset real sin errores y sus decisiones hayan sido documentadas.
