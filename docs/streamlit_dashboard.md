# Dashboard Streamlit — Inteligencia de Retención de Clientes

## Objetivo

La aplicación transforma los resultados analíticos del proyecto en una herramienta ejecutiva de apoyo a decisiones. El recorrido visual sigue una narrativa de negocio:

**Qué está pasando → quién está en riesgo → por qué → a quién priorizar → qué acción tomar.**

## Secciones

1. **Resumen ejecutivo** — KPIs, concentración del riesgo, segmentos y prioridades.
2. **Riesgo y priorización** — muestra anonimizada de clientes priorizados, filtros y acción recomendada.
3. **Desempeño de modelos** — comparación de ROC-AUC, PR-AUC, recall y F1.
4. **IA explicable** — importancia global SHAP y explicaciones individuales.
5. **Segmentación de clientes** — perfiles no supervisados y riesgo observado por segmento.
6. **Hipótesis e insights** — resultados estadísticos e interpretación de negocio.
7. **Calidad de datos y EDA** — trazabilidad del proceso analítico.

## Datos usados por la app

La aplicación no publica el CSV crudo original. Para la visualización se incluyen únicamente snapshots analíticos derivados y una muestra anonimizada del ranking de retención. Las métricas de desempeño corresponden a las ejecuciones validadas sobre el conjunto de test de 3.000 clientes.

## Diseño

- Interfaz completamente en español.
- Layout ancho y navegación lateral.
- Tarjetas KPI.
- Gráficos Plotly interactivos.
- Paleta sobria orientada a un dashboard corporativo.
- Descarga del ranking visible.
- Separación entre lectura ejecutiva y detalle metodológico.

## Validaciones realizadas

- El código de la app pasó validación sintáctica con Python.
- Todos los archivos de datos requeridos por el dashboard fueron verificados por nombre y esquema.
- `puntaje_prioridad` se validó en el rango 0–100.
- `probabilidad_abandono` se validó en el rango 0–1.
- Las métricas mostradas coinciden con las ejecuciones previamente validadas del pipeline.
- Los clientes disponibles para explicación SHAP están contenidos en la muestra visible del ranking.

## Ejecución local

Desde la raíz del repositorio:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Despliegue en Streamlit Community Cloud

1. Ingresar a Streamlit Community Cloud con la cuenta vinculada a GitHub.
2. Crear una nueva aplicación desde el repositorio `ferfava/PPIV_Proyecto_Integrador_2026`.
3. Seleccionar la rama `main`.
4. Indicar como archivo principal `app/streamlit_app.py`.
5. Desplegar.

`requirements.txt` ya contiene `streamlit`, `plotly`, `pandas` y el resto de dependencias del proyecto.

## Nota de reproducibilidad

El dashboard consume snapshots generados a partir de los notebooks y scripts versionados. El CSV crudo se mantiene fuera del repositorio público, mientras que el proceso que genera métricas, modelos, segmentación y priorización permanece documentado y reproducible dentro del proyecto.
