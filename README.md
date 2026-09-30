# PPIV — Proyecto Integrador 2026

## Ciencia de Datos e Inteligencia Artificial

# Inteligencia de Retención de Clientes

**Predicción de abandono · IA explicable · Segmentación · Priorización de acciones**

Proyecto integrador orientado a construir una solución completa de Ciencia de Datos aplicada a retención de clientes: desde la auditoría y limpieza de datos hasta el modelado predictivo, la explicabilidad, la segmentación, la priorización comercial y una aplicación interactiva desplegada en Streamlit.

## Aplicación desplegada

**Streamlit:** https://ppivproyectointegrador2026-3nmnzhsbpyrbvub9rtrwoc.streamlit.app/

La aplicación incluye dashboard ejecutivo, comparación de modelos, explicabilidad, segmentación, hipótesis, calidad de datos y una sección de **predicción en vivo** para clientes nuevos.

## Objetivo

Analizar el comportamiento de clientes de un negocio de comercio electrónico para:

- identificar patrones asociados al abandono (`churn`);
- estimar la probabilidad de abandono;
- explicar los factores que influyen en cada predicción;
- descubrir segmentos de clientes mediante aprendizaje no supervisado;
- priorizar acciones de retención según riesgo y valor comercial;
- convertir el análisis en una herramienta operativa para la toma de decisiones.

## Dataset

El dataset contiene **15.000 clientes** y **30 variables originales** relacionadas con:

- comportamiento de compra;
- marketing y engagement;
- satisfacción;
- soporte;
- navegación;
- valor económico;
- datos demográficos y temporales.

La tasa observada de abandono es de aproximadamente **15,3%**, por lo que el problema presenta desbalance de clases.

Durante la auditoría se detectaron, entre otros problemas:

- valores faltantes;
- edades inválidas;
- inconsistencias temporales;
- combinaciones país/ciudad poco confiables;
- variables con posible riesgo de fuga de información (`data leakage`).

El dataset crudo no se publica en el repositorio. Los resultados de la aplicación utilizan datos procesados, snapshots analíticos y muestras anonimizadas.

## Metodología

El flujo implementado sigue estas etapas:

1. Auditoría y calidad de datos.
2. Limpieza y preprocesamiento.
3. Análisis exploratorio de datos (EDA).
4. Estadística descriptiva.
5. Formulación y contraste de hipótesis.
6. Feature engineering.
7. Modelos baseline.
8. Modelos avanzados.
9. Validación cruzada y optimización de hiperparámetros.
10. Ensemble de modelos.
11. Explicabilidad con SHAP.
12. Segmentación no supervisada.
13. Motor de priorización de retención.
14. Aplicación Streamlit.
15. Predicción interactiva y scoring masivo por CSV.

## Hipótesis analizadas

### H1 — Baja satisfacción y abandono

Los clientes con satisfacción baja presentan una tasa de abandono sustancialmente mayor que los clientes con satisfacción media o alta.

- Grupo de riesgo: **49,7%**
- Grupo de referencia: **9,3%**
- Riesgo relativo: **5,33×**
- V de Cramér: **0,402**

### H2 — Fricción de soporte y abandono

Los clientes con 5 o más tickets de soporte presentan una tasa de abandono superior a quienes tienen entre 0 y 4 tickets.

- Grupo de riesgo: **50,4%**
- Grupo de referencia: **13,3%**
- Riesgo relativo: **3,79×**
- V de Cramér: **0,232**

### H3 — Bajo gasto y abandono

Los clientes pertenecientes al quintil inferior de gasto muestran una tasa de abandono considerablemente mayor que los clientes de los quintiles superiores.

- Grupo de riesgo: **40,5%**
- Grupo de referencia: **9,5%**
- Riesgo relativo: **4,25×**
- V de Cramér: **0,340**

> Los resultados se interpretan como asociaciones estadísticas y no como relaciones causales.

## Modelos implementados

Se compararon distintos algoritmos supervisados:

- DummyClassifier
- Regresión Logística
- Regresión Logística balanceada
- Random Forest
- Gradient Boosting
- XGBoost
- XGBoost optimizado
- CatBoost
- Ensemble de Gradient Boosting + XGBoost + CatBoost

Dado el desbalance de clases, la evaluación no se basa únicamente en accuracy. Se utilizaron métricas como:

- Sensibilidad (`recall`)
- Precision
- F1
- Balanced Accuracy
- ROC-AUC
- PR-AUC
- Matriz de confusión

## Resultados principales

El ensemble alcanzó aproximadamente:

- **ROC-AUC: 0,921**
- **PR-AUC: 0,547**
- **Recall: 96,5%**
- **F1: 0,675**

El análisis también mostró que el **20% de clientes con mayor prioridad concentra aproximadamente el 69,6% de los abandonos reales** del conjunto de test.

Esto permite enfocar recursos de retención sobre una fracción reducida de la cartera con una concentración de riesgo significativamente superior a una selección aleatoria.

## IA explicable

Se implementó explicabilidad global e individual mediante **SHAP**.

Las variables que aparecen de manera más consistente como relevantes son:

- satisfacción;
- gasto total;
- tickets de soporte;
- gasto por visita;
- tasa de tickets de soporte.

La aplicación permite analizar cómo distintas variables aumentan o reducen el riesgo estimado para clientes individuales.

## Segmentación no supervisada

Se implementó clustering sin utilizar la variable `churn` para formar los grupos.

Se identificaron cuatro perfiles interpretables:

- **Insatisfechos en riesgo**
- **Alta fricción de soporte**
- **Inactivos satisfechos**
- **Activos satisfechos**

Los clusters mostraron alta estabilidad entre semillas, con un **Adjusted Rand Index promedio superior a 0,96**.

## Motor de priorización de retención

Se construyó un puntaje operativo que combina:

- riesgo de abandono;
- valor comercial;
- riesgo asociado al segmento.

El objetivo no es solo predecir quién puede abandonar, sino determinar **a quién conviene priorizar primero** y qué acción sugerir.

Ejemplos de acciones:

- recuperación de experiencia;
- escalamiento de soporte;
- reactivación personalizada;
- seguimiento preventivo;
- fidelización.

## Predicción en vivo

La aplicación incluye una sección de inferencia interactiva donde se pueden modificar variables de un cliente y obtener:

- probabilidad de abandono;
- nivel de riesgo;
- acción recomendada;
- explicación local de la predicción.

También permite:

- descargar una plantilla CSV;
- subir nuevos clientes;
- ejecutar scoring masivo;
- descargar los resultados priorizados.

Para la inferencia interactiva se utiliza un modelo XGBoost operativo compacto. El ensemble completo se conserva como benchmark de máximo desempeño.

## Aplicación Streamlit

La app está organizada en las siguientes vistas:

- Resumen ejecutivo
- Riesgo y priorización
- Desempeño de modelos
- IA explicable
- Segmentación de clientes
- Hipótesis e insights
- Calidad de datos y EDA
- Predicción en vivo

## Estructura del repositorio

```text
PPIV_Proyecto_Integrador_2026/
├── app/
│   ├── data/                     # Snapshots y datos de la aplicación
│   ├── model/                    # Modelo operativo de inferencia
│   ├── pages/                    # Página de predicción en vivo
│   └── streamlit_app.py          # Dashboard principal
├── data/
│   ├── raw/                      # Datos originales (no versionados)
│   └── processed/                # Datos procesados
├── docs/                         # Documentación metodológica
├── models/                       # Modelos y artefactos analíticos
├── notebooks/                    # Desarrollo analítico paso a paso
├── reports/                      # Figuras, tablas e informe final
├── scripts/                      # Ejecuciones reproducibles
├── src/                          # Código reutilizable
├── .streamlit/
│   └── config.toml
├── .gitignore
├── README.md
└── requirements.txt
```

## Flujo analítico implementado

1. `01_data_audit.ipynb` — Auditoría inicial y calidad de datos.
2. `02_data_cleaning.ipynb` — Limpieza y preprocesamiento.
3. `03_eda.ipynb` — Análisis exploratorio.
4. `04_hypothesis_testing.ipynb` — Contraste de hipótesis.
5. `05_feature_engineering.ipynb` — Ingeniería y selección de variables.
6. `06_baseline_models.ipynb` — Modelos base.
7. `07_advanced_models.ipynb` — Modelos avanzados, optimización y ensemble.
8. `08_explainability_shap.ipynb` — Explicabilidad global e individual.
9. `09_customer_segmentation.ipynb` — Segmentación no supervisada.
10. `10_retention_engine.ipynb` — Priorización de clientes y acciones.

## Prevención de fuga de información

El proyecto aplica medidas explícitas para reducir riesgo de `data leakage`:

- split estratificado antes del preprocessing;
- imputación ajustada solo sobre train;
- exclusión de identificadores;
- revisión de variables con información potencialmente posterior al evento objetivo;
- separación entre variables explicativas y variables de control/calidad.

## Ejecución local

```bash
git clone https://github.com/ferfava/PPIV_Proyecto_Integrador_2026.git
cd PPIV_Proyecto_Integrador_2026
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Tecnologías

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- XGBoost
- CatBoost
- SHAP
- Plotly
- Streamlit
- GitHub

## Estado del proyecto

🟢 **Aplicación funcional y desplegada.**

Completado:

- auditoría;
- limpieza;
- EDA;
- hipótesis;
- feature engineering;
- modelos baseline y avanzados;
- ensemble;
- explicabilidad;
- segmentación;
- motor de retención;
- dashboard;
- predicción en vivo;
- scoring masivo.

En preparación:

- informe final;
- presentación oral;
- ajustes finales de calibración y umbral de negocio.
