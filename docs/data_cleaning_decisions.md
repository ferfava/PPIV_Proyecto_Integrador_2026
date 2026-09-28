# Decisiones de limpieza y preprocesamiento

Este documento registra las decisiones metodológicas tomadas sobre el dataset para garantizar trazabilidad, reproducibilidad y evitar data leakage.

## Principios

- No inventar valores cuando no existe evidencia para reconstruirlos.
- No eliminar filas de forma masiva por valores faltantes.
- Mantener indicadores de calidad cuando una anomalía puede ser informativa.
- Separar la limpieza del dato de la imputación estadística para Machine Learning.
- Aprender imputaciones, escalado y codificación únicamente con el conjunto de entrenamiento.

## Reglas aplicadas

| Problema | Decisión |
|---|---|
| `age` faltante | Se conserva como NaN hasta el pipeline de modelado. |
| `age < 18` | Se considera inválida para el alcance del análisis y se transforma en NaN. Se crea `age_invalid_flag`. |
| `gender` faltante | Se reemplaza por `Unknown` y se crea `gender_missing_flag`. |
| `coupon_code` faltante | Se representa como `NO_COUPON` y se crea `coupon_missing_flag`. |
| `total_spent` faltante | Se conserva como NaN y se crea `total_spent_missing_flag`. |
| `satisfaction_score` faltante | Se conserva como NaN y se crea `satisfaction_score_missing_flag`. |
| `last_purchase_date < signup_date` | Se crea `temporal_inconsistency_flag`; no se inventa una fecha. Se genera `last_purchase_date_clean` con NaT para esos casos. |
| Inconsistencia `country` / `city` | No se corrige sin una fuente externa confiable. Ambas se conservan para EDA; su uso en modelos se evaluará aparte. |
| `customer_id` | Se conserva para trazabilidad, pero se excluirá como predictor. |

## Fecha de referencia

Como el dataset no incluye una fecha explícita de extracción, se utiliza como `REFERENCE_DATE` un día posterior a la máxima `last_purchase_date` observada. En este dataset la fecha resultante es **2025-03-11**.

A partir de ella se generan:

- `customer_tenure_days`
- `recency_days`

La regla es determinística y hace que el análisis sea reproducible en cualquier fecha de ejecución.

## Validación ejecutada

El notebook `02_data_cleaning.ipynb` fue ejecutado completo sobre el dataset real y pasó todos los controles automáticos:

- Se conservaron las 15.000 filas.
- `customer_id` continúa siendo único.
- `churn` no tiene faltantes y permanece binario.
- No quedan edades menores a 18 entre los valores considerados válidos.
- `customer_tenure_days` no contiene valores negativos.
- `recency_days` no contiene valores negativos.

Resultado del dataset limpio: **15.000 filas y 39 columnas**.
