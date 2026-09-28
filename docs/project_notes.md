# Notas del proyecto

## Tema
Customer Churn & Marketing Intelligence.

## Problema de negocio
Analizar el comportamiento de clientes para identificar patrones asociados al abandono, anticipar riesgo de churn y convertir los resultados en acciones de retención priorizadas.

## Variable objetivo
`churn` — clasificación binaria.

## Líneas de trabajo previstas

- Calidad de datos y reglas de validación.
- EDA y estadística descriptiva.
- Tres hipótesis contrastables.
- Feature engineering.
- Modelos supervisados: baseline + modelos avanzados.
- Evaluación para clases desbalanceadas.
- Explicabilidad con SHAP.
- Segmentación no supervisada.
- Motor de priorización de retención.
- Aplicación Streamlit.

## Hallazgos iniciales a validar

- Existen valores faltantes en múltiples variables.
- Hay edades imposibles que requieren reglas de calidad.
- Existen inconsistencias temporales entre fecha de alta y última compra.
- Hay combinaciones geográficas potencialmente incoherentes entre país y ciudad.
- La variable objetivo presenta desbalance de clases.

> Estas observaciones son preliminares y deberán reproducirse y documentarse mediante código en los notebooks del proyecto.
