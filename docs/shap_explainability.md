# Explainable AI — SHAP

## Objetivo

Explicar por qué el modelo de churn asigna mayor o menor riesgo a cada cliente, evitando tratar al modelo como una caja negra.

## Modelo utilizado

La explicabilidad se aplica sobre **XGBoost**, no directamente sobre el ensemble. La razón es metodológica: el ensemble promedia probabilidades de Gradient Boosting, XGBoost y CatBoost, y una suma ingenua de valores SHAP de modelos distintos no preservaría necesariamente la aditividad ni produciría una explicación fiel.

XGBoost mantiene un rendimiento muy cercano al ensemble:

- ROC-AUC: **0,9202**
- PR-AUC: **0,5378**

y permite utilizar `TreeSHAP`, diseñado para modelos basados en árboles.

## Importancia global

Las variables con mayor media de |SHAP| fueron:

1. `satisfaction_score` — 1,6016
2. `total_spent` — 1,5636
3. `support_tickets` — 0,9372
4. `spend_per_visit` — 0,3297
5. `support_ticket_rate` — 0,1351

Este resultado es coherente con el análisis exploratorio y con las tres hipótesis planteadas previamente.

## Dirección de los efectos

El análisis de dependencia muestra:

- puntuaciones de satisfacción 1–2 empujan fuertemente el riesgo de churn hacia arriba;
- niveles bajos de gasto total tienden a aumentar el riesgo;
- niveles altos de tickets de soporte aumentan el riesgo en numerosos casos.

SHAP describe cómo el modelo utiliza las variables; **no demuestra causalidad**.

## Explicación individual

Se seleccionó de forma reproducible un cliente del conjunto de test que efectivamente hizo churn y al que el modelo asignó riesgo alto.

- customer_id: **12884**
- churn real: **1**
- probabilidad estimada: **94,9 %**
- satisfacción: **1**
- total_spent: **110,28**
- tickets de soporte: **2**

Los principales factores que incrementaron su predicción fueron la baja satisfacción y el bajo gasto total.

## Validación técnica

Se verificó la propiedad aditiva de SHAP para el ejemplo local:

`base_value + suma(SHAP) ≈ margen bruto del modelo`

La comprobación se realiza con `assert np.isclose(...)` en el script.

## Uso posterior

Las explicaciones globales e individuales se reutilizarán en la aplicación Streamlit para mostrar:

- riesgo estimado;
- factores que aumentan el riesgo;
- factores protectores;
- gráficos de importancia global;
- explicación de un cliente individual.
