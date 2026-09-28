# Modelos avanzados — PPIV Proyecto Integrador 2026

## Objetivo

Comparar modelos no lineales capaces de capturar interacciones y relaciones complejas del comportamiento de clientes, manteniendo una evaluación consistente frente al desbalance de la variable `churn`.

## Modelos evaluados

- Regresión Logística balanceada como referencia.
- Random Forest.
- Gradient Boosting.
- XGBoost.
- CatBoost.
- Ensemble de probabilidades entre Gradient Boosting, XGBoost y CatBoost.

## Validación

Se utilizó validación cruzada estratificada de 3 folds sobre el conjunto de entrenamiento. La estratificación conserva la proporción de churn en cada fold.

Con las versiones actuales de scikit-learn y CatBoost, `cross_validate` presenta una incompatibilidad de tags con `CatBoostClassifier`. Para evitar sesgos o tratamientos distintos entre algoritmos, la validación cruzada se implementó manualmente para todos los modelos.

## Métricas principales

Dado que churn representa aproximadamente el 15 % de los casos, accuracy no es suficiente. Se priorizan:

- ROC-AUC.
- PR-AUC.
- Recall.
- F1.
- Balanced Accuracy.
- Precision.

## Resultados de validación cruzada

Valores aproximados obtenidos sobre train mediante 3-fold CV:

| Modelo | ROC-AUC | PR-AUC | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic balanced | 0.825 | 0.410 | 0.758 | 0.481 |
| Random Forest | 0.907 | 0.491 | 0.456 | 0.473 |
| Gradient Boosting | 0.906 | 0.497 | 0.402 | 0.442 |
| XGBoost | **0.909** | **0.504** | **0.923** | **0.644** |
| CatBoost | 0.907 | 0.497 | 0.919 | 0.642 |

Los algoritmos de boosting superan con claridad al baseline lineal en capacidad discriminante.

## Optimización de XGBoost

Se realizó una búsqueda aleatoria reproducible de 12 configuraciones variando:

- número de árboles;
- profundidad;
- learning rate;
- subsampling;
- proporción de columnas;
- `min_child_weight`;
- `gamma`;
- regularización L2.

La selección se efectuó por PR-AUC y, en segundo término, ROC-AUC. El objetivo no fue maximizar accuracy, sino mejorar la detección de la clase minoritaria sin degradar la calidad global.

La búsqueda no produjo una mejora material frente a la configuración inicial, lo que también es un resultado válido: evita sobreoptimizar el modelo sobre pequeñas diferencias de validación.

## Resultados sobre test

| Modelo | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic balanced | 0.737 | 0.478 | 0.825 | 0.421 |
| Random Forest | 0.533 | 0.532 | 0.918 | 0.529 |
| Gradient Boosting | 0.407 | 0.462 | **0.920** | 0.547 |
| XGBoost | **0.967** | 0.672 | 0.920 | 0.538 |
| XGBoost tuned | 0.907 | 0.655 | 0.920 | 0.543 |
| CatBoost | 0.961 | 0.668 | 0.919 | 0.542 |
| Ensemble GB + XGB + CatBoost | 0.965 | **0.675** | **0.921** | **0.547** |

## Ensemble

El ensemble promedia las probabilidades generadas por Gradient Boosting, XGBoost y CatBoost.

No se incorporó por complejidad estética: se conserva porque mejora ligeramente el equilibrio global del conjunto de test respecto de los modelos individuales, especialmente F1 y ROC-AUC, manteniendo un recall muy alto.

En test, el ensemble detectó 444 de 460 churn reales, dejando 16 falsos negativos.

## Interpretación

Los resultados muestran que la relación entre las variables y churn no es puramente lineal. Los modelos de boosting capturan interacciones y umbrales que la regresión logística no representa tan bien.

Sin embargo, el mejor modelo predictivo no se adopta todavía como solución final. Antes de usar sus probabilidades para decisiones de negocio se deben completar:

1. calibración de probabilidades;
2. optimización del umbral según costos de negocio;
3. explicabilidad global e individual con SHAP;
4. evaluación de estabilidad y segmentación.

## Decisión técnica

El ensemble queda como candidato principal para el motor de retención, mientras que XGBoost y CatBoost se conservan como modelos individuales de referencia y explicabilidad.
