# Baseline models — Paso 6

## Objetivo

Establecer un piso de comparación antes de evaluar modelos de mayor complejidad.

## Modelos incluidos

1. **DummyClassifier (most_frequent)**: referencia mínima. Predice siempre la clase mayoritaria.
2. **Regresión Logística estándar**: baseline interpretable.
3. **Regresión Logística balanceada**: incorpora `class_weight="balanced"` para compensar el desbalance de `churn`.

## Métricas

No se utiliza accuracy como criterio principal porque la clase positiva representa aproximadamente 15,3 % de los casos.

Se reportan:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Matriz de confusión

## Resultado sobre el test estratificado (20 %)

| Modelo | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy-most-frequent | 0.847 | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 0.153 |
| Logistic | 0.846 | 0.565 | 0.493 | 0.161 | 0.243 | 0.821 | 0.417 |
| Logistic-balanced | 0.753 | 0.746 | 0.353 | 0.737 | 0.477 | 0.825 | 0.421 |

## Lectura

El Dummy alcanza una accuracy aparentemente alta porque predice siempre `no churn`, pero no identifica ningún cliente que abandona. Esto demuestra que accuracy no es una métrica suficiente para este problema.

La Regresión Logística balanceada detecta aproximadamente 73,7 % de los churn reales en el test y mejora fuertemente balanced accuracy, F1 y capacidad discriminativa.

## Interpretabilidad inicial

En la regresión logística balanceada, las señales de mayor magnitud absoluta incluyen:

- `total_spent`: coeficiente negativo fuerte; mayor gasto se asocia con menor probabilidad de churn.
- `satisfaction_score`: coeficiente negativo; mayor satisfacción se asocia con menor probabilidad de churn.
- `support_tickets`: coeficiente positivo; más tickets se asocian con mayor probabilidad de churn.

Estas relaciones son consistentes con el EDA y las hipótesis previas. No se interpretan como causalidad.

## Próximo paso

Comparar este baseline contra modelos no lineales y de ensamble: Random Forest, Gradient Boosting, XGBoost y CatBoost, con validación cruzada y ajuste de hiperparámetros.
