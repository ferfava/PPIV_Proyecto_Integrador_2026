# Calibración de probabilidades y política de decisión

## Objetivo

Separar dos problemas que suelen confundirse:

1. **Predicción:** estimar correctamente el riesgo de abandono.
2. **Decisión:** definir a qué clientes intervenir según costo y capacidad operativa.

Para esta etapa se utilizó un modelo XGBoost operativo con 10 variables disponibles al momento de evaluar al cliente. El dataset se dividió de forma estratificada en **64% entrenamiento, 16% calibración y 20% test final**.

## Calibración

Se compararon tres alternativas sobre el conjunto de test:

| Método | ROC-AUC | PR-AUC | Brier | Log Loss | ECE |
|---|---:|---:|---:|---:|---:|
| Sin calibrar | 0,9184 | 0,5347 | 0,1039 | 0,2973 | 0,1062 |
| Platt | 0,9184 | 0,5347 | **0,0755** | **0,2098** | **0,0098** |
| Isotónica | 0,9174 | 0,5249 | 0,0752 | 0,2083 | 0,0040 |

Se seleccionó **Platt scaling** como calibrador operativo. La razón es que mejora fuertemente Brier Score y ECE sin alterar el orden de los scores, por lo que conserva exactamente ROC-AUC y PR-AUC. La calibración isotónica presenta una mejora mínima adicional en calibración pura, pero introduce pequeñas pérdidas de discriminación y una transformación menos suave.

## Por qué 0,50 no es automáticamente el mejor umbral

Con las probabilidades calibradas, un corte fijo de 0,50 produce aproximadamente:

- Recall: **54,6%**
- Precision: **53,4%**
- F1: **0,540**
- Clientes intervenidos: **15,7%** del test

Esto muestra que el umbral 0,50 no debe adoptarse por costumbre. El punto de corte depende de cuánto cuesta perder un cliente frente a cuánto cuesta contactar a uno que finalmente no abandona.

En los escenarios simulados donde un falso negativo cuesta 2x, 5x o 10x un falso positivo, el criterio económico favorece una política agresiva de intervención. En este dataset existe además una separación muy marcada de scores, por lo que distintos umbrales bajos seleccionan prácticamente el mismo grupo de alto riesgo.

## Política recomendada: capacidad antes que umbral fijo

Para una operación real con capacidad limitada, resulta más estable definir cuántos clientes se pueden contactar y priorizar por ranking calibrado.

| Capacidad de intervención | Clientes | Abandonos capturados | Tasa de abandono del grupo |
|---:|---:|---:|---:|
| Top 10% | 300 | 35,7% | 54,7% |
| Top 20% | 600 | **67,4%** | 51,7% |
| Top 25% | 750 | **84,8%** | 52,0% |
| Top 30% | 900 | 100,0% | 51,1% |

Por lo tanto, la decisión de negocio puede expresarse como:

> El modelo estima una probabilidad calibrada; la empresa define la intensidad de la intervención según presupuesto y capacidad.

Esto evita convertir una probabilidad en una decisión automática sin contexto económico.

## Validación

El script `scripts/run_calibration_threshold.py` fue ejecutado dos veces de punta a punta sobre el dataset real. Los resultados fueron idénticos y contiene assertions que verifican que Platt mejora Brier/ECE sin modificar ROC-AUC.

## Archivos relacionados

- `notebooks/11_calibration_threshold.ipynb`
- `scripts/run_calibration_threshold.py`
- `app/data/calibration_comparison.csv`
- `app/data/capacity_thresholds.csv`
