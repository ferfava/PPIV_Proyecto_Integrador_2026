# Retention Engine — priorización accionable de clientes

## Objetivo

Transformar las predicciones de churn en una herramienta de decisión comercial. El motor no busca únicamente identificar clientes con riesgo, sino ordenar a quién conviene intervenir primero.

## Componentes del score

El `priority_score` se expresa en una escala de 0 a 100 y combina:

- 65 %: probabilidad de churn del ensemble (Gradient Boosting + XGBoost + CatBoost).
- 25 %: valor comercial relativo del cliente, calculado a partir de percentiles de `total_spent` y `avg_order_value` aprendidos sobre train.
- 10 %: riesgo contextual del segmento obtenido con K-Means, calculado a partir de la tasa de churn observada en los segmentos del conjunto de entrenamiento.

La fórmula es:

`priority_score = 100 * (0.65 * churn_probability + 0.25 * value_score + 0.10 * segment_risk_score)`

El score no debe interpretarse como una probabilidad. Es un índice operativo de priorización.

## Bandas de prioridad

- Baja: score < 40
- Media: 40 a < 60
- Alta: 60 a < 75
- Crítica: >= 75

## Acciones sugeridas

Las acciones se asignan con reglas simples y auditables:

- satisfacción <= 2: recuperación de experiencia y resolución de causa;
- 5 o más tickets: escalamiento de soporte;
- baja frecuencia reciente: campaña de reactivación;
- alto valor comercial: beneficio de fidelización/VIP;
- resto: seguimiento preventivo multicanal.

Estas acciones son recomendaciones de negocio basadas en señales observadas y no implican causalidad.

## Validación del ranking

Se evaluó el motor exclusivamente sobre el conjunto de test reservado previamente.

Resultados reproducidos en dos ejecuciones consecutivas:

| Fracción priorizada | Clientes | Churn dentro del grupo | Lift vs. base | Churn reales capturados |
|---|---:|---:|---:|---:|
| Top 5 % | 150 | 55,33 % | 3,61x | 18,04 % |
| Top 10 % | 300 | 55,00 % | 3,59x | 35,87 % |
| Top 20 % | 600 | 53,33 % | 3,48x | 69,57 % |
| Top 30 % | 900 | 51,00 % | 3,33x | 99,78 % |

La tasa base de churn del test es 15,33 %.

El resultado más accionable es que actuar sobre el 20 % con mayor prioridad concentra aproximadamente el 70 % de los churn reales del conjunto de test.

## Métricas del ensemble utilizado

- ROC-AUC: 0,9209
- PR-AUC: 0,5473

## Consideraciones metodológicas

- El split train/test fue definido antes del modelado.
- El preprocessing se ajustó exclusivamente sobre train.
- Los percentiles de valor se calculan contra la distribución de train.
- El K-Means del motor se ajusta sobre train y luego asigna segmentos al test.
- El churn no participa en la creación de los clusters; solo se usa en train para estimar el riesgo contextual de cada segmento.
- La evaluación del ranking se realiza sobre test, que no participa en el ajuste de los modelos.

## Uso en la aplicación final

En Streamlit este motor alimentará una tabla ordenable con:

- cliente;
- probabilidad de churn;
- segmento;
- score de valor;
- score de prioridad;
- banda de prioridad;
- acción sugerida.

También permitirá filtrar, por ejemplo, solo prioridades `Alta` y `Crítica` para construir una cola de intervención comercial.
