# Diseño de variables y prevención de leakage

## Objetivo

Preparar una matriz de modelado reproducible para predecir `churn` sin contaminar el conjunto de test ni incorporar variables de trazabilidad o campos con riesgo de fuga de información.

## Fecha de referencia

Se utiliza como fecha de referencia el día posterior a la fecha más reciente observada en el dataset: **2025-03-11**.

Esto permite calcular recencia y antigüedad de manera reproducible sin depender de la fecha actual del sistema.

## Variables derivadas

- `customer_tenure_days`: antigüedad del cliente desde `signup_date`.
- `recency_days`: días desde `last_purchase_date` hasta la fecha de referencia.
- `has_coupon`: indicador de presencia de `coupon_code`.
- `email_engagement_ratio`: relación click/open cuando existe apertura.
- `spend_per_visit`: gasto acumulado por visita.
- `support_ticket_rate`: tickets de soporte por visita.
- `purchase_frequency_per_month`: frecuencia de compra de los últimos 3 meses expresada como equivalente mensual.
- indicadores de missingness para `age`, `total_spent`, `gender` y `satisfaction_score`.

## Variables excluidas del modelo principal

- `customer_id`: identificador sin valor predictivo generalizable.
- `city`: inconsistencia geográfica detectada en la auditoría.
- `signup_date` y `last_purchase_date`: se reemplazan por variables temporales derivadas.
- `coupon_code`: alta cardinalidad y semántica promocional específica; se conserva `has_coupon`.
- `lifetime_value`: se excluye del modelo principal porque no está documentado si fue calculado exclusivamente con información disponible al momento de predicción. Se reserva para análisis de negocio y experimentos secundarios.
- `age_invalid` y `temporal_inconsistency`: flags de calidad utilizados para auditoría, no como señales de negocio.
- `churn`: variable objetivo.

## Split y preprocessing

Se utiliza un split 80/20 estratificado con `random_state=42`.

El orden es deliberado:

1. Separar train/test.
2. Ajustar imputadores, escaladores y codificadores solo con train.
3. Transformar train y test con el pipeline ajustado.

### Numéricas

- Imputación por mediana.
- Estandarización con `StandardScaler`.

### Categóricas

- Imputación por moda.
- `OneHotEncoder(handle_unknown='ignore')`.

## Resultado validado

- Train: 12.000 clientes.
- Test: 3.000 clientes.
- Churn train: ~15,32 %.
- Churn test: ~15,33 %.
- Variables de entrada antes de encoding: 34.
- Variables finales luego del preprocessing: 51.
- Valores faltantes en matrices transformadas: 0.

## Criterio metodológico

Toda transformación que aprende parámetros de los datos se ajusta únicamente sobre el conjunto de entrenamiento. Esto evita data leakage y permite una evaluación honesta de los modelos posteriores.
