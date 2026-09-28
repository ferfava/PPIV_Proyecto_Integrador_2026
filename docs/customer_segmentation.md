# Segmentación no supervisada de clientes

## Objetivo

Construir perfiles de clientes sin utilizar la variable objetivo `churn` durante la formación de los grupos. El churn se observa recién después para interpretar los segmentos y conectar la segmentación con decisiones de retención.

## Variables utilizadas

- `total_spent`
- `last_3_month_purchase_freq`
- `recency_days`
- `satisfaction_score`
- `support_tickets`

Estas variables representan valor, frecuencia, recencia y experiencia del cliente.

## Preprocesamiento

- Imputación por mediana para valores faltantes.
- Winsorización moderada (percentiles 1 y 99) en variables susceptibles a extremos para evitar que unos pocos outliers dominen K-Means.
- Estandarización con `StandardScaler`.

## Selección de k

Se compararon soluciones entre k=2 y k=8 usando:

- Silhouette score.
- Davies-Bouldin index.
- Calinski-Harabasz index.
- Inercia.

La estructura natural de clusters es moderada: para k=4 el silhouette es aproximadamente 0,157. Por lo tanto, los segmentos deben interpretarse como perfiles útiles de negocio y no como poblaciones completamente separadas.

Se seleccionó **k=4** por equilibrio entre calidad interna, estabilidad e interpretabilidad.

## Estabilidad

Se repitió K-Means con distintas semillas. El Adjusted Rand Index respecto de la solución de referencia fue:

- seed 1: 0,998
- seed 7: 0,902
- seed 21: 0,931
- seed 42: 1,000
- seed 99: 0,978

La estabilidad media supera 0,96, lo que indica que la estructura obtenida es altamente reproducible frente a cambios en la inicialización.

## Perfiles encontrados

### Segmento 0 — Activos satisfechos

- 4.802 clientes (32,0%).
- Churn observado: 6,9%.
- Recencia aproximada: 187 días.
- Satisfacción media: 3,97.
- Tickets de soporte: 1,38.

Perfil con buena satisfacción, menor fricción y actividad relativamente reciente.

### Segmento 1 — Insatisfechos en riesgo

- 2.342 clientes (15,6%).
- Churn observado: 46,6%.
- Satisfacción media: 1,77.
- Tickets de soporte: 1,87.

Es el segmento de mayor riesgo observado. La principal diferencia frente a los demás grupos es la baja satisfacción.

### Segmento 2 — Inactivos satisfechos

- 4.701 clientes (31,3%).
- Churn observado: 6,7%.
- Recencia aproximada: 622 días.
- Satisfacción media: 3,94.
- Tickets de soporte: 1,38.

Presenta mayor tiempo desde la última compra, pero mantiene buena satisfacción y baja fricción.

### Segmento 3 — Alta fricción de soporte

- 3.155 clientes (21,0%).
- Churn observado: 17,6%.
- Satisfacción media: 3,91.
- Tickets de soporte: 3,95.

Aunque la satisfacción declarada es buena, la elevada interacción con soporte coincide con una tasa de churn mayor que la de los segmentos satisfechos de baja fricción.

## Insight principal

La segmentación refuerza dos señales encontradas previamente por métodos supervisados y por el EDA: **satisfacción** y **fricción de soporte**. Además permite distinguir clientes con baja actividad pero todavía satisfechos, que podrían requerir estrategias de reactivación distintas de una acción de retención por insatisfacción.

## Limitación metodológica

El silhouette moderado indica que los clientes no forman grupos completamente separados en el espacio de variables elegido. Por eso la segmentación se usa como herramienta de perfilado y priorización, no como una clasificación absoluta o causal.
