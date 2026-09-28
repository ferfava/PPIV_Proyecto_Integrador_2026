# EDA — Hallazgos preliminares

Este documento resume los principales patrones encontrados en el análisis exploratorio del dataset de 15.000 clientes. Las asociaciones descriptivas se contrastarán formalmente en `04_hypothesis_testing.ipynb`.

## 1. Variable objetivo

La tasa global de churn es **15,32 %**. Existe desbalance de clases, por lo que en la etapa de modelado no se utilizará accuracy como única métrica.

## 2. Satisfacción

La satisfacción es la señal descriptiva más fuerte observada:

- Score 1: churn ≈ **48,74 %**.
- Score 2: churn ≈ **50,17 %**.
- Score 3: churn ≈ **9,66 %**.
- Score 4: churn ≈ **8,89 %**.
- Score 5: churn ≈ **9,71 %**.

El salto entre satisfacción baja (1–2) y satisfacción media/alta (3–5) es muy marcado.

## 3. Tickets de soporte

El grupo con 5 o más tickets muestra un comportamiento diferencial:

- 0 tickets: churn ≈ **12,75 %**.
- 1–2 tickets: churn ≈ **13,70 %**.
- 3–4 tickets: churn ≈ **12,84 %**.
- 5+ tickets: churn ≈ **50,43 %**.

Esto sugiere que una intensidad alta de contacto con soporte puede funcionar como señal de fricción o problemas no resueltos.

## 4. Gasto total

La tasa de churn por quintil de `total_spent` muestra una concentración muy fuerte en el grupo de menor gasto:

- Q1: churn ≈ **40,50 %**.
- Q2: churn ≈ **10,25 %**.
- Q3: churn ≈ **9,71 %**.
- Q4: churn ≈ **8,64 %**.
- Q5: churn ≈ **9,50 %**.

La relación parece estar concentrada en el quintil inferior y no presenta una caída lineal simple entre todos los quintiles.

## 5. Interacción satisfacción + soporte

La interacción revela un patrón especialmente interesante:

- Satisfacción media/alta + 0–4 tickets: churn ≈ **7,01 %**.
- Satisfacción media/alta + 5+ tickets: churn ≈ **50,00 %**.
- Satisfacción baja + 0–4 tickets: churn ≈ **49,67 %**.
- Satisfacción baja + 5+ tickets: churn ≈ **50,00 %**.

Esto indica que tanto una satisfacción baja como una cantidad muy alta de tickets identifican grupos de riesgo elevado. La interacción deberá estudiarse luego con modelos multivariados para determinar cuánto aporta cada señal controlando por las restantes variables.

## 6. Variables con menor diferenciación descriptiva

Canal de adquisición, dispositivo, tipo de suscripción, condición premium, uso de descuento, método de pago y país muestran diferencias mucho menores en churn que satisfacción, tickets de soporte y gasto.

No se eliminarán todavía: un modelo multivariado puede detectar interacciones que un análisis univariado no captura.

## 7. Hipótesis propuestas

### H1 — Satisfacción
Los clientes con satisfacción baja (1–2) presentan una tasa de churn mayor que los clientes con satisfacción media/alta (3–5).

### H2 — Soporte
Los clientes con 5 o más tickets de soporte presentan una tasa de churn mayor que los clientes con 0–4 tickets.

### H3 — Gasto
Los clientes del quintil inferior de `total_spent` presentan una tasa de churn mayor que los clientes pertenecientes a los cuatro quintiles superiores.

## 8. Interpretación de negocio

Los patrones preliminares apuntan a tres ejes accionables:

1. **Experiencia:** baja satisfacción.
2. **Fricción operativa:** volumen elevado de tickets de soporte.
3. **Vinculación económica:** clientes con muy bajo gasto acumulado.

Estas asociaciones no se interpretan como causalidad. El siguiente paso es cuantificar significancia estadística y tamaño de efecto antes de convertirlas en conclusiones formales.
