# Prueba de concepto de Uplift — Impacto Incremental y Decisión

## Objetivo

Explorar cómo complementar la predicción tradicional de `churn` con una priorización prescriptiva. En un experimento válido, el modelado de Uplift busca responder no solo *"¿quién tiene riesgo de abandonar?"*, sino también:

> **¿Qué clientes cambiarán favorablemente su comportamiento al recibir una intervención comercial, y a quiénes NO contactar?**

## Alcance y limitación de la evaluación

Esta implementación es una **prueba de concepto sobre un experimento sintético**, no un experimento de campo. La asignación a control/tratamiento y los resultados se generan artificialmente; las reglas de simulación introducen deliberadamente efectos positivos y adversos según señales de satisfacción y soporte. Por ello, los deciles, cuadrantes y diferencias entre políticas muestran cómo funciona el flujo bajo esas reglas, pero **no demuestran que una campaña real retendría clientes ni estiman su ROI**. Antes de tomar decisiones operativas se necesita una prueba A/B real, con tratamiento definido, resultados observados y evaluación de incertidumbre.

---

## 1. Fundamentos: Churn Tradicional vs. Uplift Modeling

| Criterio | Predicción de Churn Tradicional | Modelado de Uplift |
| :--- | :--- | :--- |
| **Pregunta analítica** | ¿Quién abandonará? ($P(\text{Churn} \mid X)$) | ¿A quién salva la intervención? ($\Delta P(\text{Churn})$) |
| **Tipo de analítica** | Predictiva (correlacional) | Prescriptiva / Causal (incremental) |
| **Riesgo operativo** | Desperdicio de presupuesto en *Causas Perdidas* y riesgo de acelerar bajas en *Perros Dormidos*. | Maximización del ROI al enfocar presupuesto exclusivamente en *Persuadibles*. |
| **Métrica de éxito** | ROC-AUC / PR-AUC | Curva Qini / Ganancia Incremental Acumulada |

---

## 2. Los 4 cuadrantes de respuesta (marco conceptual)

Los cuadrantes describen arquetipos teóricos de respuesta a una intervención. Sus porcentajes en este proyecto provienen de resultados sintéticos y no identifican tipos conductuales verificados en clientes reales:

```text
                          COMPORTAMIENTO SIN CONTACTO
                          No Abandona              Abandona
COMPORTAMIENTO   No Abandona | SEGUROS (70,3 %)     | PERSUADIBLES (23,8 %) |  <-- ¡Foco Comercial!
CON CONTACTO        Abandona | PERROS DORMIDOS (2,6%)| CAUSAS PERDIDAS (3,4%)|
```

1. **Persuadibles (23,8 % de la base):**
   - *Diagnóstico:* Clientes con fricción reversible (insatisfacción puntual o soporte técnico recurrente).
   - *Acción:* **Intervención prioritaria**. Resolución proactiva de reclamos y beneficio comercial personalizado.
2. **Seguros (*Sure Things*, 70,3 % de la base):**
   - *Diagnóstico:* Clientes satisfechos y estables que no abandonarán.
   - *Acción:* **Fidelización pasiva**. Comunicación de valor y agradecimiento. Evitar descuentos que canibalizan el margen.
3. **Causas Perdidas (*Lost Causes*, 3,4 % de la base):**
   - *Diagnóstico:* Clientes desvinculados o enojados sin intención de retorno.
   - *Acción:* **Cero gasto comercial**. Ahorro directo de llamadas o incentivos.
4. **Perros Dormidos (*Sleeping Dogs*, 2,6 % de la base):**
   - *Diagnóstico:* Clientes pasivos para los que el contacto actúa como recordatorio de cancelación.
   - *Acción:* **Lista estricta de exclusión (*Do Not Disturb*)**.

---

## 3. Metodología: Two-Model Approach (T-Learner)

Con datos experimentales válidos, un T-Learner estima resultados potenciales por separado para tratamiento y control. En esta prueba se aplica el mismo procedimiento a resultados sintéticos:

1. Se entrena un modelo $M_0$ (Gradient Boosting) exclusivamente sobre el grupo **Control** ($T=0$).
2. Se entrena un modelo $M_1$ (Gradient Boosting) exclusivamente sobre el grupo **Tratamiento** ($T=1$).
3. Para cada cliente nuevo, su score de impacto incremental es:
   $$\widehat{\text{Uplift}}(X) = \hat{P}_{M_0}(\text{Churn} \mid X) - \hat{P}_{M_1}(\text{Churn} \mid X)$$

---

## 4. Resultados en el test sintético y comparación de políticas

Evaluado sobre 3.000 clientes reservados de test con tratamiento y resultados generados por simulación:

### Deciles de Uplift

- **Decil 1 (Top 10 %):** Churn observado en la simulación: Control = 53,7 %, Tratados = 15,7 % $\rightarrow$ **diferencia simulada = +38,1 %**.
- **Decil 2:** Churn observado en la simulación: Control = 53,6 %, Tratados = 22,1 % $\rightarrow$ **diferencia simulada = +31,5 %**.
- **Decil 10:** Churn observado en la simulación: Control = 13,6 %, Tratados = 18,9 % $\rightarrow$ **diferencia simulada = -5,3 %** (perfil adverso generado bajo las reglas de simulación).

### Eficiencia Operativa: Churn Tradicional vs. Uplift

| Capacidad Operativa | Clientes Contactados | Retenciones simuladas: Churn | Retenciones simuladas: Uplift | Diferencia simulada | Mejora en la simulación |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Top 5 %** | 150 | 33,1 | 69,6 | +36,5 clientes | **+110,2 %** |
| **Top 10 %** | 300 | 65,1 | 114,2 | +49,1 clientes | **+75,5 %** |
| **Top 20 %** | 600 | 132,0 | 208,9 | +76,9 clientes | **+58,2 %** |

> **Lectura del resultado:** En este escenario sintético de test, con capacidad equivalente al Top 20 % (600 contactos), la política ordenada por uplift arroja **209 retenciones simuladas frente a 132** con la política tradicional (+58,2 %). No es una estimación de impacto real ni una conclusión de efectividad comercial.
