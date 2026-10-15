# Modelado de Uplift — Impacto Incremental y Decisión Causal

## Objetivo

Superar las limitaciones de la predicción tradicional de `churn` transformando la priorización reactiva en una **estrategia causal y prescriptiva**. El modelado de Uplift responde no a *"¿quién tiene riesgo de abandonar?"*, sino a:

> **¿Qué clientes cambiarán favorablemente su comportamiento al recibir una intervención comercial, y a quiénes NO contactar?**

---

## 1. Fundamentos: Churn Tradicional vs. Uplift Modeling

| Criterio | Predicción de Churn Tradicional | Modelado de Uplift |
| :--- | :--- | :--- |
| **Pregunta analítica** | ¿Quién abandonará? ($P(\text{Churn} \mid X)$) | ¿A quién salva la intervención? ($\Delta P(\text{Churn})$) |
| **Tipo de analítica** | Predictiva (correlacional) | Prescriptiva / Causal (incremental) |
| **Riesgo operativo** | Desperdicio de presupuesto en *Causas Perdidas* y riesgo de acelerar bajas en *Perros Dormidos*. | Maximización del ROI al enfocar presupuesto exclusivamente en *Persuadibles*. |
| **Métrica de éxito** | ROC-AUC / PR-AUC | Curva Qini / Ganancia Incremental Acumulada |

---

## 2. Los 4 Cuadrantes Estratégicos de la Cartera

El modelo clasifica a los clientes en cuatro arquetipos conductuales ante la intervención:

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

Para estimar el impacto contrafactual sin requerir dependencias no estándar:

1. Se entrena un modelo $M_0$ (Gradient Boosting) exclusivamente sobre el grupo **Control** ($T=0$).
2. Se entrena un modelo $M_1$ (Gradient Boosting) exclusivamente sobre el grupo **Tratamiento** ($T=1$).
3. Para cada cliente nuevo, su score de impacto incremental es:
   $$\widehat{\text{Uplift}}(X) = \hat{P}_{M_0}(\text{Churn} \mid X) - \hat{P}_{M_1}(\text{Churn} \mid X)$$

---

## 4. Resultados en Test y Comparativa Comercial

Evaluado sobre 3.000 clientes reservados de test:

### Deciles de Uplift

- **Decil 1 (Top 10 %):** Churn en Control = 53,7 %, Churn en Tratados = 15,7 % $\rightarrow$ **Uplift real = +38,1 %**.
- **Decil 2:** Churn en Control = 53,6 %, Churn en Tratados = 22,1 % $\rightarrow$ **Uplift real = +31,5 %**.
- **Decil 10:** Churn en Control = 13,6 %, Churn en Tratados = 18,9 % $\rightarrow$ **Uplift real = -5,3 %** (*Perros dormidos confirmados*).

### Eficiencia Operativa: Churn Tradicional vs. Uplift

| Capacidad Operativa | Clientes Contactados | Salvados Modelo Churn | Salvados Modelo Uplift | Ganancia Neta | Mejora de Eficiencia |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Top 5 %** | 150 | 33,1 | 69,6 | +36,5 clientes | **+110,2 %** |
| **Top 10 %** | 300 | 65,1 | 114,2 | +49,1 clientes | **+75,5 %** |
| **Top 20 %** | 600 | 132,0 | 208,9 | +76,9 clientes | **+58,2 %** |

> **Conclusión ejecutiva:** Con exactamente el mismo equipo de retención y la misma capacidad operativa (Top 20 %, 600 contactos), la política de Uplift salva **209 clientes frente a 132 del enfoque tradicional**, lo que representa un **+58,2 % de incremento en la efectividad neta del negocio**.
