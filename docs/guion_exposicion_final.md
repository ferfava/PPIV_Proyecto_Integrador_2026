# Guion de exposición final — PPIV Proyecto Integrador 2026

## Objetivo de la defensa

Explicar el proyecto como una historia de negocio y no como una lista de algoritmos.

La idea central que debe quedar clara es:

> No construí solamente un modelo de churn. Construí una solución que identifica riesgo, explica por qué existe y ayuda a decidir a quién intervenir cuando los recursos son limitados.

## Estructura sugerida de la exposición

### 1. Problema

“Partí de un problema de retención de clientes. El objetivo era anticipar abandono y transformar esa predicción en una decisión de negocio.”

Mencionar:

- 15.000 clientes;
- 30 variables originales;
- churn aproximado de 15,3%;
- problema desbalanceado.

### 2. Calidad de datos

“Antes de modelar hice una auditoría porque un modelo sofisticado sobre datos defectuosos sigue siendo un mal modelo.”

Hallazgos:

- faltantes;
- edades inválidas;
- inconsistencias temporales;
- geografía dudosa;
- posible data leakage.

Explicar que el split se realiza antes del preprocessing para evitar fuga de información.

### 3. EDA e hipótesis

Mostrar las tres señales principales:

- baja satisfacción;
- muchos tickets de soporte;
- bajo gasto.

Frase útil:

“Estas tres señales aparecen de forma consistente en EDA, hipótesis, modelos y SHAP.”

Aclarar:

“Asociación no significa causalidad.”

### 4. Modelos

Explicar que primero se construyó un baseline y después se compararon modelos avanzados.

Modelos destacados:

- Regresión Logística;
- Random Forest;
- Gradient Boosting;
- XGBoost;
- CatBoost;
- ensemble.

Pregunta probable: “¿Por qué no accuracy?”

Respuesta:

“Porque con 15% de churn un modelo puede acertar mucho prediciendo casi siempre que el cliente no abandona. Por eso prioricé recall, F1, ROC-AUC y PR-AUC.”

### 5. Resultado principal

Enfatizar:

- ROC-AUC ensemble ≈ 0,921;
- recall ≈ 96,5%;
- PR-AUC ≈ 0,547;
- Top 20% concentra cerca de 70% de los abandonos reales.

Frase clave:

“El valor no está solamente en clasificar, sino en concentrar el riesgo donde el negocio puede actuar.”

### 6. Explicabilidad

Explicar SHAP de manera simple:

“SHAP permite ver qué variables empujan una predicción hacia mayor o menor riesgo.”

Variables importantes:

- satisfacción;
- gasto;
- soporte.

Si preguntan por qué XGBoost para SHAP:

“El ensemble es el benchmark de mejor desempeño, pero XGBoost tiene rendimiento muy cercano y permite TreeSHAP fiel y eficiente.”

### 7. Segmentación

Explicar que los clusters se generan sin usar churn.

Perfiles:

- Insatisfechos en riesgo;
- Alta fricción de soporte;
- Inactivos satisfechos;
- Activos satisfechos.

Frase útil:

“El clustering no reemplaza al modelo de churn; agrega contexto para personalizar la acción.”

### 8. Calibración

Pregunta probable: “¿Qué significa calibrar?”

Respuesta:

“Un modelo puede ordenar bien a los clientes y aun así producir probabilidades poco fiables. La calibración busca que, por ejemplo, un grupo estimado cerca de 30% de riesgo tenga una frecuencia observada cercana a ese valor.”

Resultado:

- Platt reduce fuertemente Brier Score y ECE;
- ROC-AUC se mantiene.

### 9. Decisión de negocio

Esta es una de las partes más fuertes del proyecto.

Explicar:

“No obligo al negocio a usar un umbral fijo de 0,50. Permito elegir capacidad de intervención y ver cuánto churn se captura.”

Ejemplos:

- Top 10% → ~35,7%;
- Top 20% → ~67,4%;
- Top 25% → ~84,8%.

Frase clave:

“El modelo estima riesgo; la política de negocio decide cuánto intervenir.”

### 10. Demo de Streamlit

Orden recomendado para mostrar la app:

1. Resumen ejecutivo.
2. Desempeño de modelos.
3. IA explicable.
4. Segmentación.
5. Predicción en vivo.
6. Decisión de negocio.

No navegar por todas las pantallas. Mostrar solamente las que sostienen el relato.

## Preguntas probables del docente

### ¿Por qué elegiste churn?

Porque permite integrar clasificación, desbalance, interpretabilidad y una decisión de negocio clara y medible.

### ¿Cuál es el mejor modelo?

El ensemble presenta el mejor rendimiento global como benchmark. XGBoost se utiliza como modelo operativo de inferencia por su relación entre performance, portabilidad y explicabilidad.

### ¿Por qué hiciste clustering?

Para descubrir perfiles sin usar la variable objetivo y diferenciar estrategias de retención.

### ¿Por qué calibraste?

Porque una buena capacidad de ranking no garantiza probabilidades confiables.

### ¿Qué significa PR-AUC?

Resume el equilibrio entre precision y recall a distintos umbrales y resulta especialmente útil cuando la clase positiva es minoritaria.

### ¿Qué es data leakage?

Es cuando el entrenamiento utiliza información que en producción no estaría disponible al momento de predecir, produciendo métricas artificialmente altas.

### ¿El modelo demuestra causalidad?

No. Detecta asociaciones y patrones predictivos. Para causalidad serían necesarios diseños adicionales, por ejemplo experimentos o métodos causales.

### ¿Qué harías en producción?

- registrar predicciones y decisiones;
- monitorear drift;
- recalibrar;
- incorporar costos reales;
- validar campañas con experimentos;
- definir gobernanza y frecuencia de reentrenamiento.

## Cierre sugerido

“Mi objetivo fue que el proyecto no terminara en una métrica. La solución conecta calidad de datos, predicción, explicación y decisión. El modelo identifica riesgo, pero la última decisión sigue siendo del negocio.”

## Tres conceptos que no deben faltar

1. **Desbalance:** explica por qué accuracy no alcanza.
2. **Explicabilidad:** explica por qué el modelo es utilizable.
3. **Capacidad operativa:** explica por qué predicción y decisión no son lo mismo.
