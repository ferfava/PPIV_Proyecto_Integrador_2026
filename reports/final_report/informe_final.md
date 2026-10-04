# Informe final — PPIV Proyecto Integrador 2026

## Inteligencia de Retención de Clientes

**Carrera:** Ciencia de Datos e Inteligencia Artificial  
**Proyecto:** PPIV — Proyecto Integrador 2026  
**Autora:** Fernanda Fava

## 1. Resumen ejecutivo

El proyecto desarrolla una solución integral de Ciencia de Datos orientada a anticipar el abandono de clientes (*customer churn*) y transformar la predicción en una herramienta concreta para la toma de decisiones.

El trabajo parte de un dataset de 15.000 clientes y 30 variables originales vinculadas con comportamiento de compra, engagement, satisfacción, soporte, navegación, valor económico y características demográficas y temporales. La tasa observada de abandono es cercana al 15,3%, por lo que se trata de un problema de clasificación desbalanceado.

La solución integra auditoría de calidad, limpieza, análisis exploratorio, contraste de hipótesis, ingeniería de variables, modelos supervisados, segmentación no supervisada, explicabilidad con SHAP, calibración de probabilidades, priorización de clientes y una aplicación interactiva en Streamlit.

El resultado principal no es solamente un modelo que estima quién podría abandonar, sino un sistema que permite responder tres preguntas de negocio:

1. ¿Qué clientes presentan mayor riesgo?
2. ¿Por qué el modelo les asigna ese riesgo?
3. ¿A quién conviene intervenir primero cuando los recursos son limitados?

## 2. Problema de negocio

Retener un cliente suele ser más eficiente que reemplazarlo, pero una estrategia de retención aplicada de forma indiscriminada consume recursos y puede ser poco efectiva.

El problema abordado consiste en identificar clientes con mayor probabilidad de abandono y priorizar acciones según riesgo, valor comercial y capacidad operativa.

La variable objetivo es:

- `churn = 1`: cliente que abandona.
- `churn = 0`: cliente que permanece.

El objetivo analítico es construir una solución predictiva e interpretable, evitando que la salida del modelo se convierta automáticamente en una decisión de negocio sin considerar restricciones reales.

## 3. Dataset y calidad de datos

El dataset contiene 15.000 registros y 30 variables originales.

Durante la auditoría se detectaron:

- valores faltantes;
- edades inválidas;
- inconsistencias temporales;
- combinaciones país/ciudad de confiabilidad dudosa;
- variables con posible riesgo de fuga de información;
- desbalance de clases.

La estrategia adoptada fue conservar trazabilidad de las decisiones y evitar correcciones arbitrarias. Por ejemplo, las relaciones geográficas inconsistentes no se corrigieron sin una fuente externa autoritativa.

Para reducir riesgo de *data leakage*:

- el split se realiza antes del preprocesamiento;
- la imputación se ajusta solamente sobre entrenamiento;
- se excluyen identificadores;
- se revisan variables potencialmente posteriores al evento objetivo;
- la calibración utiliza un conjunto separado del test final.

## 4. Análisis exploratorio e hipótesis

El EDA permitió identificar patrones relevantes en satisfacción, interacción con soporte, gasto y actividad.

Se analizaron tres hipótesis principales.

### H1 — Baja satisfacción y abandono

Los clientes con satisfacción baja presentan una tasa de abandono de aproximadamente 49,7%, frente a 9,3% en el grupo de referencia.

- Riesgo relativo: 5,33×
- V de Cramér: 0,402

### H2 — Fricción de soporte y abandono

Los clientes con 5 o más tickets de soporte presentan una tasa de abandono aproximada de 50,4%, frente a 13,3% entre quienes tienen entre 0 y 4 tickets.

- Riesgo relativo: 3,79×
- V de Cramér: 0,232

### H3 — Bajo gasto y abandono

Los clientes del quintil inferior de gasto presentan una tasa de abandono aproximada de 40,5%, frente a 9,5% para los quintiles superiores.

- Riesgo relativo: 4,25×
- V de Cramér: 0,340

Estos resultados se interpretan como asociaciones estadísticas, no como evidencia causal.

## 5. Ingeniería de variables y modelado

Se construyó un flujo reproducible de feature engineering y preprocesamiento.

Se evaluaron:

- DummyClassifier;
- Regresión Logística;
- Regresión Logística balanceada;
- Random Forest;
- Gradient Boosting;
- XGBoost;
- XGBoost optimizado;
- CatBoost;
- ensemble de Gradient Boosting + XGBoost + CatBoost.

Debido al desbalance, la evaluación no se basó solamente en accuracy. Se priorizaron:

- recall;
- precision;
- F1;
- balanced accuracy;
- ROC-AUC;
- PR-AUC;
- matriz de confusión.

El DummyClassifier alcanza una accuracy elevada por efecto de la clase mayoritaria, pero no detecta abandonos. Esto demuestra por qué accuracy por sí sola sería una métrica inadecuada.

## 6. Resultados predictivos

El ensemble alcanzó aproximadamente:

- ROC-AUC: 0,921;
- PR-AUC: 0,547;
- recall: 96,5%;
- F1: 0,675.

Además, el 20% de clientes con mayor prioridad concentra aproximadamente el 69,6% de los abandonos reales en la evaluación del ranking.

Esto indica que el sistema logra concentrar una proporción alta del riesgo en una porción reducida de la cartera.

## 7. Explicabilidad con SHAP

Para evitar una solución de caja negra, se implementó explicabilidad global e individual.

Las variables más relevantes incluyen de forma consistente:

- satisfacción;
- gasto total;
- tickets de soporte;
- gasto por visita;
- tasa de tickets de soporte.

SHAP permite explicar qué variables aumentan o reducen el riesgo de cada cliente.

Para la explicación individual formal se utiliza XGBoost, cuyo rendimiento es próximo al ensemble y permite aplicar TreeSHAP de manera fiel y aditiva.

## 8. Segmentación no supervisada

Se implementó clustering sin utilizar `churn` para crear los grupos.

Se identificaron cuatro perfiles:

- Insatisfechos en riesgo;
- Alta fricción de soporte;
- Inactivos satisfechos;
- Activos satisfechos.

La variable objetivo se observó luego de formar los grupos para interpretar sus perfiles.

La solución mostró alta estabilidad entre semillas, con un Adjusted Rand Index promedio superior a 0,96.

## 9. Motor de priorización

El proyecto incorpora un puntaje operativo que combina:

- 65% riesgo de abandono;
- 25% valor comercial;
- 10% riesgo del segmento.

El resultado es un índice de prioridad de 0 a 100. No debe interpretarse como probabilidad.

El sistema también sugiere acciones diferenciadas, por ejemplo:

- recuperación de experiencia;
- escalamiento de soporte;
- reactivación personalizada;
- seguimiento preventivo;
- fidelización.

## 10. Calibración de probabilidades

Se compararon:

- probabilidades sin calibrar;
- Platt scaling;
- isotonic regression.

Se utilizó una partición estratificada de:

- 64% entrenamiento;
- 16% calibración;
- 20% test final.

Resultados del modelo operativo:

- Brier Score sin calibrar: 0,1039;
- Brier Score con Platt: 0,0755;
- ECE sin calibrar: 0,1062;
- ECE con Platt: 0,0098;
- ROC-AUC: 0,9184 antes y después de calibrar.

Platt scaling fue seleccionado porque mejora fuertemente la calidad probabilística sin alterar el ranking de riesgo.

## 11. Política de decisión por capacidad

El proyecto separa deliberadamente predicción de decisión.

Un umbral fijo de 0,50 no se considera una regla universal. En cambio, la empresa puede definir cuántos clientes intervenir según presupuesto y capacidad.

Sobre el conjunto de test:

- Top 10% captura aproximadamente 35,7% de los abandonos.
- Top 20% captura aproximadamente 67,4%.
- Top 25% captura aproximadamente 84,8%.
- Top 30% alcanza prácticamente el 100% en ese conjunto.

Esta capa permite convertir el modelo en una herramienta operativa bajo restricciones reales.

## 12. Aplicación Streamlit

La solución se desplegó en Streamlit Community Cloud.

Aplicación:

https://ppivproyectointegrador2026-3nmnzhsbpyrbvub9rtrwoc.streamlit.app/

Incluye:

- resumen ejecutivo;
- riesgo y priorización;
- desempeño de modelos;
- IA explicable;
- segmentación;
- hipótesis e insights;
- calidad de datos y EDA;
- predicción individual en vivo;
- scoring masivo por CSV;
- calibración;
- simulación de capacidad operativa.

## 13. Predicción sobre clientes nuevos

La sección de predicción en vivo utiliza un XGBoost operativo compacto.

Permite modificar variables de un cliente y obtener:

- score de riesgo;
- banda orientativa;
- acción sugerida;
- explicación local.

También permite cargar un CSV de nuevos clientes, realizar scoring masivo y descargar los resultados ordenados.

El ensemble se conserva como benchmark de máximo desempeño, mientras que XGBoost se utiliza para inferencia por su simplicidad operativa y capacidad de explicación.

## 14. Conclusiones

El proyecto demuestra que una solución de churn útil requiere más que un clasificador con buenas métricas.

Los principales aprendizajes son:

1. La calidad del dato condiciona todo el análisis.
2. El desbalance obliga a seleccionar métricas adecuadas.
3. La explicabilidad es necesaria para convertir una predicción en una herramienta confiable.
4. El clustering aporta contexto para diferenciar acciones.
5. La calibración permite interpretar mejor las probabilidades.
6. La decisión final debe considerar capacidad, costo y estrategia.
7. El mayor valor aparece al integrar predicción, explicación y priorización.

La solución final cumple el recorrido completo desde datos crudos hasta una aplicación desplegada orientada a negocio.

## 15. Limitaciones

- El dataset no representa necesariamente una empresa real específica.
- Las asociaciones observadas no implican causalidad.
- No se incorporaron costos reales de campañas o incentivos.
- La política óptima de intervención debería recalibrarse con datos productivos.
- El monitoreo de drift todavía no está implementado.
- Las explicaciones contrafactuales quedan como extensión futura.

## 16. Próximos pasos

Como extensiones posibles:

- incorporar costos y beneficio esperado por acción;
- construir contrafactuales accionables;
- monitorear drift de variables y performance;
- agregar seguimiento temporal del modelo;
- evaluar causalidad de acciones de retención;
- realizar experimentos A/B sobre campañas.

## 17. Reproducibilidad

El repositorio incluye notebooks, scripts, documentación, snapshots analíticos, modelo operativo y aplicación.

Ejecución local:

```bash
git clone https://github.com/ferfava/PPIV_Proyecto_Integrador_2026.git
cd PPIV_Proyecto_Integrador_2026
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

**Repositorio:** https://github.com/ferfava/PPIV_Proyecto_Integrador_2026
