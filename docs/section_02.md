# Framework de Machine Learning

Esta sección se centra en la creación de un framework claro para abordar proyectos de Machine Learning y en el proceso técnico del modelado de datos. La estructura presentada aquí servirá como referencia práctica y organizada para implementar estos conceptos en proyectos futuros.

---

## **Índice**

1. [Framework en 6 Pasos para Proyectos de ML](#framework-en-6-pasos-para-proyectos-de-ml)
2. [Paso 1: Definición del Problema](#paso-1-definición-del-problema)
3. [Paso 2: Datos](#paso-2-datos)
4. [Paso 3: Evaluación](#paso-3-evaluación)
   - [Métricas en Regresión](#métricas-en-regresión)
   - [Métricas en Clasificación](#métricas-en-clasificación)
   - [Métricas en Problemas de Recomendación](#métricas-en-problemas-de-recomendación)
5. [Paso 4: Features o Características](#paso-4-features-o-características)
6. [Paso 5: Modelado](#paso-5-modelado)
   - [División de datos (splitting)](#1-división-de-datos-splitting)
   - [Elección del Modelo](#2-elección-del-modelo)
   - [Mejora y Ajuste del Modelo](#3-mejora-y-ajuste-del-modelo)
   - [Comparación de Modelos](#4-comparación-de-modelos)
   - [Overfitting vs Underfitting](#overfitting-vs-underfitting)
7. [Paso 6: Experimentación y Mejora](#paso-6-experimentación-y-mejora)
8. [Herramientas y Recursos Clave](#herramientas-y-recursos-clave)

---

## **Framework en 6 Pasos para Proyectos de ML**

Este framework proporciona una guía clara y organizada para abordar cualquier proyecto de Machine Learning, desde la definición inicial hasta la experimentación y mejora continua. Se puede utilizar como referencia para garantizar que cada paso esté alineado con los objetivos del problema y los datos disponibles.

1. **Definición del Problema:**

   - ¿Qué problema práctico o empresarial estamos tratando de resolver?
   - ¿Cómo puede reformularse como un problema de Machine Learning?
   - ¿Supervisado o no supervisado? ¿Problema de clasificación o de regresión?
   - Ejemplo: "Queremos predecir el abandono de clientes" → "¿Podemos usar datos históricos para identificar patrones de abandono?"

2. **Datos:**

   - ¿Qué datos tenemos disponibles? ¿Son suficientes para resolver el problema?
   - ¿Cómo se relacionan estos datos con la definición del problema?
   - Tipos de datos:
     - **Estructurados o no estructurados:** Tablas vs. imágenes, texto, audio.
     - **Estáticos o en flujo:** Datos almacenados vs. datos en tiempo real.

3. **Evaluación:**

   - ¿Qué métrica definirá el éxito del modelo?
   - Ejemplo: Una precisión del 95% puede ser excelente en algunos contextos, pero insuficiente en otros como la detección de fraudes.
   - Asegúrate de que los criterios de evaluación reflejen los objetivos prácticos.

4. **Características (Features):**

   - ¿Qué partes de nuestros datos serán relevantes para entrenar el modelo?
   - Ejemplo: Para predecir precios de casas, las características relevantes podrían incluir ubicación, tamaño y antigüedad.
   - ¿Cómo pueden influir nuestros conocimientos previos en la selección y creación de estas características?

5. **Modelado:**

   - ¿Qué modelo es el más adecuado para el problema (clasificación, regresión, clustering)?
   - ¿Cómo podemos mejorar el modelo ajustando sus hiperparámetros?
   - ¿Cómo se compara este modelo con otros en términos de métricas y rendimiento?

6. **Experimentación:**
   - ¿Qué nuevas estrategias, modelos o configuraciones podríamos probar para mejorar los resultados?
   - ¿El modelo implementado está funcionando como esperábamos en producción?
   - ¿Cómo podrían las observaciones actuales modificar las decisiones en las etapas anteriores?

Esta estructura está basada en el [6-Step Field Guide](https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/).

<img src="/assets/section-2/framework.png" alt="Framework" width="600">

> 🔗 Estructura completa en [Whimsical](https://whimsical.com/6-step-field-guide-to-machine-learning-projects-flowcharts-9g65jgoRYTxMXxDosndYTB)

---

## **Paso 1: Definición del Problema**

Definir un problema como de Machine Learning requiere identificar qué tipo de aprendizaje y enfoque será más adecuado para abordarlo. A continuación, se resumen los tipos principales de aprendizaje y problemas en ML:

### **Tipos de Aprendizaje en ML**

1. **Aprendizaje Supervisado:**

   - Utiliza datos **etiquetados** (inputs con resultados conocidos) para entrenar un modelo que pueda predecir resultados futuros.
   - _Ejemplo: Predecir si un paciente tiene o no una enfermedad cardíaca, basándose en historiales médicos etiquetados._
   - El modelo devuelve predicciones probabilísticas, como "70% de probabilidad de tener la enfermedad".

2. **Aprendizaje No Supervisado:**

   - Trabaja con datos **no etiquetados** para identificar patrones o relaciones entre ellos.
   - _Ejemplo: Agrupar clientes de una tienda según su historial de compras (clustering). Las etiquetas (categorías) son añadidas posteriormente por el experto en dominio._
   - Útil para descubrimientos en grandes volúmenes de datos.

3. **Transfer Learning:**

   - Aprovecha un modelo **previamente entrenado** y lo ajusta a un problema nuevo y específico.
   - _Ejemplo: Utilizar un modelo preentrenado en texto (como uno basado en Wikipedia) para clasificar si un reclamo de seguro está "a favor" o "en contra" del cliente._

4. **Reinforcement Learning (Aprendizaje por Refuerzo)**:
   - Menos común en negocios estándar debido a:
     - Requisitos de computación
     - Complejidad del diseño
     - Largo tiempo de entrenamiento

### **Tipos de Problemas en ML**

1. **Clasificación:** _(supervisado)_

   - Predice etiquetas discretas para los datos.
   - Ejemplos:
     - Clasificar correos como "spam" o "no spam" (**binaria**).
     - Determinar niveles como "bajo", "medio", "alto" (**multi-clase**).
     - Asignar múltiples etiquetas a un dato (**multi-etiqueta**).

2. **Regresión:**

   - Predice valores numéricos continuos.
   - Ejemplo: Estimar el precio de una casa o predecir ventas futuras.

3. **Recomendación:**

   - Sugiere elementos personalizados en función de datos previos.
   - Ejemplo: Recomendar productos en una tienda online según historial de compras.

4. **Clustering:** _(no supervisado)_
   - Agrupa datos no etiquetados en categorías basadas en similitudes.
   - Ejemplo: Segmentar clientes en grupos según su comportamiento de compra.

<img src="/assets/section-2/tipos-ml.png" alt="Tipos de problemas en ML" width="600">

### **Ejemplo Aplicado: Reclamos de Seguros**

**Problema:** Un gran número de reclamos de seguros está llegando más rápido de lo que el personal puede manejarlos. Hay datos históricos etiquetados que indican si un reclamo fue "con culpa" o "sin culpa".

**Definición en términos de ML:**
Queremos clasificar los reclamos como "con culpa" o "sin culpa". Esto convierte el problema en un caso de **clasificación supervisada**.

> Para definir un problema de negocio como un problema de Machine Learning:
>
> 1. Reformúlalo en términos simples (p. ej., "¿podemos clasificar esto?").
> 2. Decide qué tipo de problema es (clasificación, regresión o recomendación).
> 3. Define el objetivo inicial en una sola frase y agrega complejidad conforme sea necesario.

---

## **Paso 2: Datos**

- **Estructurados:** Tablas con columnas bien definidas.
- **No estructurados:** Datos complejos como imágenes, audio o texto.
- **Datos estáticos:** Datos históricos que rara vez cambian o se actualizan.
- **Datos en streaming**: Datos que se generan y actualizan de manera continua.
- **Datos mixtos:** Combinación de datos estructurados y no estructurados.

---

## **Paso 3: Evaluación**

La evaluación de modelos de Machine Learning es fundamental para determinar su desempeño y utilidad en un problema práctico. Las métricas de evaluación varían según el tipo de problema: clasificación, regresión o recomendación.

### **Métricas en Regresión**

- **MAE (Mean Absolute Error - Error Absoluto Medio)**
  - Promedio de las diferencias absolutas entre las predicciones del modelo y los valores reales.
  - Fórmula: \( MAE = \frac{1}{n} \sum\_{i=1}^n | y_i - \hat{y}\_i | \)
- **RMSE (Root Mean Square Error)**
  - Promedio de los errores al cuadrado. Penaliza más los errores grandes.
  - Fórmula: \( MSE = \frac{1}{n} \sum\_{i=1}^n (y_i - \hat{y}\_i)^2 \)
- **Raíz del Error Cuadrático Medio (Root Mean Squared Error, RMSE):**
  - Raíz cuadrada del MSE. Penaliza los errores grandes más que el MAE.
  - Fórmula: \( RMSE = \sqrt{MSE} \)
- **\( R^2 \) (Coeficiente de determinación)**
  - Mide qué proporción de la variabilidad de los datos es explicada por el modelo.
  - Valores cercanos a 1.0 indican un ajuste casi perfecto.

#### **Diferencia entre MAE y RMSE:**

- **MAE:** Trata todos los errores con igual peso.
- **RMSE:** Penaliza más los errores grandes. Útil cuando estos son críticos.

### **Métricas en Clasificación**

- **Accuracy**
  - Proporción de predicciones correctas sobre el total.
  - Ejemplo: Si un modelo predice correctamente 90 de 100 correos como "spam" o "no spam", la precisión es 90%.
- **Precision**
  - Proporción de predicciones positivas que son realmente correctas.
  - Fórmula: \( \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} \)
  - Útil cuando es importante minimizar los **falsos positivos** (por ejemplo, predecir una enfermedad que no existe).
- **Recall (sensibilidad)**
  - Proporción de verdaderos positivos correctamente identificados.
  - Fórmula: \( \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \)
  - Útil cuando es crucial minimizar los **falsos negativos** (por ejemplo, no detectar una enfermedad grave).
- **F1-Score**
  - Promedio armónico de precisión y recall. Balancea la importancia de ambas métricas.
  - Fórmula: \( F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \)
- **ROC/AUC** (Receiver Operating Characteristic / Área bajo la curva)
  - La curva ROC compara la tasa de verdaderos positivos frente a la tasa de falsos positivos a diferentes umbrales.
  - El AUC mide el área bajo la curva ROC:
    - **1.0:** Predicciones perfectas.
    - **0.5:** Modelo aleatorio.
    - **0.0:** Predicciones 100% incorrectas.

#### **Errores Comunes:**

- **Falsos negativos:** Predice "negativo" pero es "positivo".
  - Ejemplo: No detectar un peatón en un sistema de visión para coches autónomos.
- **Falsos positivos:** Predice "positivo" pero es "negativo".
  - Ejemplo: Diagnosticar erróneamente una enfermedad.

### **Métricas en Problemas de Recomendación**

Los problemas de recomendación presentan desafíos únicos, ya que el objetivo no es solo acertar, sino también optimizar el orden de las sugerencias.

- **Precisión en K (Precision @ K):**

  - Proporción de recomendaciones correctas dentro de las **K mejores opciones**.
  - Ejemplo: Si un modelo recomienda 5 productos y 3 son relevantes, la precisión en K es \( \frac{3}{5} = 0.6 \).

- **Cobertura:**

  - Proporción de elementos del conjunto total que son recomendados al menos una vez.

- **Índice de Diversidad:**
  - Evalúa cuán variadas son las recomendaciones. Ayuda a evitar sugerencias repetitivas.

#### **Método de Evaluación:**

- **División Temporal:**
  - Usar datos históricos para entrenar el modelo (por ejemplo, datos de 2010-2018).
  - Probar el modelo con datos recientes (2019) para simular su rendimiento en el futuro.

---

### **Elección de la Métrica**

La métrica adecuada depende del contexto:

- **Clasificación:** Usa F1-Score o AUC si el balance entre precisión y recall es importante.
- **Regresión:** Usa MAE si los errores pequeños son más importantes que los grandes; RMSE si los errores grandes son críticos.
- **Recomendación:** Usa Precision @ K si el orden de las recomendaciones es relevante.

---

## **Paso 4: Features o Características**

Las **features** representan los atributos de los datos que se utilizan para construir un modelo de Machine Learning. Identificar, seleccionar y crear características relevantes es crucial para mejorar el rendimiento del modelo. Existen tres tipos principales de características:

### **Tipos de Features:**

1. **Categorical (Categóricas):**

   - Representan categorías o grupos.
   - Ejemplo: Sexo del paciente (hombre/mujer) o si un cliente realizó una compra (sí/no).

2. **Continuous (Continuas o Numéricas):**

   - Representan valores numéricos.
   - Ejemplo: Frecuencia cardíaca promedio o número de veces que un usuario inició sesión.

3. **Derived (Derivadas):**
   - Características creadas a partir de los datos existentes mediante ingeniería de características (feature engineering).
   - Ejemplo:
     - Combinar fechas y tiempos para calcular "tiempo desde el último inicio de sesión".
     - Transformar fechas en "día laboral (sí/no)".

<img src="/assets/section-2/features.png" alt="Ejemplo Features" width="600">

### **Consideraciones Importantes:**

- **Consistencia en el entrenamiento y la producción:** Las características utilizadas en la fase de entrenamiento deben representar fielmente las condiciones del entorno real donde se usará el modelo.
- **Colaboración con expertos en la materia:** Incorporar conocimientos del dominio para identificar y diseñar características relevantes.
- **Cobertura de datos:** Dar preferencia a características que cubran la mayor cantidad de muestras. Si solo un 10% de los datos contienen una característica, podría no ser útil para el modelo.
- **Evitar fugas de datos (feature leakage):** Si el modelo alcanza un rendimiento perfecto, podría estar utilizando información del conjunto de prueba durante el entrenamiento, lo cual no refleja un uso realista.

#### **Uso de Features en Modelos:**

- **Establecimiento de una línea base:**
  - Usar conocimiento del dominio para crear predicciones iniciales simples.
  - Ejemplo: "Un cliente que no inicia sesión en tres semanas tiene un 80% de probabilidad de cancelar su suscripción".
- **Transformación en números:**
  - Todas las características, incluidas imágenes o texto, deben convertirse a valores numéricos antes de ser utilizadas en un modelo.

> Las características son la base de cualquier modelo de Machine Learning. Una buena selección y diseño de features puede marcar la diferencia entre un modelo mediocre y uno excelente. Este paso requiere tanto conocimientos técnicos como colaboración con expertos en el dominio del problema.

---

## **Paso 5: Modelado**

El modelado es el núcleo de Machine Learning y consiste en convertir datos procesados en predicciones útiles. Este proceso se divide en las siguientes etapas:

### **1. **División de datos (splitting):\*\*

- Separar el conjunto de datos en **entrenamiento**, **validación** y **prueba** para evitar sobreajuste.
- Proporción típica: 70% entrenamiento, 15% validación, 15% prueba.

<img src="/assets/section-2/splitting-data.png" alt="División de los datos" width="600">

> **Conjunto de datos de entrenamiento:**
>
> - Se utiliza para entrenar el modelo.
> - Lo habitual es asignar el **70-80%** de los datos.
>   **Conjunto de datos de validación/desarrollo:**
> - Se utiliza para ajustar los hiperparámetros del modelo y evaluar los experimentos.
> - Lo habitual es asignar el **10-15%** de los datos.
>   **Conjunto de datos de prueba:**
> - Se utiliza para probar y comparar el modelo.
> - Lo habitual es asignar el **10-15%** de los datos.

### **2. Elección del Modelo _(Training)_**

- Elegir un algoritmo adecuado según el tipo de problema (regresión, clasificación, clustering, etc.).
- Seleccionar el modelo adecuado implica considerar los siguientes factores clave:

1. **Interpretabilidad y facilidad de depuración:**

   - ¿Por qué el modelo tomó una decisión específica?
   - ¿Qué tan fácil es identificar y corregir errores?

2. **Cantidad de datos:**

   - ¿Cuántos datos tienes disponibles?
   - ¿Es probable que esta cantidad aumente en el futuro?

3. **Limitaciones de entrenamiento y predicción:**
   - ¿Cuánto tiempo y recursos computacionales tienes para entrenar y usar el modelo?

> **Recomendación:** Comienzar con modelos simples. Los modelos complejos pueden ofrecer mejoras marginales a costa de mayores tiempos de entrenamiento y predicción.

**Tipos de Modelos:**

- **Modelos lineales (p. ej., regresión logística):**

  - Rápidos, fáciles de interpretar y depurar.
  - Útiles para problemas simples y datos lineales.

- **Modelos basados en árboles y boosting (p. ej., [Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2), [XGBoost](https://xgboost.ai/?ref=mrdbourke.com), [CatBoost](https://catboost.ai/?ref=mrdbourke.com)):**

  - Ideales para datos estructurados (tablas).
  - Suelen ofrecer un buen rendimiento en la mayoría de los casos prácticos.

- **Redes neuronales profundas:**

  - Adecuadas para datos no estructurados (imágenes, audio, texto).
  - Requieren más recursos y son más difíciles de depurar, pero son efectivas en problemas complejos.

- **Transfer Learning (aprendizaje por transferencia):**
  - Aprovecha modelos preentrenados (disponibles en plataformas como [PyTorch Hub](https://pytorch.org/hub/?ref=mrdbourke.com), [TensorFlow Hub](https://www.tensorflow.org/hub?ref=mrdbourke.com&hl=es), [ModelZoo](https://modelzoo.co/?ref=mrdbourke.com) y [Fast.ai](https://github.com/fastai/fastai?ref=mrdbourke.com)) para reducir tiempos de entrenamiento y mejorar la eficiencia.
  - Combina las ventajas de los modelos profundos y lineales.

### **3. Mejora y Ajuste del Modelo (Tuning and improving) _(Validation)_**

Los modelos pueden mejorarse ajustando hiperparámetros y configuraciones específicas. Este proceso se denomina **tuning** y puede incluir:

1. **Hiperparámetros:**

   - **Tasa de aprendizaje:** Ajusta la velocidad a la que el modelo aprende.
   - **Optimizador:** Algoritmo que controla cómo se actualizan los pesos del modelo.

2. **Específico del modelo:**
   - Número de árboles en Random Forest.
   - Número y tipo de capas en redes neuronales.

**Automatización:**  
Muchas herramientas modernas automatizan el ajuste de hiperparámetros, lo que mejora la eficiencia y reproducibilidad de los modelos.

**Prioridad:**

- **Reproducibilidad:** Documentar los pasos de ajuste para que puedan replicarse.
- **Eficiencia:** Minimizar el tiempo de entrenamiento para probar más ideas rápidamente.

### **4. Comparación de Modelos _(Test)_**

- Evaluar múltiples algoritmos y elegir el que mejor se ajuste a los datos con base en métricas relevantes.
- Comparar modelos requiere consistencia en los datos utilizados durante el entrenamiento y la evaluación.
- Es recomendable seguir estas reglas:
  - **Entrenar modelos con los mismos datos de entrenamiento (X).**
  - **Evaluarlos con los mismos datos de prueba o validación (Y).**
  - **Mantener métricas consistentes:** Comparar modelos con la misma métrica de evaluación (por ejemplo, accuracy, F1-score, RMSE, etc.).

> **Nota:**
>
> - Cambiar los datos de entrenamiento o validación puede generar comparaciones inválidas ("manzanas con naranjas").
> - Asegúrate de que los resultados reflejen las diferencias en los modelos, no en los datos.

---

### **Overfitting vs. Underfitting**

Al entrenar modelos de Machine Learning, el equilibrio entre **overfitting** y **underfitting** es clave para obtener resultados que generalicen bien a nuevos datos. Ambos conceptos representan extremos problemáticos en el proceso de modelado:

### **Overfitting**

**Definición:** Ocurre cuando un modelo se ajusta demasiado a los datos de entrenamiento, aprendiendo incluso patrones irrelevantes o ruido. Como resultado, funciona muy bien con los datos de entrenamiento, pero falla al generalizar con datos nuevos.
**Causas comunes:**

- Modelo **demasiado complejo** para la cantidad o calidad de los datos.
- Uso excesivo de características **irrelevantes**.
- Entrenamiento del modelo durante demasiadas iteraciones.
  **Indicadores:**
- Muy baja pérdida en el conjunto de entrenamiento pero alta pérdida en el conjunto de prueba.
- Métricas de evaluación significativamente mejores en entrenamiento que en prueba.
  **Cómo mitigarlo:**
- Usar regularización (p. ej., L1, L2 o dropout en redes neuronales).
- Simplificar el modelo (menos parámetros o menos capas en redes profundas).
- Incrementar la cantidad o calidad de los datos de entrenamiento.
- Utilizar técnicas como validación cruzada para evaluar el modelo durante el entrenamiento.

### **Underfitting**

**Definición:** Ocurre cuando el modelo no logra capturar los patrones importantes en los datos. Esto lleva a un desempeño pobre tanto en entrenamiento como en prueba.
**Causas comunes:**

- Modelo demasiado **simple** para la complejidad de los datos.
- **Insuficiente** tiempo de entrenamiento o mala configuración de hiperparámetros.
- Selección **incorrecta** de características (falta de información relevante).
  **Indicadores:**
- Alta pérdida tanto en el conjunto de entrenamiento como en el de prueba.
- Métricas de evaluación bajas en ambos conjuntos.
  **Cómo mitigarlo:**
- Incrementar la complejidad del modelo (más parámetros o capas).
- Asegurarse de que los datos incluyan suficiente información relevante para el problema.
- Ajustar hiperparámetros como la tasa de aprendizaje o la arquitectura del modelo.

#### **Ejemplo Visual**

|                  | **Entrenamiento** | **Prueba**     | **Problema**          |
| ---------------- | ----------------- | -------------- | --------------------- |
| **Overfitting**  | Alta precisión    | Baja precisión | Generalización pobre. |
| **Underfitting** | Baja precisión    | Baja precisión | Modelo poco útil.     |
| **Buen ajuste**  | Alta precisión    | Alta precisión | Modelo equilibrado.   |

<img src="/assets/section-2/over-under-fitting.png" alt="Overfitting vs Underfitting" width="600">

> El objetivo en el modelado es **encontrar un balance** donde el modelo capture patrones significativos (evitando underfitting) sin ajustarse en exceso a los datos de entrenamiento (evitando overfitting). Técnicas como la **validación cruzada, la regularización y una cuidadosa selección del modelo** son esenciales para lograr este equilibrio.

### Interpretación del rendimiento del modelo

**Bajo rendimiento en los datos de entrenamiento:** El modelo no ha aprendido correctamente y está underfitting. Intenta usar un modelo diferente, mejorar el modelo existente ajustando hiperparámetros o recopilar más datos.

**Alto rendimiento en los datos de entrenamiento pero bajo rendimiento en los datos de prueba:** Esto indica que el modelo no generaliza bien y podría estar overfitting los datos de entrenamiento. Intenta usar un modelo más simple o asegúrate de que los datos de prueba sean similares en estilo a los datos de entrenamiento.

**Mejor rendimiento en los datos de prueba que en los datos de entrenamiento:** Esto podría indicar que los datos de prueba están filtrándose en los datos de entrenamiento (división incorrecta de datos) o que has pasado demasiado tiempo optimizando el modelo para el conjunto de prueba. Asegúrate de mantener siempre separados los conjuntos de entrenamiento y prueba, y evita optimizar el rendimiento del modelo en los datos de prueba (usa los conjuntos de entrenamiento y validación para mejorar el modelo).

**Bajo rendimiento en producción (entorno real):** Esto indica que existe una diferencia entre los datos usados durante el entrenamiento y prueba, y los datos reales en producción. Asegúrate de que los datos utilizados durante la experimentación sean representativos de los datos que el modelo encontrará en producción.

---

## **Paso 6: Experimentación y Mejora**

La experimentación es esencial para optimizar el rendimiento del modelo. Los aspectos clave incluyen:

1. **Validación cruzada (Cross-validation):**

   - Dividir los datos en varios subconjuntos para probar el modelo de manera consistente.

2. **Técnicas de búsqueda de hiperparámetros:**

   - **Grid Search:** Prueba exhaustiva de combinaciones de parámetros.
   - **Random Search:** Selección aleatoria de combinaciones para mayor eficiencia.

3. **Documentación de experimentos:**
   - Registrar cambios en parámetros, métricas y observaciones para análisis comparativo.

---

## **Herramientas y Recursos Clave**

En esta sección, hemos explorado cómo abordar el modelado de datos en Machine Learning de manera estructurada, utilizando un marco de seis pasos. Para llevar a cabo cada etapa del proceso de manera efectiva, contamos con un conjunto de **herramientas y recursos** esenciales:

#### **Sistema**

- **Anaconda:**

  - Entorno integrado que facilita la gestión de paquetes y entornos virtuales.
  - Ideal para mantener un espacio de trabajo limpio y organizado durante todo el proyecto.
  - **Uso:** Configuración inicial del entorno para la **definición del problema** y la **preparación de datos**.

- **Jupyter Notebook:**
  - Herramienta interactiva para escribir, ejecutar y documentar código.
  - Facilita el análisis exploratorio, la experimentación y la presentación de resultados.
  - **Uso:** Durante la **preparación de datos**, la **experimentación** y la **visualización** de resultados.

#### **Framework y Guías**

- **[6-Step Field Guide](https://www.mrdbourke.com/a-6-step-field-guide-for-building-machine-learning-projects/):**
  - Una referencia estructurada para abordar proyectos de Machine Learning.
  - Proporciona un marco claro y repetible para todos los pasos del proceso.

#### **Bibliotecas**

- **Scikit-learn:**

  - Algoritmos clásicos de ML para clasificación, regresión y clustering.
  - Incluye herramientas para dividir datos, ajustar modelos y evaluar resultados.
  - **Uso:** Modelado, ajuste de hiperparámetros y comparación de modelos.

- **TensorFlow y PyTorch:**

  - Frameworks avanzados para construir y entrenar redes neuronales.
  - TensorFlow es ideal para modelos escalables y producción; PyTorch es preferido para investigación y experimentación.
  - **Uso:** Modelado avanzado y experimentación con datos no estructurados (imágenes, texto, audio).

- **Pandas y NumPy:**

  - **Pandas:** Manipulación de datos tabulares, limpieza y transformación.
  - **NumPy:** Cálculos numéricos eficientes, como operaciones matriciales.
  - **Uso:** En la **preparación de datos** y la creación de características.

- **dmlc XGBoost:**
  - Biblioteca especializada en algoritmos de boosting, ideal para datos estructurados.
  - Ofrece alta precisión y eficiencia en problemas de clasificación y regresión.
  - **Uso:** Modelado y experimentación con algoritmos basados en árboles.

#### **Visualización**

- **Matplotlib y Seaborn:**

  - Creación de gráficos estáticos para análisis y presentación.
  - **Uso:** Durante la **exploración de datos** y la interpretación de resultados.

- **Plotly:**
  - Herramienta de visualización interactiva, útil para análisis más dinámicos.
  - **Uso:** Presentación de resultados y visualización avanzada en la etapa de **experimentación**.

---

## **Recurso Opcional: Elements of AI**

> Un recurso adicional es la página web de **[Elements of AI](https://www.elementsofai.com/)**. Tiene excelentes explicaciones introductorias de muchos conceptos relacionados con el aprendizaje automático, la ciencia de datos y la inteligencia artificial.
