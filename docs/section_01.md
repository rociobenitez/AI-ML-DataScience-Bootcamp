## Sección 2 : Machine Lerning 101

### **¿Qué es Machine Learning?**

**Machine Learning (ML)** es un campo de la **inteligencia artificial (AI)** que permite a los sistemas aprender y mejorar automáticamente a partir de datos sin necesidad de ser programados explícitamente. Se basa en algoritmos y modelos que identifican patrones en datos y toman decisiones o predicciones basadas en ellos.

---

### **Diferencia entre AI, ML y Data Science:**

| **Campo**                        | **Definición**                                                                  | **Enfoque**                                 |
| -------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------- |
| **Inteligencia Artificial (AI)** | Simula la inteligencia humana en sistemas para resolver problemas complejos.    | General (pensar y actuar como humanos).     |
| **Machine Learning (ML)**        | Subcampo de AI que enseña a las máquinas a aprender de datos.                   | Predicciones y automatización de tareas.    |
| **Data Science**                 | Combina estadísticas, programación y ML para analizar datos y extraer insights. | Análisis, modelado y comunicación de datos. |

<img src="/assets/section-1/ai-ml-datascience.png" alt="Diferencia entre AI, ML y Data Science" width="600">

---

### **Tipos de Machine Learning:**

1. **Aprendizaje Supervisado:**

   - Los datos tienen etiquetas o resultados conocidos.
   - Ejemplo: Clasificar correos como "spam" o "no spam".

2. **Aprendizaje No Supervisado:**

   - Los datos no tienen etiquetas; se buscan patrones ocultos.
   - Ejemplo: Agrupación de clientes en grupos de comportamiento.

3. **Aprendizaje por Refuerzo:**
   - Un agente aprende interactuando con un entorno y recibe recompensas por acciones correctas.
   - Ejemplo: Robots aprendiendo a caminar.

---

### **Recursos adicionales de formación:**

1. **[Zero to Mastery](https://zerotomastery.io/):**

   - Cursos estructurados para aprender habilidades técnicas, desde Python hasta ML.
   - Excelente para principiantes y avanzados.

2. **[Zero to Mastery Blog](https://zerotomastery.io/blog/):**

   - Artículos sobre tendencias, herramientas y consejos prácticos en tecnología.

3. **[Zero to Mastery Resources](https://zerotomastery.io/resources/):**

   - Recursos gratuitos como guías y hojas de referencia.

4. **[Zero to Mastery en YouTube](https://www.youtube.com/@ZeroToMastery):**
   - Videos educativos sobre desarrollo, ciencia de datos y más.

---

### **Herramienta: [Teachable Machine](https://teachablemachine.withgoogle.com/)**

Teachable Machine es una herramienta de Google que permite entrenar modelos de aprendizaje automático sin necesidad de conocimientos avanzados.

- **Uso principal:**
  Entrenar modelos simples usando imágenes, audio o poses y exportarlos para usar en proyectos reales.

- **Ventajas:**
  - No requiere programación.
  - Fácil de usar y accesible desde el navegador.
  - Ideal para demostraciones y prototipos rápidos.

### Proyecto práctico: Detector de Postura

Entrené un modelo con **Teachable Machine** para identificar si estoy trabajando con buena postura o mala postura frente al ordenador.

**Objetivo del modelo**

- **Categorías**: Buena postura vs. Mala postura.
- **Datos utilizados**: Imágenes tomadas desde la cámara del ordenador en diferentes posiciones. Se encuentran en la carpeta `[assets/section-1](/assets/section-1/)`.
- **Aplicación**: Este modelo puede generar recordatorios para corregir mi postura durante largas sesiones de trabajo.

**Resultados iniciales**
A continuación, se muestra una captura de la prueba realizada con el modelo:

![Resultados de la prueba realizada en Teachable Machine](/assets/section-1/teachable-machine.jpeg)

---

### **Práctica con Machine Learning Playground**

#### **Simulación: Clasificación de Patrones de Movimiento**

En esta práctica, se utilizó [**Machine Learning Playground**](https://ml-playground.com/#) para explorar la clasificación de datos relacionados con patrones de movimiento humano. Aunque los datos fueron ingresados manualmente, se simulan situaciones relacionadas con biomecánica, como la separación de movimientos de **alta intensidad** y **baja intensidad** en función de características como:

- **Eje X (Velocidad):** Representa la velocidad promedio de un movimiento.
- **Eje Y (Amplitud):** Representa el rango de movimiento articular.

#### **Pruebas realizadas:**

1. **K-Nearest Neighbors (KNN):**

   - Configuración inicial con \(k = 3\), donde los puntos fueron clasificados según los vecinos más cercanos.
   - Resultado: Se generaron divisiones razonables entre puntos de alta y baja intensidad, pero algunas zonas de frontera quedaron con solapamientos.

2. **Variación del conjunto de datos:**
   - Al ajustar manualmente las posiciones de los puntos en el gráfico, se observó cómo los cambios en la distribución de los datos afectaron las fronteras de clasificación.

#### **Resultados:**

Los modelos lograron separar las clases principales (alta y baja intensidad) con un nivel aceptable de precisión, destacando la importancia de ajustar los parámetros para mejorar la clasificación en fronteras complejas.

#### **Imágenes de la práctica:**

1. Clasificación inicial con puntos y frontera generada:
   ![Práctica 1](/assets/section-1/practica.jpeg)

2. Clasificación ajustada con cambios manuales en los puntos:
   ![Práctica 2](/assets/section-1/practica-2.jpeg)

---

### **Aprendizaje**

- **Conceptos clave:** Machine Learning permite a las máquinas aprender de datos, clasificando o prediciendo sin instrucciones explícitas.
- **Tipos de ML:** Supervisado (datos etiquetados), no supervisado (patrones ocultos) y por refuerzo (aprendizaje basado en recompensas).
- **Diferencias:** ML se centra en algoritmos, AI abarca inteligencia más amplia, y Data Science analiza datos para obtener insights.
- **Herramientas útiles:**
  - **[Teachable Machine](https://teachablemachine.withgoogle.com/):** Creación rápida de modelos basados en imágenes, poses o audio.
  - **[ML Playground](https://ml-playground.com/#):** Visualización y experimentación con modelos básicos como KNN, SVM y redes neuronales.
- **Prácticas realizadas:**
  - Detector de postura: Identificación de buena y mala postura.
  - Clasificación de movimientos: Separación manual de datos según intensidad y amplitud.
