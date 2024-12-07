# **Scikit-Learn**

[Scikit-learn](https://scikit-learn.org/stable/user_guide.html) es una de las bibliotecas más populares de Python para Machine Learning. Ofrece herramientas simples y eficientes para el análisis de datos y modelado predictivo. Compatible con NumPy, pandas y matplotlib, es ideal para **construir y evaluar modelos de Machine Learning**.

---

## **Índice**

1. [¿Qué es Machine Learning?](#1-qué-es-machine-learning)
2. [¿Qué es Scikit-Learn?](#2-qué-es-scikit-learn)
3. [Workflow en Scikit-Learn](#3-workflow-en-scikit-learn)
4. [Debugging Warnings en Jupyter](#4-debugging-warnings-en-jupyter)
5. [División de Datos (Splitting Data)](#5-división-de-datos-splitting-data)
6. [Limpieza y Transformación de Datos](#6-limpieza-y-transformación-de-datos-clean-transform-reduce)
7. [Convertir Datos en Números](#6-convertir-datos-en-números)
8. [Manejo de Valores Faltantes](#7-manejo-de-valores-faltantes)
9. [Feature Scaling (Escalado de Características)](#8-feature-scaling-escalado-de-características)
10. [Elegir el Modelo Correcto (Regresión)](#9-elegir-el-modelo-correcto-regresión)
11. [Árboles de Decisión (Decision Trees)](#10-árboles-de-decisión-decision-trees)
12. [Funcionamiento de los Algoritmos de ML](#11-funcionamiento-de-los-algoritmos-de-ml)
13. [Elegir el Modelo Correcto (Clasificación)](#12-elegir-el-modelo-correcto-clasificación)
14. [Ajustar un Modelo a los Datos](#13-ajustar-un-modelo-a-los-datos)
15. [Predicciones con un Modelo](#14-predicciones-con-un-modelo)
16. [Evaluación de Modelos de Machine Learning](#15-evaluación-de-modelos-de-machine-learning)
17. [Mejorar un Modelo de Machine Learning](#16-mejorar-un-modelo-de-machine-learning)
18. [Guardar y Cargar Modelos](#17-guardar-y-cargar-modelos)
19. [Resumen Completo y Práctica Final](#18-resumen-completo-y-práctica-final)

> [!NOTE] > **¿Cómo obtener ayuda?**
>
> - **Experimenta:** Usa lo que sabes y prueba. Aprender haciendo es clave en ML.
> - **SHIFT + TAB:** Usa esta función en Jupyter para obtener información sobre una función o método.
> - **Busca online:** Investiga en la documentación de [Scikit-Learn](<(https://scikit-learn.org/stable/user_guide.html)>) o en Stack Overflow.
> - **Pregunta:** Si te atascas, pregunta en foros o comunidades de Machine Learning.

---

## **1. ¿Qué es Machine Learning?**

Machine Learning (ML) es una rama de la inteligencia artificial que permite a las máquinas **aprender patrones de datos sin ser explícitamente programadas.**

- **Tipos de ML:**
  - **Supervisado:** Entrenado con datos etiquetados (ej. predicción de precios).
  - **No supervisado:** Identifica patrones en datos sin etiquetas (ej. clustering).
  - **Aprendizaje por refuerzo:** Aprende interactuando con el entorno para maximizar recompensas.

<img src="../assets/section-7/machine-learning.webp" alt="Qué es machine learning" width="800" style="padding:24px; margin: 24px auto; background: white;">

---

## **2. ¿Qué es Scikit-Learn?**

**Scikit-Learn**, comúnmente llamado `sklearn`, es una biblioteca de código abierto de Python para Machine Learning.

Está construido sobre:

- **NumPy:** Biblioteca de Python para cálculos numéricos.
- **Matplotlib:** Biblioteca para visualización de datos.

Ofrece herramientas para realizar todas las etapas principales de un proyecto de Machine Learning, desde la preparación de datos hasta la construcción y evaluación de modelos.

### ¿Por qué usar Scikit-Learn?

El objetivo principal del Machine Learning es **encontrar patrones en los datos** y usar esos patrones para hacer **predicciones**.

Algunos tipos comunes de problemas de Machine Learning incluyen:

- **Clasificación:** Predecir una categoría (ej. si un email es spam o no).
- **Regresión:** Predecir un número (ej. precio de casas).
- **Clustering:** Agrupar elementos similares sin etiquetas previas.

Para cualquier problema, las etapas son similares:

1. Dividir los datos en conjuntos de entrenamiento y prueba.
2. Elegir un modelo.
3. Ajustar el modelo a los datos.
4. Evaluar el modelo para ver si ha aprendido algo.

Scikit-Learn ofrece implementaciones en Python para realizar todas estas tareas, evitando la necesidad de construirlas desde cero.

---

## **3. Workflow en Scikit-Learn**

1. **Obtener y preparar los datos**
2. **Dividir los datos**
   - en características (`X`) y etiquetas (`Y`)
   - en conjuntos de entrenamiento y prueba
3. **Elegir un modelo y sus hiperparámetros**
   - Clasificación : `RandomForestClassifier`, `LogisticRegression`, `SVC`.
   - Regresión: `LinearRegression`, `RandomRegressor`.
4. **Ajustar el modelo** a los datos de entrenamiento
5. **Hacer predicciones**: predice etiquetas en datos no vistos
6. **Evaluar el modelo**
   - Clasificación: Usa métricas como precisión, matriz de confusión, etc.
   - Regresión: Usa métricas como `r2_score`, `mean_squared_error`.
7. **Experimentar y mejorar** el modelo si es necesario
   - Ajusta los hiperparámetros con `GridSerachCV`o prueba modelos más avanzados.
   - Experimenta con diferentes técnicas de preprocesamiento.
8. **Guardar y cargar el modelo**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Ejemplo simple de workflow
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Dividir los datos
model = LinearRegression() # Modelo elegido
model.fit(X_train, y_train) # Ajustar el modelo
predictions = model.predict(X_test) # Hacer predicciones
error = mean_absolute_error(y_test, predictions) # Evaluar el modelo
```

---

## **4. Debugging Warnings en Jupyter**

Mientras trabajas en Jupyter Notebook, puedes encontrarte con **mensajes de advertencia** que te alertan sobre **posibles problemas o cambios futuros** en las bibliotecas que utilizas. Estos mensajes son útiles para mejorar tu código, pero a veces pueden ser molestos si estás experimentando o trabajando con código heredado.

### **Tipos comunes de advertencias**

1. **`FutureWarning`:** Indica que algo en tu código será obsoleto en una versión futura.
2. **`DeprecationWarning`:** Similar a `FutureWarning`, pero se usa para elementos que ya están obsoletos.
3. **`UserWarning`:** Mensajes de advertencia personalizados o relacionados con configuraciones específicas.

### **Cómo manejar advertencias en Scikit-Learn**

1. **Lee el mensaje de advertencia completo:**

   - Identifica qué está causando la advertencia. Por ejemplo:
     ```
     FutureWarning: The parameter 'normalize' in function 'LinearRegression' is deprecated and will be removed in version 1.2. Please use 'StandardScaler' instead.
     ```
   - Esto sugiere reemplazar `normalize=True` por el uso de `StandardScaler`.

2. **Consulta la documentación oficial:**

   - La advertencia suele mencionar una solución recomendada. Busca el término o función en la [documentación oficial de Scikit-Learn](https://scikit-learn.org/stable/documentation.html).

3. **Actualiza tu código para evitar advertencias futuras:**

   - Adapta tu código siguiendo las recomendaciones.
   - Ejemplo de corrección:

     ```python
     # Antes (genera FutureWarning)
     from sklearn.linear_model import LinearRegression
     model = LinearRegression(normalize=True)

     # Después (solución recomendada)
     from sklearn.pipeline import make_pipeline
     from sklearn.preprocessing import StandardScaler
     model = make_pipeline(StandardScaler(), LinearRegression())
     ```

4. **Ignorar advertencias temporalmente:**

   - Si necesitas ignorar advertencias durante el desarrollo, puedes usar:
     ```python
     import warnings
     warnings.filterwarnings("ignore", category=FutureWarning)
     ```
     Esto suprime **solo** los mensajes de `FutureWarning`.

5. **Habilitar advertencias para depuración:**
   - Si necesitas volver a habilitar las advertencias:
     ```python
     warnings.filterwarnings("default")
     ```

#### **Ejemplo práctico para `FutureWarning` en Scikit-Learn**

Si estás trabajando con una versión más antigua de Scikit-Learn y ves un `FutureWarning`, actualiza la biblioteca para evitar el problema:

```bash
pip install -U scikit-learn
```

### **Actualizar Scikit-Learn con conda**

Si estás usando **conda** como gestor de entornos, puedes actualizar **Scikit-Learn** y otras bibliotecas directamente desde la terminal. Aquí tienes los pasos para hacerlo:

1. **Activar el entorno Conda:**
   Primero asegúrate de activar el entorno en el que deseas actualizar Scikit-Learn:

   ```bash
   conda activate nombre_del_entorno
   ```

2. **Actualizar Scikit-Learn:**
   Ejecuta el siguiente comando para actualizar Scikit-Learn a la última versión disponible en los canales de Conda:

   ```bash
   conda update scikit-learn
   ```

3. **Confirmar la instalación:**
   Si Conda encuentra una versión más reciente, pedirá confirmación para actualizar. Escribe `y` y presiona **Enter** para proceder.

#### **Actualizar Scikit-Learn a una versión específica:**

Si necesitas una versión específica de Scikit-Learn, usa:

```bash
conda install scikit-learn=1.2.0
```

(Reemplaza `1.2.0` con la versión que necesites.)

#### **Actualizar todo el entorno Conda**

Si prefieres actualizar todas las bibliotecas de tu entorno al mismo tiempo, puedes usar:

```bash
conda update --all
```

#### **Verificar la versión instalada**

Después de actualizar, verifica que tienes la versión correcta:

```python
import sklearn
print(sklearn.__version__)
```

> [!Note] > **Notas importantes**
>
> - Si no encuentras la versión más reciente de Scikit-Learn en los canales predeterminados de Conda, puedes intentar instalarla desde el canal `conda-forge`:
>
> ```bash
> conda install -c conda-forge scikit-learn
> ```
>
> - Actualizar Scikit-Learn puede requerir actualizar otras bibliotecas como **NumPy** y **SciPy**, ya que Scikit-Learn depende de ellas. Conda manejará estas dependencias automáticamente.
> - Si no puedes actualizar, puedes ignorar las advertencias de forma temporal o adaptarte a las nuevas recomendaciones que aparecen en la advertencia.

#### **Buenas prácticas**

- Usa `warnings.filterwarnings("ignore")` solo como última opción o mientras experimentas.
- Actualiza tus bibliotecas regularmente para evitar problemas de compatibilidad.
- Consulta siempre la documentación oficial y las notas de la versión:
  - 🔗 [Notas de versión de Scikit-Learn](https://scikit-learn.org/stable/whats_new.html).

---

## **5. División de Datos (Splitting Data)**

Un paso crucial en cualquier proyecto de machine learning es dividir los datos en conjuntos de entrenamiento y prueba para evaluar cómo se comporta el modelo con datos no vistos.

La función `train_test_split` de Scikit-Learn divide los datos en dos o más conjuntos:

- **Entrenamiento (`train`):** Datos que el modelo utiliza para aprender patrones.
- **Prueba (`test`):** Datos reservados para evaluar el modelo después del entrenamiento.

Ejemplo básico:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # Proporción del conjunto de prueba (20%)
    random_state=42 # Fijar semilla para resultados reproducibles
)
```

- **`test_size`:** Define el porcentaje de datos asignados al conjunto de prueba.
- **`random_state`:** Asegura que los datos se dividan de la misma forma en cada ejecución, útil para experimentos reproducibles.

---

## **6. Limpieza y Transformación de Datos (Clean, Transform, Reduce)**

#### **1. Limpieza (`Clean`):**

- Elimina valores faltantes o erróneos para evitar que distorsionen los resultados del modelo.
- Por ejemplo, puedes usar Pandas para eliminar filas con valores nulos:
  ```python
  X.dropna(inplace=True)
  ```

#### **2. Transformación (`Transform`):**

- Convierte los datos a formatos adecuados, como escalar valores numéricos o codificar variables categóricas.
- Ejemplo: Escalar características numéricas usando `StandardScaler`:
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  ```

#### **3. Reducción (`Reduce`):**

- Simplifica los datos, por ejemplo, reduciendo la dimensionalidad con PCA si el conjunto de datos tiene muchas características.
- Ejemplo:
  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  X_reduced = pca.fit_transform(X)
  ```

> [!NOTE] > **Nota Práctica**
>
> - Una división mal realizada puede generar un modelo que no generalice bien.
> - Si los datos son muy pequeños, considera **validación cruzada** en lugar de dividir en `train/test`.
> - Siempre inspecciona los datos después de dividirlos para garantizar que los conjuntos sean representativos:
>
> ```python
> print(X_train.shape, X_test.shape)
> print(y_train.value_counts(), y_test.value_counts())
> ```

---

## **7. Convertir Datos en Números**

Para trabajar con datos categóricos, utiliza `LabelEncoder` o `OneHotEncoder`:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
encoded = encoder.fit_transform(data[['column_name']])
```

---

## **7. Manejo de Valores Faltantes**

### Con pandas:

```python
df["column"].fillna(value, inplace=True)
```

### Con Scikit-Learn:

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
imputed = imputer.fit_transform(df)
```

---

## **8. Feature Scaling (Escalado de Características)**

Usa `StandardScaler` o `MinMaxScaler` para normalizar los datos.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```

---

## **9. Elegir el Modelo Correcto (Regresión)**

- Modelos comunes para regresión:
  - `LinearRegression`
  - `Ridge`
  - `RandomForestRegressor`

---

## **10. Árboles de Decisión (Decision Trees)**

Un modelo que utiliza reglas condicionales para dividir los datos en ramas.

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
```

---

## **11. Funcionamiento de los Algoritmos de ML**

### Tipos:

- Algoritmos lineales (ej. regresión lineal).
- Modelos basados en árboles (ej. Random Forest).
- Redes neuronales para datos complejos.

---

## **12. Elegir el Modelo Correcto (Clasificación)**

- Modelos comunes para clasificación:
  - `LogisticRegression`
  - `SVC`
  - `RandomForestClassifier`

---

## **13. Ajustar un Modelo a los Datos**

```python
model.fit(X_train, y_train)
```

---

## **14. Predicciones con un Modelo**

```python
predictions = model.predict(X_test)
```

---

## **15. Evaluación de Modelos de Machine Learning**

### Clasificación:

- `accuracy_score`
- `confusion_matrix`
- `classification_report`

### Regresión:

- `r2_score`
- `mean_absolute_error`
- `mean_squared_error`

---

## **16. Mejorar un Modelo de Machine Learning**

- Ajustar hiperparámetros con `GridSearchCV`.
- Añadir más datos o limpiar los existentes.

---

## **17. Guardar y Cargar Modelos**

```python
import joblib
joblib.dump(model, 'model.pkl')
loaded_model = joblib.load('model.pkl')
```

---

## **18. Resumen Completo y Práctica Final**

Integra todos los pasos para resolver un problema completo de Machine Learning:

1. Limpia y transforma los datos.
2. Divide los datos.
3. Escala características.
4. Ajusta un modelo.
5. Evalúa su desempeño.
6. Mejora si es necesario.

---

Este archivo está diseñado para que sea un recurso práctico y útil a medida que avances en tu aprendizaje con Scikit-learn.
