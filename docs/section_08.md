# Problema de Clasificación

## Introducción

Esta sección explora cómo abordar un **problema de clasificación** utilizando el dataset de Kaggle sobre **enfermedades cardíacas**. El objetivo es desarrollar un **flujo de trabajo completo que aborde las etapas clave de un proyecto de machine learning**, desde la exploración de los datos hasta la optimización del modelo.

Este proyecto aborda un problema de **clasificación binaria** utilizando un [dataset sencillo de Kaggle relacionado con enfermedades cardíacas](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset). Los datos originales provienen de la [base de datos de Cleveland del Repositorio de Aprendizaje Automático de UCI](https://archive.ics.uci.edu/dataset/45/heart+disease). El objetivo es **predecir la presencia o ausencia de enfermedad cardíaca en función de características clínicas.**

El flujo de trabajo para este proyecto se explica de manera resumida en el cuaderno [heart-disease-classification.ipynb](../notebooks/5-structured-data-projects/heart-disease-classification.ipynb). Sin embargo, para un análisis más detallado y estructurado, se puede consultar [el repositorio completo en GitHub](https://github.com/rociobenitez/heart-disease-prediction), donde se incluyen optimizaciones, experimentos adicionales y una estructura más modular.

## Índice

1. [Configuración del Entorno](#1-configuración-del-entorno)
2. [Exploración de Datos (EDA)](#2-exploración-de-datos-eda)
3. [Flujo de Trabajo de Machine Learning](#3-flujo-de-trabajo-de-machine-learning)
4. [Resultados del Modelado](#4-resultados-del-modelado)
5. [Conclusión Final](#5-conclusión-final)
6. [Recursos Complementarios](#6-recursos-complementarios)

## 1. Configuración del Entorno

### Importancia de un entorno reproducible

La gestión de entornos es fundamental para garantizar que los experimentos sean consistentes y que los resultados sean reproducibles en diferentes sistemas. Este proyecto utiliza `conda` para gestionar las dependencias.

### Pasos para configurar el entorno:

1. **Exportar el entorno existente:**
   Si deseas duplicar el entorno usado para este proyecto, puedes exportarlo con:

   ```bash
   conda env export > environment.yml
   ```

2. **Crear un nuevo entorno a partir del archivo exportado:**

   ```bash
   conda env create --file environment.yml
   ```

3. **Activar el entorno:**

   ```bash
   conda activate nombre-del-entorno
   ```

4. **Verificar dependencias instaladas:**
   ```bash
   conda list
   ```

### Dependencias clave del proyecto:

- Python 3.10+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

> [!Nota]
> Al ser un dataset relativamente sencillo y pequeño, no se requiere una gran etapa de preprocesamiento. Los datos ya están **preparados y limpios**, sin valores faltantes, lo que simplifica el flujo de trabajo.

## 2. Exploración de Datos (EDA)

### Propósito

El EDA permite comprender la estructura del dataset, identificar patrones relevantes y detectar posibles problemas, como valores atípicos.

### Insights clave:

- **Distribución de las variables:** Se analizaron variables como `age`, `sex`, `thalach` (frecuencia cardíaca máxima) y su relación con `target` (presencia o ausencia de enfermedad cardíaca).
- **Matriz de correlación:** Mostró relaciones clave entre variables como `cp` (tipo de dolor torácico), `thalach` y `oldpeak` con la variable objetivo.
- **Visualizaciones:** Se utilizaron gráficos de barras y de dispersión para identificar tendencias iniciales.

## 3. Flujo de Trabajo de Machine Learning

### Resumen del Flujo

El flujo de trabajo se resume en el cuaderno [**heart-disease-classification.ipynb**](../notebooks/5-structured-data-projects/heart-disease-classification.ipynb), mientras que en el [repositorio completo](https://github.com/rociobenitez/heart-disease-prediction) se explica con más detalle y estructura. A continuación, se presenta un resumen de los pasos principales:

### 3.1 División de los datos

Los datos fueron divididos en conjuntos de entrenamiento y prueba:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### 3.2 Entrenamiento inicial de modelos

Se probaron tres modelos principales:

- **Random Forest**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**

Cada modelo fue entrenado con los datos y evaluado inicialmente con métricas clave como precisión (`Accuracy`), `Recall` y `ROC-AUC`.

### 3.3 Optimización de hiperparámetros

Se ajustaron los hiperparámetros de los modelos utilizando `GridSearchCV` y `RandomizedSearchCV`. Por ejemplo, para Random Forest:

```python
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
grid = RandomizedSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
```

### 3.4 Evaluación de los modelos

Se evaluaron los modelos en el conjunto de prueba con métricas clave:

- `Accuracy`
- `Precision`
- `Recall`
- `F1-Score`
- `ROC-AUC`

## 4. Resultados del Modelado

### Métricas Finales

| Modelo              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.79     | 0.75      | 0.91   | 0.82     | 0.88    |
| KNN                 | 0.82     | 0.76      | 0.94   | 0.85     | 0.90    |
| Random Forest       | 0.85     | 0.80      | 0.97   | 0.88     | 0.91    |

## 5. Conclusión Final

Aunque los tres modelos analizados (Random Forest, Logistic Regression y KNN) mostraron un rendimiento competitivo, **Random Forest** fue seleccionado como el modelo final debido a:

- Su equilibrio entre métricas clave, como `Recall` (0.97) y `ROC-AUC` (0.91).
- Su capacidad para manejar relaciones no lineales en los datos.

Este análisis proporciona una base sólida para futuras aplicaciones o mejoras en proyectos relacionados con predicción en el ámbito médico. Además, el flujo simplificado presentado aquí es ideal para datasets pequeños y bien preparados, como este.

## 6. Recursos Complementarios

1. [Repositorio Completo de Clasificación](https://github.com/rociobenitez/heart-disease-prediction)
2. [Cuaderno Resumido del Proyecto](../../notebooks/5-structured-data-projects/heart-disease-classification.ipynb)

Este proyecto sirve como referencia para abordar problemas similares en el futuro, adaptándose fácilmente a datasets más complejos o tareas más exigentes.
