# Problema de regresión (en curso)

## Introducción

Esta sección explora cómo abordar un **problema de regresión** utilizando el dataset de la competición de Kaggle [Bluebook for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data). Este dataset contiene información histórica sobre ventas de bulldozers, y nuestro objetivo será construir un modelo de machine learning para predecir el **precio de venta de bulldozers futuros** basándonos en sus características.

- **Entradas**: Características, como el año de fabricación, modelo base, serie del modelo, estado de venta (por ejemplo, en qué estado de EE.UU. se vendió), sistema de tracción y más.
- **Salidas**: Precio de venta (en USD).

Este problema incluye un _componente temporal_, ya que intentamos predecir precios futuros basándonos en datos históricos, convirtiéndolo también en un **problema de series temporales o de predicción**.

La métrica de evaluación será el **RMSLE** (Root Mean Squared Log Error), comúnmente utilizada para problemas donde los valores objetivo (como precios) pueden variar mucho. El objetivo es minimizar este valor, lo que indicará que las predicciones del modelo son cercanas a los valores reales.

Las técnicas de esta sección están inspiradas y adaptadas del curso de machine learning de [fas.ai](https://course18.fast.ai/ml).

## Flujo de trabajo

El flujo de trabajo para este proyecto se explica en el cuaderno [bluebook-bulldozer-price-regression.ipynb](../notebooks/5-structured-data-projects/bluebook-bulldozer-price-regression.ipynb).

1. **Definición del problema:** ¿Qué tan bien podemos predecir el precio de venta de un bulldozer dado su historial y características?
2. **Exploración inicial (EDA):** Entender los datos disponibles y su estructura.
3. **Preparación de datos:** Preprocesar las características y manejar valores faltantes.
4. **División de datos:** Separar conjuntos de entrenamiento, validación y prueba, respetando el componente temporal.
5. **Modelado:** Entrenar y evaluar modelos, ajustando hiperparámetros si es necesario.
6. **Optimización y evaluación final:** Comparar el rendimiento del modelo con la métrica RMSLE.
7. **Guardado del modelo y predicciones finales.**

## 1. Definición del problema

La pregunta central que queremos responder es:

> _¿Qué tan bien podemos predecir el precio de venta futuro de un bulldozer, dadas sus características y ejemplos anteriores de cuánto se han vendido bulldozers similares?_

## 2. Exploración de Datos (EDA)

### Datos disponibles

El dataset incluye tres archivo principales:

1. **Train.csv:** Es el conjunto de entrenamiento

- Contiene ~400,000 ejemplos de ventas hasta finales de 2011.
- Incluye más de 50 características, como:
  - `SalePrice` (la **variable objetivo**).
  - Características categóricas como `fiModelDesc`, `State`, `UsageBand`...
    Fechas como `saledate`.

2. **Valid.csv:** Es el conjunto de validación

- Contiene datos del 1 de enero de 2012 al 30 de abril de 2012.
- Cerca de 12,000 ejemplos con los mismos atributos que **Train.csv**.
- Se utiliza para realizar predicciones durante la mayor parte de la competición.
- Tu puntuación en este conjunto se utiliza para generar el ranking público.

3. **Test.csv:** Es el conjunto de prueba

- Cerca de 12,000 ejemplos
- No incluye la columna `SalePrice`, ya que eso es lo que intentaremos predecir.
- Contiene datos del 1 de mayo de 2012 a noviembre de 2012.
- Tu puntuación en este conjunto determina tu clasificación final en la competición.

> **Nota**: Los datos están disponibles en la carpeta [data](../data/raw/scikit-learn-data/bluebook-for-bulldozers.zip).

### Exploración inicial

Cargamos los datos con `pandas` y exploramos su estructura básica:

```python
import pandas as pd
df = pd.read_csv("data/raw/TrainAndValid.csv", low_memory=False)

# Inspección inicial
print(df.shape())
print(df.info())
print(df.head())
```

Observamos que muchas columnas están en formato categórico (`object`), mientras que `saledate` está en formato de texto, y varias columnas tienen valores faltantes (`NaN`).

## 3. Preparación de Datos

### 3.1 Manejo de fechas

Convertimos `saledate` a formato `datetime` para trabajar fácilmente con componentes de fechas:

```python
df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce")
```

Creamos nuevas características a partir de la columna `saledate` para aprovechar su información:

- Año, mes, día, día de la semana, día del año.
- Estas características pueden capturar patrones temporales importantes.

```python
df["saleYear"] = df["saledate"].dt.year
df["saleMonth"] = df["saledate"].dt.month
df["saleDay"] = df["saledate"].dt.day
df["saleDayOfWeek"] = df["saledate"].dt.dayofweek
df["saleDayOfYear"] = df["saledate"].dt.dayofyear
```

Eliminamos la columna `saledate` original:

```python
df.drop("saledate", axis=1, inplace=True)
```

### 3.2 Manejo de valores faltantes

Identificamos las columnas con valores nulos:

```python
missing_cols = df.isnull().sum()
print(missing_cols[missing_cols > 0])
```

Llenamos los valores faltantes:

- **Variables numéricas**: Reemplazamos con la mediana de cada columna.
- **Variables categóricas**: Reemplazamos con una categoría como `"Missing"`.

```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
```

### 3.3 Codificación de variables categóricas

Convertimos las variables categóricas a códigos numéricos usando `pd.Categorical`:

```python
for col in cat_cols:
    df[col] = pd.Categorical(df[col]).codes
```

## 4. División de Datos

Dividimos los datos respetando el componente temporal:

```python
df_train = df[df["saleYear"] < 2012]
df_valid = df[df["saleYear"] == 2012]
```

Separación de características (`X`) y variable objetivo (`y`):

```python
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train["SalePrice"]
X_valid, y_valid = df_valid.drop("SalePrice", axis=1), df_valid["SalePrice"]
```

## 5. Modelado

Entrenamos un modelo base con `RandomForestRegressor`:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Evaluación inicial
print(f"Train R^2: {model.score(X_train, y_train)}")
print(f"Valid R^2: {model.score(X_valid, y_valid)}")
```

## 6. Evaluación y Optimización

Calculamos el RMSLE para evaluar el modelo:

```python
from sklearn.metrics import mean_squared_log_error
rmsle = mean_squared_log_error(y_valid, model.predict(X_valid)) ** 0.5
print(f"RMSLE: {rmsle}")
```

Realizamos una búsqueda de hiperparámetros para mejorar el modelo:

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {"n_estimators": [100, 200, 300],
              "max_depth": [None, 10, 20],
              "max_features": ["sqrt", "log2"]}

grid_search = RandomizedSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_iter=10)
grid_search.fit(X_train, y_train)
```

## 7. Guardado del modelo

Guardamos el modelo entrenado para uso futuro:

```python
import joblib
joblib.dump(model, "models/random_forest_model.pkl")
```

### 8. Lecciones Aprendidas

Trabajar en este proyecto de regresión no solo fue un reto técnico, sino también una oportunidad para aprender sobre las particularidades de los problemas del mundo real.

A continuación detallo algunos puntos clave que he aprendido y que podrían ser útiles en proyectos similares:

#### 8.1. El tiempo importa

Este es un problema de regresión que tiene un componente temporal. No podemos simplemente dividir los datos de manera aleatoria. Si lo hacemos, corremos el riesgo de usar datos del futuro para entrenar nuestro modelo, lo cual lo haría parecer mucho mejor de lo que realmente es.

> **Lección:** En problemas temporales, siempre separar los datos respetando el orden cronológico. Usar datos del pasado para entrenar y datos del futuro para validar.

#### 8.2. Pequeñas transformaciones pueden marcar la diferencia

Al trabajar con datos temporales, agregar columnas adicionales como el año, mes o día de la semana de una venta tuvo un impacto significativo en el modelo. Esto es lo que se conoce como **ingeniería de características**.

> **Lección:** Aunque parezcan simples, las transformaciones bien pensadas pueden darle al modelo la información que necesita para encontrar patrones más claros.

#### 8.3. Manejar valores faltantes

El dataset original tiene varias columnas llenas de valores `NaN`. Rellenar valores numéricos con la mediana ha sido una solución rápida y confiable, mientras que agregar una columna binaria (0 ó 1) con filas que reflejen si falta o no un valor para valores categóricos puede ayudar al modelo a captar información que de otro modo se perdería.

> **Lección:** No ignorar los valores faltantes; enfrentarlos con estrategias claras y consistentes. La **mediana es más robusta que la media** frente a valores extremos **(outliers)**.

#### 8.4. La métrica adecuada

El **RMSLE** considera las diferencias relativas en lugar de las absolutas. Esto tiene sentido en problemas como este, donde los precios pueden variar.

> **Lección:** Entender la métrica que usamos. A veces, una métrica mal elegida puede hacernos creer que nuestro modelo es mejor (o peor) de lo que realmente es.

#### 8.5. Aprender a interpretar tus datos

Explorar los datos no solo es útil para preprocesarlos, sino también para entender el problema. Por ejemplo:

- ¿Qué estados venden más bulldozers?
- ¿Existen patrones estacionales en las ventas?

Estas preguntas no solo ayudan a mejorar el modelo, sino que también te conectan con el contexto del problema.

> **Lección:** Dedicar tiempo al análisis de datos (EDA). Es donde descubriremos las pistas clave para construir un buen modelo.

### Reflexión Final

Este proyecto me enseñó que el machine learning no es solo cuestión de algoritmos, sino también de entender los datos y cómo abordarlos. Cada paso, desde manejar los valores faltantes hasta agregar nuevas características, tuvo un impacto directo en el modelo. Pero lo más importante fue recordar que no se trata solo de optimizar números, sino de resolver problemas reales con datos reales.

> **Consejo:** Abordar cada proyecto como una oportunidad para explorar, cometer errores y aprender en el proceso. Los mejores resultados suelen venir de esa mezcla entre técnica y curiosidad.
