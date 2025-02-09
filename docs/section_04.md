# **Pandas**

[Pandas](https://pandas.pydata.org/) es una biblioteca de Python diseñada para la manipulación y el análisis de datos. Proporciona estructuras de datos rápidas, flexibles y expresivas, como **Series** y **DataFrames**, que son fundamentales para cualquier proyecto de ciencia de datos y aprendizaje automático.

---

## **¿Por qué usar Pandas?**

- **Fácil de usar:** Ofrece una sintaxis intuitiva para la manipulación de datos.
- **Rendimiento optimizado:** Basado en NumPy, lo que lo hace eficiente para el manejo de grandes conjuntos de datos.
- **Compatible con ML:** Permite preparar datos para algoritmos de Machine Learning de manera rápida y estructurada.
- **Integrado:** Funciona perfectamente con otras bibliotecas de Python como NumPy, Matplotlib y Scikit-learn.

---

## **Índice**

1. [Funciones más útiles en Pandas](#1-funciones-más-útiles-en-pandas)
2. [Tipos de datos en Pandas](#2-tipos-de-datos-en-pandas)
   - [Series](#series)
   - [DataFrame](#dataframe)
   - [CSV](#csv)
3. [Importación y Exportación de Datos](#3-importación-y-exportación-de-datos)
4. [Descripción de Datos](#4-descripción-de-datos)
5. [Visualización y Selección de Datos](#5-visualización-y-selección-de-datos)
6. [Manipulación de Datos](#6-manipulación-de-datos)
7. [Manejo de Tipos de Datos](#7-manejo-de-tipos-de-datos)
8. [Enlaces y Recursos de Interés](#8-enlaces-y-recursos-de-interés)

---

## **1. Funciones más útiles en Pandas**

- **Cargar datos desde múltiples formatos:**
  ```python
  pd.read_csv('file.csv')         # Leer un archivo CSV
  pd.read_excel('file.xlsx')      # Leer un archivo Excel
  pd.read_sql(query, connection)  # Leer datos de una base de datos
  ```
- **Operaciones básicas:**
  ```python
  df.head()      # Primeras filas
  df.info()      # Información del DataFrame
  df.describe()  # Estadísticas básicas
  ```
- **Selección y filtrado:**
  ```python
  df['columna']           # Selección de una columna
  df[df['columna'] > 10]  # Filtrado por condiciones
  ```
- **Manipulación de datos:**
  ```python
  df.drop('columna', axis=1)  # Eliminar columnas
  df.sort_values('columna')   # Ordenar datos
  df.fillna(0)                # Rellenar valores NaN
  ```

---

## **2. Tipos de Datos en Pandas**

### **Series**

- Una estructura unidimensional similar a una columna en una hoja de cálculo.
- Ejemplo:
  ```python
  s = pd.Series([1, 2, 3, 4, 5])
  print(s)
  ```

### **DataFrame**

- Una estructura bidimensional similar a una tabla.
- Ejemplo:
  ```python
  data = {'col1': [1, 2], 'col2': [3, 4]}
  df = pd.DataFrame(data)
  print(df)
  ```

### **CSV**

- Pandas permite importar y exportar datos fácilmente en formato CSV:
  ```python
  df = pd.read_csv('file.csv')  # Importar
  df.to_csv('output.csv')  # Exportar
  ```

<img src="/assets/section-4/pandas-structures-annotated.png" alt="Series vs DataFrames en Pandas" width="600">

---

## **3. Importación y Exportación de Datos**

- **Importar datos:**
  ```python
  df = pd.read_csv('file.csv')
  df = pd.read_excel('file.xlsx')
  ```
- **Exportar datos:**
  ```python
  df.to_csv('output.csv', index=False)
  df.to_excel('output.xlsx', index=False)
  ```

---

## **4. Descripción de Datos**

- **Ver estadísticas generales:**
  ```python
  df.describe()
  ```
- **Obtener información del DataFrame:**
  ```python
  df.info()
  ```
- **Dimensiones:**
  ```python
  df.shape
  ```

---

## **5. Visualización y Selección de Datos**

- **Seleccionar columnas:**
  ```python
  df['columna']
  ```
- **Seleccionar filas:**
  ```python
  df.iloc[0]  # Primera fila
  df.loc[df['columna'] > 10]  # Filtrado por condiciones
  ```
- **Indexación múltiple:**
  ```python
  df.loc[0:5, ['columna1', 'columna2']]
  ```
- **Agrupar por categorías:**
  ```python
  df.groupby('columna')['otra_columna'].mean(numeric_only=True)
  ```
- **Tablas cruzadas:**
  ```python
  pd.crosstab(df['categoria'], df['resultado'])
  ```

---

## **6. Manipulación de Datos**

- **Añadir una nueva columna:**
  ```python
  df['nueva_columna'] = df['columna1'] + df['columna2']
  ```
- **Eliminar columnas:**
  ```python
  df.drop('columna', axis=1, inplace=True)
  ```
- **Ordenar datos:**
  ```python
  df.sort_values(by='columna', ascending=True, inplace=True)
  ```
- **Rellenar valores nulos:**
  ```python
  df.fillna(0, inplace=True)
  ```
- **Renombrar columnas:**
  ```python
  df.rename(columns={'columna_antigua': 'columna_nueva'}, inplace=True)
  ```
- **Resetear índices:**
  ```python
  df.reset_index(inplace=True)
  ```
- **Seleccionar una muestra aleatoria del DataFrame:**

  ```python
  df.sample(frac=0.5)
  ```

  - Posibles parámetros:
    - `frac=0.5`: Especifica la fracción del total de filas que quieres en la muestra. En este caso, seleccionará el 50% de las filas (de manera aleatoria).
    - `n`: Especifica el número exacto de filas a tomar en la muestra.
    - `random_state`: Establece una semilla para la generación de números aleatorios, asegurando que la muestra sea reproducible.
    - `axis`: Controla si la muestra se toma de las filas (axis=0, por defecto) o de las columnas (axis=1).

- **Aplicar funciones personalizadas:**
  ```python
  df['columna'] = df['columna'].apply(lambda x: x * 2)
  ```

El argumento `inplace=True` en Pandas se utiliza para realizar **modificaciones directamente sobre el objeto original**, sin necesidad de crear una nueva copia. Esto puede ser útil para **ahorrar memoria y evitar tener que reasignar** el resultado de una operación a una nueva variable.

Cuando `inplace=True`, la operación cambia el DataFrame o la Serie original. Si se omite o se establece en `False` (por defecto), Pandas devuelve una copia modificada, dejando el objeto original sin cambios.

Por ejemplo en:

```python
df.drop('columna', axis=1, inplace=True)
```

La columna se elimina directamente del DataFrame `df`.

**Desventajas:**

- **Pérdida de datos originales**
- **Menor claridad** En algunos casos, puede ser menos claro que el objeto original ha sido modificado.

---

## **7. Manejo de Tipos de Datos**

- **Cambiar tipo de datos:**
  ```python
  df['columna'] = df['columna'].astype('float')
  ```
- **Trabajar con fechas:**
  ```python
  df['fecha'] = pd.to_datetime(df['fecha'])
  df['año'] = df['fecha'].dt.year
  ```

---

## **8. Enlaces y Recursos de Interés**

**Jupyter Notebooks**:

- [Introducción a Pandas](/notebooks/pandas/1-introduccion-pandas.ipynb)
- [Manipulación de datos en Pandas](/notebooks/pandas/2-manipulacion-pandas.ipynb)
- [Cuaderno con ejercicios para practicar con Pandas](/notebooks/pandas/3-pandas-exercises.ipynb)
- [Solución de los ejercicios](/notebooks/pandas/4-pandas-exercises-solutions.ipynb)

**Enlaces y Recursos de interés**:

- [Pandas Docs](https://pandas.pydata.org/pandas-docs/stable/)
- [10 Minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#min)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Data Manipulation with Pandas](https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html)

---
