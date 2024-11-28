# **Pandas**

Pandas es una biblioteca de Python diseñada para la manipulación y el análisis de datos. Proporciona estructuras de datos rápidas, flexibles y expresivas, como **Series** y **DataFrames**, que son fundamentales para cualquier proyecto de ciencia de datos y aprendizaje automático.

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
7. [Enlaces y Recursos de Interés](#7-enlaces-y-recursos-de-interés)

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
- Ver estadísticas generales:
  ```python
  df.describe()
  ```
- Obtener información del DataFrame:
  ```python
  df.info()
  ```
- Dimensiones:
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

---

## **7. Enlaces y Recursos de Interés**
- [Pandas Docs](https://pandas.pydata.org/pandas-docs/stable/)
- [10 Minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#min)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Data Manipulation with Pandas](https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html)

---