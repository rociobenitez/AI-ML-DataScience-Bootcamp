# **NumPy**

[NumPy](https://numpy.org/doc/) es una biblioteca esencial para la computación científica en Python, especialmente en proyectos de ciencia de datos, aprendizaje automático y análisis numérico. Proporciona soporte para **arrays multidimensionales** y una amplia colección de **funciones matemáticas** para manipular estos datos de manera eficiente.

---

## **¿Por qué usar NumPy?**

- **Eficiencia:** Opera sobre datos de manera más rápida y eficiente que las listas nativas de Python.
- **Versatilidad:** Ofrece funciones matemáticas, estadísticas y de álgebra lineal.
- **Compatibilidad:** Se integra fácilmente con otras bibliotecas como Pandas, Matplotlib y Scikit-learn.

---

## **Índice**

1. [Introducción a NumPy](#1-introducción-a-numpy)
2. [DataTypes y Attributes](#2-datatypes-y-attributes)
3. [Creación de Arrays](#3-creación-de-arrays)
4. [Uso de Random Seed](#4-uso-de-random-seed)
5. [Arrays y Matrices](#5-arrays-y-matrices)
6. [Manipulación de Arrays](#6-manipulación-de-arrays)
7. [Desviación Estándar y Varianza](#7-desviación-estándar-y-varianza)
8. [Reshape y Transpose](#8-reshape-y-transpose)
9. [Dot Product vs Element-wise Operations](#9-dot-product-vs-element-wise-operations)
10. [Operadores de Comparación](#10-operadores-de-comparación)
11. [Ordenación de Arrays](#11-ordenación-de-arrays)
12. [Convertir Imágenes en Arrays de NumPy](#12-convertir-imágenes-en-arrays-de-numpy)

---

## **1. Introducción a NumPy**

NumPy (Numerical Python) es una biblioteca utilizada para trabajar con **arrays** y **matrices**. A diferencia de las listas de Python, los arrays de NumPy son **más rápidos y consumen menos memoria**.

Instalación:

```bash
pip install numpy
```

Importar NumPy:

```python
import numpy as np
```

---

## **2. DataTypes y Attributes**

Los arrays en NumPy tienen atributos importantes que proporcionan información sobre los datos que contienen:

**Atributos comunes:**

```python
array = np.array([1, 2, 3])
print(array.ndim)      # Dimensiones del array
print(array.shape)     # Forma del array
print(array.dtype)     # Tipo de dato
print(array.size)      # Número total de elementos
print(array.itemsize)  # Tamaño de cada elemento en bytes
```

**Tipos de datos soportados:** `int`, `float`, `bool`, `complex`, etc.

<img src="../assets/section-5/anatomy-numpy-array.webp" alt="Arrays Numpy" width="600">

---

## **3. Creación de Arrays**

NumPy ofrece múltiples formas de crear arrays:

**Desde listas o tuplas:**

```python
np.array([1, 2, 3])  # Array 1D
np.array([[1, 2], [3, 4]])  # Array 2D
```

**Arrays vacíos o inicializados:**

```python
np.zeros((2, 3))    # Array de ceros
np.ones((3, 3))     # Array de unos
np.full((2, 2), 7)  # Array lleno de un valor
```

**Secuencias:**

```python
np.arange(0, 10, 2)   # Array de 0 a 10 con pasos de 2
np.linspace(0, 1, 5)  # 5 valores equidistantes entre 0 y 1
```

---

## **4. Uso de Random Seed**

El uso de `random.seed()` asegura reproducibilidad en las operaciones aleatorias.

**Generar números aleatorios reproducibles:**

```python
np.random.seed(42)
random_array = np.random.randint(0, 10, (3, 3))
print(random_array)
```

---

## **5. Arrays y Matrices**

**Diferencia entre arrays 1D, 2D y matrices:**

```python
array_1d = np.array([1, 2, 3])  # Array 1D
array_2d = np.array([[1, 2], [3, 4]])  # Array 2D
matrix = np.mat('1 2; 3 4')  # Matriz
```

**Indexación:**

```python
print(array_2d[0, 1])  # Elemento en la fila 0, columna 1
```

**Matrices de más de dos dimensiones:** Puedes crear matrices multidimensionales utilizando `np.random.randint` o funciones similares, especificando las dimensiones deseadas con el argumento `size`:

```python
matrix_3d = np.random.randint(0, 10, size=(3, 3, 3))  # Matriz 3D (3 bloques de 3x3)
print(matrix_3d)
print(matrix_3d.shape)  # (3, 3, 3)

matrix_4d = np.random.randint(0, 10, size=(2, 3, 4, 5))  # Matriz 4D
print(matrix_4d)
print(matrix_4d.shape)  # (2, 3, 4, 5)
```

**Indexación en matrices multidimensionales:** Para acceder a elementos específicos en matrices de dimensiones mayores:

```python
print(matrix_3d[0, 1, 2])     # Elemento en el bloque 0, fila 1, columna 2
print(matrix_4d[1, 2, 3, 4])  # Elemento en la posición especificada de la matriz 4D
```

> **Nota:** La función `np.random.randint` genera valores enteros aleatorios entre un rango definido (en este caso, de 0 a 10). Es muy útil para crear datos de prueba en matrices de cualquier dimensión.

### **Ventaja de Matrices Multidimensionales**

Las matrices de más de dos dimensiones son útiles en:

- **Procesamiento de imágenes:** Trabajar con datos de múltiples canales (RGB).
- **Redes neuronales:** Tensores multidimensionales que representan datos de entrenamiento.
- **Modelos científicos complejos:** Representación de datos en múltiples capas o dimensiones.

<img src="../assets/section-5/arrays-numpy.png" alt="Arrays Numpy" width="600" style="padding:24px; margin: 24px 0; background: white;">

---

## **6. Manipulación de Arrays**

- **Operaciones comunes:**

  ```python
  array = np.array([1, 2, 3])
  print(array + 10)  # Suma escalar
  print(array * 2)   # Producto escalar
  ```

- **Funciones matemáticas:**

  ```python
  print(np.sum(array))
  print(np.mean(array))
  print(np.max(array))
  print(np.min(array))
  ```

- **Producto escalar entre arrays de diferentes dimensiones:** El producto escalar **(dot product)** entre dos arrays con diferentes dimensiones utiliza las [reglas de broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) de NumPy para ajustar las dimensiones automáticamente cuando sea posible.

  ```python
  array_2d = np.array([[1, 2, 3], [4, 5, 6]])  # Array 2D (2x3)
  array_1d = np.array([1, 2, 3])  # Array 1D (3 elementos)

  result = np.dot(array_2d, array_1d)  # Producto escalar
  print(result)  # [14 32]
  ```

  - Forma expandida del cálculo:
    - Para la fila `[1, 2, 3]`: `(1*1) + (2*2) + (3*3) = 14`
    - Para la fila `[4, 5, 6]`: `(4*1) + (5*2) + (6*3) = 32`

  ```python
  array_2d = np.array([[1, 2], [3, 4]])  # Array 2D de forma (2, 2)
  array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Array 3D de forma (2, 2, 2)

  result = np.dot(array_2d, array_3d)
  print(result)
  ```

  - Para calcular manualmente, tomamos las filas de `array_2d` y hacemos el producto punto con cada matriz del `array_3d`:

    - Primero:

    ```python
    [[(1*1 + 2*3), (1*2 + 2*4)],   # Primera fila de array_2d
    [(3*1 + 4*3), (3*2 + 4*4)]]   # Segunda fila de array_2d
    Resultado: [[7, 10], [15, 22]]
    ```

    - Después:

    ```python
    [[(1*5 + 2*7), (1*6 + 2*8)],   # Primera fila de array_2d
    [(3*5 + 4*7), (3*6 + 4*8)]]   # Segunda fila de array_2d
    Resultado: [[19, 22], [43, 50]]
    ```

    - Resultado final:

    ```python
    [[[ 7, 10],   # Primera matriz resultado
    [15, 22]],

    [[19, 22],   # Segunda matriz resultado
    [43, 50]]]
    ```

> [!NOTE]
> ✍🏼 Si las dimensiones no son compatibles, NumPy arrojará un error. Asegúrate de que las dimensiones cumplan las reglas del producto escalar (el número de columnas del primer array debe coincidir con el número de elementos del segundo array).

---

## **7. Desviación Estándar y Varianza**

**Desviación estándar:** La desviación estándar es la raíz cuadrada de la varianza y se usa para interpretar la **dispersión** en las mismas unidades que los datos originales. Una desviación estándar baja indica que los valores están muy cerca de la media, mientras que una alta indica lo contrario. Es **más fácil de interpretar** porque está en las **mismas unidades** que los datos, lo que permite evaluar qué tan lejos están los valores de la media en promedio.

```python
np.std(array)
```

<img src="../assets/section-5/desviacion-estandar.png" alt="Desviación estándar" width="400" style="padding:24px; margin: 24px 0; background: white;">

**Varianza:** La varianza mide qué tan dispersos están los datos respecto a su media. Se calcula como el promedio de las diferencias elevadas al cuadrado entre cada valor y la media del conjunto de datos. Una **varianza alta** indica que los **datos están más dispersos**; una varianza baja indica que los datos están más cerca de la media. Sus valores son **más difíciles de interpretar** directamente debido a las **unidades al cuadrado**.

```python
np.var(array)
```

> [!NOTE]
> La varianza y la desviación estándar son **medidas de dispersión**, pero tienen diferencias. La desviación estándar es la raíz cuadrada de la varianza y ofrece una interpretación más directa de la dispersión.

- 🔗 [Desviación Estándar y Varianza](https://www.mathsisfun.com/data/standard-deviation.html)
- 🔗 [Varianza - Wikipedia](https://es.wikipedia.org/wiki/Varianza)

---

## **8. Reshape y Transpose**

**Reshape:** Este método cambia la forma (dimensiones) de un array sin alterar sus datos. Es útil para reorganizar arrays en diferentes estructuras, siempre que el número total de elementos se mantenga constante.

```python
array = np.array([1, 2, 3, 4, 5, 6])
reshaped = array.reshape(2, 3)  # Cambia a una matriz de 2 filas y 3 columnas
print(reshaped)
# [[1 2 3]
#  [4 5 6]]
```

> [!TIP] > **Nota:** Si no estás seguro del tamaño en una dimensión, usa `-1` y NumPy calculará automáticamente:
>
> ```python
> reshaped = array.reshape(-1, 3)  # NumPy ajusta las filas automáticamente
> ```

**Transpose:** Este método T intercambia filas por columnas en un array 2D o cambia las dimensiones en arrays de mayor dimensión. Es útil para reorientar matrices o preparar datos para cálculos matemáticos.

```python
transposed = reshaped.T  # Intercambia filas por columnas
print(transposed)
# [[1 4]
#  [2 5]
#  [3 6]]
```

- 🔗 [Broadcasting NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html)

---

## **9. Dot Product vs Element-wise Operations**

**Dot Product (producto punto):** Es una operación matemática que combina elementos de dos arrays de manera algebraica (acumulativa). En el caso de vectores, multiplica los valores correspondientes y suma los resultados; para matrices, aplica las reglas del álgebra lineal.

```python
np.dot(array1, array2)

# Ejemplo con vectores:
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
result = np.dot(array1, array2)  # (1*4) + (2*5) + (3*6) = 32

# Ejemplo con matrices:
python
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
result = np.dot(matrix1, matrix2)

```

**Element-wise (elemento a elemento):** En una operación elemento a elemento, los valores correspondientes de dos arrays se operan independientemente (suma, resta, multiplicación, etc.) (sin acumulación). Los arrays **deben tener la misma forma o ser compatibles** mediante [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

```python
array1 * array2 # Multiplicación elemento a elemento

# Ejemplo:
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
result = array1 * array2  # [1*4, 2*5, 3*6] = [4, 10, 18]
```

<img src="../assets/section-5/dotproduct-elementwise.jpg" alt="Dot product vs. Element-wise" width="500" style="margin: 24px 0;">

- 🔗 [Cómo multiplicar matrices](https://www.mathsisfun.com/algebra/matrix-multiplying.html)
- 🔗 [Matriz Multiplication](http://matrixmultiplication.xyz/)

---

## **10. Operadores de Comparación**

Los **operadores de comparación** en NumPy permiten comparar arrays elemento a elemento, devolviendo un array booleano (`True` o `False`) que indica si la condición se cumple para cada elemento. Son útiles para filtrar, analizar o manipular datos basados en condiciones.

### **Comparaciones básicas**

- Comparar si los elementos de un array cumplen una condición:

  ```python
  array = np.array([3, 6, 9])
  result = array > 5
  print(result)  # [False  True  True]
  ```

- Comparar elementos entre dos arrays:
  ```python
  array1 = np.array([1, 2, 3])
  array2 = np.array([3, 2, 1])
  print(array1 == array2)  # [False  True  False]
  print(array1 > array2)   # [False False  True]
  ```

### **Aplicaciones prácticas**

1. **Filtrar datos en un array:**

   - Puedes usar operadores de comparación junto con indexación para extraer elementos que cumplan ciertas condiciones:
     ```python
     array = np.array([3, 6, 9])
     filtered = array[array > 5]
     print(filtered)  # [6 9]
     ```

2. **Comparaciones múltiples:**

   - Combina múltiples condiciones usando operadores lógicos (`&`, `|`, `~`):
     ```python
     array = np.array([3, 6, 9])
     result = (array > 5) & (array < 10)
     print(result)  # [False  True  True]
     ```

3. **Comprobar si todos o algún elemento cumple una condición:**

   - `np.all()`: Verifica si **todos** los elementos cumplen una condición.
   - `np.any()`: Verifica si **algún** elemento cumple una condición.
     ```python
     array = np.array([3, 6, 9])
     print(np.all(array > 5))  # False
     print(np.any(array > 5))  # True
     ```

4. **Comparaciones con arrays multidimensionales:**
   - Funciona igual, pero aplica la comparación a cada elemento individual:
     ```python
     matrix = np.array([[1, 2, 3], [4, 5, 6]])
     print(matrix > 3)
     # [[False False False]
     #  [ True  True  True]]
     ```

> [!NOTE]
>
> - El resultado de las comparaciones es un array booleano del mismo tamaño que el array original.
> - Los operadores (`>`, `<`, `>=`, `<=`, `==`, `!=`) funcionan de manera nativa con arrays de NumPy.
> - 🔗 [Logic functions NumPy](https://numpy.org/doc/2.1/reference/routines.logic.html)

---

## **11. Ordenación de Arrays**

La **ordenación** en NumPy permite reorganizar los valores de un array en orden **ascendente o descendente**, según tus necesidades. También puedes obtener los índices que representarían ese orden.

### **1. Ordenar valores en un array**

- **`np.sort()`**: Devuelve un nuevo array con los elementos ordenados en **orden ascendente**.

  ```python
  array = np.array([3, 1, 2])
  sorted_array = np.sort(array)
  print(sorted_array)  # [1 2 3]
  ```

- Para arrays multidimensionales:
  - Ordena los elementos a lo largo del último eje por defecto:
    ```python
    matrix = np.array([[3, 2, 1], [6, 5, 4]])
    print(np.sort(matrix))
    # [[1 2 3]
    #  [4 5 6]]
    ```

### **2. Ordenar por índices**

- **`np.argsort()`**: Devuelve los **índices** que ordenarían el array.
  ```python
  array = np.array([3, 1, 2])
  sorted_indices = np.argsort(array)
  print(sorted_indices)  # [1 2 0] -> índices de [1, 2, 3]
  ```

### **3. Uso práctico con los índices**

- Puedes usar los índices para reordenar un array manualmente:
  ```python
  sorted_array = array[np.argsort(array)]
  print(sorted_array)  # [1 2 3]
  ```

---

## **12. Convertir Imágenes en Arrays de NumPy**

NumPy puede procesar imágenes convirtiéndolas en arrays, lo que permite realizar análisis, manipulación y procesamiento de imágenes directamente como datos numéricos.

### **Cargar una imagen como array**

Puedes utilizar `matplotlib.image` para cargar imágenes en formato NumPy:

```python
from matplotlib.image import imread

# Cargar la imagen
image = imread('imagen.jpg')

# Información de la imagen
print(type(image))  # Clase numpy.ndarray
print(image.shape)  # Dimensiones de la imagen (alto, ancho, canales)
```

### **Ejemplo práctico:**

Supongamos que tienes una imagen RGB (`imagen.jpg`) de tamaño 256x256. Cuando se carga con `imread`, se convierte en un array 3D donde:

- El primer eje representa las filas (alto de la imagen).
- El segundo eje representa las columnas (ancho de la imagen).
- El tercer eje representa los canales de color (por ejemplo, RGB con 3 canales).

```python
image = imread('imagen.jpg')
print(image.shape)  # (256, 256, 3)
```

### **Operaciones comunes con imágenes como arrays**

1. **Acceder a píxeles específicos:**

   - Los píxeles se representan como valores numéricos en los canales de color.

   ```python
   print(image[0, 0])  # Valor del píxel en la esquina superior izquierda
   ```

2. **Escala de grises:**

   - Si la imagen es en escala de grises, el array será 2D (alto y ancho):

   ```python
   gray_image = imread('imagen_gris.jpg')
   print(gray_image.shape)  # (256, 256)
   ```

3. **Normalización de valores:**

   - Las imágenes suelen tener valores entre 0 y 255. Para normalizarlos entre 0 y 1:

   ```python
   normalized_image = image / 255.0
   ```

4. **Mostrar la imagen usando `matplotlib`:**

   - Visualiza la imagen después de cargarla como array:

   ```python
   import matplotlib.pyplot as plt

   plt.imshow(image)
   plt.axis('off')  # Ocultar ejes
   plt.show()
   ```

5. **Modificar la imagen:**
   - Por ejemplo, establecer todos los valores de un canal (como el rojo) a 0:
   ```python
   image[:, :, 0] = 0  # Elimina el canal rojo
   plt.imshow(image)
   plt.show()
   ```

> [!NOTE]
>
> - La manipulación de imágenes en NumPy es útil para tareas de visión por computadora y aprendizaje automático.
> - Usa bibliotecas adicionales como **OpenCV** o **Pillow** para un procesamiento de imágenes más avanzado.

---

## **Recursos Adicionales**

- [Documentación oficial de NumPy](https://numpy.org/doc/)
- [Guía rápida de NumPy](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy en profundidad](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html)
- [Jupyter Notebooks de la sección](/notebooks/numpy/)
- [A visual Introduction to NumPy by Jay Alammar](https://jalammar.github.io/visual-numpy/)
- [The Basics of NumPy Arrays](https://jakevdp.github.io/PythonDataScienceHandbook/02.02-the-basics-of-numpy-arrays.html)
- [NumPy Quickstart tutorial](https://numpy.org/doc/1.17/user/quickstart.html)

---
