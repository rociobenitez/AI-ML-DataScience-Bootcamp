# **Matplotlib**

[Matplotlib](https://matplotlib.org/stable/users/index.html) es una biblioteca de visualización de datos en Python que permite **crear gráficos estáticos, animados e interactivos**. Es altamente personalizable y es ampliamente utilizada en ciencia de datos, aprendizaje automático y análisis de datos.

---

## **¿Qué es Matplotlib?**
- **Propósito:** Crear gráficos de alta calidad, desde visualizaciones simples hasta complejas.
- **Versatilidad:** Compatible con NumPy, Pandas y muchas otras bibliotecas.
- **Uso común:** Generar gráficos como líneas, barras, dispersión, histogramas y más.

---

## **Importar y Usar Matplotlib**
Para comenzar con Matplotlib, normalmente se importa la biblioteca `pyplot` como un alias para simplificar el código.

```python
import matplotlib.pyplot as plt
```

- **Crear un gráfico básico:**
  ```python
  x = [1, 2, 3, 4]
  y = [10, 20, 25, 30]
  plt.plot(x, y)
  plt.title("Gráfico Básico")
  plt.xlabel("Eje X")
  plt.ylabel("Eje Y")
  plt.show()
  ```

---

## **Anatomía de una Figura Matplotlib**
Una figura en Matplotlib está compuesta por varios elementos clave:

- **Figure:** El contenedor principal de todo el gráfico.
- **Axes:** La zona donde se dibujan los datos.
- **Axis:** Los ejes X e Y.
- **Ticks:** Las marcas en los ejes.
- **Labels:** Los nombres de los ejes.
- **Legend:** Leyendas para identificar elementos del gráfico.

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Título del Gráfico")
ax.set_xlabel("Eje X")
ax.set_ylabel("Eje Y")
plt.show()
```

<img src="../assets/section-6/matplotlib-anatomy-of-a-plot.png" alt="Arrays Numpy" width="800" style="padding:24px; margin: 24px auto; background: white;">

---

## **Scatter Plot y Bar Plot**

### **Scatter Plot (Gráfico de Dispersión):**
Se utiliza para mostrar la relación entre dos variables.

```python
x = [5, 7, 8, 7]
y = [99, 86, 87, 88]
plt.scatter(x, y)
plt.title("Scatter Plot")
plt.show()
```

### **Bar Plot (Gráfico de Barras):**
Se utiliza para comparar diferentes categorías.

```python
labels = ['A', 'B', 'C']
values = [10, 15, 7]
plt.bar(labels, values)
plt.title("Bar Plot")
plt.show()
```

---

## **Histogramas y Subplots**

### **Histogramas:**
Distribución de un conjunto de datos.

```python
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
plt.hist(data, bins=4)
plt.title("Histograma")
plt.show()
```

### **Subplots:**
Múltiples gráficos en una misma figura.

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot([1, 2, 3], [4, 5, 6])
axs[1].scatter([1, 2, 3], [6, 5, 4])
plt.show()
```

---

## **Data Visualizations**
Matplotlib permite crear visualizaciones interactivas con datos para explorar patrones y relaciones.

- Combina gráficos como barras y líneas.
- Utiliza estilos predefinidos para mejorar la estética.

```python
plt.style.use('ggplot')  # Estilo predefinido
plt.plot([1, 2, 3], [1, 4, 9], label="Línea 1")
plt.bar([1, 2, 3], [2, 5, 8], alpha=0.5, label="Barra")
plt.legend()
plt.show()
```

---

## **Plotting desde DataFrames de Pandas**
Matplotlib se integra perfectamente con Pandas para graficar datos directamente desde DataFrames.

```python
import pandas as pd
data = {'Col1': [1, 2, 3], 'Col2': [4, 5, 6]}
df = pd.DataFrame(data)

df.plot(x='Col1', y='Col2', kind='line')  # Gráfico de línea
plt.show()
```

---

## **Personalizar Mis Plots**
Matplotlib permite personalizar casi cualquier aspecto de un gráfico.

- **Colores y estilos de líneas:**
  ```python
  plt.plot([1, 2, 3], [3, 2, 1], color='red', linestyle='--', marker='o')
  plt.show()
  ```

- **Ajustar tamaños y márgenes:**
  ```python
  plt.figure(figsize=(8, 6))
  plt.plot([1, 2, 3], [3, 2, 1])
  plt.show()
  ```

- **Anotaciones:**
  ```python
  plt.plot([1, 2, 3], [3, 2, 1])
  plt.text(2, 2, "Anotación")
  plt.show()
  ```

---

## **Saving and Sharing Plots**
Guarda tus gráficos para compartirlos o usarlos en otros documentos.

```python
plt.savefig('mi_grafico.png', dpi=300)
```

- **Formatos comunes:**
  - `.png`
  - `.jpg`
  - `.pdf`
  - `.svg`

---

## **Recursos Adicionales**

- [Documentación de Matplotlib](https://matplotlib.org/stable/contents.html)
- [Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html#sphx-glr-tutorials-pyplot-py)
- [Galería de ejemplos de Matplotlib](https://matplotlib.org/stable/gallery/index.html)
- [Matplotlib en Stack Overflow](https://stackoverflow.com/questions/tagged/matplotlib)