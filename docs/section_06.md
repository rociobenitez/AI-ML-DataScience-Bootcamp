# **Matplotlib**

[Matplotlib](https://matplotlib.org/stable/users/index.html) es una biblioteca de visualización de datos en Python que permite **crear gráficos estáticos, animados e interactivos**. Es altamente personalizable y es ampliamente utilizada en ciencia de datos, aprendizaje automático y análisis de datos.

---

## **Índice**

1. [Qué es Matplotlib](#qué-es-matplotlib)
2. [Importar y Usar Matplotlib](#importar-y-usar-matplotlib)
3. [Anatomía de una Figura Matplotlib](#anatomía-de-una-figura-matplotlib)
4. [Gráficos Comunes](#gráficos-comunes)
   - [Line Plot(Gráfico de Línea)](#line-plotgráfico-de-línea)
   - [Scatter Plot (Gráfico de Dispersión)](#scatter-plot-gráfico-de-dispersión)
   - [Bar Plot (Gráfico de Barras)](#bar-plot-gráfico-de-barras)
   - [Histogramas](#histogramas)
   - [Subplots](#subplots)
5. [Data Visualizations](#data-visualizations)
6. [Plotting desde DataFrames de Pandas](#plotting-desde-dataframes-de-pandas)
7. [Cuándo Usar `pyplot` vs. OO Method](#cuándo-usar-pyplot-vs-oo-method)
8. [Personalizar Mis Plots](#personalizar-mis-plots)
9. [Guardar y Compartir Plots](#guardar-y-compartir-plots)
10. [Recursos Adicionales](#recursos-adicionales)

---

## **Qué es Matplotlib**

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

## **Gráficos Comunes**

### **Line Plot(Gráfico de Línea)**

Ideal para representar series temporales o la relación continua entre datos.

```python
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y, color='green', marker='o', linestyle='--')
plt.title("Line Plot")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.show()
```

### **Scatter Plot (Gráfico de Dispersión)**

Útil para visualizar relaciones entre dos variables.

```python
x = [5, 7, 8, 7]
y = [99, 86, 87, 88]
plt.scatter(x, y, color='purple')
plt.title("Scatter Plot")
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.show()
```

### **Bar Plot (Gráfico de Barras)**

Muestra comparaciones entre diferentes categorías.

```python
categories = ['A', 'B', 'C']
values = [10, 20, 15]
plt.bar(categories, values, color='orange')
plt.title("Bar Plot")
plt.xlabel("Categorías")
plt.ylabel("Valores")
plt.show()
```

### **Histogramas**

Un histograma muestra la distribución de un conjunto de datos dividiéndolos en intervalos (`bins`) y contando la frecuencia de cada intervalo. Es útil para observar patrones y distribuciones.

```python
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5]
plt.hist(data, bins=5, color='blue', edgecolor='black')
plt.title("Histograma")
plt.xlabel("Valores")
plt.ylabel("Frecuencia")
plt.show()
```

### **Subplots**

Los subplots permiten visualizar múltiples gráficos dentro de una misma figura, organizados en una cuadrícula.

```python
x = [1, 2, 3, 4]
y1 = [1, 4, 9, 16]
y2 = [1, 2, 3, 4]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Subplot 1: Line Plot
axs[0, 0].plot(x, y1, color='blue')
axs[0, 0].set_title("Line Plot")

# Subplot 2: Scatter Plot
axs[0, 1].scatter(x, y1, color='green')
axs[0, 1].set_title("Scatter Plot")

# Subplot 3: Bar Plot
axs[1, 0].bar(x, y2, color='orange')
axs[1, 0].set_title("Bar Plot")

# Subplot 4: Histogram
axs[1, 1].hist(y1, bins=4, color='purple', edgecolor='black')
axs[1, 1].set_title("Histogram")

plt.tight_layout()  # Ajusta los márgenes
plt.show()
```

**Explicación:**

- `fig, axs = plt.subplots(2, 2)`: Crea una cuadrícula de 2x2 para los gráficos.
- `axs[x, y]`: Accede a cada subplot individualmente.
- `set_title()`: Añade títulos a cada subplot.

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

> [!NOTE]
>
> - 🔗 [How to Plot a DataFrame using Pandas](https://datatofish.com/plot-dataframe-pandas/)
> - 🔗 [Chart visualization](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html)

---

## **Cuándo Usar `pyplot` vs. OO Method**

### **1. `pyplot`**

La interfaz `pyplot` de Matplotlib es similar a la de MATLAB. Proporciona un conjunto de funciones simples para crear y modificar gráficos.

**Ventajas**:

- Fácil de usar para gráficos rápidos y simples.
- Ideal para principiantes o para tareas de visualización sencillas.

**Cuándo Usarlo**:

- Cuando estás trabajando en un **script corto** o en un **notebook interactivo**.
- Para prototipos o gráficos exploratorios donde la personalización avanzada no es necesaria.

**Ejemplo**:

```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [4, 5, 6]

plt.plot(x, y)
plt.title("Gráfico simple con pyplot")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.show()
```

### **2. Object-Oriented (OO) Method**

En el enfoque orientado a objetos, trabajas directamente con objetos `Figure` y `Axes`, lo que proporciona más control sobre la personalización de los gráficos.

**Ventajas**:

- Permite **personalización avanzada** y control detallado.
- Escalable para gráficos complejos con múltiples subplots.

**Cuándo Usarlo**:

- Cuando necesitas crear gráficos **complejos** o con **múltiples subplots**.
- Para proyectos donde la consistencia y el control son importantes.
- En producción, donde los gráficos deben ser más robustos y manejables.

**Ejemplo**:

```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [4, 5, 6]

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Gráfico con OO Method")
ax.set_xlabel("Eje X")
ax.set_ylabel("Eje Y")
plt.show()
```

#### **Comparación Práctica**

| Característica              | `pyplot`                        | OO Method                       |
| --------------------------- | ------------------------------- | ------------------------------- |
| Facilidad de uso            | Alta (ideal para principiantes) | Requiere más configuración      |
| Personalización             | Básica                          | Avanzada                        |
| Gráficos con múltiples ejes | Limitado                        | Muy flexible                    |
| Uso común                   | Rápidos y exploratorios         | Producción y gráficos complejos |

#### **Recomendación General**

- Usar `pyplot` para tareas rápidas y exploratorias.
- Elegir el método OO para gráficos avanzados o cuando necesitemos control total sobre los elementos del gráfico.

---

## **Personalizar Mis Plots**

Matplotlib ofrece una amplia flexibilidad para personalizar gráficos de manera detallada. Aquí algunos aspectos clave:

#### **Estilos Disponibles**

- Explora los estilos predefinidos de Matplotlib:
  ```python
  plt.style.available
  ```
- Aplica un estilo específico:
  ```python
  plt.style.use('seaborn-v0_8-pastel')
  ```

#### **Colores, Estilos de Líneas y Marcadores**

- Personaliza líneas, colores y marcadores:
  ```python
  plt.plot([1, 2, 3], [3, 2, 1], color='red', linestyle='--', marker='o', label="Serie 1")
  plt.legend()
  plt.show()
  ```

#### **Tamaños y Márgenes**

- Ajusta el tamaño del gráfico o los márgenes de la figura:
  ```python
  plt.figure(figsize=(8, 6))
  plt.plot([1, 2, 3], [3, 2, 1])
  plt.tight_layout()
  plt.show()
  ```

#### **Anotaciones**

- Añade anotaciones dentro del gráfico:
  ```python
  plt.plot([1, 2, 3], [3, 2, 1])
  plt.text(2, 2, "Anotación", fontsize=12, color="blue", weight="bold")
  plt.show()
  ```

#### **Otros Elementos Personalizables**

- Configura títulos y etiquetas:
  ```python
  plt.title("Mi Gráfico", fontsize=16, fontweight="bold")
  plt.xlabel("Eje X", fontsize=12)
  plt.ylabel("Eje Y", fontsize=12)
  ```

---

## **Guardar y Compartir Plots**

Guardar gráficos en Matplotlib es sencillo, y hay diferentes formas de hacerlo:

#### **Desde Código**

1. Guarda el gráfico como un archivo con un formato específico:

   ```python
   plt.savefig('mi_grafico.png', dpi=300, bbox_inches='tight')
   ```

   - **`dpi`**: Resolución del gráfico (recomendado 300 para alta calidad).
   - **`bbox_inches='tight'`**: Ajusta los márgenes automáticamente.

2. Formatos soportados:
   - `.png` (gráficos rasterizados para la web)
   - `.jpg` (fotografía, calidad más baja)
   - `.pdf` (ideal para impresión)
   - `.svg` (formato vectorial)

#### **Desde Jupyter Notebook**

1. **Clic derecho sobre el gráfico:**

   - Puedes guardar un gráfico directamente desde Jupyter haciendo clic derecho sobre él y seleccionando "Guardar imagen como".

2. **Exportar con código:**

   - Usa el mismo comando `plt.savefig()` antes de mostrar el gráfico:
     ```python
     plt.plot([1, 2, 3], [3, 2, 1])
     plt.savefig("mi_grafico.png", dpi=300)
     plt.show()
     ```

3. **Impresión automática:**
   - Para guardar todos los gráficos generados en una sesión:
     ```python
     from IPython.display import display
     plt.plot([1, 2, 3], [3, 2, 1])
     display(plt.gcf())  # Guarda la figura actual (Get Current Figure)
     ```

> [!NOTE]
>
> - Usa `.svg` para gráficos escalables que no pierdan calidad al ser redimensionados.
> - Si planeas editar el gráfico más adelante, guarda en formatos como `.pdf` o `.svg`. ```

---

## **Recursos Adicionales**

- [Documentación de Matplotlib](https://matplotlib.org/stable/contents.html)
- [Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html#sphx-glr-tutorials-pyplot-py)
- [Galería de ejemplos de Matplotlib](https://matplotlib.org/stable/gallery/index.html)
- [Matplotlib en Stack Overflow](https://stackoverflow.com/questions/tagged/matplotlib)
- [Expresiones regulares (opcional)](https://regexone.com/)
