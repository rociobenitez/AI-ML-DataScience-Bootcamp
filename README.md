
# Complete A.I. & Machine Learning, Data Science Bootcamp

Este repositorio contiene todo el trabajo realizado durante el curso _Complete A.I. & Machine Learning, Data Science Bootcamp_. La estructura del proyecto y los contenidos están organizados para facilitar la navegación y el aprendizaje.

---

## **Índice**
1. [Introducción](#introducción)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Instalación y configuración](#instalación-y-configuración)
4. [Progreso del curso](#progreso-del-curso)
5. [Secciones del curso](#secciones-del-curso)

---

## Introducción
Este curso cubre temas esenciales de Machine Learning, Inteligencia Artificial y Ciencia de Datos. Está dividido en 20 secciones, cada una con notebooks y scripts detallados. A medida que avanzamos, se añadirá más información en este archivo.

---

## Estructura del proyecto

```
project/
├── data/                   # Datos crudos o procesados
│   ├── raw/                # Datos originales sin procesar
│   ├── processed/          # Datos procesados listos para análisis
├── notebooks/              # Jupyter notebooks organizados por sección
│   ├── 01-introduction.ipynb
│   ├── 02-data-cleaning.ipynb
│   └── ...                 # Archivos por cada sección del curso
├── scripts/                # Scripts de Python para tareas específicas
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── utils.py
├── models/                 # Modelos entrenados y checkpoints
│   ├── model_v1.pkl
│   └── model_v2.pkl
├── results/                # Resultados generados (gráficos, métricas, etc.)
│   ├── plots/
│   ├── metrics/
│   └── logs/
├── references/             # Recursos externos como papers, enlaces, etc.
├── environment/            # Configuración de entorno
│   ├── environment.yml     # Archivo de Conda con dependencias
│   └── requirements.txt    # Alternativa para instalar dependencias con pip
├── README.md               # Documentación principal
└── .gitignore              # Archivos y carpetas a ignorar en Git
```

## Estructura del proyecto
La estructura del repositorio está organizada en diferentes directorios para una navegación eficiente:

```
assets/                 # Imágenes
data/                   # Datos crudos y procesados
notebooks/              # Notebooks organizados por sección
scripts/                # Scripts para tareas repetitivas
models/                 # Modelos entrenados
results/                # Gráficos y métricas generadas
references/             # Recursos externos y papers
environment/            # Archivos de configuración del entorno
README.md               # Documentación principal
```

Cada carpeta está diseñada para contener recursos específicos y mantenerse modular.

---

## Instalación y configuración
### Requisitos:
- Python >= 3.9
- Conda instalado (Miniconda o Anaconda)

### Instrucciones:
1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/tu_proyecto.git
   cd tu_proyecto
   ```
2. Crea y activa el entorno de Conda:
   ```bash
   conda env create -f environment/environment.yml
   conda activate tu_entorno
   ```
3. Instala las dependencias adicionales (si es necesario):
   ```bash
   pip install -r environment/requirements.txt
   ```

---

## Progreso del curso
El curso está dividido en 20 secciones. Este es un resumen del progreso realizado hasta ahora:

| Sección        | Título                           | Descripción breve            | Estado  |
|----------------|----------------------------------|------------------------------|---------|
| 1              | Introducción                     | Introducción al Curso        | ✅      |
| 2              | Machine Learning 101             | Qué es ML y tipos de ML      | ✅      |
| 3              | Visualización de datos           | Uso de Matplotlib y Seaborn  | ❌      |
| ...            | ...                              | ...                          | ...     |

- **✅ Completado**
- **🚧 En progreso**
- **❌ Pendiente**

---

## Secciones detalladas

### Sección 2

### Sección 5 : Data Science Environment Setup

### **1. ¿Qué es Conda?**
Conda es un **gestor de entornos y paquetes** de código abierto. Fue diseñado inicialmente para Python, pero también gestiona otros lenguajes como R. Sus principales características son:

- **Gestión de entornos:** Puedes crear, clonar, exportar y eliminar entornos virtuales con diferentes versiones de Python y paquetes instalados.
- **Gestión de dependencias:** Se asegura de que los paquetes instalados sean compatibles entre sí.
- **Versatilidad:** Funciona con cualquier sistema operativo (Windows, macOS, Linux) y no está limitado a paquetes de Python (también puede gestionar bibliotecas de C, C++, etc.).

**Ventajas principales:**
- Instalación sencilla de paquetes complicados.
- Ideal para proyectos de data science y machine learning debido a la gestión eficiente de dependencias.

> 🔗 [Conda Documentación](https://docs.conda.io/en/latest/)

### **2. Conda CheatSheet**
El cheat sheet es un resumen visual y rápido con los comandos más comunes de Conda. Es una herramienta muy útil para tener a mano mientras trabajas con Conda.

#### Comandos básicos:
- **Gestión de entornos:**
  - Crear un entorno: `conda create --name myenv`
  - Activar un entorno: `conda activate myenv`
  - Desactivar un entorno: `conda deactivate`
  - Eliminar un entorno: `conda remove --name myenv --all`

- **Gestión de paquetes:**
  - Instalar un paquete: `conda install package_name`
  - Actualizar un paquete: `conda update package_name`
  - Listar paquetes instalados: `conda list`

- **Información del entorno:**
  - Listar entornos: `conda env list`

> 🔗 Consulta el [Conda CheatSheet](https://docs.conda.io/en/latest/) para más comandos.

### **3. Getting started with Conda**
Esta guía introduce cómo empezar a usar Conda, desde su instalación hasta el uso básico.

#### Pasos iniciales:
1. **Instalación de Miniconda o Anaconda:**
   - Miniconda: Una instalación ligera con Conda y Python básicos.
   - Anaconda: Incluye Conda y un amplio ecosistema de herramientas y bibliotecas.

2. **Creación de entornos virtuales:** 
   - Los entornos virtuales te permiten trabajar con diferentes versiones de paquetes y Python en proyectos aislados.

3. **Instalación de paquetes:** 
   - Los paquetes pueden instalarse desde el canal principal de Conda o desde canales externos como conda-forge.

4. **Exportar un entorno:**
   - Puedes compartir tus entornos con otros mediante el comando `conda env export > environment.yml`.

> Para más detalles, revisa la guía completa: [Getting started with conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

### **4. Preparando tu equipo para Machine Learning**
En este artículo, Daniel Bourke explica cómo usar **Anaconda, Miniconda y Conda** para preparar tu equipo para proyectos de Machine Learning.

#### Resumen práctico:
- **Anaconda** es ideal para principiantes porque incluye una gran cantidad de herramientas preinstaladas. 
- **Miniconda** es una versión más ligera, ideal para usuarios avanzados que prefieren instalar solo lo necesario.
- **Conda** es el motor que impulsa ambas plataformas.

**Recomendaciones clave:**
- Si no necesitas todas las herramientas de Anaconda, utiliza Miniconda para mantener tu equipo más limpio y ligero.
- Usa entornos específicos para cada proyecto para evitar conflictos entre paquetes.

> Consulta el artículo completo: [Getting your computer ready for machine learning](https://www.mrdbourke.com/get-your-computer-ready-for-machine-learning-using-anaconda-miniconda-and-conda/)

### **5. Miniconda para macOS**
Miniconda es una versión compacta de Anaconda. Contiene solo lo esencial: Python, Conda y paquetes básicos. Es ideal para mantener un sistema limpio y personalizar las herramientas según tus necesidades.

#### Instalación:
1. Descarga el instalador para macOS desde [Miniconda](https://docs.anaconda.com/miniconda/).
2. Sigue los pasos de instalación estándar (incluyendo añadir Conda al PATH).
3. Verifica la instalación: 
   ```bash
   conda --version
   ```

#### Ventajas:
- **Ligereza**: A diferencia de Anaconda, no instala decenas de paquetes que quizás no necesites.
- **Flexibilidad**: Puedes instalar solo los paquetes necesarios usando `conda install`.

#### Uso típico en macOS:
- Crear un entorno con una versión específica de Python: 
  ```bash
  conda create --name myenv python=3.9
  ```

- Activar el entorno:
  ```bash
  conda activate myenv
  ```

---