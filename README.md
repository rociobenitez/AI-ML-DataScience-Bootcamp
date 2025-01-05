# Complete A.I. & Machine Learning, Data Science Bootcamp

Este repositorio contiene todo el trabajo realizado durante el curso _Complete A.I. & Machine Learning, Data Science Bootcamp_. La estructura del proyecto y los contenidos están organizados para facilitar la navegación y el aprendizaje.

---

## **Índice**

1. [Introducción](#introducción)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Instalación y configuración](#instalación-y-configuración)
4. [Progreso del curso](#progreso-del-curso)

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
assets/                 # Imágenes y archivos
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

| Sección | Título                                                     | Descripción breve                                                                                                                                   | Estado |
| ------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 1       | [Machine Learning 101](/docs/section_01.md)                | Qué es ML y tipos de ML                                                                                                                             | ✅     |
| 2       | [ML and Data Science Framework](/docs/section_02.md)       | Framework de 6 pasos, features, modelado, overfitting vs. underfitting, y herramientas clave                                                        | ✅     |
| 3       | [Data Science Environment Setup](/docs/section_03.md)      | Configuración de entornos, herramientas clave y uso de Jupyter Notebook                                                                             | ✅     |
| 4       | [Pandas](/docs/section_04.md)                              | Introducción y práctica con Pandas: manipulación, análisis y preparación de datos.                                                                  | ✅     |
| 5       | [Numpy](/docs/section_05.md)                               | Arrays, Matrices, Operaciones, Estadísticas, Transformaciones                                                                                       | ✅     |
| 6       | [Matplotlib](/docs/section_06.md)                          | Data visualization, customization, saving plots                                                                                                     | ✅     |
| 7       | [Scikit-learn](/docs/section_07.md)                        | Workflow en Scikit-Learn: obtener y preparar los datos, elegir estimador e hiperparámetros, ajustar, hacer predicciones, evaluar, mejorar el modelo | ✅     |
| 8       | [Supervised Learning - Clasificación](/docs/section_08.md) | Flujo de trabajo para abordar etapas clavo de un proyecto de clasificación de machine learning                                                      | ✅     |
| 9       | Supervised Learning - Regresión                            | Flujo de trabajo para abordar etapas clavo de un proyecto de regresión de machine learning                                                          | 🚧     |
| 10      | [Data Engineering](/docs/section_10.md)                    | Data Engineering, Tipos de Datos, Tipos de Bases de Datos, Hadoop, Apache Spark, y Stream Processing                                                | ✅     |
| 12      | Redes neuronales                                           | Deep Learning, Transfer Learning and TensorFlow 2                                                                                                   | ❌     |
| ...     | ...                                                        | ...                                                                                                                                                 | ...    |

- **✅ Completado**
- **🚧 En progreso**
- **❌ Pendiente**

---

## Recursos y Enlaces de interés

- 🔗 [Jupyter Notebook vs Google Colab](/docs/extra/jupyter-vs-colab.md)
