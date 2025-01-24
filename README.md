# Complete A.I. & Machine Learning, Data Science Bootcamp (en proceso)

Este repositorio contiene todo el trabajo realizado durante el curso [_Complete A.I. & Machine Learning, Data Science Bootcamp_](https://zerotomastery.io/courses/machine-learning-and-data-science-bootcamp/). La estructura del proyecto y los contenidos están organizados para facilitar la navegación y el aprendizaje.

Este curso cubre temas esenciales de Machine Learning, Inteligencia Artificial y Ciencia de Datos. Está dividido en 20 secciones, cada una con notebooks y scripts detallados.

## Índice

1. [Estructura del proyecto](#estructura-del-proyecto)
2. [Instalación y configuración](#instalación-y-configuración)
3. [Progreso del curso](#progreso-del-curso)

## Estructura del proyecto

```
project/
├── assets/                # Imágenes y recursos gráficos adicionales
├── data/                  # Datos crudos o procesados
│   ├── raw/               # Datos originales sin procesar
│   ├── processed/         # Datos procesados listos para análisis
├── docs/                  # Documentación de cada sección del curso
├── scripts/               # Scripts de Python para tareas específicas
├── models/                # Modelos entrenados y checkpoints
├── notebooks/             # Jupyter notebooks organizados por sección
├── references/            # Recursos externos como papers, enlaces, etc.
├── environment/           # Configuración del entorno
│   ├── environment.yml    # Archivo para configurar el entorno Conda
│   └── requirements.txt   # Dependencias opcionales para pip
├── requirements.txt       # Alternativa para instalar dependencias con pip
├── README.md              # Documentación principal
└── .gitignore             # Archivos y carpetas a ignorar en Git
```

## Instalación y configuración

### Requisitos:

- Python >= 3.9
- Conda instalado (Miniconda o Anaconda)
- Jupyter Notebook

### Instrucciones:

1. Clona el repositorio:
   ```bash
   git clone git@github.com:rociobenitez/AI-ML-DataScience-Bootcamp.git
   cd AI-ML-DataScience-Bootcamp
   ```
2. Crea y activa el entorno de Conda:
   Si estás trabajando con Conda, utiliza el archivo `environment.yml` para crear un entorno con todas las dependencias:
   ```bash
   conda env create -f environment/environment.yml
   conda activate env
   ```
   Si prefieres usar pip en lugar de Conda, o necesitas un entorno compatible con otros sistemas, puedes usar el archivo `requirements.txt`:
   ```bash
   pip install -r environment/requirements.txt
   ```
3. Registra el entorno en Jupyter Notebook:
   ```bash
   python -m ipykernel install --user --name=env --display-name "Python (env)"
   ```
4. Inicia Jupyter Notebook:
   ```bash
   jupyter notebook
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
