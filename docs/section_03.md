# Data Science Environment Setup

Este documento guía paso a paso la configuración de un entorno de desarrollo para proyectos de **Ciencia de Datos y Machine Learning** utilizando herramientas como **Conda** y **Jupyter Notebook**.

Incluye:

- Cómo instalar y gestionar entornos virtuales con Conda.
- Cómo configurar y optimizar tu equipo para trabajar en proyectos avanzados de ML.
- Cómo compartir y colaborar eficientemente utilizando archivos `.yml`.

Puedes seguir esta guía para asegurar un entorno limpio, reproducible y escalable, fundamental para trabajar con modelos de Machine Learning y grandes volúmenes de datos.

## Índice

1. [¿Qué es Conda?](#1-qué-es-conda)
2. [Conda CheatSheet](#2-conda-cheatsheet)
3. [Getting started with Conda](#3-getting-started-with-conda)
4. [Preparando el equipo para ML](#4-preparando-el-equipo-para-ml)
5. [Miniconda para macOS](#5-miniconda-para-macos)
   - [Uso típico en macOs](#uso-típico-en-macos)
6. [Cómo Saber si Tienes Conda o Miniconda Instalados](#6-cómo-saber-si-tienes-conda-o-miniconda-instalados)
7. [Pasos para Configurar un Proyecto de ML en Mac](#7-pasos-para-configurar-un-proyecto-de-ml-en-mac) ⭐️
8. [Usar el Entorno en Jupyter Notebook](#8-usar-el-entorno-en-jupyter-notebook)
9. [Compartir tu Entorno Conda](#9-compartir-tu-entorno-conda)
10. [Jupyter Notebook](#10-jupyter-notebook)
11. [Atajos Esenciales para Jupyter Notebook](#11-atajos-esenciales-para-jupyter-notebook)

## 1. ¿Qué es Conda?

Conda es un **gestor de entornos y paquetes** de código abierto. Fue diseñado inicialmente para Python, pero también gestiona otros lenguajes como R. Sus principales características son:

- **Gestión de entornos:** Puedes crear, clonar, exportar y eliminar entornos virtuales con diferentes versiones de Python y paquetes instalados.
- **Gestión de dependencias:** Se asegura de que los paquetes instalados sean compatibles entre sí.
- **Versatilidad:** Funciona con cualquier sistema operativo (Windows, macOS, Linux) y no está limitado a paquetes de Python (también puede gestionar bibliotecas de C, C++, etc.).

**Ventajas principales:**

- Instalación sencilla de paquetes complicados.
- Ideal para proyectos de data science y machine learning debido a la gestión eficiente de dependencias.

> 🔗 [Conda Documentación](https://docs.conda.io/en/latest/)

## 2. Conda CheatSheet

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

> 🔗 Consulta el [Conda CheatSheet](/references/conda-cheatsheet.pdf) para más comandos.

## 3. Getting started with Conda

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

## 4. Preparando el equipo para ML

Configurar tu equipo correctamente es crucial para garantizar que los proyectos de Machine Learning y Ciencia de Datos se desarrollen de manera eficiente y sin conflictos entre paquetes.

El artículo [Getting your computer ready for machine learning](https://www.mrdbourke.com/get-your-computer-ready-for-machine-learning-using-anaconda-miniconda-and-conda/) de Daniel Bourke describe cómo usar herramientas como **Anaconda**, **Miniconda** y **Conda** para preparar tu entorno de trabajo.

### Resumen Técnico:

1. **Anaconda:** Ideal para principiantes porque incluye un ecosistema completo con herramientas preinstaladas.
2. **Miniconda:** Más ligera y flexible, adecuada para usuarios avanzados que prefieren instalar solo lo necesario.
3. **Conda:** El motor que impulsa ambas plataformas, permite gestionar entornos y dependencias de manera eficiente.

<img src="/assets/section-3/conda-miniconda-anaconda.png" alt="Conda, Miniconda y Anaconda" width="600">

### Recomendaciones Prácticas:

- Usa **Miniconda** si quieres mantener tu equipo más limpio y personalizar las herramientas necesarias.
- Crea **entornos específicos** para cada proyecto para evitar conflictos de dependencias.
- Asegúrate de que la versión de Python y las bibliotecas instaladas sean compatibles con los requerimientos de tu proyecto.

> 🔗 Consulta el artículo completo para más detalles y configuraciones avanzadas: [Getting your computer ready for machine learning](https://www.mrdbourke.com/get-your-computer-ready-for-machine-learning-using-anaconda-miniconda-and-conda/)

## 5. Miniconda para macOS

Miniconda es una versión compacta y ligera de Anaconda.

- **Incluye lo esencial:** Python, Conda y paquetes básicos.
- Es ideal para mantener un sistema limpio y personalizar las herramientas que necesitas para tus proyectos.
- 🔗 [Installing Miniconda Docs](https://docs.anaconda.com/miniconda/install/)

### Instalación

1. **Descarga el instalador para macOS:**  
   Según tu arquitectura, utiliza el siguiente comando en la terminal para descargar Miniconda:

   - **Apple Silicon (M1/M2):**

     ```bash
     curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
     ```

   - **Intel:**
     ```bash
     curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
     ```

2. **Ejecuta el instalador:**  
   Instala Miniconda ejecutando el script descargado:

   - Para Apple Silicon:
     ```bash
     bash Miniconda3-latest-MacOSX-arm64.sh
     ```
   - Para Intel:
     ```bash
     bash Miniconda3-latest-MacOSX-x86_64.sh
     ```

3. **Sigue las instrucciones de instalación:**

   - Presiona **Enter** para revisar y aceptar el acuerdo de licencia.
   - Acepta la ubicación predeterminada para la instalación (generalmente `~/miniconda3`), o elige una ruta personalizada.
   - El instalador te preguntará si deseas inicializar Conda automáticamente al abrir la terminal:
     - Escribe `yes` para añadir Conda al perfil de tu shell, lo que facilita el uso de comandos de Conda.
     - Si eliges `no`, deberás configurarlo manualmente más adelante.

4. **Cierra y vuelve a abrir tu terminal:**  
   Esto aplica los cambios realizados durante la instalación.

5. **Verifica la instalación:**  
   Asegúrate de que Conda esté instalado correctamente ejecutando:
   ```bash
   conda --version
   ```
   Si todo está en orden, deberías ver algo como `conda 24.9.2`.

### Ventajas de Miniconda

- **Ligereza:** A diferencia de Anaconda, Miniconda no incluye decenas de paquetes preinstalados. Esto reduce el consumo de espacio en disco y permite una mayor personalización.
- **Flexibilidad:** Puedes instalar solo las bibliotecas necesarias para tu proyecto, optimizando recursos y minimizando conflictos entre dependencias.
- **Compatibilidad:** Funciona en múltiples sistemas operativos (Windows, macOS, Linux) y admite lenguajes más allá de Python, como R y Julia.
- **Velocidad en entornos personalizados:** Crear entornos específicos con Miniconda es rápido y eficiente, lo que facilita la reproducción de proyectos en otros sistemas.
- **Soporte de Canales:** Permite utilizar canales como `conda-forge`, ampliando la disponibilidad de paquetes actualizados y especializados.

## 6. Cómo Saber si Tienes Conda o Miniconda Instalados

Para verificar si ya tienes Conda o Miniconda instalados en tu equipo, sigue estos pasos:

1. **Verifica Conda:**
   En tu terminal, ejecuta el siguiente comando:

   ```bash
   conda --version
   ```

   - Si Conda está instalado, verás algo como: `conda 24.9.2` (el número puede variar).
   - Si no está instalado, la terminal mostrará un error indicando que el comando no se encuentra.

2. **Verifica Miniconda:**
   Miniconda utiliza Conda como núcleo, por lo que el comando anterior también confirma su instalación. Sin embargo, puedes verificar su presencia buscando la carpeta de instalación típica:

   - En **macOS/Linux**:
     ```bash
     ls ~/miniconda3
     ```
   - En **Windows**:  
     Busca `Miniconda3` en el directorio de instalación (generalmente `C:\Users\TuUsuario\Miniconda3`).

3. **Verifica si Conda está en tu PATH:**
   Si Conda no responde pero está instalado, puede que no esté en el `PATH`. Para verificar:
   ```bash
   echo $PATH
   ```
   Busca una ruta como `/home/usuario/miniconda3/bin` (en macOS/Linux) o `C:\Users\TuUsuario\Miniconda3\Scripts` (en Windows).

> **Si Conda o Miniconda No Están Instalados**: Si no están instalados, sigue los pasos de instalación indicados en la sección ["5. Miniconda para macOS"](#5-miniconda-para-macos) o consulta las guías específicas de instalación para tu sistema operativo.

## 7. Pasos para Configurar un Proyecto de ML en Mac

Estos son los pasos clave para iniciar un proyecto de Machine Learning utilizando **Conda**, asegurando un entorno bien configurado y aislado.

**1. Crear la Carpeta del Proyecto**

- Navega al directorio donde deseas guardar tu proyecto y crea una carpeta específica:
  ```bash
  mkdir nombre-del-proyecto
  cd nombre-del-proyecto
  ```

**2. Crear un Entorno con una Versión Específica de Python**

- Utiliza Conda para crear un entorno virtual que incluya las bibliotecas necesarias desde el inicio:
  ```bash
  conda create --prefix ./env python=3.10 pandas numpy matplotlib scikit-learn seaborn
  ```

<img src="/assets/section-3/conda-terminal.png" alt="Crear entorno en un proyecto de ML (terminal)" width="600">

**¿Qué sucede al ejecutar este comando?**

- **`--prefix ./env`:** Indica que el entorno se guardará en la carpeta del proyecto (`./env`).
- **`python=3.9`:** Instala una versión específica de Python.
- **Bibliotecas adicionales (`pandas numpy matplotlib scikit-learn`):** Estas herramientas esenciales para proyectos de ciencia de datos se instalan automáticamente.
- Al ejecutar el comando, Conda descargará los paquetes y resolverá dependencias. Cuando se te solicite, confirma la instalación introduciendo `y` y presionando **Enter**.

**3. Activar el Entorno**

- Una vez creado, activa el entorno para trabajar en un entorno aislado:
  ```bash
  conda activate ./env
  ```

<img src="/assets/section-3/conda-terminal-2.png" alt="Activar entorno en un proyecto de ML (terminal)" width="600">

> **Nota:** Si activas el entorno correctamente, deberías ver algo como `(env)` al inicio del prompt de tu terminal, indicando que estás dentro del entorno.

**4. Verificar los Entornos Disponibles**

- Lista todos los entornos instalados en tu sistema para confirmar que el nuevo entorno está activo:
  ```bash
  conda env list
  ```
  - Verás una lista de entornos y sus ubicaciones. El entorno activo estará marcado con un `*`.

<img src="/assets/section-3/conda-env-list.png" alt="Verificar los Entornos Disponibles (terminal)" width="600">

**5. Añadir el Entorno al Proyecto**

- **Opcional:** Para asegurar que el entorno está documentado en el proyecto, exporta la configuración a un archivo YAML:
  ```bash
  conda env export > environment.yml
  ```
  Este archivo se puede compartir para que otros usuarios repliquen el entorno con:
  ```bash
  conda env create -f environment.yml
  ```

**6. Instalar Bibliotecas Adicionales**

- Si necesitas más bibliotecas en el futuro, puedes instalarlas fácilmente dentro del entorno activado:
  ```bash
  conda install nombre-de-la-biblioteca
  ```

**7. Desactivar el Entorno**

- Cuando termines de trabajar, desactiva el entorno para volver al sistema base:
  ```bash
  conda deactivate
  ```

**8. Eliminar el Entorno (si es necesario)**

- Para borrar un entorno específico, usa:
  ```bash
  conda remove --prefix ./env --all
  ```

<img src="/assets/section-3/project-setup.png" alt="Project Setup Environment" width="600">

> Estos pasos te permiten configurar un entorno limpio y aislado para cada proyecto. Esto asegura **compatibilidad de dependencias y evita conflictos** con otros proyectos en tu sistema.

## 8. Usar el Entorno en Jupyter Notebook

Una vez configurado tu entorno de Conda, puedes integrarlo con **Jupyter Notebook** para realizar análisis interactivos y trabajar en tus proyectos de Machine Learning de manera eficiente.

**1. Instalar Jupyter Notebook en el Entorno**

- Si Jupyter Notebook no está instalado en tu entorno, asegúrate de añadirlo:
  ```bash
  conda install notebook
  ```
  Esto instalará Jupyter Notebook y las dependencias necesarias dentro del entorno activo.

**2. Añadir el Entorno a Jupyter como un Kernel**

- Instala la biblioteca `ipykernel` para registrar el entorno en Jupyter:

  ```bash
  conda install ipykernel
  ```

- Registra el entorno como un kernel en Jupyter:
  ```bash
  python -m ipykernel install --user --name=nombre-del-entorno --display-name "Python (nombre-del-entorno)"
  ```
  - **`--name`:** Nombre interno del entorno en Jupyter (usa el mismo nombre del entorno Conda).
  - **`--display-name`:** Nombre visible en Jupyter Notebook.

**3. Iniciar Jupyter Notebook**

- Desde el directorio del proyecto, inicia Jupyter Notebook con:
  ```bash
  jupyter notebook
  ```
  Esto abrirá una interfaz web donde podrás gestionar y crear notebooks.

**4. Seleccionar el Entorno en Jupyter**

1. Crea o abre un notebook nuevo.
2. En la barra superior, haz clic en **Kernel > Change Kernel**.
3. Selecciona el entorno que registraste (aparecerá como `Python (nombre-del-entorno)`).

**5. Verificar el Entorno**
Para confirmar que estás usando el entorno correcto, ejecuta el siguiente comando dentro de una celda de Jupyter:

```python
!which python
```

Deberías ver la ruta al Python dentro de tu entorno Conda, algo como:

```
/Users/tu-usuario/nombre-del-proyecto/env/bin/python
```

**6. Instalar Bibliotecas Adicionales desde Jupyter**
Si necesitas instalar nuevas bibliotecas mientras trabajas, puedes hacerlo directamente desde una celda de Jupyter usando:

```python
!conda install nombre-de-la-biblioteca -y
```

**7. Cerrar Jupyter Notebook**
Cuando termines de trabajar, puedes cerrar Jupyter Notebook desde la interfaz web o detenerlo desde la terminal con `Ctrl + C`.

> 🎯 Estos pasos permiten usar el entorno Conda como un kernel en Jupyter Notebook, asegurando que todo el trabajo en tus notebooks utilice las bibliotecas y configuraciones específicas del proyecto.

## 9. Compartir tu Entorno Conda

Puede llegar un momento en el que necesites compartir el contenido de tu entorno Conda. Esto puede ser útil para compartir el flujo de trabajo de un proyecto con un colega o con alguien que necesite configurar su sistema para tener acceso a las mismas herramientas que tú.

Hay un par de formas de hacerlo:

1. **Compartir toda la carpeta del proyecto**, incluida la carpeta del entorno que contiene todos los paquetes de Conda.
2. **Compartir un archivo `.yml`** (se pronuncia _YAM-L_) que describa tu entorno Conda.

### Método 1: Compartir la Carpeta Completa

- **Ventaja:** Es un método sencillo. Comparte la carpeta, activa el entorno y ejecuta el código.
- **Desventaja:** Las carpetas de entorno pueden ser muy grandes y difíciles de compartir.

### Método 2: Compartir un Archivo `.yml`

Un archivo `.yml` es básicamente un archivo de texto con instrucciones que indican a Conda cómo configurar un entorno. Este método es más ligero y práctico para compartir configuraciones.

**Ejemplo: Exportar un entorno a un archivo `.yml`**
Si tienes un entorno en `/Users/rocio/Desktop/project_1/env` y deseas exportarlo como un archivo `environment.yml`, usa el siguiente comando:

```bash
conda env export --prefix /Users/rocio/Desktop/project_1/env > environment.yml
```

Al ejecutar este comando, se generará un archivo `.yml` llamado `environment.yml`.

Un archivo `.yml` de ejemplo podría verse así:

```yaml
name: my_ml_env
dependencies:
  - numpy
  - pandas
  - scikit-learn
  - jupyter
  - matplotlib
```

El contenido del archivo dependerá de los paquetes instalados en tu entorno.

### Crear un Entorno desde un Archivo `.yml`

Para crear un nuevo entorno llamado `env_from_file` a partir de un archivo llamado `environment.yml`, utiliza el siguiente comando:

```bash
conda env create --file environment.yml --name env_from_file
```

Esto configurará un nuevo entorno en tu sistema con los paquetes especificados en el archivo `.yml`.

### Más Información

- **Para compartir entornos:** Consulta la [documentación de Conda sobre compartir entornos](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment).
- **Para crear entornos desde un archivo `.yml`:** Consulta la [documentación de Conda sobre la creación de entornos desde archivos `.yml`](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

## 10. Jupyter Notebook

**Jupyter Notebook** es una herramienta interactiva ampliamente utilizada en Ciencia de Datos y Machine Learning. Permite combinar código ejecutable, visualizaciones y texto en un solo documento, ideal para análisis exploratorio, desarrollo de modelos y documentación.

### Uso Básico

1. **Iniciar Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

   Esto abre una interfaz web desde donde puedes crear y gestionar notebooks.

2. **Seleccionar un Kernel:**

   - Desde el menú de Jupyter, selecciona el kernel correspondiente a tu entorno de Conda.
   - Esto asegura que el código se ejecute con las bibliotecas instaladas en ese entorno.

3. **Escribir y Ejecutar Código:**

   - Divide el trabajo en celdas que pueden contener texto o código Python.
   - Ejecuta una celda con `Shift + Enter`.

4. **Guardar y Exportar:**
   - Guarda tu trabajo como un archivo `.ipynb`.
   - Exporta como HTML o PDF si necesitas compartir resultados.

> **Recursos adicionales:**

- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/en/stable/)
- [Tutorial para principiantes](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)

> 🎯 Jupyter Notebook es esencial para trabajar de forma colaborativa y documentar todo el flujo de trabajo en proyectos de ML.

## 11. Atajos Esenciales para Jupyter Notebook

Conocer los atajos de teclado en **Jupyter Notebook** puede aumentar significativamente tu productividad.

### Modo de Edición (dentro de una celda):

- `Ctrl + Enter`: Ejecuta la celda actual sin mover el cursor.
- `Shift + Enter`: Ejecuta la celda y mueve el cursor a la siguiente.
- `Alt + Enter`: Ejecuta la celda y crea una nueva celda debajo.
- `Esc`: Cambia al modo de comando.

### Modo de Comando (fuera de una celda):

- `A`: Inserta una celda arriba de la actual.
- `B`: Inserta una celda debajo de la actual.
- `D + D`: Elimina la celda seleccionada.
- `M`: Convierte la celda en Markdown.
- `Y`: Convierte la celda en código.
- `Z`: Deshacer eliminación de una celda.

### Atajos Globales:

- `Ctrl + S`: Guarda el notebook.
- `Shift + M`: Combina celdas seleccionadas.
- `Ctrl + Shift + -`: Divide una celda en el punto del cursor.

> 📌 Para más atajos, presiona `H` en el modo de comando para abrir la lista completa.
