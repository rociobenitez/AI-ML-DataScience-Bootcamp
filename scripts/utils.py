import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------ EDA ------------

def plot_categorical_grid(data, columns, n_cols=3, figsize=(15, 10)):
    """
    Genera un grid de gráficos de barras para variables categóricas.

    Args:
        data (DataFrame): Dataset que contiene los datos.
        columns (list): Lista de nombres de las columnas categóricas.
        n_cols (int): Número de columnas en el grid (por defecto, 3).
        figsize (tuple): Tamaño del gráfico (por defecto, (15, 10)).
    """
    # Calcular número de filas
    n_rows = (len(columns) + n_cols - 1) // n_cols

    # Crear subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Asegurar que los ejes sean iterables

    # Iterar sobre las columnas categóricas
    for i, col in enumerate(columns):
        sns.countplot(data=data, x=col, ax=axes[i])
        axes[i].set_title(f'Distribución de {col}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Conteo')

    # Eliminar gráficos vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar el layout
    plt.tight_layout()

    plt.show()