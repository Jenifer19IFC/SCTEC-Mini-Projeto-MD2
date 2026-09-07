import os

import matplotlib.pyplot as plt
import numpy as np

from config.config import DIR_FIGURES


def _salvar_figura(nome_arquivo, dir_figuras=DIR_FIGURES):
    os.makedirs(dir_figuras, exist_ok=True)
    plt.savefig(os.path.join(dir_figuras, nome_arquivo), dpi=150, bbox_inches='tight')


def plot_digito(X, y, indice):
    """Plota a imagem de um dígito e mostra seu rótulo real"""
    digito_imagem = X[indice].reshape(28, 28)

    plt.figure(figsize=(4, 4))
    plt.imshow(digito_imagem, cmap='binary')
    plt.title(f"Rótulo Real (Target): {y[indice]}", fontsize=14)
    plt.axis('off')
    _salvar_figura(f'digito_{indice}.png')
    plt.show()


def plot_distribuicao(classes, contagens, titulo="Distribuição das classes no MNIST",
                       nome_arquivo='distribuicao_classes.png'):
    """Plota um gráfico de barras com a distribuição das classes"""
    plt.figure(figsize=(8, 5))
    plt.bar(classes, contagens, color='steelblue')
    plt.xticks(classes)
    plt.xlabel("Dígito")
    plt.ylabel("Quantidade de amostras")
    plt.title(titulo)
    _salvar_figura(nome_arquivo)
    plt.show()


def plot_grade_digitos(X, y):
    """Plota uma grade contendo dígito (0 a 9)"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        # Encontra o índice da primeira ocorrência do dígito 'i'
        indice = np.where(y == i)[0][0]

        ax.imshow(X[indice].reshape(28, 28), cmap='binary')
        ax.set_title(f"Dígito: {y[indice]}")
        ax.axis('off')

    plt.suptitle("Exemplos de cada dígito do MNIST", fontsize=16)
    plt.tight_layout()
    _salvar_figura('grade_digitos.png')
    plt.show()