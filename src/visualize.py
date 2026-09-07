import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

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


def plot_matriz_confusao(y_true, y_pred, nome_modelo):
    """Plota a matriz de confusão (10x10) de um modelo como mapa de calor"""
    matriz = confusion_matrix(y_true, y_pred, labels=range(10))

    plt.figure(figsize=(7, 6))
    plt.imshow(matriz, cmap='Blues')
    plt.colorbar()
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.xlabel("Rótulo Previsto")
    plt.ylabel("Rótulo Real")
    plt.xticks(range(10))
    plt.yticks(range(10))

    limite = matriz.max() / 2
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            cor_texto = 'white' if matriz[i, j] > limite else 'black'
            plt.text(j, i, matriz[i, j], ha='center', va='center', color=cor_texto)

    plt.tight_layout()

    nome_arquivo = nome_modelo.lower().replace(' ', '_')
    _salvar_figura(f'matriz_confusao_{nome_arquivo}.png')
    plt.show()

    return matriz


def plot_confianca(confiancas, titulo, nome_arquivo, cor='#d9534f'):
    """Plota o histograma da confiança (probabilidade máxima prevista) de um modelo"""
    plt.figure(figsize=(7, 4))
    plt.hist(confiancas, bins=20, color=cor, edgecolor='black')
    plt.axvline(confiancas.mean(), color='black', linestyle='--', label=f"Média: {confiancas.mean():.2%}")
    plt.title(titulo)
    plt.xlabel("Probabilidade máxima atribuída pelo modelo")
    plt.ylabel("Quantidade de amostras")
    plt.legend()
    plt.tight_layout()
    _salvar_figura(nome_arquivo)
    plt.show()


def plot_previsao_digito(imagem, probabilidades, titulo="Imagem processada", nome_arquivo='previsao_digito.png'):
    """Plota a imagem processada ao lado do gráfico de probabilidades previstas pelo modelo"""
    rotulo_previsto = int(np.argmax(probabilidades))

    fig, (eixo_imagem, eixo_probs) = plt.subplots(1, 2, figsize=(10, 4))

    eixo_imagem.imshow(imagem, cmap='binary')
    eixo_imagem.set_title(titulo)
    eixo_imagem.axis('off')

    classes = range(len(probabilidades))
    cores = ['#5cb85c' if classe == rotulo_previsto else 'steelblue' for classe in classes]
    eixo_probs.bar(classes, probabilidades, color=cores)
    eixo_probs.set_xticks(classes)
    eixo_probs.set_xlabel("Dígito")
    eixo_probs.set_ylabel("Probabilidade")
    eixo_probs.set_title(f"Previsão do modelo: {rotulo_previsto}")

    plt.tight_layout()
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