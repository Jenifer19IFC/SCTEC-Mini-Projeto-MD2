import numpy as np


def resumo_shapes(X, y):
    """Imprime o formato de X e y e as classes existentes"""
    print(f"Formato de X (amostras, pixels por imagem): {X.shape}")
    print(f"Formato de y (rótulos): {y.shape}")
    print(f"Classes únicas existentes: {np.unique(y)}")


def distribuicao_classes(y):
    """Retorna as classes e a contagem de amostras de cada uma"""
    return np.unique(y, return_counts=True)
