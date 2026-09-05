import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist():
    """Baixa o MNIST do OpenML e retorna X (imagens) e y (rótulos)"""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target
    y = y.astype(np.uint8) # Target de string para inteiro
    return X, y
