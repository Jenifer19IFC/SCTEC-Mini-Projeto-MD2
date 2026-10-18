import os
import pickle

import numpy as np

from config.config import DIR_MODELOS

CAMINHO_MODELO = os.path.join(DIR_MODELOS, 'mlp', 'mlp.pkl')


def carregar_modelo(caminho=CAMINHO_MODELO):
    """Carrega o MLP serializado do disco (chamado uma única vez, no lifespan da API)"""
    with open(caminho, 'rb') as f:
        return pickle.load(f)


def prever_digito(modelo, pixels):
    """Recebe os 784 pixels (0 a 255) e retorna digito e confianca"""
    if max(pixels) == 0:
        raise ValueError('Imagem vazia: todos os pixels são 0, nenhum dígito foi desenhado.')

    # Mesmo pré-processamento do treino: 1 imagem com 784 pixels, normalizada para 0 a 1
    X = np.array(pixels).reshape(1, -1) / 255.0

    probabilidades = modelo.predict_proba(X)[0]  # 10 probabilidades, uma para cada dígito
    posicao = probabilidades.argmax()            

    digito = int(modelo.classes_[posicao])
    confianca = float(probabilidades[posicao])
    return digito, confianca
