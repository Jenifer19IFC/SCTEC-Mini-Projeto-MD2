import os
import pickle

from config.config import DIR_MODELOS

CAMINHO_MODELO = os.path.join(DIR_MODELOS, 'mlp', 'mlp.pkl')


def carregar_modelo(caminho=CAMINHO_MODELO):
    """Carrega o MLP serializado do disco (chamado uma única vez, no lifespan da API)"""
    with open(caminho, 'rb') as f:
        return pickle.load(f)
