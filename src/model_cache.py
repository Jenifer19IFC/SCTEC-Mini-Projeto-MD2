import json
import os
import pickle
import time

from config.config import CAMINHO_MELHORES_HIPERPARAMETROS as CAMINHO_PADRAO
from config.config import DIR_MODELOS


def _carregar_todos(caminho):
    if not os.path.exists(caminho):
        return {}
    with open(caminho) as f:
        return json.load(f)


def _salvar_todos(dados, caminho):
    with open(caminho, 'w') as f:
        json.dump(dados, f, indent=2)


def _dir_modelo(nome_modelo, dir_modelos):
    dir_modelo = os.path.join(dir_modelos, nome_modelo)
    os.makedirs(dir_modelo, exist_ok=True)
    return dir_modelo


def _salvar_modelo(modelo, nome_modelo, dir_modelos):
    caminho_modelo = os.path.join(_dir_modelo(nome_modelo, dir_modelos), f'{nome_modelo}.pkl')
    with open(caminho_modelo, 'wb') as f:
        pickle.dump(modelo, f)


def salvar_metricas_modelo(metricas, nome_modelo, dir_modelos=DIR_MODELOS):
    """Salva as métricas de um modelo"""
    metricas = {chave: (valor.item() if hasattr(valor, 'item') else valor) for chave, valor in metricas.items()}

    caminho_metricas = os.path.join(_dir_modelo(nome_modelo, dir_modelos), 'metricas.json')
    with open(caminho_metricas, 'w') as f:
        json.dump(metricas, f, indent=2)


def obter_ou_buscar_modelo(nome_modelo, get_fn, buscar_fn, X_train, y_train,
                            buscar_novamente=False, caminho=CAMINHO_PADRAO, dir_modelos=DIR_MODELOS):
    """Reaproveita os melhores hiperparâmetros já encontrados para um modelo.
    Busca melhores hiperparâmetros se é para buscar novamente ou se o modelo ainda não tiver hiperparâmetros salvos.

    O modelo final treinado é salvo em `dir_modelos` (pasta `models` na raiz do projeto).

    Retorna (modelo_treinado, melhores_hiperparametros, tempo_treino_segundos)
    """
    dados         = _carregar_todos(caminho)
    params_salvos = dados.get(nome_modelo)

    if buscar_novamente or params_salvos is None:
        grid            = buscar_fn(X_train, y_train)
        melhores_params = grid.best_params_
        modelo          = grid.best_estimator_
        tempo_treino    = grid.refit_time_

        dados[nome_modelo] = melhores_params
        _salvar_todos(dados, caminho)
    else:
        melhores_params = params_salvos
        modelo = get_fn(**melhores_params)

        inicio = time.time()
        modelo.fit(X_train, y_train)
        tempo_treino = time.time() - inicio

    _salvar_modelo(modelo, nome_modelo, dir_modelos)

    return modelo, melhores_params, tempo_treino
