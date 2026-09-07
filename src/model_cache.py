import json
import os
import time

from config.config import CAMINHO_MELHORES_HIPERPARAMETROS as CAMINHO_PADRAO


def _carregar_todos(caminho):
    if not os.path.exists(caminho):
        return {}
    with open(caminho) as f:
        return json.load(f)


def _salvar_todos(dados, caminho):
    with open(caminho, 'w') as f:
        json.dump(dados, f, indent=2)


def obter_ou_buscar_modelo(nome_modelo, get_fn, buscar_fn, X_train, y_train,
                            buscar_novamente=False, caminho=CAMINHO_PADRAO):
    """Reaproveita os melhores hiperparâmetros já encontrados para um modelo. 
    Busca melhores hiperparâmetros se é para buscar novamente ou se o modelo ainda não tiver hiperparâmetros salvos.

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

    return modelo, melhores_params, tempo_treino
