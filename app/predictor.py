import io
import os
import pickle
import sys
from collections import Counter

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DIR_APP_MODELOS, MODELOS_APP
from src.manuscript import preprocessar_imagem_propria

NOMES_EXIBICAO = {
    'knn': 'KNN',
    'random_forest': 'Random Forest',
    'mlp': 'MLP',
}


def carregar_modelos(dir_modelos=DIR_APP_MODELOS):
    """Carrega os `.pkl` de `dir_modelos`. Retorna (modelos, faltando)
    """
    modelos = {}
    faltando = []

    for nome in MODELOS_APP:
        caminho = os.path.join(dir_modelos, f'{nome}.pkl')
        if not os.path.exists(caminho):
            faltando.append(nome)
            continue
        with open(caminho, 'rb') as f:
            modelos[nome] = pickle.load(f)

    return modelos, faltando


def detectar_traco_escuro(imagem_array, largura_borda=0.05, margem_ambiguidade=0.15):
    """Decide se o traço é escuro em fundo claro (True) ou claro em fundo escuro (False)

    Retorna (traco_escuro, ambiguo) -- ambiguo=True quando o sinal não é confiável.
    """
    altura, largura = imagem_array.shape
    faixa = max(3, int(min(altura, largura) * largura_borda))

    # A borda quase sempre é fundo, já que o dígito fica centralizado
    borda = np.concatenate([
        imagem_array[:faixa, :].ravel(), imagem_array[-faixa:, :].ravel(),
        imagem_array[:, :faixa].ravel(), imagem_array[:, -faixa:].ravel(),
    ])
    mediana_borda = np.median(borda)  # mediana: pouco sensível a sombra/iluminação irregular

    # Otsu separa a imagem em dois grupos: fundo e traço
    limiar = threshold_otsu(imagem_array)
    grupo_baixo = imagem_array[imagem_array <= limiar]
    grupo_alto = imagem_array[imagem_array > limiar]
    mediana_baixo = np.median(grupo_baixo) if grupo_baixo.size else 0.0
    mediana_alto = np.median(grupo_alto) if grupo_alto.size else 255.0

    # Onde a borda cai entre os dois grupos: 0 = grupo escuro, 1 = grupo claro
    amplitude = max(mediana_alto - mediana_baixo, 1e-6)
    posicao = (mediana_borda - mediana_baixo) / amplitude

    traco_escuro = posicao > 0.5
    ambiguo = abs(posicao - 0.5) < margem_ambiguidade  # perto demais do meio pra confiar

    return traco_escuro, ambiguo


def votar_maioria(previsoes_por_modelo):
    """Decide o dígito final por voto da maioria entre os 3 modelos"""
    contagem = Counter(previsoes_por_modelo.values())
    rotulo_mais_comum, votos = contagem.most_common(1)[0]

    if votos == len(previsoes_por_modelo):
        return {'vencedor': rotulo_mais_comum, 'unanime': True, 'divergente': False}
    if votos == 1:
        return {'vencedor': None, 'unanime': False, 'divergente': True}  # 3 modelos, 3 rótulos diferentes
    return {'vencedor': rotulo_mais_comum, 'unanime': False, 'divergente': False}


def _prever_com_orientacao(dados_imagem, modelos, traco_escuro):
    """Pré-processa e prevê com os 3 modelos numa orientação específica de traco_escuro"""
    imagem_28x28, imagem_normalizada = preprocessar_imagem_propria(
        io.BytesIO(dados_imagem), traco_escuro=traco_escuro
    )

    previsoes = {}
    for nome, modelo in modelos.items():
        probabilidades = modelo.predict_proba(imagem_normalizada)[0]
        rotulo = int(probabilidades.argmax())
        previsoes[nome] = {'rotulo': rotulo, 'confianca': float(probabilidades[rotulo])}

    confianca_media = float(np.mean([p['confianca'] for p in previsoes.values()]))
    return imagem_28x28, previsoes, confianca_media


def prever_digito(dados_imagem, modelos, forcar_traco_escuro=None):
    """Classifica uma imagem com os 3 modelos, decidindo a orientação do traço"""
    if forcar_traco_escuro is not None:
        traco_escuro, ambiguo = forcar_traco_escuro, False  # ajuste manual, pula a detecção
    else:
        imagem_cinza = np.array(Image.open(io.BytesIO(dados_imagem)).convert('L'))
        traco_escuro, ambiguo = detectar_traco_escuro(imagem_cinza)

    if ambiguo:
        # Orientação errada apaga o dígito no pré-processamento -- testa as duas e
        # fica com a que os modelos reconhecem com mais confiança
        imagem_a, previsoes_a, confianca_a = _prever_com_orientacao(dados_imagem, modelos, True)
        imagem_b, previsoes_b, confianca_b = _prever_com_orientacao(dados_imagem, modelos, False)

        if confianca_a >= confianca_b:
            imagem_28x28, previsoes, traco_escuro_usado = imagem_a, previsoes_a, True
        else:
            imagem_28x28, previsoes, traco_escuro_usado = imagem_b, previsoes_b, False
    else:
        imagem_28x28, previsoes, _ = _prever_com_orientacao(dados_imagem, modelos, traco_escuro)
        traco_escuro_usado = traco_escuro

    return {
        'imagem_28x28': imagem_28x28,
        'previsoes': previsoes,
        'traco_escuro_usado': traco_escuro_usado,
        'ambiguo': ambiguo,
        'votacao': votar_maioria({nome: p['rotulo'] for nome, p in previsoes.items()}),
    }
