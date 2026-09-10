import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, gaussian_filter, shift as deslocar
from skimage.filters import threshold_otsu

TAMANHO_FINAL = 28
TAMANHO_DIGITO = 20  # deixa uma margem de 4px de cada lado, como no MNIST original


def carregar_imagem(caminho):
    """Abre a imagem a partir do caminho informado"""
    return Image.open(caminho)


def converter_escala_cinza(imagem):
    """Converte a imagem para escala de cinza (0-255)"""
    return np.array(imagem.convert('L'))


def inverter_cores(imagem_array):
    """Inverte as cores da imagem (branco vira preto e vice-versa)"""
    return 255 - imagem_array


def remover_sombra(imagem_array, sigma=31):
    """Remove variações suaves de iluminação (sombras) da foto, preservando o traço do número"""
    fundo = gaussian_filter(imagem_array.astype(np.float64), sigma=sigma)
    residuo = np.clip(imagem_array.astype(np.float64) - fundo, 0, None)

    if residuo.max() > 0:
        residuo = residuo / residuo.max() * 255

    return residuo.astype(np.uint8)


def centralizar_digito(imagem_array, tamanho_final=TAMANHO_FINAL, tamanho_digito=TAMANHO_DIGITO):
    """Redimensiona para 28x28 com centralização de massa/bounding box"""

    imagem_array = remover_sombra(imagem_array)

    # Binariza com o limiar de Otsu para separar o traço do fundo
    limiar = threshold_otsu(imagem_array)
    mascara_tinta = imagem_array > limiar

    linhas, colunas = np.where(mascara_tinta)
    if len(linhas) == 0:
        return np.zeros((tamanho_final, tamanho_final), dtype=np.uint8)

    # Zera o fundo, mantendo só a intensidade do traço
    imagem_array = np.where(mascara_tinta, imagem_array, 0)

    # Recorta pelo bounding box do traço
    topo, base = linhas.min(), linhas.max()
    esquerda, direita = colunas.min(), colunas.max()
    digito_recortado = imagem_array[topo:base + 1, esquerda:direita + 1]

    # Redimensiona mantendo a proporção, para caber numa caixa tamanho_digito x tamanho_digito
    altura, largura = digito_recortado.shape
    escala = tamanho_digito / max(altura, largura)
    nova_altura = max(1, round(altura * escala))
    nova_largura = max(1, round(largura * escala))

    digito_redimensionado = np.array(Image.fromarray(digito_recortado).resize((nova_largura, nova_altura), Image.LANCZOS)).astype(np.float64)

    # Realça o contraste, perdido no redimensionamento de um traço fino
    if digito_redimensionado.max() > 0:
        digito_redimensionado = digito_redimensionado / digito_redimensionado.max() * 255

    # Centraliza o dígito redimensionado num canvas tamanho_final x tamanho_final
    canvas = np.zeros((tamanho_final, tamanho_final), dtype=np.float64)
    topo_canvas = (tamanho_final - nova_altura) // 2
    esquerda_canvas = (tamanho_final - nova_largura) // 2
    canvas[topo_canvas:topo_canvas + nova_altura, esquerda_canvas:esquerda_canvas + nova_largura] = digito_redimensionado

    # Desloca o canvas para que o centro de massa fique no centro da imagem
    centro_y, centro_x = center_of_mass(canvas)
    deslocamento = (tamanho_final / 2 - centro_y, tamanho_final / 2 - centro_x)
    canvas = deslocar(canvas, deslocamento, mode='constant', cval=0.0)

    return np.clip(canvas, 0, 255).astype(np.uint8)


def normalizar_imagem(imagem_array):
    """Redimensiona os valores de pixel de [0, 255] para a escala [0.0, 1.0]"""
    return imagem_array / 255.0


def preprocessar_imagem_propria(caminho, traco_escuro=True):
    """Pipeline: escala de cinza -> inversão de cores (se necessário) ->
    redimensionamento para 28x28 com centralização de massa/bounding box -> normalização.

    traco_escuro=True  -> traço escuro e fundo branco 
    traco_escuro=False -> traço branco e fundo escuro (como no MNIST original) 

    Retorna (imagem_28x28, imagem_normalizada):
    - imagem_28x28: array 28x28 (0-255) - visualização;
    - imagem_normalizada: vetor de 1x784 normalizado em [0.0, 1.0] - modelo
    """
    imagem_array = converter_escala_cinza(carregar_imagem(caminho))

    if traco_escuro:
        imagem_array = inverter_cores(imagem_array)

    imagem_28x28 = centralizar_digito(imagem_array)
    imagem_normalizada = normalizar_imagem(imagem_28x28).reshape(1, -1)

    return imagem_28x28, imagem_normalizada
