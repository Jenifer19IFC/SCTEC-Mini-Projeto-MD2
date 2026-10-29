from typing import Annotated, List
from pydantic import BaseModel, Field

# Exemplo exibido no Swagger (/docs): imagem 28x28 com um traço vertical no meio, parecido com o dígito 1
LINHA_VAZIA = [0] * 28
LINHA_COM_TRACO = [0] * 13 + [255, 255] + [0] * 13
EXEMPLO_PIXELS = LINHA_VAZIA * 4 + LINHA_COM_TRACO * 20 + LINHA_VAZIA * 4

# Cada pixel é um número de 0 (preto) a 255 (branco); strict=True recusa textos como "12"
Pixel = Annotated[float, Field(ge=0, le=255, strict=True)]

class EntradaPredicao(BaseModel):
    """Corpo da requisição do POST /predict"""
    model_config = {'json_schema_extra': {'examples': [{'pixels': EXEMPLO_PIXELS}]}}

    pixels: List[Pixel] = Field(
        min_length=784, max_length=784,
        description='Imagem 28x28 achatada em 784 pixels de 0 a 255 (dígito branco em fundo preto, como no MNIST).',
    )

class SaidaPredicao(BaseModel):
    """Resposta do POST /predict"""
    digito: int = Field(ge=0, le=9, description='Dígito previsto pelo modelo (0 a 9).')
    confianca: float = Field(ge=0, le=1, description='Confiança do modelo no dígito previsto (de 0 a 1).')

class SaidaHealth(BaseModel):
    """Resposta do GET /health"""
    model_config = {'json_schema_extra': {'examples': [{'status': 'OK', 'modelo_carregado': True}]}}
    status: str = Field(description='Situação do serviço (ex.: "OK").')
    modelo_carregado: bool = Field(description='Indica se o modelo de ML está carregado na memória RAM.')
