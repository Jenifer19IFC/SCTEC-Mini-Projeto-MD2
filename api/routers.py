from fastapi import APIRouter, HTTPException, Request
from api.schemas import EntradaPredicao, SaidaHealth, SaidaPredicao
from api.services import prever_digito

router = APIRouter()


@router.get('/health', response_model=SaidaHealth)
def health(request: Request):
    """Retorna se a API está no ar e se o modelo está carregado na memória"""
    modelo_carregado = request.app.state.modelo is not None
    return SaidaHealth(status='OK', modelo_carregado=modelo_carregado)


@router.post('/predict', response_model=SaidaPredicao)
def predict(entrada: EntradaPredicao, request: Request):
    """Classifica o dígito da imagem enviada"""
    modelo = request.app.state.modelo  # já está na memória
    if modelo is None:
        raise HTTPException(status_code=500, detail='Modelo não está carregado no servidor.')

    try:
        digito, confianca = prever_digito(modelo, entrada.pixels)
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))  # dado válido no formato, mas sem sentido
    except Exception:
        raise HTTPException(status_code=500, detail='Erro interno ao executar a predição.')

    return SaidaPredicao(digito=digito, confianca=confianca)
