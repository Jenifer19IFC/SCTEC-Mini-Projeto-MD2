import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers import router
from api.services import carregar_modelo

logger = logging.getLogger('uvicorn.error')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida da API: o modelo é lido do disco uma única vez, na inicialização,
    e fica na memória RAM (app.state) para ser reaproveitado por todas as requisições
    """
    try:
        app.state.modelo = carregar_modelo()
        logger.info('Modelo MLP carregado na memória')
    except (FileNotFoundError, OSError) as erro:
        # API sobe mesmo sem modelo e o health informa que ele não está disponível
        app.state.modelo = None
        logger.error(f'Falha ao carregar o modelo: {erro}')

    yield  # atende requisições enquanto está aqui

    # Ao final, libera o modelo da memória
    app.state.modelo = None


app = FastAPI(
    title='API de Classificação de Dígitos (MNIST)',
    description='Classifica dígitos manuscritos (0-9) com o MLP (Perceptron Multicamadas).',
    version='1.0.0',
    lifespan=lifespan,
    openapi_tags=[
        {'name': 'Monitoramento', 'description': 'Saúde da API e do modelo.'},
        {'name': 'Inferência', 'description': 'Predição do dígito a partir dos pixels.'},
    ],
)
app.include_router(router)
