import os

DIR_CONFIG = os.path.dirname(__file__)
DIR_RAIZ = os.path.dirname(DIR_CONFIG)

CAMINHO_MELHORES_HIPERPARAMETROS = os.path.join(DIR_CONFIG, 'melhores_hiperparametros.json')

DIR_MODELOS = os.path.join(DIR_RAIZ, 'models')

DIR_OUTPUTS = os.path.join(DIR_RAIZ, 'outputs')
DIR_FIGURES = os.path.join(DIR_OUTPUTS, 'figures')

DIR_IMAGENS_PROPRIAS = os.path.join(DIR_RAIZ, 'my_images')

DIR_APP = os.path.join(DIR_RAIZ, 'app')
DIR_APP_MODELOS = os.path.join(DIR_APP, 'models')

MODELOS_APP = ('knn', 'random_forest', 'mlp')  
