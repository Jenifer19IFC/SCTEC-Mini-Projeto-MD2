import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def get_knn(n_neighbors=5, weights='distance'):
    """Cria um classificador KNN (K-Nearest Neighbors)

    Hiperparâmetros:
    - n_neighbors: quantidade de vizinhos mais próximos consultados para
      classificar uma imagem;
    - weights: forma de ponderar o voto dos vizinhos. 'uniform' dá peso
      igual a todos; 'distance' dá mais peso aos vizinhos mais próximos.
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)


def buscar_melhor_knn(X_train, y_train, param_grid=None, cv=3, n_jobs=-1):
    """Executa GridSearchCV para encontrar a melhor combinação de
    n_neighbors e weights, usando validação cruzada
    (Retorna o GridSearchCV já ajustado)
    """
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 4, 5, 6, 7, 9, 11],
            'weights': ['uniform', 'distance'],
        }

    grid_knn = GridSearchCV(get_knn(), param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    grid_knn.fit(X_train, y_train)
    return grid_knn


def get_random_forest(n_estimators=200, max_depth=20, random_state=42):
    """Cria um classificador Random Forest

    Hiperparâmetros:
    - n_estimators: número de árvores de decisão no ensemble;
    - max_depth: profundidade máxima de cada árvore.
    """
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)


def buscar_melhor_random_forest(X_train, y_train, param_grid=None, cv=3, n_jobs=-1):
    """Executa GridSearchCV para encontrar a melhor combinação de
    n_estimators e max_depth, usando validação cruzada
    (Retorna o GridSearchCV já ajustado)
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
        }

    grid_rf = GridSearchCV(get_random_forest(), param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    grid_rf.fit(X_train, y_train)
    return grid_rf


def get_mlp(hidden_layer_sizes=(128, 64), learning_rate_init=0.001, random_state=42, max_iter=200):
    """Cria um classificador MLP (Perceptron Multicamadas)

    Hiperparâmetros:
    - hidden_layer_sizes: quantidade de neurônios em cada camada oculta;
    - learning_rate_init: taxa de aprendizado inicial do otimizador
      baseado em gradiente.


    max_iter = limite máximo de tempo/iterações que dou para a rede encontrar uma solução
    """
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        random_state=random_state,
        max_iter=max_iter,
    )


def buscar_melhor_mlp(X_train, y_train, param_grid=None, cv=3, n_jobs=1):
    """Executa GridSearchCV para encontrar a melhor combinação de
    hidden_layer_sizes e learning_rate_init, usando validação cruzada
    (Retorna o GridSearchCV já ajustado)
    """
    if param_grid is None:
        param_grid = {
            'hidden_layer_sizes': [(64,), (128, 64), (128, 64, 32)],
            'learning_rate_init': [0.001, 0.0005, 0.0001],
        }

    grid_mlp = GridSearchCV(get_mlp(), param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        grid_mlp.fit(X_train, y_train)

    return grid_mlp
