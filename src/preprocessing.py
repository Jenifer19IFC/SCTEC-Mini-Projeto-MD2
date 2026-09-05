from sklearn.model_selection import train_test_split


def split_treino_teste(X, y, test_size=0.2, random_state=42):
    """Divide X e y em treino (80%) e teste (20%) de forma estratificada por classe
    OBS: stratify=y garante que a proporção de classes seja mantida em ambos os conjuntos
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def normalizar_pixels(*conjuntos):
    """Redimensiona os valores de pixel de [0, 255] para a escala [0.0, 1.0]"""
    return tuple(X / 255.0 for X in conjuntos)
