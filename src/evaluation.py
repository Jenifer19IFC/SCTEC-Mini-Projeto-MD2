import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


def plot_matriz_confusao(y_true, y_pred, nome_modelo):
    """Plota a matriz de confusão (10x10) de um modelo como mapa de calor"""
    matriz = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    plt.imshow(matriz, cmap='Blues')
    plt.colorbar()
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.xlabel("Rótulo Previsto")
    plt.ylabel("Rótulo Real")
    plt.xticks(range(10))
    plt.yticks(range(10))

    limite = matriz.max() / 2
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            cor_texto = 'white' if matriz[i, j] > limite else 'black'
            plt.text(j, i, matriz[i, j], ha='center', va='center', color=cor_texto)

    plt.tight_layout()
    plt.show()

    return matriz


def calcular_metricas(y_true, y_pred, nome_modelo, tempo_treino=None):
    """Calcula métricas:
        - Acurácia Global (Accuracy)
        - Precisão Média Ponderada (Precision)
        - Revocação/Sensibilidade Média Ponderada (Recall)
        - F1-Score Ponderado (F1-Score)
    """
    precisao, revocacao, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'Modelo': nome_modelo,
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão (ponderada)': precisao,
        'Revocação (ponderada)': revocacao,
        'F1-Score (ponderado)': f1,
        'Tempo de Treino (s)': tempo_treino,
    }


def montar_tabela_comparativa(lista_metricas):
    """Monta a tabela comparativa de métricas de todos os modelos avaliados"""
    return pd.DataFrame(lista_metricas).set_index('Modelo').round(4)


def par_mais_confundido(y_true, y_pred):
    """Identifica o par (rótulo real, rótulo previsto) com a maior contagem de confusão entre dígitos diferentes"""
    matriz = confusion_matrix(y_true, y_pred).astype(float)
    np.fill_diagonal(matriz, 0)

    real, previsto = np.unravel_index(np.argmax(matriz), matriz.shape)
    return int(real), int(previsto), int(matriz[real, previsto])


def gerar_conclusao_tecnica(tabela_comparativa, y_test, previsoes):
    """Monta o texto de conclusão técnica:
        - para cada modelo, aponta o par de dígitos mais confundido entre si;
        - aponta o modelo mais rápido de treinar.
    """
    linhas = ["Confusão mais frequente por modelo:"]
    for nome_modelo, y_pred in previsoes.items():
        real, previsto, qtd = par_mais_confundido(y_test, y_pred)
        linhas.append(f"- {nome_modelo}: dígito real {real} classificado como {previsto} " f"em {qtd} casos.")

    mais_rapido       = tabela_comparativa['Tempo de Treino (s)'].idxmin()
    tempo_mais_rapido = tabela_comparativa.loc[mais_rapido, 'Tempo de Treino (s)']

    linhas.append("")
    linhas.append(f"Menor custo computacional de treino: {mais_rapido}, treinado em "f"{tempo_mais_rapido:.2f}s.")

    return "\n".join(linhas)
