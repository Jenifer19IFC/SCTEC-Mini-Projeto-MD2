import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # garante `import predictor` mesmo fora de `streamlit run`

from predictor import NOMES_EXIBICAO, carregar_modelos, prever_digito

st.set_page_config(page_title="Classificador de Dígitos MNIST", page_icon="🔢")
st.title("Classificador de Dígitos Manuscritos")
st.caption("Envie fotos ou desenhos de dígitos (0-9) para classificar com KNN, Random Forest e MLP.")


@st.cache_resource
def _carregar_modelos_cache():
    return carregar_modelos()


modelos, faltando = _carregar_modelos_cache()

if faltando:
    # Se algum .pkl falta em app/models/ é exibida mensagem
    st.error(
        "Modelo(s) não encontrado(s) em `app/models/`: "
        f"{', '.join(NOMES_EXIBICAO[nome] for nome in faltando)}. "
        "Rode o notebook `mnist.ipynb` primeiro para treinar e salvar os modelos."
    )
    st.stop()

arquivos = st.file_uploader("Imagens de dígitos", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True,)

for arquivo in arquivos or []:
    st.divider()
    st.subheader(arquivo.name)

    dados_imagem = arquivo.getvalue()
    resultado = prever_digito(dados_imagem, modelos)

    # Orientação detectada automaticamente, mostrada de forma discreta
    orientacao_texto = "traço escuro em fundo claro" if resultado['traco_escuro_usado'] else "traço claro em fundo escuro"
    aviso_ambiguo = " (detecção ambígua — testadas as duas orientações)" if resultado['ambiguo'] else ""
    st.caption(f"Orientação detectada: {orientacao_texto}{aviso_ambiguo}")

    # Ajuste manual, opcional: só refaz a previsão se alterado aqui
    if st.checkbox("Corrigir orientação manualmente", key=f"corrigir_{arquivo.file_id}"):
        opcao = st.radio(
            "Traço:",
            ["Escuro em fundo claro", "Claro em fundo escuro"],
            index=0 if resultado['traco_escuro_usado'] else 1,
            key=f"radio_{arquivo.file_id}",
            horizontal=True,
        )
        resultado = prever_digito(dados_imagem, modelos, forcar_traco_escuro=(opcao == "Escuro em fundo claro"))

    col_original, col_processada = st.columns(2)
    with col_original:
        st.image(dados_imagem, caption="Imagem original", width=200)
    with col_processada:
        st.image(resultado['imagem_28x28'], caption="Pré-processada (28x28)", width=200)

    colunas_modelos = st.columns(3)
    for coluna, nome in zip(colunas_modelos, ('knn', 'random_forest', 'mlp')):
        previsao = resultado['previsoes'][nome]
        with coluna:
            st.metric(NOMES_EXIBICAO[nome], previsao['rotulo'], f"{previsao['confianca']:.1%} confiança")

    # Resultado: unânime, maioria 2/3, ou sem maioria
    votacao = resultado['votacao']
    if votacao['unanime']:
        st.success(f"Resultado final (unânime): **{votacao['vencedor']}**")
    elif votacao['divergente']:
        st.warning("Os 3 modelos divergem entre si — sem maioria clara.")
    else:
        st.info(f"Resultado final (maioria 2/3): **{votacao['vencedor']}**")
