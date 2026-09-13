# SCTEC Mini Projeto MD2 — Classificação de Dígitos MNIST

Treinamento e avaliação de KNN, Random Forest e MLP para classificação de dígitos
manuscritos (MNIST), em `mnist.ipynb` + módulos em `src/`.

## App de classificação (Docker)

Há uma aplicação web (Streamlit) em `app/` que usa os 3 modelos já treinados para
classificar imagens de dígitos enviadas pelo usuário.

**Antes de rodar o app, é preciso rodar o notebook `mnist.ipynb` até o fim pelo
menos uma vez**, para treinar os modelos e gerar os arquivos `.pkl` em `app/models/`.

Para subir o app:

```bash
docker compose up --build
```

Depois acesse **http://localhost:8501**.

Os modelos em `app/models/` são montados como volume — ao retreinar no notebook,
o app passa a usar os modelos novos automaticamente, sem precisar reconstruir a
imagem Docker.
