import json
import urllib.request

import numpy as np
from PIL import Image

# Converte a imagem para 28x28 em escala de cinza (dígito claro em fundo escuro)
img = Image.open('./my_images/dark/6_dark.png').convert('L').resize((28, 28))
pixels = np.array(img).flatten().tolist()

req = urllib.request.Request(
    'http://127.0.0.1:8000/predict',
    data=json.dumps({'pixels': pixels}).encode(),
    headers={'Content-Type': 'application/json'},
)
print(json.load(urllib.request.urlopen(req)))  # {'digito': ..., 'confianca': ...}