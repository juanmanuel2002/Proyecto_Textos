import os
import fitz  # PyMuPDF
import nltk
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import re

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Configuración
carpeta = "./Archivos"
modelo_bert = "distilbert-base-uncased"
stop_words = set(stopwords.words('english'))

# Inicializar modelo BERT
tokenizer = BertTokenizer.from_pretrained(modelo_bert)
model = BertForSequenceClassification.from_pretrained(modelo_bert, num_labels=2)  # 0 = no negativo, 1 = negativo

# Cargar textos
def extraer_texto_pdf(path):
    texto = ""
    doc = fitz.open(path)
    for pagina in doc:
        texto += pagina.get_text()
    return texto

# Limpieza básica
def limpiar_texto(texto):
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'\[[^\]]*\]', '', texto)  # eliminar referencias tipo [1]
    return texto.strip()

# Extraer frases sobre consecuencias
def detectar_consecuencias(texto):
    frases = sent_tokenize(texto)
    claves = ["as a result", "therefore", "consequently", "leads to", "results in", "due to", "causes"]
    consecuencias = [f for f in frases if any(k in f.lower() for k in claves)]
    return consecuencias

# Analizar sentimiento con BERT
def analizar_sentimiento(frases):
    negativos = 0
    total = 0
    for frase in frases:
        inputs = tokenizer(frase, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        negativo = probs[0][1].item()
        if negativo > 0.5:
            negativos += 1
        total += 1
    if total == 0:
        return 0
    return negativos / total  # proporción de frases negativas

# Clasificador Bayesiano (solo como comparación)
def clasificador_bayes(frases, etiquetas):
    vectorizer = CountVectorizer(binary=True, stop_words="english")
    X = vectorizer.fit_transform(frases)
    clf = BernoulliNB()
    clf.fit(X, etiquetas)
    return clf, vectorizer

# Procesamiento completo
resultados = []
for archivo in tqdm(os.listdir(carpeta)):
    if archivo.endswith(".pdf"):
        ruta = os.path.join(carpeta, archivo)
        texto = limpiar_texto(extraer_texto_pdf(ruta))
        consecuencias = detectar_consecuencias(texto)
        probabilidad_negativa = analizar_sentimiento(consecuencias)
        resultados.append({
            "archivo": archivo,
            "frases_consecuencia": len(consecuencias),
            "probabilidad_melanoma_malo": round(probabilidad_negativa * 100, 2),
            "conclusion": "Probablemente perjudicial" if probabilidad_negativa > 0.5 else "No concluyente o benigno"
        })

# Guardar resultados
df = pd.DataFrame(resultados)
df.to_csv("resultado_melanomas.csv", index=False)
df.to_csv("resultado_melanomas.txt", sep="\t", index=False)

# Mostrar resumen
print(df.to_string(index=False))
