import os
import fitz  # PyMuPDF
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
from transformers import AutoTokenizer

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# 1. Leer PDFs
def extract_text_from_pdfs(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(folder_path, filename)) as doc:
                text = " ".join([page.get_text() for page in doc])
                texts[filename] = text
    return texts

# 2. Limpiar y normalizar
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,]", "", text)
    text = text.lower()
    return text

# 3. Extraer frases relevantes
consequence_keywords = [
    "as a result", "this leads to", "consequently", "therefore", "thus", "resulting in",
    "which causes", "leading to", "may cause", "associated with", "linked to", "due to"
]

def extract_consequence_sentences(text):
    sentences = re.split(r"[.?!]", text)
    relevant = [s.strip() for s in sentences if any(kw in s for kw in consequence_keywords)]
    return relevant

# 4. Análisis de implicación con BERT
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Inicializar tokenizer y pipeline una sola vez
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)

def analyze_sentiment(sentences):
    results = []
    for s in sentences:
        # Tokenizar para obtener la longitud real en tokens
        encoded = tokenizer.encode(s, truncation=True, max_length=512, return_tensors="pt")
        if encoded.shape[1] > 5:  # mínimo 5 tokens
            try:
                analysis = classifier(s)[0]
                if analysis["label"] == "NEGATIVE" and analysis["score"] > 0.75:
                    results.append((s, analysis["score"]))
            except Exception as e:
                print(f"Error al analizar la frase:\n{s}\n{e}")
    return results

# 5. Probabilidad bayesiana estimada
def estimate_probability(neg_count, total_count, alpha=1.0):
    return (neg_count + alpha) / (total_count + 2 * alpha)

# 6. Procesamiento completo
def analyze_documents(folder_path):
    pdf_texts = extract_text_from_pdfs(folder_path)
    summary = []

    for filename, raw_text in pdf_texts.items():
        cleaned = clean_text(raw_text)
        consequence_sentences = extract_consequence_sentences(cleaned)
        negative_sentences = analyze_sentiment(consequence_sentences)

        total = len(consequence_sentences)
        negs = len(negative_sentences)
        probability = estimate_probability(negs, total)

        summary.append({
            "File": filename,
            "Total Consequence Sentences": total,
            "Negative Sentences": negs,
            "Probability Melanoma is Harmful": round(probability, 2),
            "Negative Examples": [s[0] for s in negative_sentences]
        })

    return summary

# 7. Guardar resultados
def export_results(results, out_txt="resultadoNegativo.txt", out_csv="resultadoNegativo.csv"):
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        for row in results:
            f.write(f"File: {row['File']}\n")
            f.write(f"Probability Melanoma is Harmful: {row['Probability Melanoma is Harmful']}\n")
            f.write("Negative Examples:\n")
            for example in row["Negative Examples"]:
                f.write(f"  - {example.strip()}\n")
            f.write("\n" + "-" * 40 + "\n\n")

# 🔁 Ejecutar todo
if __name__ == "__main__":
    folder = "./Archivos"
    results = analyze_documents(folder)
    export_results(results)
    print("Análisis completado y exportado.")
