import os
from PyPDF2 import PdfReader
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from googletrans import Translator
from deep_translator import GoogleTranslator

# Descargar las stop words de nltk (solo la primera vez)
nltk.download('stopwords')

# Obtener las stop words en inglés
stop_words = set(stopwords.words('english'))

palabras_personalizadas = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "10", "a", "b", "c", "d",
                            "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", 
                            "s", "t", "u", "v", "w", "x", "y", "z","et", "etc", "al", "al.", "et.",
                            "al", "et", "al", "al.", "etc.", "etc", "i.e", "e.g", "e.g.", "i.e.", 
                            "i.e", "e.g", "nm","0O","o0","doi", "lm","e0", "e1", "e2", "e3", "e4",
                            "e5", "e6", "e7", "e8", "e9", "et al", "et al.", "et. al", "et. al."}
stop_words.update(palabras_personalizadas)

# Lista de stop words en inglés (respaldo)
'''
stop_words = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "can't", "cannot",
    "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll",
    "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "a", "b", "c", "d",
    "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z","et", 
    "etc", "al", "al.", "et.", "al", "et", "al", "al.", "etc.", "etc", "i.e", "e.g", "e.g.", "i.e.", "i.e", "e.g", "nm"
}

'''


def leer_pdfs(carpeta):
    textos = {}
    for archivo in os.listdir(carpeta):
        if archivo.endswith('.pdf'):
            ruta = os.path.join(carpeta, archivo)
            try:
                lector = PdfReader(ruta)
                texto = ""
                for pagina in lector.pages:
                    texto += pagina.extract_text()
                textos[archivo] = texto
            except Exception as e:
                print(f"Error al leer {archivo}: {e}")
    return textos

def palabras_repetidas(texto, top_n=10):
    # Eliminar caracteres especiales y convertir a minúsculas
    palabras = re.findall(r'\b\w+\b', texto.lower())
    # Filtrar palabras irrelevantes
    palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
    contador = Counter(palabras_filtradas)
    return contador.most_common(top_n)

def extraer_causas_consecuencias(texto):
    # Palabras clave para identificar causas y consecuencias
    causas_claves = ["cause", "causes", "due to", "because", "leads to", "results in"]
    consecuencias_claves = ["consequence", "consequences", "result", "effect", "impact"]

    # Buscar frases relacionadas con causas
    causas = []
    for clave in causas_claves:
        patrones = re.findall(rf".*?{clave}.*?[.]", texto, re.IGNORECASE)
        causas.extend(patrones)

    # Buscar frases relacionadas con consecuencias
    consecuencias = []
    for clave in consecuencias_claves:
        patrones = re.findall(rf".*?{clave}.*?[.]", texto, re.IGNORECASE)
        consecuencias.extend(patrones)

    return causas, consecuencias

# Ruta de la carpeta donde están los archivos PDF
carpeta_pdfs = "Archivos/"
textos_pdfs = leer_pdfs(carpeta_pdfs)

# Analizar palabras más repetidas y extraer causas/consecuencias
for nombre, texto in textos_pdfs.items():
    print(f"Palabras más repetidas en {nombre}:")
    palabras_comunes = palabras_repetidas(texto)
    for palabra, frecuencia in palabras_comunes:
        print(f"{palabra}: {frecuencia}")
    print("\n")

    # Extraer causas y consecuencias
    causas, consecuencias = extraer_causas_consecuencias(texto)
    print(f"Causas encontradas en {nombre}:")
    for causa in causas:
        print(f"- {causa}")
    print(f"\nConsecuencias encontradas en {nombre}:")
    for consecuencia in consecuencias:
        print(f"- {consecuencia}")
    print("\n")


def mostrar_tabla_comparacion(comparacion_palabras):
    """
    Muestra una tabla comparando las palabras más repetidas entre los PDFs.
    :param comparacion_palabras: Diccionario con el nombre del archivo como clave y las palabras más repetidas como valor.
    """
    # Crear un DataFrame con las palabras y frecuencias
    palabras = set()
    for palabras_frecuentes in comparacion_palabras.values():
        palabras.update([palabra for palabra, _ in palabras_frecuentes])
    
    palabras = list(palabras)
    data = {archivo: {palabra: dict(comparacion_palabras[archivo]).get(palabra, 0) for palabra in palabras} for archivo in comparacion_palabras}
    df = pd.DataFrame(data).fillna(0).astype(int)
    
    # Mostrar la tabla
    print("\nTabla de comparación de palabras más repetidas:")
    print(df)

def exportar_tabla_comparacion_a_csv(comparacion_palabras, archivo_salida="comparacion_palabras.csv"):
    """
    Exporta la tabla de comparación de palabras más repetidas a un archivo CSV.
    :param comparacion_palabras: Diccionario con el nombre del archivo como clave y las palabras más repetidas como valor.
    :param archivo_salida: Nombre del archivo CSV de salida.
    """
    # Crear un DataFrame con las palabras y frecuencias
    palabras = set()
    for palabras_frecuentes in comparacion_palabras.values():
        palabras.update([palabra for palabra, _ in palabras_frecuentes])
    
    palabras = list(palabras)
    data = {archivo: {palabra: dict(comparacion_palabras[archivo]).get(palabra, 0) for palabra in palabras} for archivo in comparacion_palabras}
    df = pd.DataFrame(data).fillna(0).astype(int)
    
    # Exportar a CSV
    df.to_csv(archivo_salida, index=True)
    print(f"Tabla de comparación exportada a {archivo_salida}")

def graficar_heatmap_comparacion(comparacion_palabras):
    """
    Genera un heatmap comparando las palabras más repetidas entre los PDFs.
    :param comparacion_palabras: Diccionario con el nombre del archivo como clave y las palabras más repetidas como valor.
    """
    # Crear un DataFrame con las palabras y frecuencias
    palabras = set()
    for palabras_frecuentes in comparacion_palabras.values():
        palabras.update([palabra for palabra, _ in palabras_frecuentes])
    
    palabras = list(palabras)
    data = {archivo: {palabra: dict(comparacion_palabras[archivo]).get(palabra, 0) for palabra in palabras} for archivo in comparacion_palabras}
    df = pd.DataFrame(data).fillna(0).astype(int)
    
    # Crear el heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=False, fmt="d", cmap="YlGnBu", cbar=True)
    plt.title("Comparación de palabras más repetidas entre PDFs")
    plt.ylabel("Palabras")
    plt.xlabel("Archivos")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def comparar_causas(causas_por_archivo):
    """
    Compara las causas obtenidas de diferentes archivos y encuentra similitudes.
    :param causas_por_archivo: Diccionario con el nombre del archivo como clave y una lista de causas como valor.
    """
    # Convertir las listas de causas en conjuntos para encontrar intersecciones
    causas_comunes = set.intersection(*[set(causas) for causas in causas_por_archivo.values() if causas])
    
    print("\nCausas comunes entre los archivos:")
    if causas_comunes:
        for causa in causas_comunes:
            print(f"- {causa}")
    else:
        print("No se encontraron causas comunes.")
    
    # Mostrar todas las causas por archivo
    print("\nCausas por archivo:")
    for archivo, causas in causas_por_archivo.items():
        print(f"{archivo}:")
        for causa in causas:
            print(f"- {causa}")


def traducir_conclusiones(conclusiones, idioma_destino="es"):
    """
    Traduce las conclusiones a un idioma específico.
    :param conclusiones: Lista de conclusiones en inglés.
    :param idioma_destino: Código del idioma al que se traducirán las conclusiones (por defecto, español: 'es').
    :return: Lista de conclusiones traducidas.
    """
    traductor = Translator()
    conclusiones_traducidas = []
    for conclusion in conclusiones:
        try:
            traduccion = traductor.translate(conclusion, dest=idioma_destino)
            conclusiones_traducidas.append(traduccion.text)
        except Exception as e:
            print(f"Error al traducir la conclusión: {conclusion}. Error: {e}")
            conclusiones_traducidas.append(conclusion)  # Dejar la conclusión en inglés si falla la traducción
    return conclusiones_traducidas


def generar_conclusiones(causas_por_archivo):
    """
    Genera conclusiones basadas en las causas obtenidas de cada archivo.
    :param causas_por_archivo: Diccionario con el nombre del archivo como clave y una lista de causas como valor.
    :return: Diccionario con las conclusiones por archivo.
    """
    conclusiones = {}
    for archivo, causas in causas_por_archivo.items():
        if causas:
            conclusion = f"The main causes identified in {archivo} are: " + "; ".join(causas)
        else:
            conclusion = f"No significant causes were identified in {archivo}."
        conclusiones[archivo] = conclusion
    return conclusiones


# Comparar palabras más repetidas entre los PDFs
comparacion_palabras = {}
for nombre, texto in textos_pdfs.items():
    palabras_comunes = palabras_repetidas(texto)
    comparacion_palabras[nombre] = palabras_comunes

# Extraer causas de cada archivo
causas_por_archivo = {}
for nombre, texto in textos_pdfs.items():
    causas, _ = extraer_causas_consecuencias(texto)
    causas_por_archivo[nombre] = causas

# Comparar las causas entre los archivos
#comparar_causas(causas_por_archivo)

# Generar conclusiones basadas en las causas
conclusiones_por_archivo = generar_conclusiones(causas_por_archivo)

# Traducir las conclusiones al español
conclusiones_traducidas = traducir_conclusiones(list(conclusiones_por_archivo.values()))
print("\nConclusiones traducidas al español:")
for archivo, conclusion in zip(conclusiones_por_archivo.keys(), conclusiones_traducidas):
    print(f"{archivo}: {conclusion}")

# Mostrar la tabla
#mostrar_tabla_comparacion(comparacion_palabras)

# Generar el heatmap
#graficar_heatmap_comparacion(comparacion_palabras)

# Exportar la tabla a un archivo CSV
#exportar_tabla_comparacion_a_csv(comparacion_palabras)