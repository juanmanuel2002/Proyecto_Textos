# Proyecto_Analiss_y_Procesamiento_de_Textos_Inteligente

## Análisis de Sentimiento en Documentos de Melanoma

Este proyecto analiza documentos PDF relacionados con melanoma para identificar frases de consecuencia y estimar la probabilidad de que el contenido sea perjudicial o no, utilizando modelos de lenguaje BERT y análisis de sentimiento.

## Descripción

El script [`positivosNegativos.py`](positivosNegativos.py) realiza las siguientes tareas principales:

1. **Lectura de PDFs:** Extrae el texto de todos los archivos PDF en la carpeta `./Archivos`.
2. **Limpieza y normalización:** El texto se limpia y normaliza para facilitar el análisis.
3. **Extracción de frases relevantes:** Se identifican frases que contienen palabras clave asociadas a consecuencias, causas o asociaciones.
4. **Análisis de sentimiento:** Utiliza un modelo BERT (`distilbert-base-uncased-finetuned-sst-2-english`) para clasificar las frases relevantes como positivas o negativas.
5. **Cálculo de probabilidades:** Estima la probabilidad de que el contenido sea perjudicial (negativo) o no perjudicial (positivo) usando un enfoque bayesiano.
6. **Exportación de resultados:** Los resultados se guardan en archivos `resultadoPositivoNegativo.txt` y `resultadoPositivoNegativo.csv`, incluyendo ejemplos de frases negativas y positivas detectadas.

El script [`resultadosNegativos.py`](resultadosNegativos.py) realiza las mismas tareas, solo que se centra solo en causas negativas excluxivamente.

Los resultados se guardan en archivos `resultadoNegativo.txt` y `resultadoNegativo.csv`, incluyendo ejemplos de frases negativas y positivas detectadas.

## Uso

1. Coloca tus archivos PDF en la carpeta `./Archivos`.
2. Ejecuta el script:

   ```bash
   python positivosNegativos.py
3. Revisa los resultados en los archivos generados:
resultadoPositivoNegativo.txt\
resultadoPositivoNegativo.csv

## Requisitos
* Python 3.7+
* PyMuPDF (fitz)
* pandas
* nltk
* transformers
* torch

Instala las dependencias con:
```bash
pip install pymupdf pandas nltk transformers torch
```

## Notas
El script descarga automáticamente las stopwords de NLTK la primera vez que se ejecuta.
El análisis de sentimiento se basa en frases que contienen palabras clave como "therefore", "due to", "associated with", entre otras.
El modelo BERT utilizado es distilbert-base-uncased-finetuned-sst-2-english.
Estructura de resultados
Cada archivo PDF procesado genera un resumen con:

Número total de frases de consecuencia.
Número de frases negativas y positivas.
Probabilidad estimada de que el contenido sea perjudicial o no.
Ejemplos de frases negativas y positivas encontradas.\


\
Para más detalles, revisa el código fuente en
[`positivosNegativos.py`](positivosNegativos.py)


