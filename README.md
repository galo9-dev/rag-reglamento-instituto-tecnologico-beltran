# Consultor del Reglamento — ISFT Instituto Tecnológico Beltrán

Proyecto personal para facilitar el acceso al reglamento académico del instituto. 
En vez de leer el PDF completo, los alumnos pueden hacer preguntas en lenguaje natural y obtener respuestas basadas en el documento oficial.

## Cómo funciona

El sistema usa RAG (Retrieval-Augmented Generation): divide el reglamento en fragmentos, 
los convierte en vectores semánticos con sentence-transformers, y cuando alguien hace una 
pregunta busca los fragmentos más relevantes con FAISS y se los pasa a LLaMA 3 (via Groq) 
para generar la respuesta.

## Tecnologías

- Python, PyMuPDF, sentence-transformers, FAISS, Groq API, Streamlit

## Instalación

1. Clonar el repo
2. Crear entorno virtual e instalar dependencias: `pip install pymupdf sentence-transformers faiss-cpu streamlit groq python-dotenv`
3. Agregar `reglamento.pdf` en la carpeta `data/`
4. Crear `.env` con tu `GROQ_API_KEY`
5. Correr `python embeddings.py` y luego `streamlit run app.py`