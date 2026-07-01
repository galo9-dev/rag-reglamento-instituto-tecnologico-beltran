# src/retriever.py
import re
import unicodedata
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from src.config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_SCORE
)

# Normaliza variantes de género comunes para mejorar el matching semántico.
# Agregar acá más términos a medida que se detecten en historial_preguntas.json
REEMPLAZOS_GENERO = {
    r"\bvestida\b": "vestido",
    r"\balumna\b": "alumno",
    r"\balumnas\b": "alumnos",
    r"\bestudianta\b": "estudiante",
    r"\binscripta\b": "inscripto",
    r"\bsancionada\b": "sancionado",
    r"\baprobada\b": "aprobado",
    r"\bdesaprobada\b": "desaprobado",
    r"\bregularizada\b": "regularizado",
}

def normalizar_genero(texto: str) -> str:
    texto_normalizado = texto
    for patron, reemplazo in REEMPLAZOS_GENERO.items():
        texto_normalizado = re.sub(patron, reemplazo, texto_normalizado, flags=re.IGNORECASE)
    return texto_normalizado

def quitar_acentos(texto: str) -> str:
    # Descompone letras con tilde (á -> a + ´) y descarta los acentos,
    # sin tocar la ñ (la separamos antes para no romperla)
    texto = texto.replace("ñ", "§").replace("Ñ", "&")
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto.replace("§", "ñ").replace("&", "Ñ")

def limpiar_puntuacion(texto: str) -> str:
    # Sacamos signos de pregunta/exclamación (¿ ? ¡ !) y espacios duplicados,
    # que a veces afectan el embedding de preguntas muy cortas
    texto = re.sub(r"[¿?¡!]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def normalizar_para_busqueda(texto: str) -> str:
    texto = limpiar_puntuacion(texto)
    texto = quitar_acentos(texto)
    texto = normalizar_genero(texto)
    return texto

def buscar_chunks(pregunta: str, indice, chunks: list, modelo, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    pregunta_normalizada = normalizar_para_busqueda(pregunta)

    vector = modelo.encode([pregunta_normalizada])
    faiss.normalize_L2(vector)

    scores, indices = indice.search(np.array(vector, dtype=np.float32), top_k)

    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score > MIN_SIMILARITY_SCORE:
            chunk = chunks[idx].copy()
            chunk["score"] = float(score)
            resultados.append(chunk)

    return resultados