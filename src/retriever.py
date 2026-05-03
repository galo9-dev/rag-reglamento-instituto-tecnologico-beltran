# src/retriever.py
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

def buscar_chunks(pregunta: str, indice, chunks: list, modelo, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    vector = modelo.encode([pregunta])
    faiss.normalize_L2(vector)

    scores, indices = indice.search(np.array(vector, dtype=np.float32), top_k)

    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score > MIN_SIMILARITY_SCORE:
            chunk = chunks[idx].copy()
            chunk["score"] = float(score)
            resultados.append(chunk)

    return resultados