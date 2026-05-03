# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from src.retriever import buscar_chunks
from src.generator import responder
from src.config import FAISS_INDEX_PATH, CHUNKS_PATH, EMBEDDING_MODEL

load_dotenv()

app = FastAPI(
    title="RAG Reglamento ISFT 197",
    description="API para consultar el reglamento académico del ISFT N° 197",
    version="1.0.0"
)

# ── CARGA DEL SISTEMA ────────────────────────────────────────────
indice = faiss.read_index(str(FAISS_INDEX_PATH))
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)
modelo = SentenceTransformer(EMBEDDING_MODEL)

# ── MODELOS ──────────────────────────────────────────────────────
class PreguntaRequest(BaseModel):
    pregunta: str
    top_k: int = 4

class FuenteResponse(BaseModel):
    chunk_id: int
    pagina: int
    preview: str
    score: float

class RespuestaResponse(BaseModel):
    respuesta: str
    fuentes: list[FuenteResponse]

# ── ENDPOINTS ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexados": len(chunks)}

@app.post("/query", response_model=RespuestaResponse)
def query(request: PreguntaRequest):
    chunks_relevantes = buscar_chunks(
        request.pregunta, indice, chunks, modelo, top_k=request.top_k
    )

    if not chunks_relevantes:
        return RespuestaResponse(
            respuesta="No encontré fragmentos relevantes para tu pregunta.",
            fuentes=[]
        )

    respuesta = responder(request.pregunta, chunks_relevantes, [])

    fuentes = [
        FuenteResponse(
            chunk_id=c["chunk_id"],
            pagina=c["pagina"],
            preview=c["preview"],
            score=round(c["score"], 4)
        )
        for c in chunks_relevantes
    ]

    return RespuestaResponse(respuesta=respuesta, fuentes=fuentes)