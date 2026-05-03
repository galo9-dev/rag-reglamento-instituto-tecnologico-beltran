# src/indexer.py
import fitz
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from src.config import (
    EMBEDDING_MODEL, FAISS_INDEX_PATH, 
    CHUNKS_PATH, PDF_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP
)

# ── PROCESAMIENTO DEL PDF ─────────────────────────────────────────

def cargar_pdf(ruta=PDF_PATH) -> list[dict]:
    doc = fitz.open(ruta)
    paginas = []
    for num_pagina, pagina in enumerate(doc, start=1):
        texto = pagina.get_text()
        if texto.strip():
            paginas.append({"texto": texto, "pagina": num_pagina})
    return paginas

def hacer_chunks(paginas: list[dict]) -> list[dict]:
    chunks = []
    chunk_id = 0

    for pagina_data in paginas:
        texto = pagina_data["texto"]
        num_pagina = pagina_data["pagina"]

        parrafos = [p.strip() for p in texto.split('\n\n') if p.strip()]

        buffer = ""
        for parrafo in parrafos:
            if len(buffer) + len(parrafo) > CHUNK_SIZE and buffer:
                chunks.append({
                    "texto": buffer.strip(),
                    "pagina": num_pagina,
                    "chunk_id": chunk_id,
                    "preview": buffer.strip()[:80] + "..."
                })
                chunk_id += 1
                buffer = buffer[-CHUNK_OVERLAP:] + "\n" + parrafo
            else:
                buffer += "\n" + parrafo

        if buffer.strip():
            chunks.append({
                "texto": buffer.strip(),
                "pagina": num_pagina,
                "chunk_id": chunk_id,
                "preview": buffer.strip()[:80] + "..."
            })
            chunk_id += 1

    return chunks

# ── GENERACIÓN DEL ÍNDICE ─────────────────────────────────────────

def construir_indice(chunks: list[dict], modelo: SentenceTransformer) -> faiss.Index:
    textos = [c["texto"] for c in chunks]
    vectores = modelo.encode(textos, show_progress_bar=True)
    faiss.normalize_L2(vectores)

    indice = faiss.IndexFlatIP(vectores.shape[1])
    indice.add(vectores.astype(np.float32))
    return indice

def guardar(indice: faiss.Index, chunks: list[dict]):
    faiss.write_index(indice, str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

# ── ENTRY POINT ───────────────────────────────────────────────────

def indexar():
    print("📄 Leyendo PDF...")
    paginas = cargar_pdf()
    
    print("✂️  Generando chunks...")
    chunks = hacer_chunks(paginas)
    print(f"   → {len(chunks)} chunks generados")

    print("🔢 Calculando embeddings...")
    modelo = SentenceTransformer(EMBEDDING_MODEL)
    indice = construir_indice(chunks, modelo)

    print("💾 Guardando índice...")
    guardar(indice, chunks)
    print(f"✅ Listo. {indice.ntotal} vectores en el índice.")

if __name__ == "__main__":
    indexar()