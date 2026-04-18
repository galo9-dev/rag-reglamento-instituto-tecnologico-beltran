from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from procesar_pdf import cargar_pdf, hacer_chunks

def crear_indice(chunks):
    print("Cargando modelo de embeddings...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Generando embeddings...")
    vectores = modelo.encode(chunks, show_progress_bar=True)
    
    dimension = vectores.shape[1]
    indice = faiss.IndexFlatL2(dimension)
    indice.add(np.array(vectores))
    
    print(f"Índice creado con {indice.ntotal} vectores")
    return indice, vectores

def guardar_indice(indice, chunks):
    faiss.write_index(indice, "data/indice.faiss")
    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("Índice guardado en data/")

if __name__ == "__main__":
    texto = cargar_pdf("data/reglamento.pdf")
    chunks = hacer_chunks(texto)
    indice, vectores = crear_indice(chunks)
    guardar_indice(indice, chunks)