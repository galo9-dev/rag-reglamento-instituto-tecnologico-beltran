#Instituto Tecnologico Beltran
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

def cargar_indice():
    indice = faiss.read_index("data/indice.faiss")
    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return indice, chunks

def buscar_chunks_relevantes(pregunta, indice, chunks, modelo, top_k=3):
    vector_pregunta = modelo.encode([pregunta])
    distancias, indices = indice.search(np.array(vector_pregunta), top_k)
    return [chunks[i] for i in indices[0]]

def responder(pregunta, chunks_relevantes):
    cliente = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    contexto = "\n\n".join(chunks_relevantes)
    
    respuesta = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Sos un asistente del reglamento académico. Respondé solo basándote en el contexto proporcionado. Si la respuesta no está en el contexto, decilo claramente. Respondé en español."
            },
            {
                "role": "user",
                "content": f"Contexto del reglamento:\n{contexto}\n\nPregunta: {pregunta}"
            }
        ]
    )
    
    return respuesta.choices[0].message.content

if __name__ == "__main__":
    print("Cargando sistema...")
    indice, chunks = cargar_indice()
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    
    pregunta = "¿Qué pasa si un alumno desaprueba una materia?"
    print(f"\nPregunta: {pregunta}")
    
    chunks_relevantes = buscar_chunks_relevantes(pregunta, indice, chunks, modelo)
    respuesta = responder(pregunta, chunks_relevantes)
    
    print(f"\nRespuesta: {respuesta}")