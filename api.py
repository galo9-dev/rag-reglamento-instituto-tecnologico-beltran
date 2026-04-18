# Instituto Tecnologico Beltran
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando sistema...")
indice = faiss.read_index("data/indice.faiss")
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
modelo = SentenceTransformer("all-MiniLM-L6-v2")
print("Sistema listo.")

class Pregunta(BaseModel):
    texto: str

def buscar_chunks(pregunta, top_k=3):
    vector = modelo.encode([pregunta])
    distancias, indices = indice.search(np.array(vector), top_k)
    return [chunks[i] for i in indices[0]]

def responder(pregunta, chunks_relevantes):
    cliente = Groq(api_key=os.getenv("GROQ_API_KEY"))
    contexto = "\n\n".join(chunks_relevantes)
    respuesta = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Sos un asistente del reglamento académico del Instituto Tecnológico Beltrán. Respondé solo basándote en el contexto proporcionado. Si la respuesta no está en el contexto, decilo claramente. Respondé en español, de forma clara y amigable para estudiantes."
            },
            {
                "role": "user",
                "content": f"Contexto del reglamento:\n{contexto}\n\nPregunta: {pregunta}"
            }
        ]
    )
    return respuesta.choices[0].message.content

@app.get("/")
def inicio():
    return {"mensaje": "API del Reglamento Instituto Tecnológico Beltrán funcionando"}

@app.post("/preguntar")
def preguntar(pregunta: Pregunta):
    chunks_relevantes = buscar_chunks(pregunta.texto)
    respuesta = responder(pregunta.texto, chunks_relevantes)
    return {"respuesta": respuesta}