import streamlit as st
import faiss
import numpy as np
import pickle
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Reglamento ISFT 197",
    page_icon="📋",
    layout="centered"
)

@st.cache_resource
def cargar_sistema():
    from sentence_transformers import SentenceTransformer
    indice = faiss.read_index("data/indice.faiss")
    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    return indice, chunks, modelo

def buscar_chunks(pregunta, indice, chunks, modelo, top_k=3):
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
                "content": "Sos un asistente del reglamento académico del ISFT 197. Respondé solo basándote en el contexto proporcionado. Si la respuesta no está en el contexto, decilo claramente. Respondé en español, de forma clara y amigable para estudiantes."
            },
            {
                "role": "user",
                "content": f"Contexto del reglamento:\n{contexto}\n\nPregunta: {pregunta}"
            }
        ]
    )
    return respuesta.choices[0].message.content

st.title("📋 Consultor del Reglamento")
st.caption("ISFT N° 197 — Hacé tu pregunta sobre el reglamento académico")

with st.spinner("Cargando sistema..."):
    indice, chunks, modelo = cargar_sistema()

if "historial" not in st.session_state:
    st.session_state.historial = []

for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.write(mensaje["contenido"])

pregunta = st.chat_input("¿Qué querés saber del reglamento?")

if pregunta:
    with st.chat_message("user"):
        st.write(pregunta)
    st.session_state.historial.append({"rol": "user", "contenido": pregunta})

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el reglamento..."):
            chunks_relevantes = buscar_chunks(pregunta, indice, chunks, modelo)
            respuesta = responder(pregunta, chunks_relevantes)
        st.write(respuesta)
    st.session_state.historial.append({"rol": "assistant", "contenido": respuesta})