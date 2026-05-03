# app.py — VERSIÓN PRODUCCIÓN COMPLETA
import streamlit as st
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Consultor Reglamento — ISFT 197",
    page_icon="📋",
    layout="wide"
)

# ── CARGA DEL SISTEMA ────────────────────────────────────────────
@st.cache_resource
def cargar_sistema():
    from sentence_transformers import SentenceTransformer
    indice = faiss.read_index("data/indice.faiss")
    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    return indice, chunks, modelo

from src.retriever import buscar_chunks
from src.generator import responder

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuración")
    
    modo_debug = st.toggle("🔍 Modo Debug", value=False, 
                           help="Muestra los fragmentos recuperados y sus scores de similitud")
    
    top_k = st.slider("Fragmentos a recuperar (top_k)", 
                      min_value=1, max_value=8, value=4,
                      help="Más fragmentos = más contexto pero respuestas más lentas")
    
    st.divider()
    st.caption("**Stack técnico:**")
    st.caption("• Embeddings: all-MiniLM-L6-v2")
    st.caption("• Vector DB: FAISS (coseno)")
    st.caption("• LLM: LLaMA 3.3 70B via Groq")
    st.caption("• UI: Streamlit")
    
    st.divider()
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.historial = []
        st.rerun()

# ── MAIN ─────────────────────────────────────────────────────────
st.title("📋 Consultor del Reglamento Académico")
st.caption("ISFT N° 197 · Respondido con fragmentos del reglamento oficial")

with st.spinner("Cargando sistema de búsqueda..."):
    indice, chunks, modelo = cargar_sistema()

if "historial" not in st.session_state:
    st.session_state.historial = []

# Renderizar historial
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.write(mensaje["contenido"])
        
        # Mostrar fuentes si el mensaje es del asistente y tiene chunks guardados
        if mensaje["rol"] == "assistant" and "chunks" in mensaje:
            with st.expander(f"📚 Ver {len(mensaje['chunks'])} fuentes consultadas"):
                for i, chunk in enumerate(mensaje["chunks"], 1):
                    score_pct = int(chunk["score"] * 100)
                    score_color = "🟢" if score_pct > 70 else "🟡" if score_pct > 50 else "🔴"
                    
                    st.markdown(f"**Fragmento {i}** — Página {chunk['pagina']} {score_color} Relevancia: {score_pct}%")
                    
                    if modo_debug:
                        st.progress(chunk["score"])
                        st.code(chunk["texto"], language=None)
                    else:
                        st.info(chunk["preview"])
                    
                    st.divider()

# Input del usuario
pregunta = st.chat_input("¿Qué querés saber del reglamento?")

if pregunta:
    # Mostrar pregunta
    with st.chat_message("user"):
        st.write(pregunta)
    st.session_state.historial.append({"rol": "user", "contenido": pregunta})
    
    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando fragmentos relevantes..."):
            chunks_relevantes = buscar_chunks(pregunta, indice, chunks, modelo, top_k=top_k)
        
        if not chunks_relevantes:
            respuesta = "No encontré fragmentos del reglamento relevantes para tu pregunta. Intentá reformularla."
            st.warning(respuesta)
        else:
            with st.spinner("Generando respuesta..."):
                # Pasar historial REAL al LLM (el fix del problema #3)
                historial_para_llm = [
                    m for m in st.session_state.historial 
                    if m["rol"] in ("user", "assistant")
                ]
                respuesta = responder(pregunta, chunks_relevantes, historial_para_llm)
            
            st.write(respuesta)
            
            # Siempre mostrar fuentes (colapsadas por default)
            with st.expander(f"📚 {len(chunks_relevantes)} fragmentos consultados"):
                for i, chunk in enumerate(chunks_relevantes, 1):
                    score_pct = int(chunk["score"] * 100)
                    score_color = "🟢" if score_pct > 70 else "🟡" if score_pct > 50 else "🔴"
                    
                    st.markdown(f"**Fragmento {i}** — Página {chunk['pagina']} {score_color} {score_pct}% relevancia")
                    
                    if modo_debug:
                        st.progress(chunk["score"])
                        st.code(chunk["texto"], language=None)
                    else:
                        st.info(chunk["preview"])
    
    # Guardar con los chunks para poder mostrarlos después
    st.session_state.historial.append({
        "rol": "assistant", 
        "contenido": respuesta,
        "chunks": chunks_relevantes
    })