# 📋 Consultor RAG — Reglamento ISFT N° 197

> Sistema de Q&A con inteligencia artificial sobre el reglamento académico del instituto,
> construido con arquitectura RAG (Retrieval-Augmented Generation) de producción.

**[🚀 Demo en vivo](https://rag-reglamento-instituto-tecnologico-beltran-sr7zxwbhtg9q6d6zy.streamlit.app)**

## 🏗️ Arquitectura

[diagrama ASCII o imagen]

Usuario → Pregunta → Embedding (MiniLM) → FAISS (coseno) → Top-K chunks
→ LLaMA 3.3 70B (Groq) → Respuesta con fuentes citadas

## ⚡ Decisiones técnicas

| Componente | Elección | Por qué |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | Balance velocidad/calidad para español |
| Vector DB | FAISS IndexFlatIP | Búsqueda coseno exacta, sin overhead |
| LLM | LLaMA 3.3 70B via Groq | 0 costo, latencia <2s, calidad GPT-3.5+ |
| Chunking | Semántico por párrafos | Preserva artículos del reglamento completos |

## 🔍 Features

- ✅ Búsqueda semántica con scores de similitud
- ✅ Fuentes citadas con número de página  
- ✅ Modo debug para inspeccionar chunks recuperados
- ✅ Historial de conversación con memoria contextual
- ✅ Filtro de calidad mínima (descarta chunks irrelevantes)

## 🧠 Mejoras pendientes / roadmap

- [ ] Re-ranking con cross-encoder
- [ ] Evaluación con RAGAS
- [ ] Caché de embeddings de preguntas frecuentes