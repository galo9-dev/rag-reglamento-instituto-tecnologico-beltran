# src/generator.py
import os
from groq import Groq
from src.config import LLM_MODEL, LLM_TEMPERATURE

def responder(pregunta: str, chunks_relevantes: list[dict], historial: list[dict]) -> str:
    cliente = Groq(api_key=os.getenv("GROQ_API_KEY"))

    contexto = "\n\n---\n\n".join([
        f"[Página {c['pagina']}]\n{c['texto']}"
        for c in chunks_relevantes
    ])

    mensajes_llm = [
        {
            "role": "system",
            "content": (
                "Sos un asistente especializado en el reglamento académico del ISFT N° 197. "
                "Respondé SOLO basándote en el contexto proporcionado. "
                "Si la información no está en el contexto, decí explícitamente: "
                "'Esta información no está en los fragmentos del reglamento que encontré.' "
                "Citá el número de página cuando sea posible. "
                "Respondé en español rioplatense, claro y amigable para estudiantes."
            )
        }
    ]

    for msg in historial[-6:]:
        mensajes_llm.append({
            "role": msg["rol"],
            "content": msg["contenido"]
        })

    mensajes_llm.append({
        "role": "user",
        "content": f"Contexto del reglamento:\n{contexto}\n\nPregunta: {pregunta}"
    })

    respuesta = cliente.chat.completions.create(
        model=LLM_MODEL,
        messages=mensajes_llm,
        temperature=LLM_TEMPERATURE,
        max_tokens=1024
    )

    return respuesta.choices[0].message.content