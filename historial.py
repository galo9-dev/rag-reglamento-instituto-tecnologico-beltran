import json
import os
from datetime import datetime

HISTORIAL_PATH = "historial_preguntas.json"

# Reglas simples de clasificación por palabras clave.
# Agregá/quitá términos según lo que más preguntan en tu instituto.
REGLAS_CATEGORIA = {
    "vestimenta": ["vestimenta", "uniforme", "ropa", "vestir"],
    "asistencia": ["asistencia", "ausente", "falta", "inasistencia"],
    "examenes": ["examen", "parcial", "final", "evaluacion", "evaluación"],
    "inscripcion": ["inscripcion", "inscripción", "matricula", "matrícula"],
    "regularidad": ["regular", "regularidad", "condicion", "condición"],
    "disciplina": ["sancion", "sanción", "disciplina", "amonestacion", "amonestación"],
}

REGLAS_PRIORIDAD = {
    "ALTA": ["sancion", "sanción", "expulsion", "expulsión", "amonestacion", "urgente"],
    "MEDIA": ["examen", "parcial", "inasistencia", "falta"],
}

def clasificar_categoria(texto):
    texto = texto.lower()
    for categoria, palabras in REGLAS_CATEGORIA.items():
        if any(p in texto for p in palabras):
            return categoria
    return "general"

def clasificar_prioridad(texto):
    texto = texto.lower()
    for prioridad, palabras in REGLAS_PRIORIDAD.items():
        if any(p in texto for p in palabras):
            return prioridad
    return "BAJA"

def cargar_historial():
    if os.path.exists(HISTORIAL_PATH):
        with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"consultas": []}

def guardar_consulta(consulta_original, respuesta, origen="app.py"):
    historial = cargar_historial()

    nuevo_id = (
        max((c["id"] for c in historial["consultas"]), default=0) + 1
    )
    ahora = datetime.now()

    nueva_entrada = {
        "id": nuevo_id,
        "fecha": ahora.strftime("%Y-%m-%d"),
        "hora": ahora.strftime("%H:%M:%S"),
        "consulta_original": consulta_original,
        "categoria": clasificar_categoria(consulta_original),
        "prioridad": clasificar_prioridad(consulta_original),
        "respuesta": respuesta,
        "origen": origen
    }

    historial["consultas"].append(nueva_entrada)

    with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

    return nueva_entrada