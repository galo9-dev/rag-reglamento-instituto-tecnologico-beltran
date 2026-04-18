import fitz

def cargar_pdf(ruta):
    doc = fitz.open(ruta)
    texto_completo = ""
    
    for numero_pagina in range(len(doc)):
        pagina = doc[numero_pagina]
        texto_completo += pagina.get_text()
    
    return texto_completo

def hacer_chunks(texto, tamano=500, superposicion=50):
    chunks = []
    inicio = 0
    
    while inicio < len(texto):
        fin = inicio + tamano
        chunk = texto[inicio:fin]
        chunks.append(chunk)
        inicio = fin - superposicion
    
    return chunks

if __name__ == "__main__":
    texto = cargar_pdf("data/reglamento.pdf")
    chunks = hacer_chunks(texto)
    
    print(f"Total de chunks: {len(chunks)}")
    print("\n--- Chunk de ejemplo ---")
    print(chunks[5])