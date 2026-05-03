from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
DEFAULT_TOP_K = 4
MIN_SIMILARITY_SCORE = 0.25
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

FAISS_INDEX_PATH = DATA_DIR / "indice.faiss"
CHUNKS_PATH = DATA_DIR / "chunks.pkl"
PDF_PATH = DATA_DIR / "reglamento.pdf"