from dotenv import load_dotenv
import os

# Cargar las variables del archivo .env
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")

}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RAWG_API_KEY=os.getenv("RAWG_API_KEY")
MAIN_USER=os.getenv("MAIN_USER")

VECTORDB_CONFIG = {
    'path': "./chroma_db",  # Directory for ChromaDB storage
    'collection_name': "game_descriptions"
}

# Embedding Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, good for descriptions