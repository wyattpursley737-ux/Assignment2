from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    app_name: str = "DuoFAQ"
    product_name: str = "Instant Pot Duo 7-in-1 Electric Pressure Cooker"

    # Paths
    data_csv_path: str = os.path.join("data", "instant_pot_duo_faq.csv")
    vectorstore_dir: str = "vectorstore_faiss"

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 4

    # LLM + Embeddings (OpenAI-compatible via langchain-openai)
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.2

settings = Settings()
