import os
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import settings

REQUIRED_COLUMNS = ["question", "answer", "category", "source", "last_updated"]

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Clean
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    df = df[(df["question"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)

    return df

def df_to_documents(df: pd.DataFrame) -> list[Document]:
    docs: list[Document] = []
    for _, row in df.iterrows():
        text = (
            f"Product: {settings.product_name}\n"
            f"Category: {row['category']}\n"
            f"Q: {row['question']}\n"
            f"A: {row['answer']}"
        )
        meta = {
            "product": settings.product_name,
            "category": row["category"],
            "source": row["source"],
            "last_updated": row["last_updated"],
        }
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def build_vectorstore(docs: list[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def main() -> None:
    if not os.path.exists(settings.data_csv_path):
        raise FileNotFoundError(f"Could not find dataset at: {settings.data_csv_path}")

    df = load_dataset(settings.data_csv_path)
    docs = df_to_documents(df)
    vs = build_vectorstore(docs)

    os.makedirs(settings.vectorstore_dir, exist_ok=True)
    vs.save_local(settings.vectorstore_dir)

    print("=== INGESTION COMPLETE ===")
    print(f"App: {settings.app_name}")
    print(f"Product: {settings.product_name}")
    print(f"Dataset path: {settings.data_csv_path}")
    print(f"FAQ records (rows): {len(df)}")
    print(f"Vectorstore saved to: {settings.vectorstore_dir}")
    print(f"Retrieval top_k: {settings.top_k}")
    print(f"Chunk size/overlap: {settings.chunk_size}/{settings.chunk_overlap}")

if __name__ == "__main__":
    main()
