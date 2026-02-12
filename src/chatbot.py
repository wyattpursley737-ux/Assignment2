import os
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.config import settings

SYSTEM_PROMPT = """You are an FAQ chatbot for {product_name}.
You MUST answer using ONLY the provided context passages.
If the answer is not contained in the context, reply exactly:
"I don't know based on the provided product information."

Rules:
- Be concise and practical.
- If steps are needed, use a short numbered list.
- Include a "Sources:" line that lists the passage numbers you used (e.g., Sources: [1], [3]).
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context passages:\n{context}\n\nUser question: {question}")
])

def load_vectorstore() -> FAISS:
    if not os.path.exists(settings.vectorstore_dir):
        raise FileNotFoundError(
            f"Vectorstore not found at '{settings.vectorstore_dir}'. "
            f"Run ingestion first: python -m src.ingest"
        )
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    return FAISS.load_local(
        settings.vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

def format_context(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "Unknown")
        cat = d.metadata.get("category", "Uncategorized")
        updated = d.metadata.get("last_updated", "Unknown")
        blocks.append(
            f"[{i}] (Category: {cat}; Source: {src}; Updated: {updated})\n{d.page_content}"
        )
    return "\n\n".join(blocks)

def main() -> None:
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": settings.top_k})

    llm = ChatOpenAI(model=settings.chat_model, temperature=settings.temperature)
    chain = PROMPT | llm | StrOutputParser()

    print(f"{settings.app_name} - FAQ Chatbot")
    print(f"Product: {settings.product_name}")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        docs = retriever.invoke(question)
        context = format_context(docs)

        answer = chain.invoke({
            "product_name": settings.product_name,
            "context": context,
            "question": question
        })

        print(f"\nBot: {answer}\n")

if __name__ == "__main__":
    main()
