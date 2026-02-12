import os
from datetime import datetime

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import settings

SYSTEM_PROMPT = """You are an FAQ chatbot for {product_name}.
You MUST answer using ONLY the provided context passages.
If the answer is not contained in the context, reply exactly:
"I don't know based on the provided product information."
End every answer with: Sources: [n], [m] (use the passage numbers you relied on).
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context passages:\n{context}\n\nUser question: {question}")
])

def load_vectorstore():
    if not os.path.exists(settings.vectorstore_dir):
        raise FileNotFoundError(
            f"Vectorstore not found at '{settings.vectorstore_dir}'. "
            f"Run: python -m src.ingest"
        )
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    return FAISS.load_local(
        settings.vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

def main():
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": settings.top_k})

    llm = ChatOpenAI(model=settings.chat_model, temperature=settings.temperature)
    chain = PROMPT | llm | StrOutputParser()

    test_questions = [
        "What is the water test and how do I do it?",
        "My Instant Pot says 'Burn'-what should I do?",
        "Why is the float valve not coming up?",
        "Can I do pressure canning in the Instant Pot Duo?",
        "How do I clean and deodorize the sealing ring?",
        "How much liquid do I need at minimum to pressure cook?",
        "Why isn’t the timer counting down yet?"
    ]

    print("=== EVALUATION RUN LOGS ===")
    print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"App: {settings.app_name}")
    print(f"Product: {settings.product_name}")
    print(f"Model: {settings.chat_model}")
    print(f"Embeddings: {settings.embedding_model}")
    print(f"top_k: {settings.top_k}\n")

    for idx, q in enumerate(test_questions, start=1):
        docs = retriever.invoke(q)

        context_blocks = []
        for i, d in enumerate(docs, start=1):
            context_blocks.append(f"[{i}] {d.page_content}")
        context = "\n\n".join(context_blocks)

        ans = chain.invoke({
            "product_name": settings.product_name,
            "context": context,
            "question": q
        })

        print(f"--- Test {idx} ---")
        print(f"Q: {q}")
        print("Retrieved passages:")
        for i, d in enumerate(docs, start=1):
            print(f"  [{i}] category={d.metadata.get('category')} source={d.metadata.get('source')}")
        print(f"Answer:\n{ans}\n")

if __name__ == "__main__":
    main()
