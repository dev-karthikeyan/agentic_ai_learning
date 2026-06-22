from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langsmith import traceable

model = ChatGroq(model="openai/gpt-oss-120b")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0
)

docs_splitter = text_splitter.split_documents(doc_list)

# Create vector store
vector_store = InMemoryVectorStore.from_documents(
    documents=docs_splitter,
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)

# Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

@traceable
def rag_bot(question: str) -> dict:
    docs = retriever.invoke(question)

    docs_string = "\n\n".join(
        doc.page_content for doc in docs
    )

    instructions = f"""
Role: You are a helpful, expert assistant designed to provide accurate information based exclusively on the documents provided.

Core Guidelines:

- Answer only using the provided documents.
- If the answer is not in the documents, say:
  "I'm sorry, but I do not have enough information in the provided documents to answer that question."
- Do not use outside knowledge.
- Cite sources whenever possible.

Documents:
{docs_string}
"""

    response = model.invoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": question}
    ])

    return {
        "answer": response.content,
        "documents": docs
    }