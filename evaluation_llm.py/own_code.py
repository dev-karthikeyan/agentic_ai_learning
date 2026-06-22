from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.tools import RetrieverInput
from langchain.document_loaders import WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = WebBaseLoader(urls)

docs_spliitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(docs)

vector_store = InMemoryVectorStore.from_documents(

          documents = docs_spliitter ,
          embeddings = HuggingFaceEmbeddings()

)


retriever = vector_store.as_retriever(k=5)