from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader1 = PyPDFLoader("rag/data/RAG_TRANSFORMER1.pdf")
loader2 = PyPDFLoader("rag/data/RAG_TRANSFORMER2.pdf")

documents = loader1.load() + loader2.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
print("Total Chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Chroma DB built successfully")