from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 15
    }
)

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

normal_prompt = PromptTemplate(
    template="""
You are an expert transformer insulation engineer.

Retrieved Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=[
        "context",
        "question"
    ]
)

prediction_prompt = PromptTemplate(
    template="""
You are an expert transformer insulation engineer.

Transformer Status:
{status}

Transformer Measurements:
{inputs}

Retrieved Context:
{context}

Question:
{question}

Use both the transformer measurements and the retrieved context to answer.

Answer:
""",
    input_variables=[
        "status",
        "inputs",
        "context",
        "question"
    ]
)

def ask_rag(question, status=None, inputs=None):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    if status and inputs:
        prompt_text = prediction_prompt.format(
            status=status,
            inputs=inputs,
            context=context,
            question=question
        )
    else:
        prompt_text = normal_prompt.format(
            context=context,
            question=question
        )

    response = llm.invoke(prompt_text)

    return response.content