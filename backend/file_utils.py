from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from pathlib import Path
import shutil


# Loading and embedding uploaded documents into ChromaDB
def process_upload_files(file_paths: List[str]):
    """
    Loads, chunks, embeds, and stores uploaded PDF/TXT files.
    
    Args:
        file_paths (List[str]): Paths to the uploaded files.
    """

    documents = []

    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            # Specify encoding explicitly to avoid decode errors
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            print(f"Unsupported file type: {file_path}")
            continue  # Skip unsupported files

        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading {file_path}: {str(e)}")

    if not documents:
        raise HTTPException(status_code=400, detail="No documents loaded from the uploaded files.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)

    # Create embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create or update the Chroma vector store
    persist_dir = "chroma_store"
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )

