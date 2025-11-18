from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import importlib.util

# Helper to import a module path if available; returns the module object or None
def _import_if_available(module_path: str):
    try:
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            return None
        return importlib.import_module(module_path)
    except (ModuleNotFoundError, ImportError, ValueError):
        return None

# Document loaders: prefer langchain_community, then langchain
_m = _import_if_available("langchain_community.document_loaders")
if _m is not None:
    PyPDFLoader = getattr(_m, "PyPDFLoader", None)
    TextLoader = getattr(_m, "TextLoader", None)
else:
    _m = _import_if_available("langchain.document_loaders")
    if _m is not None:
        PyPDFLoader = getattr(_m, "PyPDFLoader", None)
        TextLoader = getattr(_m, "TextLoader", None)
    else:
        PyPDFLoader = None
        TextLoader = None

if PyPDFLoader is None or TextLoader is None:
    raise RuntimeError(
        "Could not find document loaders. Install 'langchain-community' package."
    )

# Text splitter
_m = _import_if_available("langchain_text_splitters")
if _m is not None:
    RecursiveCharacterTextSplitter = getattr(_m, "RecursiveCharacterTextSplitter", None)
else:
    _m = _import_if_available("langchain.text_splitter")
    if _m is not None:
        RecursiveCharacterTextSplitter = getattr(_m, "RecursiveCharacterTextSplitter", None)
    else:
        RecursiveCharacterTextSplitter = None

if RecursiveCharacterTextSplitter is None:
    raise RuntimeError(
        "Could not find text splitter. Install 'langchain-text-splitters' package."
    )

# Embeddings
_m = _import_if_available("langchain_community.embeddings")
if _m is not None and hasattr(_m, "HuggingFaceEmbeddings"):
    HuggingFaceEmbeddings = getattr(_m, "HuggingFaceEmbeddings")
else:
    _m = _import_if_available("langchain.embeddings")
    if _m is not None and hasattr(_m, "HuggingFaceEmbeddings"):
        HuggingFaceEmbeddings = getattr(_m, "HuggingFaceEmbeddings")
    else:
        raise RuntimeError(
            "Could not find HuggingFaceEmbeddings. Install 'langchain-community'."
        )

# Vectorstore (Chroma)
_m = _import_if_available("langchain_community.vectorstores")
if _m is not None and hasattr(_m, "Chroma"):
    Chroma = getattr(_m, "Chroma")
else:
    _m = _import_if_available("langchain.vectorstores")
    if _m is not None and hasattr(_m, "Chroma"):
        Chroma = getattr(_m, "Chroma")
    else:
        raise RuntimeError(
            "Could not find Chroma vectorstore. Install 'langchain-community' and 'chromadb'."
        )

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
        chunk_overlap=200,
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

