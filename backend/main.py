from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path  # For handling file paths
import shutil  # For saving uploaded files to disk
from .file_utils import process_upload_files
from .rag_chain import initialize_rag_chain, ask_question


# Pydantic models for clean API responses
class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool
    response: str
    message: str = ""


class UploadResponse(BaseModel):
    """Response model for upload endpoint"""
    success: bool
    message: str
    files_processed: int


app = FastAPI(title="RAG Backend API", version="1.0.0")

# Enabling CORS so the frontend (which may run on a different port) can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to the actual frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "RAG Backend API is running",
        "rag_initialized": rag_chain is not None
    }


# Initializing RAG chain

# File upload endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Handles uploaded PDF or TXT files:
    - Saves them to /data directory 
    - Reprocesses and embeds them into the vector DB
    """
    upload_dir = Path("data")
    upload_dir.mkdir(exist_ok=True)

    saved_files = []
    for file in files:
        file_path = upload_dir / file.filename
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(str(file_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save {file.filename}: {str(e)}")
        finally:
            file.file.close()
    
    try:
        process_upload_files(saved_files)
        return UploadResponse(
            success=True,
            message="Files uploaded and embedded successfully",
            files_processed=len(saved_files)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Chat endpoint
chat_history = []  # Initialize chat history globally


# Initialize placeholders for RAG chain and chat history; initialize at startup
rag_chain = None
chat_history = []


@app.on_event("startup")
def startup_event():
    """Initialize the RAG chain when the FastAPI app starts.

    Doing this at startup (instead of module import) prevents import-time
    crashes when optional dependencies (like Groq integration) are missing.
    """
    global rag_chain, chat_history
    try:
        rag_chain, chat_history = initialize_rag_chain()
        print("RAG chain initialized successfully")
    except Exception as e:
        # Print a clear warning; the /chat endpoint will return a 503 until
        # the issue is resolved or optional integrations are installed.
        print(f"Warning: RAG chain initialization failed: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(user_input: str = Form(...)):
    """
    Accepts user prompt and returns an AI response.
    Maintains chat history for context.
    Returns clean JSON with success status and response text.
    """
    global chat_history

    try:
        # Ensure RAG chain is initialized
        if rag_chain is None:
            return ChatResponse(
                success=False,
                response="",
                message="RAG chain not initialized. Check server logs for details."
            )

        # Get AI response from RAG chain (returns clean string)
        result = ask_question(rag_chain, user_input, chat_history)

        # Append user input and AI response to chat history as tuples
        chat_history.append(("human", user_input))
        chat_history.append(("ai", result))

        return ChatResponse(
            success=True,
            response=result,
            message="Response generated successfully"
        )

    except Exception as e:
        # Return structured error response
        return ChatResponse(
            success=False,
            response="",
            message=f"Chat processing failed: {str(e)}"
        )
