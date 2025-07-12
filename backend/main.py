from typing import List
from fastapi import FastAPI, UploadFile, File, Form,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path # For handling file paths
import shutil # For saving uploaded files to disk
from file_utils import process_upload_files
from rag_chain import initialize_rag_chain, ask_question



app = FastAPI()

# Enabling CORS so the frontend (which may run on a different port) can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to the actual frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Initializing RAG chain

# File upload endpoint
@app.post("/upload")
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return {'message': 'Files uploaded and embedded successfully'}

# Chat endpoint
chat_history = []  # Initialize chat history globally


# Initialize RAG chain and chat history once at startup
rag_chain, chat_history = initialize_rag_chain()
@app.post("/chat")
async def chat(user_input: str = Form(...)):
    """
    Accepts user prompt and returns an AI response.
    Maintains chat history for context.
    """
    global chat_history

    try:
        # Get AI response from RAG chain.
        result = ask_question(rag_chain, user_input, chat_history)

        # Append user input and AI response to chat history as tuples.
        chat_history.append(("human", user_input))
        chat_history.append(("ai", result))

        return JSONResponse(content={"response": result})

    except Exception as e:
        # Return error details for debugging.
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
