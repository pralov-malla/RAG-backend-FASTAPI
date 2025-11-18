import os
from dotenv import load_dotenv
import importlib.util

def _find_and_get(module_path: str, attr_name: str):
    try:
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            return None
        m = importlib.import_module(module_path)
        return getattr(m, attr_name) if hasattr(m, attr_name) else None
    except (ModuleNotFoundError, ImportError, ValueError):
        return None

# Embeddings
HuggingFaceEmbeddings = _find_and_get("langchain_community.embeddings", "HuggingFaceEmbeddings")
if HuggingFaceEmbeddings is None:
    HuggingFaceEmbeddings = _find_and_get("langchain.embeddings", "HuggingFaceEmbeddings")
if HuggingFaceEmbeddings is None:
    HuggingFaceEmbeddings = _find_and_get("langchain_huggingface", "HuggingFaceEmbeddings")
if HuggingFaceEmbeddings is None:
    raise RuntimeError("HuggingFaceEmbeddings not found. Install 'langchain-community' or 'langchain_huggingface'.")

# Chroma vectorstore
Chroma = _find_and_get("langchain_community.vectorstores", "Chroma")
if Chroma is None:
    Chroma = _find_and_get("langchain.vectorstores", "Chroma")
if Chroma is None:
    Chroma = _find_and_get("langchain_chroma.vectorstores", "Chroma")
if Chroma is None:
    raise RuntimeError("Chroma vectorstore not found. Install 'langchain-community' or 'chromadb'.")

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

ChatPromptTemplate = _find_and_get("langchain_core.prompts", "ChatPromptTemplate")
MessagesPlaceholder = _find_and_get("langchain_core.prompts", "MessagesPlaceholder")
if ChatPromptTemplate is None or MessagesPlaceholder is None:
    # try classic fallback
    ChatPromptTemplate = _find_and_get("langchain_classic.prompts", "ChatPromptTemplate")
    MessagesPlaceholder = _find_and_get("langchain_classic.prompts", "MessagesPlaceholder")
    if ChatPromptTemplate is None or MessagesPlaceholder is None:
        raise RuntimeError("Could not find ChatPromptTemplate/MessagesPlaceholder in langchain. Ensure langchain is installed.")

# ChatGroq is optional
ChatGroq = _find_and_get("langchain_groq", "ChatGroq")

load_dotenv()


def initialize_rag_chain():
    """
    Initializes:
    - HuggingFace embeddings
    - Chroma vector store
    - Groq LLM with prompt templates
    - LangChain RAG chain
    Returns the full chain and empty chat history
    """

    # Embeddings for vector search
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store (Chroma)
    vector_store = Chroma(
        persist_directory='chroma_store',
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever()

    # Prompt to convert follow-up questions into standalone questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the following conversation and a follow-up question, "
         "rephrase the follow-up into a standalone question. "
         "If it's already standalone, return it unchanged."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Main system prompt for answering questions based on context only
    system_prompt = """
    You are an assistant that only answers questions using the provided context below.
    Do NOT use any prior knowledge or assumptions beyond the context.
    If the context does not contain enough information to answer, respond with:
    "I'm sorry, the provided documents do not contain enough information to answer that question."
    Context:
    {context}
    """

    # QA prompt combining system prompt and conversation history
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Initialize the LLM using ChatGroq with the openai/gpt-oss-20b model
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    
    if ChatGroq is None:
        raise RuntimeError(
            "ChatGroq integration not available.\n"
            "Install langchain-groq: pip install langchain-groq"
        )

    try:
        llm = ChatGroq(
            model=model_name,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=1,
            max_tokens=8192,
        )
        print(f"Initialized ChatGroq with model: {model_name}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize ChatGroq with model '{model_name}'.\n"
            f"Ensure GROQ_API_KEY is set correctly in .env file.\n"
            f"Error: {e}"
        )

    # Create combine documents chain
    combine_documents_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create QA chain
    qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_documents_chain
    )

    # Retriever that is aware of chat history to contextualize follow-up questions
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )

    # Create final RAG chain combining history-aware retriever and QA chain
    rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=qa_chain
    )


    # Return the chain and an empty chat history list
    return rag_chain, []


def ask_question(rag_chain, user_input, chat_history):
    """
    Invoke the RAG chain and return a clean string response.
    
    Args:
        rag_chain: The initialized RAG chain
        user_input: User's question
        chat_history: List of (role, message) tuples for conversation context
    
    Returns:
        str: Clean answer text from the LLM
    """
    result = rag_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    # Extract the answer from the result dictionary
    answer = result.get("answer", "")
    
    # Handle case where answer might be empty or None
    if not answer:
        return "I'm sorry, I couldn't generate a response. Please try again."
    
    # If answer is already a clean string, return it
    if isinstance(answer, str):
        # Remove any extra whitespace
        return answer.strip()
    
    # If answer is some other type, convert to string
    return str(answer).strip()
    