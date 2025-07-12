import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

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

    # Initialize the LLM (Groq)
    llm = ChatGroq(
        model="llama3-70b-8192",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
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
    result = rag_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    # If result["answer"] is a stringified dict, parse it safely
    answer = result.get("answer")

    # If answer looks like a stringified dict, try to parse it
    if isinstance(answer, str) and answer.startswith("{") and answer.endswith("}"):
        import ast
        try:
            answer_dict = ast.literal_eval(answer)
            # Extract the actual answer text from the dict if available
            if "answer" in answer_dict:
                return str(answer_dict["answer"])
            else:
                # fallback to string representation
                return str(answer)
        except Exception:
            # If parsing fails, return raw string
            return str(answer)
    else:
        # If answer is already a string, return as is
        return str(answer)
    