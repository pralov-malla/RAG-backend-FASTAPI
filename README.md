History-Aware PDF/Text Chatbot using LLaMA3 (LangChain + Groq)

This project implements a Retrieval-Augmented Generation (RAG) chatbot that can answer user queries based on the contents of a PDF and text file, while retaining chat history for better context. It's powered by LangChain, Groq's LLaMA3, and HuggingFace embeddings, with storage managed using ChromaDB.
🚀 Features

    🧠 Chat history-aware questioning using create_history_aware_retriever

    📄 PDF and Text file ingestion using LangChain loaders

    🧩 Chunked text splitting with overlap for better semantic understanding

    🔍 Dense vector embeddings with HuggingFace (MiniLM)

    🏪 Persistent vector store using ChromaDB

    🤖 Groq-hosted LLaMA3 as the language model backend

    📚 RAG pipeline with source printing to show where answers came from
