# 📄 Offline PDF RAG Assistant

An offline Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content.

The application uses **LangChain, Ollama, ChromaDB and Streamlit** to retrieve relevant information from the uploaded PDF and generate answers using a locally running **Phi-3 Mini** LLM.

The complete application runs locally without requiring an external LLM API.

---

## 🚀 Project Overview

The Offline PDF RAG Assistant follows a simple RAG pipeline:

```text
PDF Document
     ↓
PDF Loader
     ↓
Text Chunking
     ↓
Local Embeddings
     ↓
ChromaDB
     ↓
User Question
     ↓
Semantic Retrieval
     ↓
Relevant Context
     ↓
Phi-3 Mini
     ↓
Generated Answer
