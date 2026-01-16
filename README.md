# ScholarChat-RAG-PROJECT

### HOW TO run this Project Intall Python

### Just Download all the folders and open the ipynb file where the codes are and follow the instrcutions there...

# Summary of Project

"Developed a multi-domain academic chatbot using a custom Retrieval-Augmented Generation pipeline. Curated and chunked 500+ segments from NLP, Cloud Computing, Pakistan Studies, and The Sealed Nectar. Employed transformer-based embeddings (intfloat/e5-base), FAISS for dense retrieval, and LLaMA 3.2B Instruct to generate context-aware, citation-backed responses."

# DESCRIPTION OF PROJECT

# 🚀 Advanced RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to provide comprehensive and accurate answers by combining information retrieval with large language model (LLM) generation. It allows you to query your extensive document collection and synthesize context-aware responses.

---

## ✨ Key Components & Features

- **Document Ingestion & Preparation:** Handles `.docx` and `.pdf` files, converting them to `.txt` for consistent processing.
- **Enhanced Intelligent Document Chunking:**
  - Splits large documents into semantically meaningful chunks (approx. 512 tokens with 100 token overlap).
  - Utilizes `intfloat/e5-base` tokenizer for accurate token estimation.
  - Incorporates semantic splitting, category-specific configurations, and enhanced keyword extraction.
- **Dense Embedding Generation:**
  - Converts each document chunk into a 768-dimensional numerical embedding using `intfloat/e5-base`.
  - Performed offline for efficiency.
- **Vector Store (FAISS) Creation:**
  - Indexes all dense embeddings in a FAISS (`IndexFlatIP`) vector store for extremely fast and accurate similarity search (cosine similarity).
  - Pre-computed offline.
- **Dense Retrieval:**
  - At runtime, embeds user queries using `intfloat/e5-base`.
  - Performs rapid similarity search in the FAISS index to retrieve the top `K` (e.g., 4) most relevant document chunks.
- **Context Assembly & LLM Generation:**

  - Compiles retrieved chunks as context for the `LLaMA 3.2-1B` LLM.
  - Generates comprehensive, accurate, and detailed responses based _only_ on the provided context.

  ***

  # Web Chatbot (HTML/CSS + minimal JS)

  This repo includes a lightweight web UI + Python API server for the 568-document RAG pipeline.

  ## Run locally

  1. Install dependencies:

  - `pip install -r requirements.txt`

  2. Start the API server (serves the UI too):

  - `uvicorn server.app:app --reload --host 127.0.0.1 --port 8000`

  Or (simpler):

  - `python -m server.app`

  3. Open:

  - `http://127.0.0.1:8000/`

  Health check:

  - `http://127.0.0.1:8000/health`
