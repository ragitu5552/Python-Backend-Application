# Document Management & RAG Q&A System

![RAG Architecture](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*5ZLci3SuR0zM_QlZOADv8Q.png)  
*A Retrieval-Augmented Generation (RAG) pipeline*

## 📌 Objectives
- **Document Ingestion**: Store and manage documents with metadata
- **Selective Activation**: Control which documents are searchable
- **Semantic Search**: Find relevant content using vector embeddings
- **Q&A Generation**: Answer questions using Groq's Llama 3 70B model

## 🛠 Tech Stack
| Component          | Technology               |
|--------------------|--------------------------|
| Backend Framework  | FastAPI (Async)          |
| Vector Database    | PostgreSQL + pgvector    |
| Embeddings         | BAAI/bge-small-en-v1.5   |
| LLM Integration    | Groq (Llama 3 70B)       |
| Text Processing    | LlamaIndex, PyPDF2, docx |

## 🌟 Key Features
### Document Management
```python
@router.put("/documents/{id}/activate")  # Toggle document visibility
@router.post("/upload")  # Ingest PDFs, Word, TXT with auto-chunking
```


# RAG Pipeline

A **Retrieval-Augmented Generation (RAG)** pipeline for intelligent document Q&A using **FastAPI**, **Sentence-Transformers**, and **Llama 3**.

---

## 🚀 Overview  

### **Pipeline Components**  
1. **Chunking**:  
   - Splits text into **512-token** segments with a **20-token overlap**.  
2. **Embedding**:  
   - Uses **Sentence-Transformers** for dense vector representations.  
3. **Retrieval**:  
   - Fetches the **Top-K** most relevant chunks across active documents.  
4. **Generation**:  
   - Uses **Llama 3** for context-aware responses.  

---

## 🔗 API Endpoints  

| Endpoint                      | Method | Description                    |
|--------------------------------|--------|--------------------------------|
| `/documents/upload`            | `POST` | Upload and embed documents     |
| `/documents/{id}/activate`     | `PUT`  | Enable document for Q&A        |
| `/qa/query`                    | `POST` | Retrieve relevant document chunks |
| `/qa/answer`                   | `POST` | Generate answers using LLM     |

---

## 📂 Code Structure
```
.
├── routers/
│   ├── documents.py  # CRUD and activation
│   └── qa.py         # RAG endpoints
├── services/
│   ├── embedding.py  # Chunking + vector generation
│   └── retriever.py  # Semantic search
├── models.py         # Database schemas
└── main.py           # FastAPI app setup
```

Setup Guide
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start PostgreSQL with pgvector
create a .env file like .env.example provided and give your own credentials

# 3. Run FastAPI
uvicorn main:app --reload
