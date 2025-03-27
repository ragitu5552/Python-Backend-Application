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
