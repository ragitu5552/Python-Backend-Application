from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from services.embedding import embedding_service
from database import get_db
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from services.retriever import retriever  # Import Retriever class
from groq import Groq
import os
import traceback
from dotenv import load_dotenv

router = APIRouter(prefix="/qa", tags=["Q&A"])

# Request model
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    min_similarity_score: Optional[float] = None

# Response model for individual chunk
class ChunkResponse(BaseModel):
    chunk_text: str
    document_title: str
    document_id: int
    file_path: Optional[str]

# Response model
class QueryResponse(BaseModel):
    question: str
    relevant_chunks: List[ChunkResponse]


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    # retriever: Retriever,
    db: AsyncSession = Depends(get_db)
):
    try:
        # Perform semantic search
        #retriever = Retriever(embedding_service=embedding_service)
        results = await retriever.semantic_search(
            db=db,
            query=request.question,
            top_k=request.top_k
        )
        
        return {
            "question": request.question,
            "relevant_chunks": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer")
async def generate_answer(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        # Generate answer using the updated function
        answer_result = await generate_answer_with_context(
            question=request.question, 
            db=db,
            retriever=retriever,  # Use the instantiated retriever
            top_k=request.top_k or 3,
            min_similarity_score=request.min_similarity_score or 0.5
        )
        
        return {
            "question": request.question,
            "answer": answer_result.get('answer', 'No answer could be generated.'),
            "context_chunks": [
                {
                    "document_title": result.get('document_title', ''),
                    "chunk_text": result.get('chunk_text', '')[:300] + "...",  # Preview
                }
                for result in answer_result.get('context_results', [])
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qa_debug.log'),  # Log to file
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

async def generate_answer_with_context(
    question: str, 
    db: AsyncSession, 
    retriever: retriever,  
    top_k: int = 3, 
    min_similarity_score: float = 0.5
) -> Dict[str, Any]:
    """
    Generate an answer using semantic search and LLM context retrieval
    """
    try:
        # Log detailed input parameters
        logger.debug(f"Input parameters:")
        logger.debug(f"Question: {question}")
        logger.debug(f"Top K: {top_k}")
        logger.debug(f"Min Similarity Score: {min_similarity_score}")
        
        # Log retriever details
        logger.debug(f"Retriever type: {type(retriever)}")
        
        # Attempt to retrieve context
        try:
            logger.debug("Starting semantic search...")
            context_results = await retriever.semantic_search(
                query=question, 
                db=db, 
                top_k=top_k
            )
            logger.debug(f"Semantic search completed. Results: {context_results}")
        except Exception as search_error:
            logger.error(f"Semantic search error: {search_error}")
            logger.error(traceback.format_exc())
            return {
                "answer": f"Error during semantic search: {str(search_error)}",
                "context_results": [],
                "raw_context": "",
                "error": str(search_error)
            }
        
        # Check if context results are empty
        if not context_results:
            logger.warning("No context results found for the given question.")
            return {
                "answer": "I couldn't find any relevant context to answer your question.",
                "context_results": [],
                "raw_context": ""
            }
        
        # Combine context chunks
        context = "\n\n".join([
            f"Document: {result.get('document_title', 'Unknown')}\n"
            f"Content: {result.get('chunk_text', '')}" 
            for result in context_results
        ])
        
        logger.debug(f"Generated context: {context}")
        
        # Generate answer using Groq's Llama 3 70B
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI assistant. "
                            "Answer the question based only on the provided context. "
                            "If the context does not contain sufficient information, "
                            "clearly state that you cannot find an answer in the given context."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ],
                model="llama3-70b-8192",
                temperature=0.3,  # Lower for more factual answers
                max_tokens=1024
            )
            
            answer = chat_completion.choices[0].message.content
            logger.debug(f"Generated answer: {answer}")
            
            return {
                "answer": answer,
                "context_results": context_results,
                "raw_context": context
            }
        
        except Exception as groq_error:
            logger.error(f"Groq API error: {groq_error}")
            logger.error(traceback.format_exc())
            return {
                "answer": f"Error generating answer: {str(groq_error)}",
                "context_results": context_results,
                "raw_context": context,
                "error": str(groq_error)
            }
    
    except Exception as e:
        logger.error(f"Unexpected error in answer generation: {e}")
        logger.error(traceback.format_exc())
        return {
            "answer": f"Unexpected error: {str(e)}",
            "context_results": [],
            "raw_context": "",
            "error": str(e)
        }
    
@router.get("/debug-context")
async def debug_context(
    question: str,
    top_k: int = 3,
    db: AsyncSession = Depends(get_db)
):
    """Diagnostic endpoint to see what context would be used"""
    results = await retriever.get_relevant_documents(
        db=db,
        query_embedding=await embedding_service.generate_embeddings(question),
        top_k=top_k
    )
    
    return {
        "active_documents": [doc.title for doc in results],
        "total_chunks": sum(len(doc.chunks) for doc in results)
    }
#Use first
# $body = @{
#     question = "What is this about?"
#     top_k = 2
# } | ConvertTo-Json -Compress

# Invoke-RestMethod -Uri "http://localhost:8000/qa/query" -Method Post -Body $body -ContentType "application/json"

#curl.exe -X PUT http://localhost:8000/documents/6/deactivate

# Invoke-RestMethod -Uri "http://localhost:8000/qa/answer" `
#     -Method Post `
#     -Body $body `
#     -ContentType "application/json"