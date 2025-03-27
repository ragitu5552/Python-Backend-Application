from typing import List, Any, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from services.embedding import embedding_service

class Retriever:
    def __init__(self, embedding_service):
        """
        Initialize a vector store retriever
        
        Args:
            embedding_service: Embedding generation service
        """
        self.embedding_service = embedding_service

    async def semantic_search(
        self, 
        query: str, 
        db: AsyncSession,
        top_k: int = 3,
        min_similarity_score: float = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search 
        
        Args:
            query (str): Search query
            db: Database session
            top_k (int): Number of top results to retrieve
            min_similarity_score (float, optional): Minimum similarity threshold
        
        Returns:
            List of semantic search results
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embeddings(query)
            
            # Perform similarity search using the previous method
            query_results = []
            
            # Query document chunks
            from sqlalchemy import select
            from models import DocumentChunk, Document
            
            # Construct query to find chunks ordered by embedding similarity
            chunk_query = select(DocumentChunk, Document).join(Document).order_by(
                DocumentChunk.embedding.l2_distance(query_embedding)
            ).limit(top_k)
            
            # Execute the query
            result = await db.execute(chunk_query)
            
            # Process results
            for chunk, document in result.tuples():
                query_results.append({
                    'chunk_text': chunk.text,
                    'document_title': document.title,
                    'document_id': document.id,
                    'file_path': document.file_path
                })
            
            return query_results
        
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

# Create retriever with embedding service
retriever = Retriever(embedding_service=embedding_service)
