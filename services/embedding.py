# from langchain.embeddings import SentenceTransformerEmbeddings
# from typing import List
# from dotenv import load_dotenv

# load_dotenv()

# class EmbeddingService:
#     def __init__(self):
#         # Using sentence-transformers directly
#         self.embed_model = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")

#     async def generate_embeddings(self, text: str) -> List[float]:
#         """Generate embeddings for a single text"""
#         return self.embed_model.embed_query(text)

#     async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
#         """Generate embeddings for multiple texts"""
#         return self.embed_model.embed_documents(texts)

# embedding_service = EmbeddingService()

from langchain.embeddings import SentenceTransformerEmbeddings
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the Embedding Service with text chunking capabilities
        
        Args:
            chunk_size (int): Maximum number of tokens/characters per chunk
            chunk_overlap (int): Number of tokens/characters to overlap between chunks
        """
        # Using sentence-transformers directly
        self.embed_model = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller segments
        
        Args:
            text (str): Input text to be chunked
        
        Returns:
            List[str]: List of text chunks
        """
        # Simple chunking strategy using character-based splitting
        chunks = []
        for start in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[start:start + self.chunk_size]
            chunks.append(chunk)
        
        return chunks

    async def chunk_and_embed(self, text: str) -> Tuple[List[str], List[List[float]]]:
        """
        Chunk text and generate embeddings for each chunk
        
        Args:
            text (str): Input text to be chunked and embedded
        
        Returns:
            Tuple of chunks and their corresponding embeddings
        """
        chunks = self.chunk_text(text)
        embeddings = await self.generate_batch_embeddings(chunks)
        return chunks, embeddings

    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for a single text"""
        return self.embed_model.embed_query(text)

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return self.embed_model.embed_documents(texts)

embedding_service = EmbeddingService()