# # from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean  # Add Boolean
# # from sqlalchemy.sql import func
# # from database import Base
# # from pgvector.sqlalchemy import Vector

# # class DocumentChunk(Base):
# #     __tablename__ = "document_chunks"
    
# #     id = Column(Integer, primary_key=True)
# #     document_id = Column(Integer, ForeignKey("documents.id"))
# #     text = Column(Text)
# #     embedding = Column(Vector(384))  # Match your embedding dimension
# #     metadata = Column(JSON)  # Stores chunk-specific metadata
    
# #     document = relationship("Document", back_populates="chunks")


# # class Document(Base):
# #     __tablename__ = "documents"
    
# #     id = Column(Integer, primary_key=True, index=True)
# #     title = Column(String(255), nullable=False)
# #     content = Column(Text)
# #     file_path = Column(String(512))
# #     embedding = Column(Vector(384))  # Adjust dimension based on your embedding model
# #     created_at = Column(DateTime(timezone=True), server_default=func.now())
# #     is_active = Column(Boolean, default=True)  # For document selection

# #     chunks = relationship("DocumentChunk", back_populates="document")
# #     def __repr__(self):
# #         return f"<Document {self.title}>"

# from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey
# from sqlalchemy.orm import relationship
# from sqlalchemy.sql import func
# from database import Base
# from pgvector.sqlalchemy import Vector

# class DocumentChunk(Base):
#     __tablename__ = "document_chunks"
    
#     id = Column(Integer, primary_key=True)
#     document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)  # Added nullable=False
#     text = Column(Text, nullable=False)  # Added nullable=False for required fields
#     embedding = Column(Vector(384))  # Match your embedding dimension
#     metadata = Column(JSON, nullable=True)  # Optional metadata
    
#     # Added index for faster lookups on document_id
#     __table_args__ = (
#         Index('idx_document_chunks_document_id', 'document_id'),
#     )
    
#     document = relationship("Document", back_populates="chunks")
    
#     def __repr__(self):
#         return f"<DocumentChunk document_id={self.document_id}, id={self.id}>"


# class Document(Base):
#     __tablename__ = "documents"
    
#     id = Column(Integer, primary_key=True, index=True)
#     title = Column(String(255), nullable=False)
#     content = Column(Text, nullable=True)  # Content might be optional
#     file_path = Column(String(512), nullable=True)  # File path might be optional
#     embedding = Column(Vector(384), nullable=True)  # Embedding might be generated later
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
#     is_active = Column(Boolean, default=True)  # For document selection
    
#     chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
#     def __repr__(self):
#         return f"<Document {self.title}>"


from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
from pgvector.sqlalchemy import Vector

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # Match your embedding dimension
    meta_data = Column(JSON, nullable=True)  # Optional metadata
    
    # Correct Index import and usage
    __table_args__ = (
        Index('idx_document_chunks_document_id', 'document_id'),
    )
    
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk document_id={self.document_id}, id={self.id}>"


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)  # Content might be optional
    file_path = Column(String(512), nullable=True)  # File path might be optional
    embedding = Column(Vector(384), nullable=True)  # Embedding might be generated later
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)  # For document selection
    
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document {self.title}>"