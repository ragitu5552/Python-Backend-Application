from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DocumentBase(BaseModel):
    title: str
    content: Optional[str] = None

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: int
    created_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True

class DocumentListResponse(BaseModel):
    id: int
    title: str
    content: Optional[str]
    embedding: Optional[list[float]]
    created_at: datetime
    is_active: bool

    class Config:
        orm_mode = True

class DocumentUpdate(DocumentBase):  
    title: Optional[str] = None
    content: Optional[str] = None
    is_active: Optional[bool] = None
