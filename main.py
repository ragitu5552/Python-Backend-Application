from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from database import get_db
from routers import documents, qa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text
from database import init_db

app = FastAPI()
app.include_router(documents.router)
app.include_router(qa.router)
# CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    await init_db()

@app.get("/")
async def root():
    return {"message": "Welcome to RAG Application"}

# Example endpoint using database
@app.get("/test-db")
async def test_db(db: AsyncSession = Depends(get_db)):
    # Test the database connection
    try:
        result = await db.execute(text("SELECT 1"))  # Wrap SQL in text()
        return {"db_connection": "success", "result": result.scalar()}
    except Exception as e:
        return {"db_connection": "failed", "error": str(e)}