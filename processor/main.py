from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import asyncio
from typing import List, Optional
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import PyPDF2
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Vector Database API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_documents")
VECTOR_SIZE = 384  # Dimension for sentence-transformers model

# Initialize Qdrant client
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class PDFProcessor:
    def __init__(self):
        self.collection_name = COLLECTION_NAME
        self.ensure_collection_exists()
    
    def ensure_collection_exists(self):
        """Ensure the Qdrant collection exists"""
        try:
            collections = qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks"""
        try:
            embeddings = model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def store_in_qdrant(self, document_id: str, filename: str, chunks: List[str], embeddings: List[List[float]]):
        """Store document chunks and embeddings in Qdrant"""
        try:
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=f"{document_id}_{i}",
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_index": i,
                        "text": chunk,
                        "total_chunks": len(chunks)
                    }
                )
                points.append(point)
            
            qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Error storing in Qdrant: {e}")
            raise

# Initialize processor
processor = PDFProcessor()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if Qdrant is accessible
        collections = qdrant_client.get_collections()
        return {"status": "healthy", "qdrant_connected": True, "collections": len(collections.collections)}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read file content
        content = await file.read()
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Process PDF in background
        background_tasks.add_task(process_pdf, document_id, file.filename, content)
        
        return {
            "message": "PDF uploaded successfully",
            "document_id": document_id,
            "filename": file.filename,
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

async def process_pdf(document_id: str, filename: str, content: bytes):
    """Process PDF file asynchronously"""
    try:
        # Extract text
        text = processor.extract_text_from_pdf(content)
        
        if not text.strip():
            logger.warning(f"No text extracted from PDF: {filename}")
            return
        
        # Chunk text
        chunks = processor.chunk_text(text)
        
        # Create embeddings
        embeddings = processor.create_embeddings(chunks)
        
        # Store in Qdrant
        processor.store_in_qdrant(document_id, filename, chunks, embeddings)
        
        logger.info(f"Successfully processed PDF: {filename}")
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {e}")

@app.get("/search")
async def search_documents(query: str, limit: int = 10):
    """Search for documents using semantic similarity"""
    try:
        # Create query embedding
        query_embedding = model.encode([query]).tolist()[0]
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "score": result.score,
                "document_id": result.payload["document_id"],
                "filename": result.payload["filename"],
                "chunk_index": result.payload["chunk_index"],
                "text": result.payload["text"],
                "total_chunks": result.payload["total_chunks"]
            })
        
        return {"query": query, "results": results}
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all documents in the database"""
    try:
        # Get all points from the collection
        points = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True
        )[0]
        
        # Group by document
        documents = {}
        for point in points:
            doc_id = point.payload["document_id"]
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": point.payload["filename"],
                    "total_chunks": point.payload["total_chunks"],
                    "chunks_processed": 0
                }
            documents[doc_id]["chunks_processed"] += 1
        
        return {"documents": list(documents.values())}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks"""
    try:
        # Delete all points for this document
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector={"filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]}}
        )
        
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 