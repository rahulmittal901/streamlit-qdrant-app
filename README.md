# PDF Vector Database with Qdrant and Streamlit

This project combines the power of Qdrant vector database with a Streamlit chat interface for PDF document processing and semantic search.

## 🏗️ Architecture

```
Streamlit App (app_with_qdrant.py) 
    ↓ HTTP API calls
PDF Processor (FastAPI + Qdrant)
    ↓ Vector storage
Qdrant Database (Docker)
```

## 🚀 Quick Start

### 1. Start the Qdrant Services

First, start the Docker containers for Qdrant and the PDF processor:

```bash
# Start all services
docker-compose up -d

# Check if services are running
docker-compose ps
```

### 2. Install Streamlit Dependencies

```bash
pip install -r requirements.streamlit.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app_with_qdrant.py
```

The app will be available at `http://localhost:8501`

## 📋 Prerequisites

- **Docker and Docker Compose** installed
- **Python 3.8+** with pip
- **Groq API Key** (get it from [Groq Console](https://console.groq.com/))

## 🔧 Configuration

### Environment Variables

The app uses these default settings:
- **Qdrant Host**: `localhost`
- **Qdrant Port**: `6333`
- **API URL**: `http://localhost:8000`

You can modify these in `app_with_qdrant.py`:

```python
QDRANT_HOST = "localhost"  # Change if needed
QDRANT_PORT = 6333
API_URL = f"http://{QDRANT_HOST}:8000"
```

## 📖 How It Works

### 1. **PDF Upload Process**
```
User uploads PDF → Streamlit → PDF Processor API → Qdrant Database
```

1. User uploads PDF through Streamlit interface
2. Streamlit sends PDF to our FastAPI processor
3. Processor extracts text, chunks it, creates embeddings
4. Embeddings stored in Qdrant vector database

### 2. **Search Process**
```
User asks question → Streamlit → Search Qdrant → Get relevant chunks → LLM generates answer
```

1. User types a question
2. Streamlit searches Qdrant for relevant document chunks
3. Relevant chunks are sent to Groq LLM with the question
4. LLM generates answer based on the context

## 🎯 Key Features

### ✅ **What's Different from Original app.py**

| Feature | Original app.py | New app_with_qdrant.py |
|---------|----------------|------------------------|
| **Vector Store** | LlamaIndex built-in | Qdrant (Docker) |
| **Scalability** | Limited | High (production-ready) |
| **Persistence** | Session-based | Permanent storage |
| **Search Quality** | Good | Excellent (better embeddings) |
| **Document Management** | Basic | Advanced (upload/delete) |
| **Health Monitoring** | None | Built-in health checks |

### 🚀 **Advantages of This Setup**

1. **Better Embeddings**: Uses `BAAI/bge-large-en-v1.5` (1024D) instead of `all-MiniLM-L6-v2` (384D)
2. **Persistent Storage**: Documents stay in Qdrant even after app restart
3. **Scalable**: Can handle thousands of documents
4. **Production Ready**: Docker containers with health checks
5. **Better Search**: More accurate semantic search results

## 🔍 API Endpoints

The PDF processor provides these endpoints:

- `GET /health` - Health check
- `POST /upload` - Upload PDF file
- `GET /search?query=...` - Search documents
- `GET /documents` - List all documents
- `DELETE /documents/{id}` - Delete document

## 🛠️ Troubleshooting

### Common Issues

1. **"Qdrant services are not running"**
   ```bash
   # Start services
   docker-compose up -d
   
   # Check logs
   docker-compose logs pdf-processor
   ```

2. **"Connection refused"**
   - Make sure Docker containers are running
   - Check if ports 8000 and 6333 are available
   - Verify firewall settings

3. **"Upload failed"**
   - Check if PDF file is valid
   - Ensure file size is reasonable (< 50MB)
   - Check processor logs: `docker-compose logs pdf-processor`

### Health Checks

```bash
# Check Qdrant health
curl http://localhost:6333/health

# Check PDF processor health
curl http://localhost:8000/health

# Check all services
docker-compose ps
```

## 📊 Performance Tips

1. **Large PDFs**: The system can handle large PDFs, but processing time increases with file size
2. **Multiple Uploads**: You can upload multiple PDFs - they'll be processed in parallel
3. **Search Quality**: More documents = better search results
4. **Memory Usage**: Qdrant uses efficient storage, but monitor Docker resource usage

## 🔄 Migration from Original app.py

If you want to migrate from your original `app.py`:

1. **Keep your original app.py** as backup
2. **Use app_with_qdrant.py** for the new functionality
3. **Upload your existing PDFs** through the new interface
4. **Test the search quality** - it should be significantly better

## 🎨 Customization

### Change Embedding Model

To use a different embedding model, modify `processor/main.py`:

```python
# Change this line
model = SentenceTransformer('all-MiniLM-L6-v2')

# To this (better quality)
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
```

### Modify Chunk Size

In `processor/main.py`, change the chunking parameters:

```python
def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200):
    # Modify chunk_size and overlap as needed
```

## 📈 Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f pdf-processor
docker-compose logs -f qdrant
```

### Database Stats

Visit `http://localhost:6333/dashboard` for Qdrant's web interface.

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Improve documentation

## 📄 License

This project is open source and available under the MIT License. 