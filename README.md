# Streamlit Qdrant App

This project provides a Streamlit UI for document chat powered by LlamaIndex embeddings and Qdrant vector database. It is containerized for easy deployment with Docker Compose and ready for use in GitHub Codespaces.

## Features
- Upload PDF documents and store embeddings in Qdrant
- Chat with your documents using LlamaIndex + Groq LLM
- All-in-one deployment with Docker Compose

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for local use)
- (Optional) [GitHub Codespaces](https://github.com/features/codespaces) for cloud development

## Quick Start (Docker Compose)

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd streamlit-qdrant-app
   ```
2. **Build and start the services:**
   ```bash
   docker compose up --build
   ```
3. **Access the app:**
   - Streamlit UI: [http://localhost:8501](http://localhost:8501)
   - Qdrant API: [http://localhost:6333](http://localhost:6333)

4. **Stop the services:**
   ```bash
   docker compose down -v
   ```

## Development in GitHub Codespaces

- This project includes a `.devcontainer` setup for Codespaces.
- On Codespaces start, Qdrant will run in Docker, and you can launch the Streamlit app with:
  ```bash
  streamlit run app_qdrant_api.py --server.port 8501 --server.address 0.0.0.0
  ```
- All dependencies are in `requirements.txt`.

## Environment Variables
- `QDRANT_HOST` and `QDRANT_PORT` are set in Docker Compose for service discovery.
- You will need a Groq API key to use the chat features (enter in the Streamlit sidebar).

## Troubleshooting
- **Docker not found:** Make sure Docker Desktop is running.
- **Port conflicts:** Ensure ports 8501 and 6333 are free.
- **Qdrant not running:** Check Docker logs for errors.

## License
MIT 