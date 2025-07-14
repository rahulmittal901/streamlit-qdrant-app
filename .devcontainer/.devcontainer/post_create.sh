#!/bin/bash
set -e

# Install Python dependencies for Streamlit app
pip install --upgrade pip
pip install -r requirements.qdrant_api.txt

# Start Qdrant Docker container (if not already running)
if [ -z "$(docker ps -q -f name=qdrant)" ]; then
  docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
fi

# Optionally, start Streamlit app in the background (uncomment if desired)
# nohup streamlit run app_qdrant_api.py --server.port 8501 --server.address 0.0.0.0 &

echo "Setup complete. Qdrant running on 6333, Streamlit can be started with:"
echo "streamlit run app_qdrant_api.py --server.port 8501 --server.address 0.0.0.0"