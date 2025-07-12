# Modified version that uses LlamaIndex for embeddings but Qdrant via API
# Best of both worlds: LlamaIndex embeddings + Qdrant API

import os
import base64
import gc
import random
import tempfile
import time
import uuid
import requests
import json
from typing import List, Dict, Any



from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.core import Document

import streamlit as st

# Qdrant API settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_API_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
COLLECTION_NAME = "pdf_documents"

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

@st.cache_resource
def load_llm():
    api_key = st.session_state.get("groq_api_key")
    if not api_key:
        st.warning("Please add your Groq API key in the sidebar to continue.")
        st.stop()
    llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        request_timeout=120.0
    )
    return llm

@st.cache_resource
def load_embedding_model():
    """Load the embedding model using LlamaIndex"""
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5", 
        trust_remote_code=True
    )
    return embed_model

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""
        <div style="width:100%; height:600px; border: 1px solid #ddd; border-radius: 5px; overflow: hidden;">
            <iframe 
                src="data:application/pdf;base64,{base64_pdf}#toolbar=1&navpanes=1&scrollbar=1&page=1&view=FitH" 
                width="100%" 
                height="100%" 
                type="application/pdf"
                style="border: none;">
                <p>Your browser does not support PDFs. 
                <a href="data:application/pdf;base64,{base64_pdf}" target="_blank">Download the PDF</a>
                </p>
            </iframe>
        </div>
        """
    st.markdown(pdf_display, unsafe_allow_html=True)

def ensure_collection_exists(collection_name: str, vector_size: int = 1024):
    """Ensure Qdrant collection exists via API"""
    try:
        # Check if collection exists
        response = requests.get(f"{QDRANT_API_URL}/collections/{collection_name}")
        if response.status_code == 200:
            return True
        
        # Create collection if it doesn't exist
        collection_config = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine"
            }
        }
        
        response = requests.put(
            f"{QDRANT_API_URL}/collections/{collection_name}",
            json=collection_config
        )
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"Failed to create collection: {response.text}")
            return False
            
    except Exception as e:
        st.error(f"Error ensuring collection exists: {e}")
        return False

def process_pdf_with_llamaindex_and_qdrant_api(file) -> bool:
    """Process PDF using LlamaIndex embeddings and store in Qdrant via API"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.name)
            
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            
            # Load documents using LlamaIndex
            loader = SimpleDirectoryReader(
                input_dir=temp_dir,
                required_exts=[".pdf"],
                recursive=True
            )
            docs = loader.load_data()
            
            # Get embedding model
            embed_model = load_embedding_model()
            
            # Create unique collection name for this document
            collection_name = f"{COLLECTION_NAME}_{file.name.replace('.pdf', '').replace(' ', '_')}"
            
            # Ensure collection exists
            if not ensure_collection_exists(collection_name, vector_size=1024):
                return False
            
            # Process each document chunk
            points = []
            for i, doc in enumerate(docs):
                # Get text content
                text = doc.text
                
                # Create embedding using LlamaIndex
                embedding = embed_model.get_text_embedding(text)
                
                # Create point for Qdrant with UUID
                point = {
                    "id": str(uuid.uuid4()),
                    "vector": embedding,
                    "payload": {
                        "text": text,
                        "document_id": collection_name,
                        "filename": file.name,
                        "chunk_index": i,
                        "total_chunks": len(docs)
                    }
                }
                points.append(point)
            
            # Upload points to Qdrant via API
            response = requests.put(
                f"{QDRANT_API_URL}/collections/{collection_name}/points",
                json={"points": points}
            )
            
            if response.status_code == 200:
                st.success(f"✅ Uploaded {len(points)} chunks to Qdrant")
                return True
            else:
                st.error(f"Failed to upload to Qdrant: {response.text}")
                return False
                
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return False

def search_qdrant_api(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search documents in Qdrant via API"""
    try:
        # Get embedding for query using LlamaIndex
        embed_model = load_embedding_model()
        query_embedding = embed_model.get_text_embedding(query)
        
        # Get all collections
        collections = get_all_collections()
        
        if not collections:
            st.warning("No collections found in Qdrant")
            return []
        
        st.info(f"🔍 Searching across {len(collections)} collections: {', '.join(collections)}")
        
        all_results = []
        
        for collection in collections:
            # Search in each collection
            search_payload = {
                "vector": query_embedding,
                "limit": limit,
                "with_payload": True
            }
            
            response = requests.post(
                f"{QDRANT_API_URL}/collections/{collection}/points/search",
                json=search_payload
            )
            
            if response.status_code == 200:
                results = response.json()["result"]
                st.info(f"📄 Found {len(results)} results in collection: {collection}")
                
                for result in results:
                    all_results.append({
                        "score": result["score"],
                        "document_id": result["payload"]["document_id"],
                        "filename": result["payload"]["filename"],
                        "chunk_index": result["payload"]["chunk_index"],
                        "text": result["payload"]["text"],
                        "total_chunks": result["payload"]["total_chunks"]
                    })
            else:
                st.error(f"Failed to search collection {collection}: {response.text}")
        
        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        st.info(f"🎯 Total results found: {len(all_results)}")
        
        # Ensure we get results from multiple documents
        if len(all_results) > limit:
            # Group results by document
            doc_results = {}
            for result in all_results:
                filename = result['filename']
                if filename not in doc_results:
                    doc_results[filename] = []
                doc_results[filename].append(result)
            
            # Take top results from each document
            balanced_results = []
            results_per_doc = max(1, limit // len(doc_results))
            
            for filename, results in doc_results.items():
                balanced_results.extend(results[:results_per_doc])
            
            # Sort by score again and return
            balanced_results.sort(key=lambda x: x["score"], reverse=True)
            st.info(f"⚖️ Balanced results from {len(doc_results)} documents: {len(balanced_results)} total")
            return balanced_results[:limit]
        
        return all_results[:limit]
        
    except Exception as e:
        st.error(f"Error searching Qdrant: {e}")
        return []

def get_all_collections() -> List[str]:
    """Get all document collections from Qdrant via API"""
    try:
        response = requests.get(f"{QDRANT_API_URL}/collections")
        if response.status_code == 200:
            collections = response.json()["result"]["collections"]
            # Filter collections that start with our prefix
            doc_collections = [
                col["name"] for col in collections 
                if col["name"].startswith(COLLECTION_NAME)
            ]
            return doc_collections
        else:
            st.error(f"Failed to get collections: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error getting collections: {e}")
        return []

def delete_collection_api(collection_name: str) -> bool:
    """Delete a collection from Qdrant via API"""
    try:
        response = requests.delete(f"{QDRANT_API_URL}/collections/{collection_name}")
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error deleting collection: {e}")
        return False

def check_qdrant_health():
    """Check if Qdrant is running via API"""
    try:
        response = requests.get(f"{QDRANT_API_URL}/collections", timeout=5)
        return response.status_code == 200
    except:
        return False

with st.sidebar:
    st.header(f"Add your documents!")
    
    # API key input
    st.text_input("Groq API Key", type="password", key="groq_api_key")
    
    # Check Qdrant health
    if not check_qdrant_health():
        st.error("⚠️ Qdrant is not running. Please start the Docker container first:")
        st.code("docker run -d -p 6333:6333 qdrant/qdrant:latest")
        st.stop()
    else:
        st.success("✅ Qdrant is running!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            st.write("Processing your document with LlamaIndex embeddings...")
            
            # Process PDF with LlamaIndex and store in Qdrant via API
            success = process_pdf_with_llamaindex_and_qdrant_api(uploaded_file)
            
            if success:
                st.success(f"✅ Document processed and stored in Qdrant!")
                
                # Display PDF preview
                display_pdf(uploaded_file)
            else:
                st.error("Failed to process document")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

    # Show stored documents
    collections = get_all_collections()
    if collections:
        st.subheader("📚 Stored Documents")
        for collection in collections:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Extract filename from collection name
                filename = collection.replace(f"{COLLECTION_NAME}_", "").replace("_", " ") + ".pdf"
                st.write(f"📄 {filename}")
            with col2:
                if st.button("🗑️", key=f"delete_{collection}"):
                    if delete_collection_api(collection):
                        st.success("Document deleted!")
                        st.rerun()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with Docs using Groq + LlamaIndex + Qdrant API")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Block chat if API key is not set
if not st.session_state.get("groq_api_key"):
    st.info("Please add your Groq API key in the sidebar to continue.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Search Qdrant via API
            search_results = search_qdrant_api(prompt, limit=5)
            
            if not search_results:
                full_response = "No documents have been uploaded yet or no relevant information found."
            else:
                # Debug: Show what documents were found
                st.info(f"🔍 Found {len(search_results)} relevant chunks from search")
                
                # Group results by document
                docs_found = set()
                for result in search_results:
                    docs_found.add(result['filename'])
                st.info(f"📚 Documents found: {', '.join(docs_found)}")
                
                # Load LLM
                llm = load_llm()
                
                # Create context from search results
                context_parts = []
                for result in search_results:
                    context_parts.append(f"Document: {result['filename']}\nChunk {result['chunk_index'] + 1}:\n{result['text']}\n")
                
                context = "\n".join(context_parts)
                
                # Create prompt with context
                qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                )
                
                # Format the prompt
                formatted_prompt = qa_prompt_tmpl_str.format(
                    context_str=context,
                    query_str=prompt
                )
                
                # Get response from LLM
                response = llm.complete(formatted_prompt)
                full_response = response.text
                
                # Show sources if available
                if search_results:
                    full_response += "\n\n**Sources:**\n"
                    for i, result in enumerate(search_results[:3], 1):
                        full_response += f"{i}. {result['filename']} (Chunk {result['chunk_index'] + 1})\n"
            
        except Exception as e:
            full_response = f"Sorry, I encountered an error: {str(e)}"
        
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add database info
if st.sidebar.checkbox("Show database info"):
    st.sidebar.subheader("📊 Qdrant Database Info")
    collections = get_all_collections()
    st.sidebar.write(f"Total documents: {len(collections)}")
    
    if collections:
        for collection in collections:
            st.sidebar.write(f"• {collection}")
    else:
        st.sidebar.write("No documents in database") 