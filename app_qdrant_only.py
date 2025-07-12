# Modified version of app.py to use LlamaIndex + Qdrant (no PDF processor)
# Uses LlamaIndex for embeddings but stores in Qdrant for persistence

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

from IPython.display import Markdown, display

from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from qdrant_client import QdrantClient

import streamlit as st

# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
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
def get_qdrant_client():
    """Get Qdrant client connection"""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Test connection
        client.get_collections()
        return client
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
        return None

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

def process_pdf_with_llamaindex(file) -> VectorStoreIndex:
    """Process PDF using LlamaIndex and store in Qdrant"""
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
            
            # Setup embedding model (using the better one from your original app)
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-large-en-v1.5", 
                trust_remote_code=True
            )
            Settings.embed_model = embed_model
            
            # Setup Qdrant vector store
            qdrant_client = get_qdrant_client()
            if not qdrant_client:
                raise Exception("Qdrant connection failed")
            
            # Create unique collection name for this document
            collection_name = f"{COLLECTION_NAME}_{file.name.replace('.pdf', '').replace(' ', '_')}"
            
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index with Qdrant storage
            index = VectorStoreIndex.from_documents(
                docs, 
                storage_context=storage_context,
                show_progress=True
            )
            
            return index
            
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def get_all_collections():
    """Get all document collections from Qdrant"""
    try:
        qdrant_client = get_qdrant_client()
        if not qdrant_client:
            return []
        
        collections = qdrant_client.get_collections()
        # Filter collections that start with our prefix
        doc_collections = [
            col.name for col in collections.collections 
            if col.name.startswith(COLLECTION_NAME)
        ]
        return doc_collections
    except Exception as e:
        st.error(f"Error getting collections: {e}")
        return []

def delete_collection(collection_name: str):
    """Delete a collection from Qdrant"""
    try:
        qdrant_client = get_qdrant_client()
        if qdrant_client:
            qdrant_client.delete_collection(collection_name)
            return True
    except Exception as e:
        st.error(f"Error deleting collection: {e}")
        return False

def check_qdrant_health():
    """Check if Qdrant is running"""
    try:
        qdrant_client = get_qdrant_client()
        if qdrant_client:
            qdrant_client.get_collections()
            return True
        return False
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
            st.write("Processing your document with LlamaIndex...")
            
            # Process PDF with LlamaIndex and store in Qdrant
            index = process_pdf_with_llamaindex(uploaded_file)
            
            if index:
                st.success(f"✅ Document processed and stored in Qdrant!")
                
                # Store index in session state
                if 'document_indexes' not in st.session_state:
                    st.session_state.document_indexes = {}
                
                collection_name = f"{COLLECTION_NAME}_{uploaded_file.name.replace('.pdf', '').replace(' ', '_')}"
                st.session_state.document_indexes[collection_name] = index
                
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
                    if delete_collection(collection):
                        st.success("Document deleted!")
                        st.rerun()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with Docs using Groq + LlamaIndex + Qdrant")

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
            # Get all available indexes
            indexes = st.session_state.get('document_indexes', {})
            
            if not indexes:
                full_response = "No documents have been uploaded yet. Please upload a PDF document first."
            else:
                # Load LLM
                llm = load_llm()
                Settings.llm = llm
                
                # Create query engines for all documents
                query_engines = []
                for collection_name, index in indexes.items():
                    query_engine = index.as_query_engine(streaming=True)
                    query_engines.append(query_engine)
                
                # Query all documents and combine results
                all_responses = []
                for i, query_engine in enumerate(query_engines):
                    try:
                        response = query_engine.query(prompt)
                        all_responses.append(response.response)
                    except Exception as e:
                        st.warning(f"Error querying document {i+1}: {e}")
                
                if all_responses:
                    # Combine responses
                    combined_response = "\n\n".join(all_responses)
                    full_response = combined_response
                else:
                    full_response = "I couldn't find relevant information in the uploaded documents."
            
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