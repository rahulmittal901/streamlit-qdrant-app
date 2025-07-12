# Modified version of app.py to connect with Qdrant Docker container
# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming

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

import streamlit as st

# Qdrant connection settings
QDRANT_HOST = "localhost"  # Docker container host
QDRANT_PORT = 6333
API_URL = f"http://{QDRANT_HOST}:8000"  # PDF processor API

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

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

def upload_pdf_to_qdrant(file) -> str:
    """Upload PDF to Qdrant via our API"""
    try:
        # Prepare file for upload
        files = {'file': (file.name, file.getvalue(), 'application/pdf')}
        
        # Upload to our PDF processor API
        response = requests.post(f"{API_URL}/upload", files=files)
        response.raise_for_status()
        
        result = response.json()
        return result['document_id']
    except Exception as e:
        st.error(f"Error uploading to Qdrant: {e}")
        return None

def search_qdrant(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search documents in Qdrant"""
    try:
        response = requests.get(f"{API_URL}/search", params={
            'query': query,
            'limit': limit
        })
        response.raise_for_status()
        
        result = response.json()
        return result['results']
    except Exception as e:
        st.error(f"Error searching Qdrant: {e}")
        return []

def get_documents_from_qdrant() -> List[Dict[str, Any]]:
    """Get list of documents from Qdrant"""
    try:
        response = requests.get(f"{API_URL}/documents")
        response.raise_for_status()
        
        result = response.json()
        return result['documents']
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return []

def create_context_from_search_results(search_results: List[Dict[str, Any]]) -> str:
    """Create context string from search results"""
    if not search_results:
        return "No relevant information found."
    
    context_parts = []
    for result in search_results:
        context_parts.append(f"Document: {result['filename']}\nChunk {result['chunk_index'] + 1}:\n{result['text']}\n")
    
    return "\n".join(context_parts)

# Check if Qdrant services are running
def check_qdrant_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

with st.sidebar:
    st.header(f"Add your documents!")
    
    # API key input
    st.text_input("Groq API Key", type="password", key="groq_api_key")
    
    # Check Qdrant health
    if not check_qdrant_health():
        st.error("⚠️ Qdrant services are not running. Please start the Docker containers first:")
        st.code("docker-compose up -d")
        st.stop()
    else:
        st.success("✅ Qdrant services are running!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            st.write("Uploading and processing your document...")
            
            # Upload to Qdrant
            document_id = upload_pdf_to_qdrant(uploaded_file)
            
            if document_id:
                st.success(f"✅ Document uploaded successfully! ID: {document_id[:8]}...")
                
                # Store document info in session state
                if 'uploaded_documents' not in st.session_state:
                    st.session_state.uploaded_documents = []
                
                st.session_state.uploaded_documents.append({
                    'id': document_id,
                    'name': uploaded_file.name,
                    'file': uploaded_file
                })
                
                # Display PDF preview
                display_pdf(uploaded_file)
            else:
                st.error("Failed to upload document")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

    # Show uploaded documents
    if 'uploaded_documents' in st.session_state and st.session_state.uploaded_documents:
        st.subheader("📚 Uploaded Documents")
        for doc in st.session_state.uploaded_documents:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {doc['name']}")
            with col2:
                if st.button("🗑️", key=f"delete_{doc['id']}"):
                    # Delete document from Qdrant
                    try:
                        response = requests.delete(f"{API_URL}/documents/{doc['id']}")
                        if response.status_code == 200:
                            st.session_state.uploaded_documents.remove(doc)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting document: {e}")

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with Docs using Groq + Qdrant")

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
            # Search Qdrant for relevant context
            search_results = search_qdrant(prompt, limit=5)
            context = create_context_from_search_results(search_results)
            
            # Load LLM
            llm = load_llm()
            
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

# Add a section to show all documents in Qdrant
if st.sidebar.checkbox("Show all documents in database"):
    st.sidebar.subheader("📊 Database Contents")
    documents = get_documents_from_qdrant()
    
    if documents:
        for doc in documents:
            st.sidebar.write(f"📄 {doc['filename']}")
            st.sidebar.write(f"   Chunks: {doc['chunks_processed']}/{doc['total_chunks']}")
            st.sidebar.write("---")
    else:
        st.sidebar.write("No documents in database") 