#!/usr/bin/env python3
"""
Test script to verify connection between Streamlit app and Qdrant services
"""

import requests
import json
import time

def test_qdrant_health():
    """Test if Qdrant is running"""
    try:
        response = requests.get("http://localhost:6333/health", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running")
            return True
        else:
            print(f"❌ Qdrant health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant: {e}")
        return False

def test_pdf_processor_health():
    """Test if PDF processor API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ PDF Processor API is running")
            return True
        else:
            print(f"❌ PDF Processor health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to PDF Processor: {e}")
        return False

def test_documents_endpoint():
    """Test the documents endpoint"""
    try:
        response = requests.get("http://localhost:8000/documents", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Documents endpoint working. Found {len(data.get('documents', []))} documents")
            return True
        else:
            print(f"❌ Documents endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot access documents endpoint: {e}")
        return False

def main():
    print("🔍 Testing Qdrant + Streamlit Connection")
    print("=" * 50)
    
    # Test Qdrant
    qdrant_ok = test_qdrant_health()
    
    # Test PDF Processor
    processor_ok = test_pdf_processor_health()
    
    # Test documents endpoint
    documents_ok = test_documents_endpoint()
    
    print("\n" + "=" * 50)
    
    if qdrant_ok and processor_ok and documents_ok:
        print("🎉 All services are running correctly!")
        print("You can now run: streamlit run app_with_qdrant.py")
    else:
        print("⚠️  Some services are not running properly.")
        print("Please run: docker-compose up -d")
        print("Then wait a few seconds and run this test again.")

if __name__ == "__main__":
    main() 