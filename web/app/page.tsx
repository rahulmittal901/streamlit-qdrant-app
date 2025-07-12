'use client'

import { useState, useEffect } from 'react'
import { Upload, Search, FileText, Trash2, Loader2 } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Document {
  document_id: string
  filename: string
  total_chunks: number
  chunks_processed: number
}

interface SearchResult {
  score: number
  document_id: string
  filename: string
  chunk_index: number
  text: string
  total_chunks: number
}

export default function Home() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [uploadProgress, setUploadProgress] = useState<string>('')

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf']
    },
    onDrop: handleFileDrop
  })

  async function handleFileDrop(acceptedFiles: File[]) {
    setIsUploading(true)
    setUploadProgress('')

    for (const file of acceptedFiles) {
      try {
        setUploadProgress(`Uploading ${file.name}...`)
        
        const formData = new FormData()
        formData.append('file', file)

        const response = await axios.post(`${API_URL}/upload`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        })

        setUploadProgress(`Processing ${file.name}...`)
        
        // Wait a bit for processing to complete
        await new Promise(resolve => setTimeout(resolve, 2000))
        
        // Refresh documents list
        await loadDocuments()
        
        setUploadProgress(`Successfully uploaded ${file.name}`)
      } catch (error) {
        console.error('Upload error:', error)
        setUploadProgress(`Error uploading ${file.name}`)
      }
    }

    setIsUploading(false)
    setTimeout(() => setUploadProgress(''), 3000)
  }

  async function loadDocuments() {
    try {
      const response = await axios.get(`${API_URL}/documents`)
      setDocuments(response.data.documents)
    } catch (error) {
      console.error('Error loading documents:', error)
    } finally {
      setIsLoading(false)
    }
  }

  async function searchDocuments() {
    if (!searchQuery.trim()) return

    setIsSearching(true)
    try {
      const response = await axios.get(`${API_URL}/search`, {
        params: {
          query: searchQuery,
          limit: 10
        }
      })
      setSearchResults(response.data.results)
    } catch (error) {
      console.error('Search error:', error)
    } finally {
      setIsSearching(false)
    }
  }

  async function deleteDocument(documentId: string) {
    try {
      await axios.delete(`${API_URL}/documents/${documentId}`)
      await loadDocuments()
    } catch (error) {
      console.error('Delete error:', error)
    }
  }

  useEffect(() => {
    loadDocuments()
  }, [])

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    searchDocuments()
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          PDF Vector Database
        </h1>
        <p className="text-gray-600">
          Upload PDF documents and search them using semantic similarity
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="card">
          <h2 className="text-2xl font-semibold mb-4 flex items-center">
            <Upload className="mr-2 h-6 w-6" />
            Upload PDF Documents
          </h2>
          
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-primary-400'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            {isDragActive ? (
              <p className="text-primary-600">Drop the PDF files here...</p>
            ) : (
              <div>
                <p className="text-gray-600 mb-2">
                  Drag & drop PDF files here, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  Only PDF files are supported
                </p>
              </div>
            )}
          </div>

          {isUploading && (
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <div className="flex items-center">
                <Loader2 className="animate-spin h-4 w-4 mr-2" />
                <span className="text-blue-700">{uploadProgress}</span>
              </div>
            </div>
          )}
        </div>

        {/* Search Section */}
        <div className="card">
          <h2 className="text-2xl font-semibold mb-4 flex items-center">
            <Search className="mr-2 h-6 w-6" />
            Search Documents
          </h2>
          
          <form onSubmit={handleSearch} className="mb-4">
            <div className="flex gap-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Enter your search query..."
                className="input-field flex-1"
                disabled={isSearching}
              />
              <button
                type="submit"
                className="btn-primary flex items-center"
                disabled={isSearching || !searchQuery.trim()}
              >
                {isSearching ? (
                  <Loader2 className="animate-spin h-4 w-4" />
                ) : (
                  <Search className="h-4 w-4" />
                )}
              </button>
            </div>
          </form>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="space-y-4">
              <h3 className="font-semibold text-gray-700">
                Search Results ({searchResults.length})
              </h3>
              {searchResults.map((result, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <h4 className="font-medium text-gray-900">
                        {result.filename}
                      </h4>
                      <p className="text-sm text-gray-500">
                        Chunk {result.chunk_index + 1} of {result.total_chunks}
                      </p>
                    </div>
                    <span className="text-sm bg-green-100 text-green-800 px-2 py-1 rounded">
                      {Math.round(result.score * 100)}% match
                    </span>
                  </div>
                  <p className="text-gray-700 text-sm line-clamp-3">
                    {result.text}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Documents List */}
      <div className="card mt-8">
        <h2 className="text-2xl font-semibold mb-4 flex items-center">
          <FileText className="mr-2 h-6 w-6" />
          Uploaded Documents
        </h2>
        
        {isLoading ? (
          <div className="flex justify-center py-8">
            <Loader2 className="animate-spin h-8 w-8 text-gray-400" />
          </div>
        ) : documents.length === 0 ? (
          <p className="text-gray-500 text-center py-8">
            No documents uploaded yet. Upload your first PDF to get started!
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 font-medium text-gray-700">
                    Filename
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">
                    Chunks
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">
                    Status
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.document_id} className="border-b border-gray-100">
                    <td className="py-3 px-4">
                      <div className="flex items-center">
                        <FileText className="h-4 w-4 text-gray-400 mr-2" />
                        {doc.filename}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-gray-600">
                      {doc.chunks_processed} / {doc.total_chunks}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        doc.chunks_processed === doc.total_chunks
                          ? 'bg-green-100 text-green-800'
                          : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {doc.chunks_processed === doc.total_chunks ? 'Complete' : 'Processing'}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <button
                        onClick={() => deleteDocument(doc.document_id)}
                        className="text-red-600 hover:text-red-800 transition-colors"
                        title="Delete document"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
} 