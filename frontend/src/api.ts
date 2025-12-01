export const API_BASE = 'http://localhost:8000'

export interface UploadResponse {
  document_id: string
  file_name: string
  num_chars: number
  num_chunks: number
  encoder_name: string
  chunk_size: number
  chunk_overlap: number
}

export interface ChunkSummary {
  chunk_id: number
  text: string
}

export interface CurrentDocumentResponse extends UploadResponse {}

export interface SearchResult {
  chunk_id: number
  score: number
  text: string
}

export interface SearchResponse {
  results: SearchResult[]
}

export interface StorageEntry {
  file_name: string
  size_bytes: number
  modified_at: string
}

export interface StorageUploadResponse {
  file_name: string
  size_bytes: number
  stored: boolean
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || 'Ошибка запроса')
  }
  return response.json()
}

export async function uploadAndIndexDocument(formData: FormData): Promise<UploadResponse> {
  const res = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData
  })
  return handleResponse<UploadResponse>(res)
}

export async function getCurrentDocument(): Promise<CurrentDocumentResponse> {
  const res = await fetch(`${API_BASE}/api/document/current`)
  return handleResponse<CurrentDocumentResponse>(res)
}

export async function getCurrentChunks(): Promise<ChunkSummary[]> {
  const res = await fetch(`${API_BASE}/api/document/current/chunks`)
  return handleResponse<ChunkSummary[]>(res)
}

export async function searchInDocument(query: string, topK: number): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK })
  })
  return handleResponse<SearchResponse>(res)
}

export async function listStorageDocuments(): Promise<StorageEntry[]> {
  const res = await fetch(`${API_BASE}/api/storage/list`)
  return handleResponse<StorageEntry[]>(res)
}

export async function uploadToStorage(formData: FormData): Promise<StorageUploadResponse> {
  const res = await fetch(`${API_BASE}/api/storage/upload`, {
    method: 'POST',
    body: formData
  })
  return handleResponse<StorageUploadResponse>(res)
}

export async function loadFromStorage(params: {
  file_name: string
  encoder_name: string
  chunk_size: number
  chunk_overlap: number
}): Promise<UploadResponse> {
  const res = await fetch(`${API_BASE}/api/storage/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  })
  return handleResponse<UploadResponse>(res)
}
