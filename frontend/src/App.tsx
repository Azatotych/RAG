import React, { useEffect, useRef, useState } from 'react'
import {
  ChunkSummary,
  CurrentDocumentResponse,
  SearchResult,
  StorageEntry,
  UploadResponse,
  loadFromStorage,
  getCurrentChunks,
  getCurrentDocument,
  listStorageDocuments,
  searchInDocument,
  uploadAndIndexDocument,
  uploadToStorage
} from './api'

const formatDate = (iso: string) => new Date(iso).toLocaleString()

function App() {
  const [encoderName, setEncoderName] = useState<'mini' | 'labse'>('mini')
  const [chunkSize, setChunkSize] = useState<number>(800)
  const [chunkOverlap, setChunkOverlap] = useState<number>(200)
  const [currentDoc, setCurrentDoc] = useState<CurrentDocumentResponse | null>(null)
  const [chunks, setChunks] = useState<ChunkSummary[]>([])
  const [storage, setStorage] = useState<StorageEntry[]>([])
  const [query, setQuery] = useState<string>('')
  const [topK, setTopK] = useState<number>(3)
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [status, setStatus] = useState<string>('Документ не загружен')
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const uploadInputRef = useRef<HTMLInputElement | null>(null)
  const storageDropRef = useRef<HTMLDivElement | null>(null)

  const refreshStorage = async () => {
    try {
      const list = await listStorageDocuments()
      setStorage(list)
    } catch (err) {
      setStatus(`Ошибка: ${(err as Error).message}`)
    }
  }

  const refreshCurrentDocument = async () => {
    try {
      const info = await getCurrentDocument()
      setCurrentDoc(info)
      setStatus(`Текущий документ: ${info.file_name}, чанков: ${info.num_chunks}, модель: ${info.encoder_name}`)
    } catch (err) {
      setCurrentDoc(null)
      setStatus('Документ не загружен')
    }
  }

  const refreshChunks = async () => {
    try {
      const data = await getCurrentChunks()
      setChunks(data)
    } catch (err) {
      setChunks([])
      setStatus('Нет документа или не удалось загрузить чанки')
    }
  }

  useEffect(() => {
    refreshStorage()
    refreshCurrentDocument().then(() => refreshChunks())
  }, [])

  useEffect(() => {
    const dropZone = storageDropRef.current
    if (!dropZone) return

    const handleDrop = async (e: DragEvent) => {
      e.preventDefault()
      if (!e.dataTransfer || !e.dataTransfer.files.length) return
      const file = e.dataTransfer.files[0]
      const formData = new FormData()
      formData.append('file', file)
      setStatus('Загрузка файла в хранилище...')
      try {
        await uploadToStorage(formData)
        setStatus('Файл добавлен в хранилище')
        await refreshStorage()
      } catch (err) {
        setStatus(`Ошибка: ${(err as Error).message}`)
      }
    }

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault()
    }

    dropZone.addEventListener('drop', handleDrop)
    dropZone.addEventListener('dragover', handleDragOver)

    return () => {
      dropZone.removeEventListener('drop', handleDrop)
      dropZone.removeEventListener('dragover', handleDragOver)
    }
  }, [])

  const handleUploadAndIndex = async () => {
    if (!uploadInputRef.current || !uploadInputRef.current.files || !uploadInputRef.current.files[0]) {
      setStatus('Выберите .txt файл для загрузки')
      return
    }
    const formData = new FormData()
    formData.append('file', uploadInputRef.current.files[0])
    formData.append('encoder_name', encoderName)
    formData.append('chunk_size', String(chunkSize))
    formData.append('chunk_overlap', String(chunkOverlap))
    setIsLoading(true)
    setStatus('Загрузка и индексация...')
    try {
      const resp: UploadResponse = await uploadAndIndexDocument(formData)
      setCurrentDoc(resp)
      await refreshChunks()
      setStatus(`Текущий документ: ${resp.file_name}, чанков: ${resp.num_chunks}, модель: ${resp.encoder_name}`)
    } catch (err) {
      setStatus(`Ошибка: ${(err as Error).message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSearch = async () => {
    if (!query.trim()) {
      setStatus('Введите запрос для поиска')
      return
    }
    setIsLoading(true)
    setStatus('Идёт поиск...')
    try {
      const res = await searchInDocument(query, topK)
      setSearchResults(res.results)
      setStatus(`Найдено результатов: ${res.results.length}`)
    } catch (err) {
      setStatus(`Ошибка: ${(err as Error).message}`)
      setSearchResults([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleStorageUploadClick = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.txt'
    input.onchange = async () => {
      if (!input.files || !input.files[0]) return
      const formData = new FormData()
      formData.append('file', input.files[0])
      setStatus('Загрузка файла в хранилище...')
      try {
        await uploadToStorage(formData)
        await refreshStorage()
        setStatus('Файл добавлен в хранилище')
      } catch (err) {
        setStatus(`Ошибка: ${(err as Error).message}`)
      }
    }
    input.click()
  }

  const handleMakeCurrent = async (fileName: string) => {
    setIsLoading(true)
    setStatus('Загрузка из хранилища...')
    try {
      const resp = await loadFromStorage({
        file_name: fileName,
        encoder_name: encoderName,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap
      })
      setCurrentDoc(resp)
      await refreshChunks()
      setStatus(`Текущий документ: ${resp.file_name}, чанков: ${resp.num_chunks}, модель: ${resp.encoder_name}`)
    } catch (err) {
      setStatus(`Ошибка: ${(err as Error).message}`)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <div>
          <h1>RAG Prototype: Chunking & Search</h1>
          <p>Этап 1 — один активный документ, библиотека .txt в хранилище</p>
        </div>
        <div className="encoder-select">
          <label>
            <input
              type="radio"
              name="encoder"
              value="mini"
              checked={encoderName === 'mini'}
              onChange={() => setEncoderName('mini')}
            />
            mini (rubert-mini-sts)
          </label>
          <label>
            <input
              type="radio"
              name="encoder"
              value="labse"
              checked={encoderName === 'labse'}
              onChange={() => setEncoderName('labse')}
            />
            labse (LaBSE-ru-sts)
          </label>
        </div>
      </header>

      <main className="main-grid">
        <section className="column">
          <h2>Документы</h2>

          <div className="card">
            <div className="card-header">
              <h3>Хранилище документов (.txt)</h3>
              <button onClick={refreshStorage}>Обновить список</button>
            </div>
            {storage.length === 0 ? (
              <p>В хранилище пока нет документов</p>
            ) : (
              <ul className="storage-list">
                {storage.map((doc) => (
                  <li key={doc.file_name}>
                    <div>
                      <strong>{doc.file_name}</strong>
                      <div className="meta">
                        {Math.round(doc.size_bytes / 1024)} КБ — {formatDate(doc.modified_at)}
                      </div>
                    </div>
                    <button onClick={() => handleMakeCurrent(doc.file_name)}>Сделать текущим</button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="card">
            <h3>Добавить документ в хранилище</h3>
            <div ref={storageDropRef} className="drop-zone">
              <p>Перетащите сюда .txt файл или нажмите для выбора</p>
              <button onClick={handleStorageUploadClick}>Выбрать файл</button>
            </div>
          </div>
        </section>

        <section className="column">
          <h2>Текущий документ и поиск</h2>

          <div className="card">
            <h3>Параметры разбиения и индексации</h3>
            <div className="form-row">
              <label>
                chunk_size
                <input type="number" value={chunkSize} onChange={(e) => setChunkSize(Number(e.target.value))} />
              </label>
              <label>
                chunk_overlap
                <input type="number" value={chunkOverlap} onChange={(e) => setChunkOverlap(Number(e.target.value))} />
              </label>
            </div>
            <div className="form-row">
              <input ref={uploadInputRef} type="file" accept=".txt" />
              <button onClick={handleUploadAndIndex} disabled={isLoading}>
                Загрузить и разбить новый .txt
              </button>
            </div>
            <div className="status">
              {currentDoc
                ? `Текущий документ: ${currentDoc.file_name}, чанков: ${currentDoc.num_chunks}, модель: ${currentDoc.encoder_name}`
                : 'Документ не загружен'}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h3>Просмотр чанков</h3>
              <button onClick={refreshChunks}>Обновить чанки</button>
            </div>
            <div className="chunk-list">
              {chunks.map((chunk) => (
                <div key={chunk.chunk_id} className="chunk-item">
                  <div className="chunk-title">Chunk #{chunk.chunk_id} (длина: {chunk.text.length} символов)</div>
                  <pre>{chunk.text}</pre>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h3>Поиск по документу</h3>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Введите запрос"
              rows={3}
            />
            <div className="form-row">
              <label>
                top_k
                <input type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
              </label>
              <button onClick={handleSearch} disabled={isLoading}>
                Искать
              </button>
            </div>
            <div className="results">
              {searchResults.map((res, idx) => (
                <div key={res.chunk_id} className="result-item">
                  <div className="result-header">
                    <strong>#{idx + 1}</strong> — chunk {res.chunk_id} — score {res.score.toFixed(4)}
                  </div>
                  <pre>{res.text}</pre>
                  <button onClick={() => navigator.clipboard.writeText(res.text)}>Копировать текст чанка</button>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">{status}</footer>
    </div>
  )
}

export default App
