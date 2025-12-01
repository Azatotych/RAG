import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from razdel import sentenize

from models_config import get_encoder

app = FastAPI(title="RAG Prototype: Chunking & Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = Path(__file__).parent / "storage"


def ensure_storage_dir() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def split_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    sentences = [s.text.strip() for s in sentenize(text)]
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue

        separator = " " if current else ""
        if len(current) + len(separator) + len(sentence) <= chunk_size:
            current = f"{current}{separator}{sentence}" if current else sentence
            continue

        if current:
            chunks.append(current)
            overlap_tail = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = f"{overlap_tail} {sentence}".strip()
        else:
            chunks.append(sentence[:chunk_size])
            overlap_tail = sentence[:chunk_overlap] if chunk_overlap > 0 else ""
            remaining = sentence[chunk_size:]
            current = f"{overlap_tail} {remaining}".strip() if remaining else ""

    if current:
        chunks.append(current)

    return chunks


def build_embeddings(chunks: List[str], encoder_name: str) -> np.ndarray:
    model = get_encoder(encoder_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def list_storage_documents() -> List[Dict[str, object]]:
    ensure_storage_dir()
    documents: List[Dict[str, object]] = []
    for path in STORAGE_DIR.iterdir():
        if path.suffix.lower() != ".txt" or not path.is_file():
            continue
        stat = path.stat()
        documents.append(
            {
                "file_name": path.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )
    documents.sort(key=lambda x: x["file_name"])  # type: ignore[arg-type]
    return documents


def load_text_from_storage(file_name: str) -> str:
    ensure_storage_dir()
    path = STORAGE_DIR / file_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Файл не найден в хранилище")
    if path.suffix.lower() != ".txt":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Поддерживаются только .txt файлы")
    return path.read_text(encoding="utf-8")


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    num_chars: int
    num_chunks: int
    encoder_name: str
    chunk_size: int
    chunk_overlap: int


class ChunkSummary(BaseModel):
    chunk_id: int
    text: str


class CurrentDocumentResponse(BaseModel):
    document_id: str
    file_name: str
    num_chars: int
    num_chunks: int
    encoder_name: str
    chunk_size: int
    chunk_overlap: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1)


class SearchResult(BaseModel):
    chunk_id: int
    score: float
    text: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


class StorageUploadResponse(BaseModel):
    file_name: str
    size_bytes: int
    stored: bool


class StorageLoadRequest(BaseModel):
    file_name: str
    encoder_name: str = Field(default="mini")
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)


document_store: Dict[str, Optional[Dict[str, object]]] = {"current": None}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_and_index(
    file: UploadFile = File(...),
    encoder_name: str = "mini",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
):
    if encoder_name not in {"mini", "labse"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Неподдерживаемая модель")
    if file.content_type not in {"text/plain", None}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Поддерживаются только .txt файлы")

    ensure_storage_dir()
    file_name = file.filename or "uploaded.txt"
    if not file_name.lower().endswith(".txt"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Файл должен иметь расширение .txt")
    file_path = STORAGE_DIR / file_name

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    file_path.write_bytes(content)

    chunks = split_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Файл не содержит текста")

    embeddings = build_embeddings(chunks, encoder_name)
    document_id = str(uuid.uuid4())
    document_store["current"] = {
        "document_id": document_id,
        "file_name": file_name,
        "raw_text": text,
        "chunks": [{"chunk_id": idx, "text": chunk} for idx, chunk in enumerate(chunks)],
        "embeddings": embeddings,
        "encoder_name": encoder_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    return UploadResponse(
        document_id=document_id,
        file_name=file_name,
        num_chars=len(text),
        num_chunks=len(chunks),
        encoder_name=encoder_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_current_document() -> Dict[str, object]:
    current = document_store.get("current")
    if not current:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Документ не загружен")
    return current


@app.get("/api/document/current", response_model=CurrentDocumentResponse)
async def current_document() -> CurrentDocumentResponse:
    current = get_current_document()
    raw_text: str = current["raw_text"]  # type: ignore[index]
    chunks: List[Dict[str, object]] = current["chunks"]  # type: ignore[index]

    return CurrentDocumentResponse(
        document_id=current["document_id"],
        file_name=current["file_name"],
        num_chars=len(raw_text),
        num_chunks=len(chunks),
        encoder_name=current["encoder_name"],
        chunk_size=current["chunk_size"],
        chunk_overlap=current["chunk_overlap"],
    )


@app.get("/api/document/current/chunks", response_model=List[ChunkSummary])
async def current_document_chunks() -> List[ChunkSummary]:
    current = get_current_document()
    chunks: List[Dict[str, object]] = current["chunks"]  # type: ignore[index]
    return [ChunkSummary(chunk_id=c["chunk_id"], text=c["text"]) for c in chunks]


@app.post("/api/search", response_model=SearchResponse)
async def search_document(request: SearchRequest) -> SearchResponse:
    current = get_current_document()
    embeddings: np.ndarray = current["embeddings"]  # type: ignore[index]
    encoder_name: str = current["encoder_name"]  # type: ignore[index]
    chunks: List[Dict[str, object]] = current["chunks"]  # type: ignore[index]

    model = get_encoder(encoder_name)
    query_vector = model.encode(request.query, convert_to_numpy=True, normalize_embeddings=True)
    scores = embeddings @ query_vector

    top_k = min(request.top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [
        SearchResult(
            chunk_id=int(idx),
            score=float(scores[idx]),
            text=chunks[int(idx)]["text"],
        )
        for idx in top_indices
    ]

    return SearchResponse(results=results)


@app.get("/api/storage/list")
async def storage_list() -> List[Dict[str, object]]:
    return list_storage_documents()


@app.post("/api/storage/upload", response_model=StorageUploadResponse)
async def upload_to_storage(file: UploadFile = File(...)) -> StorageUploadResponse:
    if file.content_type not in {"text/plain", None}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Поддерживаются только .txt файлы")
    file_name = file.filename or "uploaded.txt"
    if not file_name.lower().endswith(".txt"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Файл должен иметь расширение .txt")

    ensure_storage_dir()
    content = await file.read()
    path = STORAGE_DIR / file_name
    path.write_bytes(content)
    return StorageUploadResponse(file_name=file_name, size_bytes=len(content), stored=True)


@app.post("/api/storage/load", response_model=UploadResponse)
async def load_from_storage(request: StorageLoadRequest) -> UploadResponse:
    if request.encoder_name not in {"mini", "labse"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Неподдерживаемая модель")
    text = load_text_from_storage(request.file_name)
    chunks = split_into_chunks(text, chunk_size=request.chunk_size, chunk_overlap=request.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Файл не содержит текста")

    embeddings = build_embeddings(chunks, request.encoder_name)
    document_id = str(uuid.uuid4())
    document_store["current"] = {
        "document_id": document_id,
        "file_name": request.file_name,
        "raw_text": text,
        "chunks": [{"chunk_id": idx, "text": chunk} for idx, chunk in enumerate(chunks)],
        "embeddings": embeddings,
        "encoder_name": request.encoder_name,
        "chunk_size": request.chunk_size,
        "chunk_overlap": request.chunk_overlap,
    }

    return UploadResponse(
        document_id=document_id,
        file_name=request.file_name,
        num_chars=len(text),
        num_chunks=len(chunks),
        encoder_name=request.encoder_name,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
