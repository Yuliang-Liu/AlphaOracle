# rag_api_server_query.py

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

from haystack.core.pipeline import Pipeline
from haystack.dataclasses import Document
from haystack.utils.device import ComponentDevice
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.rankers import SentenceTransformersSimilarityRanker

try:
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
except Exception:
    from qdrant_haystack import QdrantDocumentStore
    from qdrant_haystack.retriever import QdrantEmbeddingRetriever

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 模型
EMBED_MODEL = "Path/to/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Path/to/Qwen3-Reranker-0.6B"
EMBED_DIM = 1024
CUDA0 = ComponentDevice.from_str("cuda:0")

# Qdrant
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
USE_GRPC = False  # 固定 HTTP

SOURCES = []  # Your data List
DEFAULT_SOURCE = ""

def qdrant_store(collection: str) -> QdrantDocumentStore:
    for k in ["HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy"]:
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"

    try:
        return QdrantDocumentStore(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            index=collection,
            embedding_dim=EMBED_DIM,
            similarity="cosine",
            recreate_index=False,
            return_embedding=False,
            hnsw_config={"m": 32, "ef_construct": 256},
            prefer_grpc=USE_GRPC,
            timeout=60,
        )
    except TypeError:
        client = None
        return QdrantDocumentStore(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            index=collection,
            embedding_dim=EMBED_DIM,
            similarity="cosine",
            recreate_index=False,
            return_embedding=False,
            hnsw_config={"m": 32, "ef_construct": 256},
            prefer_grpc=USE_GRPC,
            timeout=60,
        )

def build_pipeline(source: str) -> Pipeline:
    store = qdrant_store(source)
    pipe = Pipeline()
    embedder_text = SentenceTransformersTextEmbedder(model=EMBED_MODEL, batch_size=1, device=CUDA0)
    retriever = QdrantEmbeddingRetriever(document_store=store, top_k=30)  # 候选 30
    ranker = SentenceTransformersSimilarityRanker(model=RERANK_MODEL, top_k=10, batch_size=1, device=CUDA0)

    pipe.add_component("embedder", embedder_text)
    pipe.add_component("retriever", retriever)
    pipe.add_component("ranker", ranker)
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "ranker.documents")
    embedder_text.warm_up(); ranker.warm_up()
    return pipe

app = FastAPI(title="RAG Query API – Qdrant Server")
PIPES: Dict[str, Pipeline] = {}

class QueryRequest(BaseModel):
    query: str
    source: str = DEFAULT_SOURCE
    top_k: int = 5

@app.on_event("startup")
def startup_event() -> None:
    pass

@app.post("/query", response_model=Dict[str, Any])
def query_docs(req: QueryRequest):
    if req.source not in SOURCES:
        raise HTTPException(400, f"未知 source '{req.source}'。有效值: {SOURCES}")

    if req.source not in PIPES:
        PIPES[req.source] = build_pipeline(req.source)

    pipe = PIPES[req.source]
    try:
        result = pipe.run({
            "embedder": {"text": req.query},
            "retriever": {"top_k": 30},
            "ranker": {"query": req.query, "top_k": req.top_k},
        })
        docs: List[Document] = result["ranker"]["documents"]

        def flat(d: Document) -> Dict[str, Any]:
            meta = d.meta or {}
            content_for_display = meta.get("classical_text", d.content)
            if meta.get("classical_text") is None and isinstance(content_for_display, str) and len(content_for_display) > 120:
                content_for_display = content_for_display[:120] + "…"
            return {
                "content": content_for_display,
                "retrieved_text": d.content,
                "score": float(getattr(d, "score", 0.0)),
                "meta": meta,
                "file_name": meta.get("file_name"),
                "page_id": meta.get("page_id"),
                "page_resolution": meta.get("page_resolution"),
                "input_image": meta.get("input_image"),
                "block_id": meta.get("block_id"),
                "block_label": meta.get("block_label"),
                "block_bbox": meta.get("block_bbox"),
                "block_bbox_norm": meta.get("block_bbox_norm"),
                "line_boxes": meta.get("line_boxes"),
            }

        return {"query": req.query, "source": req.source, "results": [flat(d) for d in docs]}
    except Exception as exc:
        print(f"❌ 查询出错: {exc}")
        raise HTTPException(500, str(exc))

if __name__ == "__main__":
    uvicorn.run("rag_api_server_query:app", host="0.0.0.0", port=8000, reload=False)
