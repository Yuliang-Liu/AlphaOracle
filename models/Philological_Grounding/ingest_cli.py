# ingest_cli.py

from __future__ import annotations

import argparse
import json
import os
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from tqdm import tqdm

# Haystack 2.x
from haystack.dataclasses import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.utils.device import ComponentDevice

# Qdrant（检索存在性用 client、写入用 DocumentStore）
try:
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
except Exception:
    from qdrant_haystack import QdrantDocumentStore
from qdrant_client import QdrantClient

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========= 常量（可按机器调）=========
FILE_CHUNK   = 500   # 每批处理的文件数；可按机器改 500~5000

# ========= 常量（可按硬件调优）=========
EMBED_MODEL   = "Path/to/Qwen3-Embedding-0.6B"   # 你的本地模型（保持不变）
EMBED_DIM     = 1024
CUDA0         = ComponentDevice.from_str("cuda:0")  # 如需 CPU 可改 "cpu"

EMBED_BATCH   = 4     # 模型内部 batch_size（显存足够可提到 256）
OUTER_CHUNK   = 4000    # 外层一次 run 的文档条数（可提到 5000）
UPSERT_BATCH  = 2000    # Qdrant upsert 每批条数

# ========= Qdrant Server =========
QDRANT_URL    = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY= os.environ.get("QDRANT_API_KEY", None)
USE_GRPC      = False   # 固定 HTTP，避免被系统代理劫持到 7890

# ========= 数据源路径 =========
JSON_FOLDER       = '***'
JSON_SOURCE_NAME  = '***'

JSON_FOLDER_1     = '***'
JSON_SOURCE_NAME_1= '***'

TXT_FOLDER_DEFAULT= '***'
TXT_SOURCE_NAME   = '***'

TXT_FOLDER_SHUOWEN= '***'
TXT_SOURCE_NAME_SHUOWEN = '***'


# ========= 工具函数 =========
def stable_id_uuid5(*parts: str) -> str:
    """确定性 UUIDv5（Qdrant 允许的 point id 类型，且稳定可复现）"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(parts)))

def content_sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def qdrant_client() -> QdrantClient:
    # 清理代理（避免走 7890）
    for k in ["HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy"]:
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=USE_GRPC, timeout=60)

def qdrant_store(collection: str) -> QdrantDocumentStore:
    client = qdrant_client()
    try:
        return QdrantDocumentStore(
            client=client,
            index=collection,
            embedding_dim=EMBED_DIM,
            similarity="cosine",
            recreate_index=False,
            return_embedding=False,
            hnsw_config={"m": 32, "ef_construct": 256},
        )
    except TypeError:
        # 兼容旧版 qdrant-haystack
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

def chunked(iterable, size):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


# ========= Splitters =========
@dataclass
class SourceConfig:
    name: str
    folder: Path
    suffix: str
    splitter: Callable[[List[Path], str], List[Document]]

def splitter_json_blocks(files: List[Path], src_name: str) -> List[Document]:
    """
    只索引：block_label == 'text' 且 文本长度 > 15 的块
    并保留位置信息等 meta 字段。
    """
    ALLOWED_LABELS = {"text"}
    MIN_TEXT_LEN = 15

    docs: List[Document] = []
    kept, skipped_short, skipped_label = 0, 0, 0

    for fp in files:
        try:
            pages = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            print(f"⚠️ 跳过损坏 JSON：{fp}")
            continue
        if not isinstance(pages, list):
            continue

        for page in pages:
            if not isinstance(page, dict):
                continue
            page_id = page.get("page_id")
            page_res = page.get("page_resolution")
            input_image = page.get("input_image")
            results = page.get("result", [])
            width = height = None
            if isinstance(page_res, dict):
                try:
                    width = int(page_res.get("width") or 0) or None
                    height = int(page_res.get("height") or 0) or None
                except Exception:
                    width = height = None

            for block_id, block in enumerate(results if isinstance(results, list) else []):
                if not isinstance(block, dict):
                    continue

                label = str(block.get("block_label") or "").strip().lower()
                if label not in ALLOWED_LABELS:
                    skipped_label += 1
                    continue

                text = (block.get("block_content") or "").strip()
                if len(text) <= MIN_TEXT_LEN:
                    skipped_short += 1
                    continue

                block_bbox = block.get("block_bbox")
                line_boxes = block.get("line_boxes", [])
                block_bbox_norm = None
                if (
                    isinstance(block_bbox, (list, tuple)) and len(block_bbox) == 4
                    and isinstance(width, int) and width and isinstance(height, int) and height
                ):
                    try:
                        x1, y1, x2, y2 = map(float, block_bbox)
                        block_bbox_norm = [x1/width, y1/height, x2/width, y2/height]
                    except Exception:
                        block_bbox_norm = None

                meta = {
                    "source": src_name,
                    "file_name": fp.name,
                    "page_id": page_id,
                    "page_resolution": page_res,
                    "input_image": input_image,
                    "block_id": block_id,
                    "block_label": label,
                    "block_bbox": block_bbox,
                    "block_bbox_norm": block_bbox_norm,
                    "line_boxes": line_boxes,
                }
                # 统一用 UUIDv5，满足 Qdrant 要求
                doc_id = stable_id_uuid5(src_name, fp.name, str(page_id or ""), str(block_id))
                docs.append(Document(id=doc_id, content=text, meta=meta))
                kept += 1

    print(f"📊 splitter_json_blocks: kept={kept}, skipped_label={skipped_label}, skipped_short={skipped_short}")
    return docs

def splitter_default(files: List[Path], src_name: str) -> List[Document]:
    docs, seen_modern = [], set()
    for fp in files:
        try:
            raw = fp.read_text(encoding="utf-8")
        except Exception:
            print(f"⚠️ 跳过损坏 TXT：{fp}")
            continue
        for blk in raw.strip().split("\n\n"):
            lines = [ln.strip() for ln in blk.split("\n") if ln.strip()]
            if len(lines) == 2 and lines[0].startswith("古文：") and lines[1].startswith("现代文："):
                klass = lines[0][3:].strip()
                modern = lines[1][4:].strip()
                if modern and modern not in seen_modern:
                    seen_modern.add(modern)
                    meta = {"source": src_name, "classical_text": klass, "file_name": fp.name}
                    # 用内容 hash 做 ID 的组成部分，但最终仍生成 UUIDv5
                    doc_id = stable_id_uuid5(src_name, fp.name, content_sha1(modern))
                    docs.append(Document(id=doc_id, content=modern, meta=meta))
    return docs

def splitter_shuowen(files: List[Path], src_name: str) -> List[Document]:
    """
    整文件写入（如“说文解字字头”），ID 用 UUIDv5。
    """
    docs = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8").strip()
        except Exception:
            print(f"⚠️ 跳过损坏 TXT：{fp}")
            continue
        if text:
            meta = {"source": src_name, "entry": fp.stem, "file_name": fp.name}
            doc_id = stable_id_uuid5(src_name, fp.name)
            docs.append(Document(id=doc_id, content=text, meta=meta))
    return docs

SOURCES: Dict[str, SourceConfig] = {
    JSON_SOURCE_NAME_1:      SourceConfig(JSON_SOURCE_NAME_1,      JSON_FOLDER_1,      ".json", splitter_json_blocks),
    JSON_SOURCE_NAME:        SourceConfig(JSON_SOURCE_NAME,        JSON_FOLDER,        ".json", splitter_json_blocks),
    TXT_SOURCE_NAME:         SourceConfig(TXT_SOURCE_NAME,         TXT_FOLDER_DEFAULT, ".txt",  splitter_default),
    TXT_SOURCE_NAME_SHUOWEN: SourceConfig(TXT_SOURCE_NAME_SHUOWEN, TXT_FOLDER_SHUOWEN, ".txt",  splitter_shuowen),
}


# ========= 对账（与 Server 比较）：找出 MISSING / CHANGED =========
def diff_against_server(collection: str, docs: List[Document]) -> Tuple[List[Document], List[Document], int]:
    """
    返回：(missing_docs, changed_docs, same_count)
    依据：id 是否存在；存在则比较 payload.content_hash 是否一致
    """
    client = qdrant_client()
    id_to_doc = {d.id: d for d in docs}

    # 预先生成 content_hash 放入 meta（payload）
    for d in docs:
        if d.meta is None:
            d.meta = {}
        d.meta["content_hash"] = content_sha1(d.content)

    missing, changed = [], []
    same_cnt = 0

    for batch_ids in tqdm(list(chunked(id_to_doc.keys(), 10000)), desc="对账:retrieve", unit="batch"):
        try:
            points = client.retrieve(collection_name=collection, ids=batch_ids, with_payload=True, with_vectors=False)
        except Exception as e:
            print(f"⚠️ retrieve 批失败：{e}")
            points = []

        exist_ids = set()
        for p in points:
            pid = str(p.id)
            exist_ids.add(pid)
            old_hash = (p.payload or {}).get("content_hash")
            new_hash = id_to_doc[pid].meta.get("content_hash")
            if old_hash == new_hash:
                same_cnt += 1
            else:
                changed.append(id_to_doc[pid])

        for pid in batch_ids:
            if pid not in exist_ids:
                missing.append(id_to_doc[pid])
    return missing, changed, same_cnt


# ========= 仅计算需要的 embedding（不使用 pkl）=========
def ensure_embeddings(source: str, docs_need: List[Document]) -> List[Document]:
    """
    Server-only 模式：
    - 不再读取/写回 pkl
    - 仅对 missing/changed 文档计算 embedding 后返回
    """
    if not docs_need:
        return []

    # 可选：开启 TF32（Ampere+ 明显提速）
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    embedder = SentenceTransformersDocumentEmbedder(
        model=EMBED_MODEL,
        batch_size=EMBED_BATCH,
        device=CUDA0
    )
    embedder.warm_up()
    print(f"🧮 需要新计算向量：{len(docs_need)}")

    embedded_docs: List[Document] = []
    for batch in tqdm(list(chunked(docs_need, OUTER_CHUNK)), desc="embed", unit="batch"):
        res = embedder.run(documents=batch)
        embedded_docs.extend(res.get("documents", batch))
    return embedded_docs


# ========= upsert 写入 =========
def upsert_documents(collection: str, docs: List[Document], batch_size: int = UPSERT_BATCH) -> int:
    store = qdrant_store(collection)
    written = 0
    for batch in tqdm(list(chunked(docs, batch_size)), desc="upsert", unit="batch"):
        try:
            store.write_documents(batch, policy="upsert")
            written += len(batch)
        except Exception as e:
            print(f"⚠️ upsert 批失败：{e}")
    return written


# ========= 主流程 =========
def ingest_one_source(source: str) -> None:
    scfg = SOURCES.get(source)
    if not scfg:
        print(f"❌ 未知 source: {source}，可选：{list(SOURCES.keys())}")
        return
    if not scfg.folder.exists():
        print(f"❌ 目录不存在：{scfg.folder}")
        return

    # 一次只拿 FILE_CHUNK 个文件，流式处理
    all_files = sorted(scfg.folder.glob(f"*{scfg.suffix}"))
    total_files = len(all_files)
    if total_files == 0:
        print(f"ℹ️ 无文件：{scfg.folder} (*{scfg.suffix})")
        return

    print(f"📁 扫描 {source}：{total_files} 个文件（按 {FILE_CHUNK} 一批处理）")

    processed_files = 0
    total_scanned = total_same = total_changed = total_missing = 0
    total_written = 0

    # 逐批处理，避免一次性内存/时间过大
    for chunk_idx, file_chunk in enumerate(chunked(all_files, FILE_CHUNK), start=1):
        # 1) 解析本批文件 → docs
        docs = scfg.splitter(file_chunk, source)
        if not docs:
            print(f"ℹ️ 第 {chunk_idx} 批未产生文本块")
            processed_files += len(file_chunk)
            continue

        # 2) 对账：只找本批的 missing/changed
        missing, changed, same_cnt = diff_against_server(source, docs)
        scanned = len(docs)
        need_write = changed + missing

        total_scanned += scanned
        total_same    += same_cnt
        total_changed += len(changed)
        total_missing += len(missing)

        print(
            f"🧾 第 {chunk_idx} 批：files={len(file_chunk)}  scanned={scanned}  "
            f"same={same_cnt}  changed={len(changed)}  missing={len(missing)}"
        )

        if need_write:
            # 3) 仅对本批需要写入的做 embedding
            docs_ready = ensure_embeddings(source, need_write)
            # 4) upsert 到 Qdrant
            written = upsert_documents(source, docs_ready, batch_size=UPSERT_BATCH)
            total_written += written
            print(f"✅ 第 {chunk_idx} 批写入：{written} / {len(docs_ready)}")
        else:
            print(f"✅ 第 {chunk_idx} 批：全部 up-to-date")

        processed_files += len(file_chunk)

        # 5) 释放显存/内存（可选，长跑更稳）
        try:
            import torch, gc
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

        # 批间进度
        print(
            f"📊 进度：{processed_files}/{total_files} files  |  "
            f"scanned={total_scanned}  same={total_same}  "
            f"changed={total_changed}  missing={total_missing}  "
            f"written={total_written}"
        )

    print(
        f"🎉 全部完成：files={total_files}  scanned={total_scanned}  same={total_same}  "
        f"changed={total_changed}  missing={total_missing}  written={total_written}"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help=f"可选：{list(SOURCES.keys())}")
    args = parser.parse_args()
    ingest_one_source(args.source)
