import base64
import copy
import io
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app_config import load_app_env
from document_chunking import build_text_chunks

load_app_env()

from model_registry import (
    get_dense_model,
    get_docling_converter,
    get_reranker,
    get_sparse_model,
    preload_all,
)
from vision_storage import (
    StorageConfigurationError,
    StorageTimeoutError,
    StorageUpstreamError,
    get_vision_storage,
    vision_bucket_enabled,
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    preload_all()
    yield


app = FastAPI(title="Model API Microservice", lifespan=lifespan)
_docling_conversion_lock = threading.Lock()
_docling_chunking_lock = threading.Lock()
_chunk_tokenizer_lock = threading.Lock()
_chunk_tokenizer = None


class DenseRequest(BaseModel):
    texts: List[str]
    normalize_embeddings: bool = True

class SparseQueryRequest(BaseModel):
    text: str

class SparsePassagesRequest(BaseModel):
    texts: List[str]

class RerankRequest(BaseModel):
    query: str
    texts: List[str]

@app.post("/embed/dense")
def embed_dense(req: DenseRequest):
    try:
        model = get_dense_model()
        vectors = model.encode(req.texts, normalize_embeddings=req.normalize_embeddings, convert_to_numpy=True)
        return {"vectors": vectors.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/sparse/query")
def embed_sparse_query(req: SparseQueryRequest):
    try:
        model = get_sparse_model()
        result = model.encode_query(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/sparse/passages")
def embed_sparse_passages(req: SparsePassagesRequest):
    try:
        model = get_sparse_model()
        vectors = model.encode_passages(req.texts)
        results = []
        for vec in vectors:
            nonzero_idx = np.nonzero(vec)[0]
            results.append({
                "indices": nonzero_idx.tolist(),
                "values": vec[nonzero_idx].tolist()
            })
        return {"vectors": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rerank")
def rerank(req: RerankRequest):
    try:
        reranker = get_reranker()
        pairs = [(req.query, t) for t in req.texts]
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        # Handle scalar or array cases safely
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = [scores]
        return {"scores": [float(s) for s in scores]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_docling_upload(
    upload_stream,
    converter,
    storage=None,
    suffix: str = ".pdf",
    chunk_tokenizer=None,
) -> Dict[str, Any]:
    """Convert one uploaded document, chunk it, and optionally store outputs."""
    safe_suffix = suffix.lower()
    if (
        not safe_suffix.startswith(".")
        or len(safe_suffix) > 11
        or not safe_suffix[1:].isalnum()
    ):
        safe_suffix = ".bin"

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=safe_suffix) as temp_file:
            temp_path = Path(temp_file.name)
            upload_stream.seek(0)
            shutil.copyfileobj(upload_stream, temp_file)

        with _docling_conversion_lock:
            result = converter.convert(str(temp_path))
        document = result.document
        total_pages = len(document.pages) if hasattr(document, "pages") else 0
        request_id = str(uuid.uuid4())

        # Export the complete document first. Chunking then uses the same merged,
        # structured DoclingDocument so hierarchy, paragraphs, lists and tables survive.
        markdown_text = document.export_to_markdown()
        markdown_file_name = "document.md"
        markdown_object_key = f"docling/{request_id}/{markdown_file_name}"
        markdown_file = None
        bucket_path = None
        if storage is not None:
            markdown_file = storage.upload_markdown(
                markdown_text,
                markdown_object_key,
            )
            bucket_path = storage.bucket_path(markdown_object_key)

        with _docling_chunking_lock:
            ordered_elements = build_text_chunks(document, chunk_tokenizer)
        item_order = {
            item.self_ref: index
            for index, (item, _) in enumerate(document.iterate_items())
            if getattr(item, "self_ref", None)
        }
        image_counter = 0
        current_page = 0
        for fallback_sequence, (element, _) in enumerate(document.iterate_items()):
            page_no = element.prov[0].page_no if element.prov else current_page
            current_page = page_no
            if element.label != "picture":
                continue
            if not hasattr(element, "image") or element.image is None:
                continue
            pil_image = element.image.pil_image
            if pil_image is None:
                continue
            if pil_image.width < 80 or pil_image.height < 80:
                continue

            image_counter += 1
            with io.BytesIO() as image_buffer:
                pil_image.save(image_buffer, format="PNG")
                png_bytes = image_buffer.getvalue()

            image_element = {
                "type": "image",
                "page": page_no,
                "_sequence": item_order.get(
                    getattr(element, "self_ref", ""), fallback_sequence
                ),
            }
            if storage is None:
                image_element["image_base64"] = base64.b64encode(png_bytes).decode(
                    "utf-8"
                )
            else:
                object_key = (
                    f"docling/{request_id}/page-{page_no}/"
                    f"picture-{image_counter}.png"
                )
                image_element["image_url"] = storage.upload_png(
                    png_bytes, object_key
                )
            ordered_elements.append(image_element)

        ordered_elements.sort(key=lambda item: item.get("_sequence", 0))
        for item in ordered_elements:
            item.pop("_sequence", None)

        return {
            "total_pages": total_pages,
            "markdown_file": markdown_file,
            "bucket_path": bucket_path,
            "file_name": markdown_file_name,
            "elements": ordered_elements,
        }
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _create_chunk_tokenizer():
    global _chunk_tokenizer
    if _chunk_tokenizer is None:
        with _chunk_tokenizer_lock:
            if _chunk_tokenizer is None:
                dense_model = get_dense_model()
                independent_tokenizer = copy.deepcopy(
                    getattr(dense_model, "tokenizer")
                )
                _chunk_tokenizer = HuggingFaceTokenizer(
                    tokenizer=independent_tokenizer,
                    max_tokens=512,
                )
    return _chunk_tokenizer


def extract_docling_sync(upload_stream, filename: Optional[str] = None) -> Dict[str, Any]:
    """Run all blocking Docling/image/storage work outside the event loop."""
    use_bucket = vision_bucket_enabled()
    storage = get_vision_storage() if use_bucket else None
    converter = get_docling_converter()
    chunk_tokenizer = _create_chunk_tokenizer()
    suffix = Path(filename or "").suffix or ".bin"
    return process_docling_upload(
        upload_stream,
        converter,
        storage,
        suffix=suffix,
        chunk_tokenizer=chunk_tokenizer,
    )


@app.post("/extract/docling")
async def extract_docling(file: UploadFile = File(...)):
    try:
        return await run_in_threadpool(extract_docling_sync, file.file, file.filename)
    except HTTPException:
        raise
    except StorageConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StorageTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except StorageUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, workers=1)
