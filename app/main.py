"""FastAPI application exposing the AI OCR module as an HTTP API."""

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .ocr_engine import extract_text, extract_full_text
from .table_detector import detect_tables
from .doc_parser import parse_document
from .document_loaders import (
    detect_document_type,
    pdf_pages_to_images,
    load_docx,
    load_xlsx,
)
from .auth import (
    router as auth_router,
    client_router,
    auth_router as user_auth_router,
    bootstrap_key,
    resolve_client,
    record_usage,
    record_ocr_run,
    enforce_limits,
    get_llm_override,
)
from .schemas import (
    OCRResponse, HealthResponse, TextBlock, Table, TableCell, PageResult,
    APIKeyInfo, DetectedDocument,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Max accepted upload size (bytes). Default 10 MB, override via env.
MAX_UPLOAD_BYTES = int(os.getenv("OCR_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
# Hard ceiling on PDF pages per request: each page is rendered at 200 dpi and
# held in memory, so an unbounded max_pages would let one caller exhaust RAM.
MAX_PDF_PAGES = int(os.getenv("OCR_MAX_PDF_PAGES", 100))


def _clamp_pages(max_pages: int) -> int:
    """Keep the requested page count within [1, MAX_PDF_PAGES]."""
    try:
        value = int(max_pages)
    except (TypeError, ValueError):
        value = 20
    return max(1, min(value, MAX_PDF_PAGES))


def _chunk_size_str() -> str:
    mb = MAX_UPLOAD_BYTES / (1024 * 1024)
    return f"{mb:.0f} MB" if mb >= 1 else f"{MAX_UPLOAD_BYTES} bytes"


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in chunks, aborting with 413 as soon as it exceeds the cap.

    Streaming the read avoids buffering an unbounded upload into memory (a
    must for a public SaaS endpoint).
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum accepted size is {_chunk_size_str()}.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _apply_limit_headers(response, limits: dict) -> None:
    """Expose quota/rate info via response headers when a Response is available."""
    if response is None:
        return
    if limits.get("monthly_quota"):
        response.headers["X-RateLimit-Limit"] = str(limits["monthly_quota"])
        response.headers["X-RateLimit-Remaining"] = str(limits.get("remaining"))
        response.headers["X-RateLimit-Reset"] = "month"
        if limits.get("over_quota"):
            # soft-cap: request is served past the monthly quota but flagged.
            response.headers["X-Over-Quota"] = "true"
    if limits.get("rpm_limit"):
        response.headers["X-RateLimit-Per-Minute"] = str(limits["rpm_limit"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI OCR Module starting up...")
    bootstrap_key()
    logger.info("Upload size limit: %s", _chunk_size_str())
    yield
    logger.info("AI OCR Module shutting down.")


app = FastAPI(
    title="AI OCR Module",
    description="Plug-and-play AI OCR API powered by PaddleOCR. "
                "Supports text extraction, table detection, multi-language, and document parsing. "
                "All /ocr endpoints require a per-client API key (X-API-Key header).",
    version="1.1.0",
    lifespan=lifespan,
    # Swagger's stock page is a standalone light-only sheet; /docs below serves a
    # themed one that follows the portal's light/dark switch.
    docs_url=None,
)

# Serve the frontend
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.middleware("http")
async def _revalidate_frontend(request: Request, call_next):
    """Make the portal shell and its assets revalidate on every load.

    FileResponse sends no Cache-Control, so browsers fall back to heuristic
    caching: an edited theme.css can keep being reused until it is considered
    stale, and a light/dark toggle then flips data-theme with no visible effect
    because the cached sheet has no light palette. no-cache keeps the ETag /
    Last-Modified round-trip (304s), so this stays cheap.
    """
    response = await call_next(request)
    if request.method == "GET":
        path = request.url.path
        if path in ("/", "/demo", "/docs") or path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-cache")
    return response


# Admin key-management endpoints (require an admin login session)
app.include_router(auth_router)
# Client self-service endpoints (API key OR client portal session)
app.include_router(client_router)
# Public authentication: signup, login, logout, session probe
app.include_router(user_auth_router)


@app.get("/", include_in_schema=False)
async def landing():
    """Unified portal: one login, role-based dashboard (client or admin)."""
    return FileResponse(static_dir / "portal.html")


@app.get("/client", include_in_schema=False)
async def client_portal():
    """Legacy client route -> unified portal."""
    return RedirectResponse(url="/", status_code=302)


@app.get("/admin", include_in_schema=False)
async def admin_portal():
    """Legacy admin route -> unified portal."""
    return RedirectResponse(url="/", status_code=302)


@app.get("/demo", include_in_schema=False)
async def demo_page():
    """Standalone demo: OCR any document, let the LLM name the type, fill a form."""
    return FileResponse(static_dir / "demo.html")


@app.get("/docs", include_in_schema=False)
async def api_docs():
    """Swagger UI wearing the portal palette (honours the light/dark switch)."""
    return FileResponse(static_dir / "api-docs.html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the service is running."""
    return HealthResponse()


def _parse_fields(raw: Optional[str]) -> Optional[list]:
    """Parse the optional 'fields' form value into a list of field names.

    Accepts a JSON array (e.g. '["vendor","total"]') or a comma-separated
    string ('vendor,total'). Returns None when empty so the generic parser is
    used. De-dupes and drops blanks while preserving order.
    """
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    items: list = []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                items = [str(x) for x in data]
        except (ValueError, TypeError):
            items = []
    if not items:
        items = raw.strip("[]").split(",")
    seen, out = set(), []
    for it in items:
        name = it.strip().strip('"').strip("'")
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out or None


# Keys the auto-detect answer uses for classification; everything else under
# "fields" is the caller's data.
_DETECTED_KEYS = {"document_type", "label", "category", "is_document", "confidence", "summary", "fields"}


def _detected_from_llm(payload: dict) -> DetectedDocument:
    """Shape an auto-detect answer into DetectedDocument.

    Models drift: confidence can arrive as "high" or 95, is_document as a
    string, and document_type may be missing — coerce instead of returning a
    500 to the client.
    """
    confidence: Optional[float] = None
    raw_conf = payload.get("confidence")
    if raw_conf is not None and not isinstance(raw_conf, bool):
        try:
            confidence = float(str(raw_conf).strip().rstrip("%"))
            if confidence > 1:
                confidence = confidence / 100
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = None

    raw_doc = payload.get("is_document")
    if isinstance(raw_doc, bool):
        is_document = raw_doc
    else:
        is_document = str(raw_doc).strip().lower() not in {"false", "0", "no", "n"}

    document_type = str(payload.get("document_type") or "unknown").strip() or "unknown"
    if document_type.lower() in {"not_a_document", "not a document", "none"}:
        document_type = "not_a_document"
        is_document = False

    def _text(key: str) -> Optional[str]:
        value = payload.get(key)
        if value is None or value == "":
            return None
        return str(value).strip() or None

    return DetectedDocument(
        document_type=document_type,
        label=_text("label"),
        category=_text("category"),
        is_document=is_document,
        confidence=confidence,
        summary=_text("summary"),
    )


async def _process_ocr_upload(
    upload: UploadFile, *, lang: str, detect_table: bool, max_pages: int
) -> tuple[str, list]:
    """Read + OCR one uploaded file into (doc_type, pages_raw), page numbers 1-based."""
    contents = await _read_upload_limited(upload, MAX_UPLOAD_BYTES)
    max_pages = _clamp_pages(max_pages)
    doc_type = detect_document_type(contents, upload.filename or "")
    if doc_type == "unsupported":
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Accepted: PNG, JPG, TIFF, BMP, WEBP, PDF, DOCX, XLSX",
        )
    pages_raw: list[tuple[int, list[dict], list[dict]]] = []
    if doc_type == "image":
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        try:
            blocks = extract_text(image, lang=lang)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")
        tables = _safe_detect_tables(image, blocks) if detect_table else []
        pages_raw.append((1, blocks, tables))
    elif doc_type == "pdf":
        try:
            page_images = pdf_pages_to_images(contents, max_pages=max_pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {e}")
        for i, page_img in enumerate(page_images, start=1):
            try:
                blocks = extract_text(page_img, lang=lang)
            except Exception as e:
                logger.error(f"OCR failed on PDF page {i}: {e}")
                blocks = []
            tables = _safe_detect_tables(page_img, blocks) if detect_table else []
            pages_raw.append((i, blocks, tables))
    else:  # docx / xlsx: native extraction, no OCR needed
        try:
            blocks, tables = load_docx(contents) if doc_type == "docx" else load_xlsx(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read {doc_type} file: {e}")
        pages_raw.append((1, blocks, tables))
    return doc_type, pages_raw


@app.post("/ocr", response_model=OCRResponse, tags=["OCR"])
async def ocr_endpoint(
    file: UploadFile = File(..., description="Primary image or document (PNG, JPG, TIFF, BMP, WEBP, PDF, DOCX, XLSX)"),
    extra_files: Optional[List[UploadFile]] = File(
        default=None,
        description="Optional additional images to OCR and merge with 'file' into one document (e.g. the back of an ID card). With parse_doc, fields from all sides are extracted together.",
    ),
    lang: str = Form(default="en", description="Language code (en, ch, japan, korean, fr, etc.)"),
    detect_table: bool = Form(default=False, description="Enable table detection"),
    parse_doc: bool = Form(default=False, description="Enable document parsing via LLM"),
    fields: Optional[str] = Form(
        default=None,
        description=(
            "When parse_doc=true, the exact form fields to extract, as a JSON array "
            'or comma-separated list (e.g. invoice_number,vendor,total,due_date). '
            "Omit to let the parser choose a generic set."
        ),
    ),
    auto_detect: bool = Form(
        default=False,
        description=(
            "With parse_doc=true, let the LLM name the document type (receipt, "
            "driving licence, passport, ...) and choose the fields itself. 'fields' "
            "is ignored and the response adds a 'detected' block."
        ),
    ),
    max_pages: int = Form(default=20, description="Max PDF pages to process"),
    client: APIKeyInfo = Depends(resolve_client),
    response: Response = None,
):
    """
    Extract text from an uploaded image or document.

    Requires a valid API key in the `X-API-Key` header. Subject to the client's
    monthly quota and per-minute rate limit (HTTP 429 when exceeded).

    - **file**: Image (PNG, JPG, TIFF, BMP, WEBP), PDF (OCR per page), DOCX or XLSX (native text/table extraction)
    - **extra_files**: Optional extra images/files to OCR and merge into the same document
      (e.g. the back of an identity card). Text is concatenated and, with parse_doc, a
      single LLM pass extracts fields from all sides together.
    - **lang**: PaddleOCR language code (applies to images and PDFs)
    - **detect_table**: Detect and extract tables
    - **parse_doc**: Parse document into structured fields (needs an OpenAI-compatible LLM endpoint; a local server such as Ollama works without an API key)
    - **fields**: With parse_doc, the exact form fields to extract (JSON array or
      comma-separated). Omit for a generic field set.
    - **auto_detect**: With parse_doc, let the LLM classify the document and pick the
      fields (works for receipts, licences, invoices — anything). The response then
      carries a `detected` block and `document` holds the extracted fields.
    - **max_pages**: Cap on the number of PDF pages processed
    """
    _apply_limit_headers(response, enforce_limits(client))

    uploads = [file] + [f for f in (extra_files or []) if f and (f.filename or "")]
    logger.info("OCR request by client id=%s name=%r (%d file(s))", client.id, client.name, len(uploads))

    # OCR each uploaded side and merge into one page list (front/back -> one document)
    pages_raw: list[tuple[int, list[dict], list[dict]]] = []
    doc_type = ""
    for up in uploads:
        dt, sub = await _process_ocr_upload(up, lang=lang, detect_table=detect_table, max_pages=max_pages)
        if not doc_type:
            doc_type = dt
        offset = len(pages_raw)
        pages_raw.extend((pn + offset, blocks, tbls) for pn, blocks, tbls in sub)

    # Assemble response
    text_blocks = [b for _, blocks, _ in pages_raw for b in blocks]
    tables_all = [t for _, _, tbls in pages_raw for t in tbls]
    full_text = "\n".join(block["text"] for block in text_blocks)

    response = OCRResponse(
        text_blocks=[TextBlock(**b) for b in text_blocks],
        full_text=full_text,
        language=lang,
        document_type=doc_type,
        pages=[
            PageResult(
                page=pn,
                text_blocks=[TextBlock(**b) for b in blocks],
                tables=[_build_table(t) for t in tbls],
            )
            for pn, blocks, tbls in pages_raw
        ],
        tables=[_build_table(t) for t in tables_all],
    )

    # Document parsing: an admin-configured per-client LLM key/endpoint wins,
    # otherwise parse_document falls back to the server-wide OCR_LLM_* settings.
    if parse_doc:
        try:
            auto = bool(auto_detect)
            parsed = parse_document(
                full_text,
                fields=None if auto else _parse_fields(fields),
                auto=auto,
                **(get_llm_override(client.id) or {}),
            )
            # The LLM may answer with a JSON array (or a bare scalar); the
            # response model requires a field->value object, so normalise.
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if auto and isinstance(parsed, dict) and parsed:
                response.detected = _detected_from_llm(parsed)
                auto_fields = parsed.get("fields")
                if isinstance(auto_fields, dict) and auto_fields:
                    response.document = auto_fields
                else:
                    # Model ignored the envelope: keep any flat values as fields.
                    flat = {k: v for k, v in parsed.items() if k not in _DETECTED_KEYS}
                    if flat:
                        response.document = flat
            elif isinstance(parsed, dict) and parsed:
                response.document = parsed
            elif parsed:
                logger.warning("Ignoring LLM output: expected a JSON object, got %s", type(parsed).__name__)
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")

    combined_name = " + ".join((u.filename or "") for u in uploads if (u.filename or "")) or None
    record_usage(client.id)
    record_ocr_run(
        client.id,
        filename=combined_name,
        document_type=doc_type,
        lang=lang,
        text_blocks=len(text_blocks),
        tables=len(tables_all),
        pages=len(pages_raw),
        chars=len(full_text),
        files_count=len(uploads),
        table_detection=bool(detect_table),
        llm_parsing=bool(parse_doc),
    )
    return response


def _safe_detect_tables(image: Image.Image, text_blocks: list[dict]) -> list[dict]:
    """Run table detection with graceful fallback to empty results."""
    try:
        return detect_tables(image, text_blocks)
    except Exception as e:
        logger.error(f"Table detection failed: {e}")
        return []


def _build_table(table_raw: dict) -> Table:
    """Convert a raw table dict into the Table response model."""
    return Table(
        cells=[
            TableCell(
                row=c["row"],
                col=c["col"],
                text=c["text"],
                bbox=c.get("bbox"),
            )
            for c in table_raw["cells"]
        ],
        bbox=table_raw.get("bbox"),
    )


@app.post("/ocr/text", tags=["OCR"])
async def ocr_text_only(
    file: UploadFile = File(..., description="Image or document file (PNG, JPG, PDF, DOCX, XLSX, ...)"),
    lang: str = Form(default="en", description="Language code"),
    max_pages: int = Form(default=20, description="Max PDF pages to process"),
    client: APIKeyInfo = Depends(resolve_client),
    response: Response = None,
):
    """
    Simple endpoint: extract text only, returns plain text string.

    Requires a valid API key in the `X-API-Key` header. Respects quota/rate limit.
    """
    _apply_limit_headers(response, enforce_limits(client))
    contents = await _read_upload_limited(file, MAX_UPLOAD_BYTES)
    max_pages = _clamp_pages(max_pages)
    logger.info("OCR(text) request by client id=%s name=%r (%d bytes)", client.id, client.name, len(contents))
    doc_type = detect_document_type(contents, file.filename or "")

    if doc_type == "unsupported":
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Accepted: PNG, JPG, TIFF, BMP, WEBP, PDF, DOCX, XLSX",
        )

    pages = 0
    text = ""
    try:
        if doc_type == "image":
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            text = extract_full_text(image, lang=lang)
            pages = 1
        elif doc_type == "pdf":
            parts = []
            for i, page_img in enumerate(pdf_pages_to_images(contents, max_pages=max_pages), start=1):
                parts.append(f"--- Page {i} ---\n{extract_full_text(page_img, lang=lang)}")
            text = "\n\n".join(parts)
            pages = len(parts)
        else:
            blocks, _ = load_docx(contents) if doc_type == "docx" else load_xlsx(contents)
            text = "\n".join(b["text"] for b in blocks)
            pages = 1
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {e}")

    record_usage(client.id)
    record_ocr_run(
        client.id,
        filename=file.filename or None,
        document_type=doc_type,
        lang=lang,
        text_blocks=len([ln for ln in text.splitlines() if ln.strip()]),
        pages=pages,
        chars=len(text),
    )
    return {"text": text, "language": lang, "document_type": doc_type}
