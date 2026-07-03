import os
import random
import uuid
from datetime import datetime
from typing import Optional
import io
import serpapi
from fastapi import HTTPException, UploadFile
from pathlib import Path
from server import crud, schemas
from server.config import settings
from server.models.workspace import DocumentChunk
from server.utils.embedding_service import load_rerank_model
from server.utils.mail import send_email
from docx import Document
import tempfile
import textract

from io import BytesIO, StringIO
from pypdf import PdfReader
import pandas as pd
import openpyxl

serp_client = serpapi.Client(api_key=settings.SERPAPI_KEY)


async def validate_email_send_otp(db, email, user_id):
    time_now = datetime.utcnow()
    key = str(random.randint(99999, 999999))
    old_otp_obj = crud.email_otp.get_by_email(db, email=email)
    if old_otp_obj:
        attempts = old_otp_obj.attempts
        if (attempts >= 10) and (
            (((time_now - old_otp_obj.updated_at).total_seconds()) / 3600) > 24
        ):
            crud.email_otp.update(
                db,
                db_obj=old_otp_obj,
                obj_in=schemas.EmailOtpUpdate(created_at=time_now, attempts=1),
            )
            await send_email(email, key)
            return True
        if (attempts >= 10) and (
            (((old_otp_obj.updated_at - old_otp_obj.created_at).total_seconds()) / 60)
            < 10
        ):
            raise HTTPException(429, "Sending otp limit exceeded. try after 24 hours")
        if attempts >= 10:
            raise HTTPException(429, "Sending otp limit exceeded. try after sometime")
        crud.email_otp.update(
            db,
            db_obj=old_otp_obj,
            obj_in=schemas.EmailOtpUpdate(
                created_at=time_now, attempts=old_otp_obj.attempts + 1, otp=key
            ),
        )
    else:
        crud.email_otp.create(
            db,
            obj_in=schemas.EmailOtpBase(
                email=email, attempts=1, otp=key, user_id=user_id
            ),
        )
    await send_email(email, key)
    return True


def rerank_documents(query, documents, top_k=3):

    pairs = [(query, doc) for doc in documents]

    scores = load_rerank_model().predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked[:top_k]]


def search_similarity(
    db, query: str, current_user, top_k: int = 3, workspace_id: Optional[str] = None
):
    import server.utils.embedding_service as emb

    model = emb.load_model()

    query_embedding = model.encode(query).tolist()

    # Scope retrieval to the current user's chunks, and to a single workspace when
    # one is in play. pgvector orders by cosine distance (``<=>``) so the nearest
    # chunks come first; we then rerank the raw candidates with the CrossEncoder.
    stmt = db.query(DocumentChunk).filter(
        DocumentChunk.user_id == str(current_user.id)
    )
    if workspace_id:
        stmt = stmt.filter(DocumentChunk.workspace_id == str(workspace_id))

    rows = (
        stmt.order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
        .all()
    )

    documents = [row.content for row in rows]
    file_names = list(dict.fromkeys(row.file_name for row in rows))

    reranked_docs = rerank_documents(query=query, documents=documents, top_k=top_k)

    return reranked_docs, file_names


def chunk_text(text: str, chunk_size: int = 2500):
    """
    Split text into chunks.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def agent_search(query: str):

    results = serp_client.search({"engine": "duckduckgo", "q": query, "kl": "in-en"})
    organic_results = results["organic_results"]

    return organic_results


def extract_text_from_docx(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_doc(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".doc", delete=True) as temp_file:
        temp_file.write(content)
        temp_file.flush()

        text = textract.process(temp_file.name)
        return text.decode("utf-8", errors="ignore")


def extract_text_from_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def extract_text_from_csv(content: bytes) -> str:
    df = pd.read_csv(StringIO(content.decode("utf-8")))
    return df.to_string(index=False)


def extract_text_from_xlsx(content: bytes) -> str:
    sheets = pd.read_excel(
        BytesIO(content),
        sheet_name=None,
        engine="openpyxl",
    )

    text = []

    for sheet_name, df in sheets.items():
        text.append(f"Sheet: {sheet_name}")
        text.append(df.fillna("").to_string(index=False))
        text.append("")

    return "\n".join(text)


def extract_text_from_xls(content: bytes) -> str:
    sheets = pd.read_excel(
        BytesIO(content),
        sheet_name=None,
        engine="xlrd",
    )

    text = []

    for sheet_name, df in sheets.items():
        text.append(f"Sheet: {sheet_name}")
        text.append(df.fillna("").to_string(index=False))
        text.append("")

    return "\n".join(text)


# extension -> (stored file_format, extractor). Single source of truth for the
# upload flow so both the /ai and /workspaces upload paths behave identically.
_EXTRACTORS = {
    ".txt": ("txt", lambda content: content.decode("utf-8")),
    ".docx": ("docx", extract_text_from_docx),
    ".doc": ("doc", extract_text_from_doc),
    ".pdf": ("pdf", extract_text_from_pdf),
    ".csv": ("csv", extract_text_from_csv),
    ".xlsx": ("xlsx", extract_text_from_xlsx),
    ".xls": ("xls", extract_text_from_xls),
}


async def ingest_file_into_workspace(
    db, *, file: UploadFile, user_id: str, workspace_id: str
):
    """Extract, chunk, embed and store an uploaded file under a workspace.

    A ``Document`` row is persisted to represent the file, and each text chunk is
    stored as a ``DocumentChunk`` row (with its pgvector embedding) tagged with
    both ``user_id`` and ``workspace_id`` so retrieval can be scoped to the
    workspace. Returns the ``Document`` row.
    """
    filename = (file.filename or "").lower()
    ext = next((e for e in _EXTRACTORS if filename.endswith(e)), None)
    if ext is None:
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .doc, .docx, .pdf, .csv, .xlsx and .xls files are allowed",
        )
    file_format, extractor = _EXTRACTORS[ext]

    # Read uploaded file
    content = await file.read()

    # Create workspace directory
    STATIC_DIR = Path("static")
    upload_dir = STATIC_DIR / "workspace_docs"
    os.makedirs(upload_dir, exist_ok=True)

    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{ext}"

    # Full file path
    file_path = upload_dir / unique_filename

    # Save file
    with open(file_path, "wb") as f:
        f.write(content)

    # Store this in DB
    storage_path = f"static/workspace_docs/{unique_filename}"

    # Extract text
    try:
        text = extractor(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    import server.utils.embedding_service as emb

    chunks = chunk_text(text=text, chunk_size=2500)
    embeddings = emb.load_model().encode(chunks).tolist()

    # Persist the document first so chunks can reference it. Chunk rows carry the
    # embeddings into pgvector; their FK to the document cascades on delete.
    document = crud.document.create(
        db=db,
        obj_in=schemas.DocumentResponse(
            user_id=str(user_id),
            workspace_id=str(workspace_id),
            file_name=file.filename,
            file_format=file_format,
            storage_path=storage_path,
        ),
    )

    chunk_rows = [
        DocumentChunk(
            document_id=document.id,
            user_id=str(user_id),
            workspace_id=str(workspace_id),
            file_name=file.filename,
            chunk_number=index,
            total_chunks=len(chunks),
            content=chunk,
            embedding=embedding,
        )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    db.add_all(chunk_rows)
    db.commit()

    return document
