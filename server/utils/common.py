import random
import uuid
from datetime import datetime
from typing import Optional
import io
import chromadb
import serpapi
from fastapi import HTTPException, UploadFile

from server import crud, schemas
from server.config import settings
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


def chromadb_client():
    client = chromadb.CloudClient(
        api_key=settings.CHROMADB_KEY,
        tenant="c87cebe2-d1c4-448e-8b12-c7745ebea77e",
        database=settings.CHROMADB_DATABASE,
    )
    return client


def rerank_documents(query, documents, top_k=3):

    pairs = [(query, doc) for doc in documents]

    scores = load_rerank_model().predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked[:top_k]]


def search_similarity(
    query: str, current_user, top_k: int = 3, workspace_id: Optional[str] = None
):
    import server.utils.embedding_service as emb

    model = emb.load_model()

    query_embedding = model.encode(query).tolist()

    client = chromadb_client()
    collection = client.get_collection("embedded")

    # Scope retrieval to a workspace's documents when a workspace is in play.
    # Chroma requires the $and operator to combine more than one condition.
    if workspace_id:
        where = {
            "$and": [
                {"user_id": {"$eq": str(current_user.id)}},
                {"workspace_id": {"$eq": str(workspace_id)}},
            ]
        }
    else:
        where = {"user_id": {"$eq": str(current_user.id)}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    documents = results["documents"][0]
    file_names = list(
        dict.fromkeys(metadata["file_name"] for metadata in results["metadatas"][0])
    )

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

    Chunks are upserted into ChromaDB tagged with both ``user_id`` and
    ``workspace_id`` (so retrieval can be scoped to the workspace), and a single
    ``Embedding`` row is persisted to represent the file. Returns that row.
    """
    filename = (file.filename or "").lower()
    ext = next((e for e in _EXTRACTORS if filename.endswith(e)), None)
    if ext is None:
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .doc, .docx, .pdf, .csv, .xlsx and .xls files are allowed",
        )
    file_format, extractor = _EXTRACTORS[ext]

    content = await file.read()
    try:
        text = extractor(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    import server.utils.embedding_service as emb

    collection = chromadb_client().get_or_create_collection(name="embedded")
    chunks = chunk_text(text=text, chunk_size=2500)
    embeddings = emb.load_model().encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "user_id": str(user_id),
            "workspace_id": str(workspace_id),
            "file_name": file.filename,
            "chunk_number": index,
            "total_chunks": len(chunks),
        }
        for index in range(len(chunks))
    ]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return crud.document.create(
        db=db,
        obj_in=schemas.DocumentResponse(
            file_name=file.filename,
            user_id=str(user_id),
            file_format=file_format,
            workspace_id=str(workspace_id),
        ),
    )


def delete_chroma_by(where: dict) -> None:
    """Delete all ChromaDB chunks matching a metadata filter."""
    chromadb_client().get_or_create_collection(name="embedded").delete(where=where)
