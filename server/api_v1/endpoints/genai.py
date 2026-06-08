import json
import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from groq import Groq

import server.utils.embedding_service as emb
from server import crud, schemas
from server.api_v1.deps import get_db
from server.config import settings
from server.utils.auth import get_current_user
from server.utils.common import (
    agent_search,
    chromadb_client,
    chunk_text,
    search_similarity,
)
from server.utils.prompt import (
    ai_assistent,
    doc_agent_system_agent,
    doom_agent_system_prompt,
)

client = Groq(api_key=settings.GROQ_API_KEY)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


ai_router = APIRouter(prefix="/ai", tags=["GenAI"])


@ai_router.post("/ai-agent")
def ai_agent(
    request: schemas.GenAIBase,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):

    if "@doom" in request.user_query:
        logger.info("This is the agentic flow. User query contains '@doom' trigger.")

        organic_results = agent_search(request.user_query.replace("@doom", "").strip())

        message = [
            {"role": "system", "content": doom_agent_system_prompt()},
            {
                "role": "system",
                "content": f"Search Results:\n\n{json.dumps(organic_results, indent=2)}",
            },
            {"role": "user", "content": request.user_query},
        ]

    elif "@doc" in request.user_query:

        logger.info(
            "grok_ai: user=%s starting AI request; query_len=%d",
            getattr(current_user, "id", None),
            len(request.user_query) if getattr(request, "user_query", None) else 0,
        )

        similarity_results = search_similarity(
            request.user_query, current_user, top_k=3
        )
        logger.debug(
            "grok_ai: user=%s similarity results=%d",
            getattr(current_user, "id", None),
            len(similarity_results) if similarity_results is not None else 0,
        )
        message = [
            {"role": "system", "content": doc_agent_system_agent(similarity_results)},
            {"role": "user", "content": request.user_query},
        ]

    else:
        logger.info("This is the AI assistent flow.")
        message = [
            {"role": "system", "content": ai_assistent()},
            {"role": "user", "content": request.user_query},
        ]

    ai_response = ""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=message,
        temperature=0,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            if content:
                ai_response += content

    logger.info(
        "grok_ai: user=%s AI response generated; response_len=%d",
        getattr(current_user, "id", None),
        len(ai_response),
    )

    crud.genai.create(
        db=db,
        obj_in=schemas.GenAICreate(
            user_query=request.user_query, ai_response=ai_response
        ),
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": {"prompt": request.user_query, "response": ai_response},
            "message": "AI response generated successfully.",
        },
    )


@ai_router.post("/upload-files")
async def upload_files(
    file: UploadFile = File(...),
    db=Depends(get_db),
    current_user=Depends(get_current_user),
):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    logger.info(
        "create_embedding: user=%s uploaded file=%s",
        getattr(current_user, "id", None),
        file.filename,
    )

    content = await file.read()

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Unable to decode file as UTF-8")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    client = chromadb_client()

    collection = client.get_or_create_collection(name="embedded")

    chunks = chunk_text(text=text, chunk_size=2500)

    model = emb.load_model()

    embeddings = model.encode(chunks).tolist()
    logger.debug(
        "create_embedding: user=%s embeddings_generated=%d",
        getattr(current_user, "id", None),
        len(embeddings),
    )

    ids = [str(uuid.uuid4()) for _ in chunks]

    metadatas = [
        {
            "user_id": str(current_user.id),
            "file_name": file.filename,
            "chunk_number": index,
            "total_chunks": len(chunks),
        }
        for index in range(len(chunks))
    ]

    collection.upsert(
        ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
    )

    crud.enbedding.create(
        db=db, obj_in=schemas.EmbeddingResponse(file_name=file.filename)
    )

    logger.info(
        "create_embedding: user=%s embedding upserted collection=embedded file=%s chunks=%d",
        getattr(current_user, "id", None),
        file.filename,
        len(chunks),
    )

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "error": None,
            "data": None,
            "message": "File upload successfully.",
        },
    )
