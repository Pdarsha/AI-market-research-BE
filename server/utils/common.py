import random
from datetime import datetime

import chromadb
import serpapi
from fastapi import HTTPException

from server import crud, schemas
from server.config import settings
from server.utils.mail import send_email

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


def search_similarity(query: str, current_user, top_k: int = 3):
    import server.utils.embedding_service as emb

    model = emb.load_model()

    query_embedding = model.encode(query).tolist()

    client = chromadb_client()
    collection = client.get_collection("embedded")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"user_id": str(current_user.id)},
    )

    return results["documents"][0]


def chunk_text(text: str, chunk_size: int = 2500):
    """
    Split text into chunks.
    """
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def agent_search(query: str):

    results = serp_client.search({"engine": "duckduckgo", "q": query, "kl": "in-en"})
    organic_results = results["organic_results"]

    return organic_results
