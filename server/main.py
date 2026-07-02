from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles
import server.utils.embedding_service as emb
from server.config import settings, create_storage_directories, STATIC_DIR
from server.db.init_db import init_db

from .api import api_router


async def app_startup():
    try:
        init_db()
    except Exception as e:
        # TODO: log error
        print(e)


async def lifespan(app: FastAPI):

    emb.load_model()

    await app_startup()
    yield


app = FastAPI(
    title="CapitalEdge AI Enterprise",
    description="AI-powered Conversational Intelligence and Workspace Analysis platform",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/v1")

# Create static folders on startup
create_storage_directories()

# Custom Middlewares
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_headers=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": exc.detail,
            "message": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": str(exc),
            "message": "Something went wrong!",
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "data": None,
            "error": exc.errors(),
            "message": "Validation error",
        },
    )


@app.get("/")
async def welcome():
    return {"status": 200, "message": "Server running"}
