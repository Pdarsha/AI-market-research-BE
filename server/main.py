import logging
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles
from server.config import STATIC_DIR, create_storage_directories, settings
from server.db.init_db import init_db
from server.services.audit_logger import AuditLogger

from .api import api_router

logger = logging.getLogger(__name__)


async def app_startup():
    try:
        init_db()
    except Exception as e:
        # TODO: log error
        print(e)


async def lifespan(app: FastAPI):

    await app_startup()

    import threading

    from server.services.ocr_service import warm_up_ocr
    threading.Thread(target=warm_up_ocr, name="ocr-warmup", daemon=True).start()
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
    allow_origins=["*"],
    allow_headers=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    AuditLogger.log(
        action="API_ERROR",
        endpoint=request.url.path,
        http_method=request.method,
        status=str(exc.status_code),
        metadata={"error": exc.detail},
    )
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
    logger.exception(
        "Unhandled error on %s %s: %s", request.method, request.url.path, exc
    )
    AuditLogger.log(
        action="API_ERROR",
        endpoint=request.url.path,
        http_method=request.method,
        status="500",
        metadata={"error": str(exc)},
    )
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
    AuditLogger.log(
        action="VALIDATION_ERROR",
        endpoint=request.url.path,
        http_method=request.method,
        status="422",
        metadata={"errors": exc.errors()},
    )
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
