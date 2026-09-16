import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response, FileResponse

from pydantic import BaseModel, Field

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from main import (
    initialize_system,
    shutdown_system,
    stream_financial_response,
    validate_configuration,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("wealth-api")

API_VERSION = "5.0.0"

REQUEST_TIMEOUT = int(
    os.getenv("REQUEST_TIMEOUT", "120")
)

FRONTEND_URLS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_URLS",
        "*"
    ).split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing system...")

    try:
        validate_configuration()
        app.state.rag = await initialize_system()
        logger.info("System initialized successfully")

    except Exception as e:
        logger.exception("Initialization failed")
        raise RuntimeError(
            "Application startup failed"
        ) from e

    yield

    logger.info("Shutting down system...")

    try:
        await shutdown_system()
        logger.info("Shutdown complete")

    except Exception:
        logger.exception("Shutdown error")


app = FastAPI(
    title="Nigerian Financial Research Assistant",
    version=API_VERSION,
    lifespan=lifespan,
)


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/hour"],
)

app.state.limiter = limiter


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=2000,
    )


class SelectiveGZipMiddleware(GZipMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        excluded_paths: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(app, **kwargs)
        self.excluded_paths = excluded_paths or []

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ):
        if (
            scope["type"] == "http"
            and scope["path"] in self.excluded_paths
        ):
            await self.app(scope, receive, send)
            return

        await super().__call__(
            scope,
            receive,
            send,
        )


app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.add_middleware(
    SelectiveGZipMiddleware,
    excluded_paths=["/chat"],
    minimum_size=1000,
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
    request: Request,
    exc: RateLimitExceeded,
):
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Rate limit exceeded"
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception("Unhandled server error")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error"
        },
    )


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("index.html")


@app.get("/style.css", include_in_schema=False)
async def css():
    return FileResponse(
        "style.css",
        media_type="text/css",
    )


@app.get("/script.js", include_in_schema=False)
async def javascript():
    return FileResponse(
        "script.js",
        media_type="application/javascript",
    )


@app.get("/app", include_in_schema=False)
async def frontend():
    return FileResponse("index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": int(time.time()),
    }


@app.get("/ready")
async def ready(request: Request):
    if not hasattr(request.app.state, "rag"):
        raise HTTPException(
            status_code=503,
            detail="Service is not ready",
        )

    return {
        "status": "ready",
        "timestamp": int(time.time()),
    }


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.post("/chat")
@limiter.limit("20/minute")
async def chat(
    request: Request,
    payload: ChatRequest,
):
    query = payload.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Query must not be blank",
        )

    rag = getattr(
        request.app.state,
        "rag",
        None,
    )

    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready",
        )

    async def event_generator():
        loop = asyncio.get_running_loop()
        heartbeat = loop.time()

        try:
            async with asyncio.timeout(REQUEST_TIMEOUT):
                generator = stream_financial_response(
                    query=query,
                    user="anonymous",
                    rag=rag,
                )

                async for chunk in generator:
                    if await request.is_disconnected():
                        logger.info("Client disconnected")
                        break

                    if (loop.time() - heartbeat) > 15:
                        yield ": ping\n\n"
                        heartbeat = loop.time()

                    clean_chunk = (
                        str(chunk)
                        .replace("\r", "\\r")
                        .replace("\n", " ")
                    )

                    yield f"data: {clean_chunk}\n\n"
                    await asyncio.sleep(0)

                yield "data: [DONE]\n\n"

        except TimeoutError:
            logger.warning(
                "Request timed out after %s seconds",
                REQUEST_TIMEOUT,
            )
            yield "data: Request timeout\n\n"

        except asyncio.CancelledError:
            logger.warning("Streaming cancelled")

        except Exception:
            logger.exception("Streaming failed")
            yield "data: Server error\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
