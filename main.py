import os
from dotenv import load_dotenv
from google import genai
import secrets
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from pydantic import BaseModel, Field
import logging
from logging.handlers import RotatingFileHandler
import time
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, make_asgi_app
from google.genai import types

class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "event"):
            record.event = "-"
        if not hasattr(record, "endpoint"):
            record.endpoint = "-"
        if not hasattr(record, "duration_seconds"):
            record.duration_seconds = "-"    
        return super().format(record)


file_handler = RotatingFileHandler(
    "app.log",
    maxBytes=1_000_000,
    backupCount=3,
    encoding="utf-8"
)

console_handler = logging.StreamHandler()

formatter = SafeFormatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s "
    "| event=%(event)s | endpoint=%(endpoint)s"
    "| duration_seconds=%(duration_seconds)s"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
    force=True
)

logger = logging.getLogger(__name__)


load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
if not GEMINI_MODEL:
    raise ValueError("GEMINI_MODEL is not configured")

client = (
    genai.Client(
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                exp_base=2,
                max_delay=4.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        )
    )
    if os.getenv("GEMINI_API_KEY")
    else None
)

app = FastAPI()

API_ACCESS_KEY = os.getenv("API_ACCESS_KEY")

if not API_ACCESS_KEY and os.getenv("CI") != "true":
    raise ValueError("API_ACCESS_KEY is not configured")


def verify_api_key(
    x_api_key: str | None = Header(default=None)
):
    if (
        not x_api_key
        or not secrets.compare_digest(x_api_key, API_ACCESS_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method"]
)

REQUEST_ERRORS = Counter(
    "api_request_errors_total",
    "Total number of HTTP error responses",
    ["endpoint", "method", "status_code"]
)

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "Duration of HTTP requests in seconds",
    ["endpoint", "method"]
)

@app.middleware("http")
async def count_requests(request: Request, call_next):
    endpoint = request.url.path
    method = request.method

    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        method=request.method
    ).inc()
    with REQUEST_DURATION.labels(
        endpoint=endpoint,
        method=method
    ).time():
        response = await call_next(request)
    if response.status_code >= 400:
        REQUEST_ERRORS.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(response.status_code)
        ).inc()
    return response
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

redis_url = os.getenv("REDIS_URL")
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=redis_url or "memory://"
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


class Question(BaseModel):
    question: str=Field(min_length=1, max_length=1000)
    temperature: float=Field(default=0.7, ge=0, le=1)


def simple_agent(question:str, temperature: float)->str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=question,
        config={
            "temperature": temperature,
        }
    )
    return response.text


@app.post("/ask")
@limiter.limit("5/minute")
def ask(request: Request, question: Question, _: None = Depends(verify_api_key)):
    logger.info(
        "Request received",
        extra={"event": "request_received", "endpoint": "/ask"}
    )
    start_time = time.perf_counter()
    try:
        answer=simple_agent(question.question, question.temperature)
        duration = time.perf_counter() - start_time
        logger.info(
            "Answer generated successfully in %.2f seconds", duration,
            extra={
                "event": "llm_request_success",
                "endpoint": "/ask",
                "duration_seconds": round(duration, 2)
            }
    )
        return{"answer":answer}
    
    except Exception:
        duration = time.perf_counter() - start_time
        logger.error(
            "LLM request failed after %.2f seconds",
            duration,
            extra={
                "event": "llm_request_failure",
                "endpoint": "/ask",
                "duration_seconds": round(duration, 2)
            }
   )
        logger.exception("LLM request failed")
        raise HTTPException(
            status_code=502,
            detail="The LLM service failed. Please try again"
        )

@app.get("/hello")
def hello():
    return {"message": "Hello, World!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ready")
def ready():
    # GEMINI_MODEL = None
    if not GEMINI_MODEL:
        raise HTTPException(
            status_code=503,
            detail="Application is not ready"
        )

    return {"status": "ready"}

@app.get("/rate-test")
@limiter.limit("5/minute")
def rate_test(request: Request):
    return {"status": "ok"}