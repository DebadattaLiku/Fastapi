import os
from dotenv import load_dotenv
from google import genai
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import logging
from logging.handlers import RotatingFileHandler
import time
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

client = genai.Client() if os.getenv("GEMINI_API_KEY") else None
app = FastAPI()
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
def ask(request: Request, question: Question):
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
            "LLM request failed",
            extra={
                "event": "llm_request_failure",
                "endpoint": "/ask"
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