import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rag_runtime import RAGServerConfig, ScholarChatRuntime


class ChatTurn(BaseModel):
    role: str = Field(..., description="user|assistant")
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: List[ChatTurn] = Field(default_factory=list)
    top_k: Optional[int] = None
    similarity_threshold: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, int]
    debug: Dict[str, Any] = Field(default_factory=dict)


BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"

config = RAGServerConfig(base_path=str(BASE_DIR))
runtime = ScholarChatRuntime(config)
init_thread: Optional[threading.Thread] = None

app = FastAPI(title="ScholarChat API", version="0.1.0")

# If you later host frontend separately, keep this. If serving from same origin, it's still safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_init() -> None:
    # Start initialization in the background so the server can respond to /health
    # while models are still loading.
    t0 = time.time()

    def _bg_init() -> None:
        runtime.initialize()

    global init_thread
    th = threading.Thread(target=_bg_init, name="scholarchat-init", daemon=True)
    th.start()
    init_thread = th
    _ = int((time.time() - t0) * 1000)


@app.get("/health")
def health() -> Dict[str, Any]:
    s = runtime.status()
    s.update(
        {
            "faiss_dir": config.faiss_dir,
            "models_dir": config.llm_models_dir,
            "pid": os.getpid(),
            "init_thread_alive": bool(init_thread and init_thread.is_alive()),
        }
    )
    return s


if __name__ == "__main__":
    import uvicorn

    # Run with: python -m server.app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    payload, retrieval_ms, generation_ms, total_ms = runtime.answer(
        message=req.message,
        history=[t.model_dump() for t in req.history],
        top_k=req.top_k,
        similarity_threshold=req.similarity_threshold,
    )

    return ChatResponse(
        answer=payload.get("answer", ""),
        sources=payload.get("sources", []),
        metrics={
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
        },
        debug=payload.get("debug", {}),
    )


# Static frontend
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        # Keep API usable even if the frontend files are missing
        return FileResponse(str((BASE_DIR / "README.md")))
    return FileResponse(str(index_file))


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(str(WEB_DIR / "app.js"))


@app.get("/styles.css")
def styles_css() -> FileResponse:
    return FileResponse(str(WEB_DIR / "styles.css"))


@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)
