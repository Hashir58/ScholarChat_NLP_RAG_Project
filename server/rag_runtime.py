import os
import time
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RAGServerConfig:
    base_path: str = str(Path(__file__).resolve().parents[1])

    faiss_dir: str = "faiss_vector_store_568"
    faiss_index_filename: str = "faiss_index_568.index"
    document_metadata_filename: str = "document_metadata_568.json"

    embedding_model_local_dir: str = os.path.join("models", "intfloat__e5-base")
    embedding_model_name: str = "intfloat/e5-base"

    llm_models_dir: str = "models"
    llm_gguf_preferred: str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    top_k_dense: int = 3
    similarity_threshold: float = 0.75

    max_tokens: int = 900
    temperature: float = 0.3
    top_p: float = 0.9
    context_length: int = 4096

    n_threads: Optional[int] = None
    n_gpu_layers: int = -1

    max_history_turns: int = 8
    max_message_chars: int = 2000


class ScholarChatRuntime:
    def __init__(self, config: Optional[RAGServerConfig] = None):
        self.config = config or RAGServerConfig()

        self._faiss = None
        self._index = None
        self._doc_meta: List[Dict[str, Any]] = []

        self._embedding_model = None
        self._llm = None

        self._init_error: Optional[str] = None
        self._initialized: bool = False

        self._init_lock = threading.Lock()
        self._init_started_at: Optional[float] = None
        self._init_completed_at: Optional[float] = None
        self._init_stage: str = "not_started"
        self._init_timings_ms: Dict[str, int] = {}
        self._init_events: List[Dict[str, Any]] = []

        self._generation_lock = threading.Lock()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def status(self) -> Dict[str, Any]:
        now = time.time()
        started = self._init_started_at
        completed = self._init_completed_at

        elapsed_ms = 0
        if started is not None:
            elapsed_ms = int(((completed or now) - started) * 1000)

        return {
            "ready": self._initialized,
            "stage": self._init_stage,
            "elapsed_ms": elapsed_ms,
            "error": self._init_error,
            "timings_ms": dict(self._init_timings_ms),
            "events": self._init_events[-25:],
            "llm_available": self._llm is not None,
        }

    def _event(self, stage: str, message: str, level: str = "info") -> None:
        self._init_stage = stage
        self._init_events.append(
            {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "level": level,
                "stage": stage,
                "message": message,
            }
        )

    def _abs(self, relative_path: str) -> str:
        return str(Path(self.config.base_path) / relative_path)

    def initialize(self) -> None:
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return
            if self._init_stage in {"starting", "loading_faiss", "loading_metadata", "loading_embedder", "loading_llm"}:
                return

            self._init_started_at = time.time()
            self._init_completed_at = None
            self._init_timings_ms = {}
            self._init_error = None
            self._event("starting", "Initialization started")

            try:
                t0 = time.time()
                self._event("loading_faiss", "Loading FAISS index")
                self._load_faiss()
                self._init_timings_ms["faiss_ms"] = int((time.time() - t0) * 1000)

                t0 = time.time()
                self._event("loading_metadata", "Loading document metadata")
                self._load_metadata()
                self._init_timings_ms["metadata_ms"] = int((time.time() - t0) * 1000)

                t0 = time.time()
                self._event("loading_embedder", "Loading embedding model")
                self._load_embedder()
                self._init_timings_ms["embedder_ms"] = int((time.time() - t0) * 1000)

                t0 = time.time()
                self._event("loading_llm", "Loading GGUF LLM (llama-cpp)")
                self._load_llm()
                self._init_timings_ms["llm_ms"] = int((time.time() - t0) * 1000)

                self._initialized = True
                self._init_completed_at = time.time()
                self._event("ready", "Initialization complete")

            except Exception as exc:
                self._initialized = False
                self._init_completed_at = time.time()
                self._init_error = f"{type(exc).__name__}: {exc}"
                self._event("error", self._init_error, level="error")

    def _load_faiss(self) -> None:
        try:
            import faiss  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "FAISS is not installed. Install `faiss-cpu` (or `faiss-gpu`) in your environment."
            ) from exc

        index_path = self._abs(os.path.join(self.config.faiss_dir, self.config.faiss_index_filename))
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        self._faiss = faiss
        self._index = faiss.read_index(index_path)

    def _load_metadata(self) -> None:
        meta_path = self._abs(os.path.join(self.config.faiss_dir, self.config.document_metadata_filename))
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Document metadata not found at {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self._doc_meta = json.load(f)

        if not isinstance(self._doc_meta, list) or not self._doc_meta:
            raise ValueError("Document metadata JSON is empty or invalid")

    def _load_embedder(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        local_dir = self._abs(self.config.embedding_model_local_dir)
        if os.path.isdir(local_dir):
            self._embedding_model = SentenceTransformer(local_dir)
        else:
            self._embedding_model = SentenceTransformer(self.config.embedding_model_name)

    def _resolve_llm_path(self) -> Optional[str]:
        models_dir = Path(self._abs(self.config.llm_models_dir))
        preferred = models_dir / self.config.llm_gguf_preferred
        if preferred.exists():
            return str(preferred)

        # Fallback: first GGUF in models folder
        ggufs = sorted(models_dir.glob("*.gguf"))
        if ggufs:
            return str(ggufs[0])
        return None

    def _load_llm(self) -> None:
        llm_path = self._resolve_llm_path()
        if llm_path is None:
            self._llm = None
            return

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception:
            self._llm = None
            return

        n_threads = self.config.n_threads or (os.cpu_count() or 4)

        # Loading GGUF can take time; keep it once per process
        self._llm = Llama(
            model_path=llm_path,
            n_ctx=self.config.context_length,
            n_threads=n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False,
        )

    def _embed_query(self, query: str) -> np.ndarray:
        prefixed = f"query: {query}"
        embedding = self._embedding_model.encode([prefixed], normalize_embeddings=True)
        return np.asarray(embedding, dtype=np.float32)[0]

    def _search(self, query_embedding: np.ndarray, top_k: int, threshold: float) -> List[Dict[str, Any]]:
        scores, indices = self._index.search(query_embedding.reshape(1, -1), top_k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self._doc_meta):
                continue
            if float(score) < threshold:
                continue

            doc = dict(self._doc_meta[idx])
            doc["similarity_score"] = float(score)
            doc["retrieval_rank"] = rank
            results.append(doc)

        results.sort(key=lambda d: d.get("similarity_score", 0.0), reverse=True)
        return results

    def _format_prompt(self, message: str, history: List[Dict[str, str]], retrieved: List[Dict[str, Any]]) -> str:
        # Keep history compact
        history = history[-(self.config.max_history_turns * 2) :]

        context_parts: List[str] = []
        for i, doc in enumerate(retrieved, start=1):
            file_name = doc.get("file_name") or Path(doc.get("source_file", "")).name or "Unknown"
            content = doc.get("content") or doc.get("text") or ""
            score = float(doc.get("similarity_score", 0.0))
            context_parts.append(
                f"<document id={i} source=\"{file_name}\" relevance={score:.3f}>\n{content}\n</document>"
            )

        context_str = "\n".join(context_parts)

        conversation_lines: List[str] = []
        for turn in history:
            role = (turn.get("role") or "").strip().lower()
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if role not in {"user", "assistant"}:
                continue
            conversation_lines.append(f"{role.upper()}: {content}")

        conversation_block = "\n".join(conversation_lines)

        system = (
            "You are ScholarChat, an AI assistant for course materials. "
            "Answer using ONLY the provided context documents. "
            "If the documents do not contain the answer, say so clearly. "
            "Cite sources like: [Document X: filename.ext]."
        )

        prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}\n\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Context Documents:\n{context_str}\n\n"
            f"Conversation (optional):\n{conversation_block}\n\n"
            f"Question: {message}\n\n"
            f"Answer based ONLY on the provided documents.\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        return prompt

    def answer(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], int, int, int]:
        self.initialize()
        if not self._initialized:
            return (
                {
                    "answer": f"Server not ready: {self._init_error}",
                    "sources": [],
                    "debug": {"ready": False},
                },
                0,
                0,
                0,
            )

        message = (message or "").strip()
        if not message:
            return ({"answer": "Please enter a question.", "sources": [], "debug": {}}, 0, 0, 0)

        if len(message) > self.config.max_message_chars:
            message = message[: self.config.max_message_chars]

        history = history or []
        used_top_k = int(top_k or self.config.top_k_dense)
        used_top_k = max(1, min(used_top_k, 10))

        used_threshold = float(similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold)
        used_threshold = max(0.0, min(1.0, used_threshold))

        t0 = time.time()
        q_emb = self._embed_query(message)
        retrieved = self._search(q_emb, used_top_k, used_threshold)
        retrieval_ms = int((time.time() - t0) * 1000)

        if not retrieved:
            return (
                {
                    "answer": "Based on the provided documents, I could not find specific information to answer your query.",
                    "sources": [],
                    "debug": {"retrieved": 0},
                },
                retrieval_ms,
                0,
                retrieval_ms,
            )

        prompt = self._format_prompt(message, history, retrieved)

        t1 = time.time()
        if self._llm is None:
            # Safe fallback (no crashes) if llama-cpp is missing
            sources = [
                {
                    "document": i + 1,
                    "file_name": (doc.get("file_name") or Path(doc.get("source_file", "")).name or "Unknown"),
                    "score": float(doc.get("similarity_score", 0.0)),
                }
                for i, doc in enumerate(retrieved)
            ]
            answer_text = (
                "[LLM NOT AVAILABLE]\n\n"
                "Retrieved relevant documents, but the LLM backend is not available.\n"
                "Install `llama-cpp-python` and ensure the GGUF model exists in `models/`.\n"
            )
            generation_ms = int((time.time() - t1) * 1000)
            total_ms = retrieval_ms + generation_ms
            return (
                {"answer": answer_text, "sources": sources, "debug": {"retrieved": len(retrieved), "simulated": True}},
                retrieval_ms,
                generation_ms,
                total_ms,
            )

        with self._generation_lock:
            out = self._llm.create_completion(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=["<|eot_id|>"],
                echo=False,
            )

        generation_ms = int((time.time() - t1) * 1000)
        total_ms = retrieval_ms + generation_ms

        text = (out.get("choices") or [{}])[0].get("text", "")
        answer_text = (text or "").strip() or "(No output)"

        sources = [
            {
                "document": i + 1,
                "file_name": (doc.get("file_name") or Path(doc.get("source_file", "")).name or "Unknown"),
                "score": float(doc.get("similarity_score", 0.0)),
            }
            for i, doc in enumerate(retrieved)
        ]

        return (
            {
                "answer": answer_text,
                "sources": sources,
                "debug": {"retrieved": len(retrieved)},
            },
            retrieval_ms,
            generation_ms,
            total_ms,
        )
