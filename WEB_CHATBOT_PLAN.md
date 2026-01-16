# ScholarChat (RAG) Web Chatbot — Option B Plan (HTML/CSS + minimal JS + Python API)

## Goal

Build a modern, responsive chatbot website for your existing RAG pipeline so you can ask multiple questions, see answers clearly, and optionally view sources/metadata.

This plan targets your current artifacts:

- Pipeline file: [pipelines/rag_pipeline_568_v2.pkl](pipelines/rag_pipeline_568_v2.pkl)
- Notebook that currently defines the classes / logic: [testing_on_orginal_Chunks.ipynb](testing_on_orginal_Chunks.ipynb)
- FAISS + metadata artifacts: [faiss_vector_store_568/](faiss_vector_store_568/)
- Local models: [models/](models/)

## Key constraint (important)

Pickle loading requires the **exact same class definitions to be importable** in the Python process that loads the `.pkl`.

Your current notebook-based workflow often pickles classes under `__main__` (Jupyter), which can cause:

- `AttributeError: Can't get attribute 'RAGPipeline568' on <module '__main__'>`

### Recommended approach for the web app

Avoid relying on unpickling the whole pipeline object. Instead:

- Load FAISS index + metadata + embedding model + GGUF model directly in the API server at startup.

If you still want to use the `.pkl`, we will stabilize it by:

- Moving the pipeline classes into a proper Python module (example: `rag/pipeline_568.py`)
- Recreating + repickling the pipeline so it points to that module path

## Architecture (Option B)

**Frontend (static):**

- `index.html` + `styles.css` (modern UI)
- `app.js` (minimal JS) to send messages to backend and render responses

**Backend (Python API):**

- `POST /api/chat` accepts a user message + optional conversation context
- Returns JSON: answer text, sources, timing, and optional retrieval details

### Why minimal JS is needed

Without JS, HTML/CSS cannot:

- Send chat messages asynchronously
- Append messages to the transcript without full page reload
- Implement “thinking…” state, retry, scroll-to-bottom

We’ll keep JS very small: `fetch()` + DOM updates only.

## Proposed folder layout

Create a small web app folder without touching your data artifacts:

- `web/`
  - `index.html`
  - `styles.css`
  - `app.js`
- `server/`
  - `app.py` (FastAPI or Flask)
  - `rag_runtime.py` (loads models/index once; exposes `answer()`)
- `rag/` (optional, recommended)
  - `pipeline_568.py` (module version of notebook classes)
- `WEB_CHATBOT_PLAN.md` (this file)

## Backend choice

Recommended: **FastAPI** (clean JSON APIs, easy to run locally)
Alternative: Flask (also fine)

The plan below assumes FastAPI, but the structure is identical with Flask.

## API contract

### Endpoint

`POST /api/chat`

### Request JSON

```json
{
  "message": "What is private cloud?",
  "session_id": "optional-session-id",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "top_k": 3,
  "similarity_threshold": 0.75
}
```

Notes:

- `history` lets the UI support multi-turn conversation.
- `top_k` and `similarity_threshold` are optional overrides; defaults come from server config.

### Response JSON

```json
{
  "answer": "...",
  "sources": [{ "document": 1, "file_name": "icc02_...txt", "score": 0.83 }],
  "metrics": {
    "retrieval_ms": 42,
    "generation_ms": 51234,
    "total_ms": 51290
  },
  "debug": {
    "retrieved": 3
  }
}
```

UI will display:

- Answer text (primary)
- Sources list (collapsible)
- Optional timing/debug in a small footer

## Runtime loading strategy (reliable)

### What the server loads at startup

- FAISS index from [faiss_vector_store_568/faiss_index_568.index](faiss_vector_store_568/faiss_index_568.index)
- Document metadata from [faiss_vector_store_568/document_metadata_568.json](faiss_vector_store_568/document_metadata_568.json)
- Embedding model `intfloat/e5-base` (prefer local cache under [models/intfloat\_\_e5-base/](models/intfloat__e5-base/), fallback to Hugging Face)
- GGUF model under [models/](models/) (your llama file)

### Why this is better than unpickling

- No dependency on notebook cell execution order
- Avoids `__main__` pickle issues
- More portable to GitHub + deployment

### Server config

Create a simple config block (or env vars) for:

- index path
- metadata path
- embedding model name/path
- llm path
- `top_k`
- `similarity_threshold`
- `max_tokens`, `temperature`, `top_p`, `context_length`

## Frontend UI plan (modern, responsive)

### Screens / layout

Single page app layout (one page):

- Header: “ScholarChat” + subtitle + status indicator (Ready / Thinking / Error)
- Chat transcript panel (scrollable)
  - User messages on right
  - Assistant messages on left
  - Each assistant answer includes a “Sources” toggle section
- Composer (sticky at bottom)
  - Textarea (Enter to send, Shift+Enter for newline)
  - Send button
- Optional: right-side drawer on desktop (sources/debug). On mobile, sources appear inline.

### Chat UX premises (minimum expected for this project)

- Clear separation of user vs assistant messages
- Loading indicator while the backend is generating
- Error banner with retry button if request fails
- Prevent double-send while a request is in-flight
- Auto-scroll to latest message
- Basic accessibility:
  - visible focus states
  - keyboard-friendly send
  - good contrast

### CSS-only design goals

- Use CSS variables for theme tokens (colors, spacing)
- Responsive layout with flex/grid
- Smooth message bubble styling
- Consistent typography and spacing

## Handling multi-turn conversation

Frontend will maintain an in-memory `history[]` array.
On each user message:

1. Append user message to transcript
2. POST `{message, history}` to backend
3. Append assistant response
4. Keep history bounded (e.g., last 10 turns) to avoid huge prompts

Backend can either:

- Use the history to build a better prompt (optional)
- Or ignore history initially and add it later

## Security / safety basics (practical)

For a portfolio project, include:

- Input length limits (e.g., 2k chars) to avoid runaway latency
- Basic sanitization: render assistant output as plain text (no HTML injection)
- CORS: allow only your frontend origin in production
- Don’t expose local absolute paths in API responses

## Performance considerations

Your notebook suggests queries can take ~50 seconds.
Plan:

- Add a frontend “Thinking…” state
- Increase server timeout accordingly (and show a helpful message)
- Load models once at server startup (do not reload per request)

Optional later:

- Streaming tokens (requires more JS + server streaming)

## Milestones (implementation sequence)

### Milestone 1 — Skeleton

- Create `web/index.html` and `web/styles.css` with a polished static layout
- Add dummy chat transcript examples for UI testing

### Milestone 2 — API server

- Create a FastAPI server with `/health` and `/api/chat`
- Implement `rag_runtime.py` that loads FAISS + metadata + embedder once
- Return a stub response first (verify the UI wiring)

### Milestone 3 — Real RAG answer

- Integrate retrieval + prompt formatting + llama generation
- Include sources in response JSON

### Milestone 4 — Polish

- Add robust error handling + retry
- Add collapsible sources UI
- Add small settings (top-k, threshold) as optional controls

## Local run workflow (target)

- Start backend: `python -m server.app`
- Open frontend:
  - simplest: serve `web/` as static from the backend, or
  - use a simple static server for `web/`

## Implementation notes (what we will do when you say “start”)

- Extract the relevant parts of [testing_on_orginal_Chunks.ipynb](testing_on_orginal_Chunks.ipynb) into importable Python code.
- Prefer artifact-based loading from [faiss_vector_store_568/](faiss_vector_store_568/) instead of unpickling.
- Keep the frontend truly “simple”: HTML/CSS for design, minimal JS only for sending/receiving chat.

---

## Decisions to confirm before implementation

1. Do you want the UI to show sources by default or behind a toggle?
2. Do you want a light theme only, or light + dark toggle (CSS variables + one JS toggle)?
3. Should we host the frontend from the Python backend (single command run), or keep it separate?
