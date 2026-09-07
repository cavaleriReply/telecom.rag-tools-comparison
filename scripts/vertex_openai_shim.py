"""
Shim OpenAI-compatible per Gemini/Vertex AI.

Perché esiste: supermemory-server (lite) accetta come "model provider" solo
chiavi OpenAI / Anthropic / Gemini(AI Studio) / Groq, e come provider di embedding
solo ``local | openai | gemini``. Non c'è modo di dirgli "usa Vertex".
Noi però abbiamo solo il service account Vertex.

Questo shim finge di essere OpenAI e inoltra tutto a Vertex, riusando lo stesso
codice degli adapter (``adapters._vertex``). Così supermemory usa:
  - lo STESSO LLM di generazione/estrazione:  gemini-2.5-flash
  - lo STESSO embedder di LightRAG:            gemini-embedding-001 (1536d)

Endpoint implementati (il sottoinsieme che serve a supermemory):
  POST /v1/chat/completions   -> _vertex.acompletion_message  (inoltra anche tools/
                                 tool_choice: l'estrazione memorie è un agente a
                                 function-calling. Stream "finto": un chunk.)
  POST /v1/embeddings         -> _vertex.aembed  (--embedding-dimensions come
                                 default: supermemory non manda `dimensions`)
  GET  /v1/models             -> elenco statico
  GET  /health

SHIM_DEBUG=1 -> logga metodo/path/body di ogni richiesta.

Uso (di solito lo lancia scripts/supermemory_local.sh):
    poetry run python scripts/vertex_openai_shim.py --port 6799 --embedding-dimensions 1536
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adapters import _vertex  # noqa: E402  (dopo aver sistemato sys.path)


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------
# Parametri OpenAI che inoltriamo a Vertex tal quali. `tools`/`tool_choice`
# servono: supermemory estrae le memorie con un agente a function-calling.
_CHAT_PASSTHROUGH = (
    "temperature",
    "max_tokens",
    "top_p",
    "response_format",
    "tools",
    "tool_choice",
)


async def handle_chat(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    messages = body.get("messages") or []
    model = body.get("model") or _vertex.llm_model()

    passthrough = {k: body[k] for k in _CHAT_PASSTHROUGH if body.get(k) is not None}
    # Gemini rifiuta response_format insieme a tools: in quel caso vince tools.
    if "tools" in passthrough:
        passthrough.pop("response_format", None)

    msg = await _vertex.acompletion_message(messages, model=model, **passthrough)
    assistant_message: dict = {"role": "assistant", "content": msg["content"]}
    if msg["tool_calls"]:
        assistant_message["tool_calls"] = msg["tool_calls"]
    finish_reason = msg["finish_reason"]

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    tag = "tool_calls" if msg["tool_calls"] else f"{len(msg['content'] or '')} char"

    # Streaming "finto": un unico chunk SSE + [DONE]. A supermemory serve il
    # messaggio finale, non il flusso incrementale.
    if body.get("stream"):
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)

        def _sse(payload: dict) -> bytes:
            return f"data: {json.dumps(payload)}\n\n".encode()

        delta = {"role": "assistant", "content": msg["content"]}
        if msg["tool_calls"]:
            delta["tool_calls"] = msg["tool_calls"]
        await response.write(
            _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            )
        )
        await response.write(
            _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
            )
        )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        print(f"[shim] chat (stream) -> {tag}", flush=True)
        return response

    print(f"[shim] chat -> {tag}", flush=True)
    return web.json_response(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "message": assistant_message, "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------
def _as_text_list(raw_input: object) -> list[str]:
    """OpenAI accetta ``input`` come stringa o lista di stringhe. Normalizziamo a lista."""
    if isinstance(raw_input, str):
        return [raw_input]
    if isinstance(raw_input, list) and all(isinstance(x, str) for x in raw_input):
        return list(raw_input)
    raise web.HTTPBadRequest(reason="`input` deve essere una stringa o una lista di stringhe")


#: dimensione di embedding di default (da --embedding-dimensions).
#: supermemory blocca il "piano" a 1536 ma NON manda `dimensions` nelle richieste,
#: quindi senza questo default lo shim restituirebbe i 3072 nativi e i vettori
#: non entrerebbero nelle colonne vector(1536) → chunk non indicizzati.
_DEFAULT_EMBED_DIM: int | None = None


async def handle_embeddings(request: web.Request) -> web.Response:
    body = await request.json()
    texts = _as_text_list(body.get("input"))
    model = body.get("model", _vertex.embedding_model())

    # Priorità: dimensione richiesta esplicitamente > default dello shim > nativa.
    requested_dim = body.get("dimensions")
    target_dim = int(requested_dim) if requested_dim is not None else _DEFAULT_EMBED_DIM

    started = time.monotonic()
    vectors = await _vertex.aembed(texts, dimensions=target_dim)  # stesso path di LightRAG
    elapsed_ms = round((time.monotonic() - started) * 1000)

    if target_dim is not None and vectors.shape[1] != target_dim:
        raise web.HTTPInternalServerError(
            reason=f"dimensione attesa {target_dim}, ottenuta {vectors.shape[1]}"
        )

    print(f"[shim] embeddings {len(texts)} testi -> {vectors.shape} in {elapsed_ms}ms", flush=True)
    return web.json_response(
        {
            "object": "list",
            "model": model,
            "data": [
                {"object": "embedding", "index": i, "embedding": vec.tolist()}
                for i, vec in enumerate(vectors)
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    )


# ---------------------------------------------------------------------------
# endpoint accessori
# ---------------------------------------------------------------------------
async def handle_models(_: web.Request) -> web.Response:
    now = int(time.time())
    ids = [_vertex.llm_model().split("/")[-1], _vertex.embedding_model().split("/")[-1]]
    return web.json_response(
        {"object": "list", "data": [{"id": i, "object": "model", "created": now, "owned_by": "vertex-shim"} for i in ids]}
    )


async def handle_health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


@web.middleware
async def _log_middleware(request: web.Request, handler):
    """Con SHIM_DEBUG=1: logga metodo/path e un estratto del body di ogni richiesta."""
    body_preview = ""
    if request.method == "POST":
        # aiohttp memorizza il body al primo read(): gli handler lo rileggono.
        raw = await request.read()
        body_preview = raw[:600].decode("utf-8", "replace")
    try:
        response = await handler(request)
        print(f"[shim] {request.method} {request.path} -> {response.status}", flush=True)
    except web.HTTPException as exc:
        print(f"[shim] {request.method} {request.path} -> {exc.status}  body={body_preview}", flush=True)
        raise
    if body_preview:
        print(f"[shim]   req body: {body_preview}", flush=True)
    return response


def build_app() -> web.Application:
    middlewares = [_log_middleware] if os.environ.get("SHIM_DEBUG") == "1" else []
    app = web.Application(middlewares=middlewares)
    app.add_routes(
        [
            web.post("/v1/chat/completions", handle_chat),
            web.post("/chat/completions", handle_chat),
            web.post("/v1/embeddings", handle_embeddings),
            web.post("/embeddings", handle_embeddings),
            web.get("/v1/models", handle_models),
            web.get("/models", handle_models),
            web.get("/health", handle_health),
        ]
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6799)
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=None,
        help="Dimensione da usare quando la richiesta non specifica `dimensions` "
        "(supermemory non lo manda mai). Es. 1536.",
    )
    args = parser.parse_args()

    global _DEFAULT_EMBED_DIM
    _DEFAULT_EMBED_DIM = args.embedding_dimensions

    load_dotenv(_PROJECT_ROOT / ".env")
    print(
        f"[shim] default embedding dim = {_DEFAULT_EMBED_DIM or 'nativa (3072)'}",
        flush=True,
    )
    print(
        f"[shim] OpenAI-compat su http://{args.host}:{args.port}/v1  "
        f"(chat={_vertex.llm_model()}, embed={_vertex.embedding_model()})",
        flush=True,
    )
    web.run_app(build_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
