"""
Accesso ai modelli Gemini su Vertex AI, condiviso da tutti gli adapter.

Un solo punto di contatto con i modelli (via ``litellm``), così ogni framework
usa lo stesso LLM e lo stesso embedding, letti dalle stesse variabili ``.env``
(``LLM_MODEL`` / ``EMBEDDING_MODEL``). Se un tool si comporta diversamente da un
altro, non è perché parla con Vertex in un modo diverso.

Qui dentro solo due cose:
  * ritentativi con backoff sugli errori transitori di Vertex (i 429 in primis);
  * un helper per scoprire la dimensione reale dei vettori di embedding.
"""

from __future__ import annotations

import asyncio
import os

import litellm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# litellm stampa un banner "Give Feedback / Get Help" e righe INFO per ogni
# chiamata (ed eccezione ritentata): rumore inutile nei log dell'ingest, dove
# LightRAG fa centinaia di chiamate e qualche timeout transitorio è normale.
litellm.suppress_debug_info = True

# Errori che ha senso ritentare: rate limit e indisponibilità temporanee di
# Vertex. Gli errori di prompt/permessi NON sono qui: devono fallire subito.
_TRANSIENT_ERRORS = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.Timeout,
)

# Politica di ritentativo comune (usata sia per completion che per embedding).
_with_retry = retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)

# gemini-embedding-001 accetta una sola istanza per richiesta: la "batch" la
# facciamo noi, con un tetto di richieste in volo per non innescare i 429.
_EMBED_MAX_CONCURRENCY = 8
_embed_semaphore = asyncio.Semaphore(_EMBED_MAX_CONCURRENCY)


def llm_model() -> str:
    """Modello LLM condiviso (es. ``vertex_ai/gemini-2.5-flash``)."""
    return os.environ["LLM_MODEL"]


def embedding_model() -> str:
    """Modello di embedding condiviso (es. ``vertex_ai/gemini-embedding-001``)."""
    return os.environ["EMBEDDING_MODEL"]


@_with_retry
async def acompletion_message(
    messages: list[dict], *, model: str | None = None, **kwargs
) -> dict:
    """
    Completion con ritentativi, ritorna il MESSAGGIO completo in forma OpenAI:
    ``{"content": str | None, "tool_calls": list | None, "finish_reason": str}``.

    Serve allo shim, che deve poter inoltrare anche le tool-call (supermemory usa
    un agente con function-calling per l'estrazione delle memorie).
    """
    response = await litellm.acompletion(
        model=model or llm_model(), messages=messages, **kwargs
    )
    choice = response.choices[0]
    message = choice.message
    tool_calls = message.tool_calls or None
    if tool_calls is not None:
        # normalizza in dict semplici serializzabili
        tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
    return {
        "content": message.content,
        "tool_calls": tool_calls,
        "finish_reason": choice.finish_reason or "stop",
    }


async def acompletion(messages: list[dict], *, model: str | None = None, **kwargs) -> str:
    """Come sopra ma ritorna solo il testo (nessuna tool-call)."""
    result = await acompletion_message(messages, model=model, **kwargs)
    return result["content"] or ""


@_with_retry
async def _aembed_one(text: str, *, model: str, dimensions: int | None) -> list[float]:
    """Embedding di un singolo testo, con ritentativi e tetto di concorrenza."""
    # `dimensions` -> `output_dimensionality` di gemini-embedding-001: se None,
    # si ottiene la dimensione nativa (3072); altrimenti quella richiesta.
    extra = {} if dimensions is None else {"dimensions": dimensions}
    async with _embed_semaphore:
        response = await litellm.aembedding(model=model, input=[text], **extra)
    return response.data[0]["embedding"]


async def aembed(
    texts: list[str], *, model: str | None = None, dimensions: int | None = None
) -> np.ndarray:
    """
    Embedding di una lista di testi -> array ``(n_testi, dim)``.

    ``dimensions`` = ``None`` (default) usa la dimensione nativa del modello.
    Lo si valorizza solo quando un consumatore esterno chiede una taglia precisa
    (vedi ``scripts/vertex_embeddings_shim.py``); LightRAG usa sempre il default.

    I testi di una stessa chiamata vengono elaborati in parallelo (fino a
    ``_EMBED_MAX_CONCURRENCY`` richieste in volo): l'ingest di LightRAG genera
    centinaia di embedding di entità/relazioni e in sequenza sarebbe lentissimo.
    """
    model = model or embedding_model()
    vectors = await asyncio.gather(
        *(_aembed_one(t, model=model, dimensions=dimensions) for t in texts)
    )
    arr = np.array(vectors, dtype=np.float32)

    # gemini-embedding-001 restituisce vettori L2-normalizzati SOLO a 3072 dim;
    # a dimensioni ridotte no (Google lo documenta). Normalizziamo sempre noi,
    # così ogni vector store si comporta uguale a prescindere dalla metrica
    # (cosine, inner product, L2) e i due tool restano confrontabili.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


async def probe_embedding_dim(*, model: str | None = None, dimensions: int | None = None) -> int:
    """Dimensione reale dei vettori restituiti, misurata con una chiamata."""
    vec = await aembed(["probe"], model=model, dimensions=dimensions)
    return int(vec.shape[1])
