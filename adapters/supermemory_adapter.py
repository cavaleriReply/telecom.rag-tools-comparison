"""
Adapter per supermemory (supermemoryai/supermemory), testato *as-is*.

Differenza chiave rispetto a LightRAG: supermemory **recupera** passaggi/memorie
ma **non genera** la risposta. Quindi:

  * il retrieval (chunking, embedding, grafo) è tutto di supermemory e NON è
    configurabile dall'esterno — a differenza di LightRAG dove iniettiamo noi
    LLM ed embedding;
  * la risposta la generiamo noi da ciò che supermemory recupera, con il prompt
    neutro condiviso in ``adapters._rag_prompt`` (stesso LLM di tutti gli altri).

Di conseguenza il confronto con supermemory misura la **qualità del retrieval**
a generazione costante. Vedi `docs/supermemory.md`.

Deployment: hosted (default, API key) o self-hosted (``base_url`` sul binario
locale). L'adapter è identico nei due casi.

Ciclo di vita (vedi ``adapters.base.RAGAdapter``):
    adapter = SupermemoryAdapter(config=SupermemoryConfig(search_mode="hybrid"))
    await adapter.setup()
    await adapter.ingest(documents)   # add asincrono + polling fino a "done"
    ans = await adapter.query("...")  # search + generazione risposta
    await adapter.teardown()
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import supermemory as _supermemory_pkg
from supermemory import AsyncSupermemory

from adapters import _vertex
from adapters._rag_prompt import answer_from_contexts
from adapters.base import AdapterAnswer, Document


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SupermemoryConfig:
    #: Cosa cercare: "memories" (fatti estratti), "documents" (chunk grezzi),
    #: "hybrid" (entrambi, consigliato da supermemory).
    search_mode: str = "hybrid"
    #: Numero massimo di risultati recuperati per domanda.
    limit: int = 20
    #: Reranker di supermemory (~+100ms). Off per un retrieval "puro".
    rerank: bool = False
    #: Soglia di similarità 0-1: sotto questo valore i risultati vengono scartati.
    threshold: float = 0.5
    #: supermemory riscrive la query in più varianti e unisce i risultati.
    rewrite_query: bool = False
    #: "instant" = grafo disponibile subito (predicibile per un benchmark);
    #: "dynamic" = raggruppa i documenti prima di estrarre le memorie (default di supermemory).
    dreaming: str = "instant"
    #: None = API hosted (api.supermemory.ai); valorizzato = binario self-hosted.
    base_url: str | None = None
    #: Alla fine del run, cancella i documenti del container (utile sull'hosted).
    cleanup_on_teardown: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# L'adapter
# ---------------------------------------------------------------------------
class SupermemoryAdapter:
    """Traduttore sottile tra l'interfaccia comune e l'SDK di supermemory."""

    name = "supermemory"

    def __init__(
        self,
        config: SupermemoryConfig | None = None,
        *,
        api_key: str | None = None,
        container_tag: str | None = None,
    ) -> None:
        self.config = config or SupermemoryConfig()
        # Sull'hosted serve la key; sul self-hosted spesso no (stringa vuota ok).
        self._api_key = api_key if api_key is not None else os.environ.get("SUPERMEMORY_API_KEY", "")
        # container_tag = namespace isolato per questo (tool, config): due run non
        # si contaminano. Default derivato dall'hash della config.
        self._container_tag = container_tag or self._default_container_tag()
        self._client: AsyncSupermemory | None = None

    def _default_container_tag(self) -> str:
        digest = hashlib.sha1(repr(sorted(self.config.as_dict().items())).encode()).hexdigest()[:8]
        return f"olivettiV0__{self.config.search_mode}__{digest}"

    # -- setup -------------------------------------------------------------
    async def setup(self) -> None:
        if self._client is not None:
            return
        client_kwargs: dict[str, Any] = {"api_key": self._api_key or "self-hosted"}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        self._client = AsyncSupermemory(**client_kwargs)

    # -- ingest -----------------------------------------------------------
    async def ingest(self, documents: Sequence[Document]) -> None:
        """
        Carica i documenti e ASPETTA che siano indicizzati.

        ``documents.add`` è asincrono: ritorna subito un id e il documento passa
        per queued → ... → indexing → done. Per ``search_mode`` "hybrid"/"memories"
        aspettiamo anche la fase "dreaming" (estrazione del grafo di memorie).
        """
        client = self._require_client()
        doc_ids: list[str] = []
        for doc in documents:
            resp = await client.documents.add(
                content=doc.text,
                container_tag=self._container_tag,
                # supermemory accetta in custom_id solo [A-Za-z0-9_:-] (i nostri
                # doc_id hanno spazi e "!"). Il doc_id vero resta in metadata.
                custom_id=_safe_id(doc.doc_id),
                metadata={"doc_id": doc.doc_id},
                dreaming=self.config.dreaming,
            )
            doc_ids.append(resp.id)

        await self._wait_until_ready(doc_ids)

    async def _wait_until_ready(
        self, doc_ids: list[str], *, poll_seconds: float = 5.0, timeout_seconds: float = 1800.0
    ) -> None:
        needs_dreaming = self.config.search_mode in ("hybrid", "memories")
        deadline = time.monotonic() + timeout_seconds
        pending = set(doc_ids)

        while pending and time.monotonic() < deadline:
            for doc_id in list(pending):
                doc = await self._require_client().documents.get(doc_id)
                if doc.status == "failed":
                    raise RuntimeError(f"supermemory: indicizzazione fallita per il documento {doc_id}")
                indexed = doc.status == "done"
                dreamed = (not needs_dreaming) or doc.dreaming_status == "done"
                if indexed and dreamed:
                    pending.discard(doc_id)
            if pending:
                await asyncio.sleep(poll_seconds)

        if pending:
            raise TimeoutError(
                f"supermemory: {len(pending)} documenti non pronti dopo {timeout_seconds:.0f}s"
            )

    # -- query ----------------------------------------------------------
    async def query(self, question: str) -> AdapterAnswer:
        client = self._require_client()

        started = time.monotonic()
        res = await client.search.memories(
            q=question,
            container_tag=self._container_tag,
            search_mode=self.config.search_mode,
            limit=self.config.limit,
            rerank=self.config.rerank,
            threshold=self.config.threshold,
            rewrite_query=self.config.rewrite_query,
        )
        retrieval_seconds = time.monotonic() - started

        contexts = _extract_contexts(res)
        answer = await answer_from_contexts(question, contexts)

        return AdapterAnswer(
            answer=answer,
            contexts=contexts,
            raw={
                "retrieval_latency_s": round(retrieval_seconds, 3),
                "n_results": res.total,
                "timing_ms": res.timing,
                "results": res.model_dump(mode="json").get("results"),
            },
        )

    # -- teardown ---------------------------------------------------------
    async def teardown(self) -> None:
        if self._client is None:
            return
        if self.config.cleanup_on_teardown:
            try:
                await self._client.documents.delete_bulk(container_tags=[self._container_tag])
            except Exception:  # noqa: BLE001 - il cleanup non deve far fallire il run
                pass
        await self._client.close()
        self._client = None

    # -- introspezione --------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "supermemory_sdk_version": _supermemory_pkg.__version__,
            "config": self.config.as_dict(),
            "deployment": "self-hosted" if self.config.base_url else "hosted",
            "base_url": self.config.base_url or "https://api.supermemory.ai",
            "container_tag": self._container_tag,
            # La risposta la generiamo noi: embedding e chunking restano di supermemory.
            "generation_llm": _vertex.llm_model(),
            "note": "retrieval-only: chunking/embedding di supermemory, risposta generata da noi",
        }

    # -- interni --------------------------------------------------------
    def _require_client(self) -> AsyncSupermemory:
        if self._client is None:
            raise RuntimeError("Chiama prima `await adapter.setup()`.")
        return self._client


def _safe_id(doc_id: str) -> str:
    """doc_id -> slug accettato da supermemory come custom_id ([A-Za-z0-9_:-])."""
    slug = re.sub(r"[^A-Za-z0-9_:-]+", "-", doc_id).strip("-")
    return slug or "doc"


def _extract_contexts(res: Any) -> list[str]:
    """
    Appiattisce i risultati di supermemory in una lista di stringhe, deduplicata.

    Un result può portare: un ``memory`` (fatto estratto), un ``chunk`` o una
    lista ``chunks`` (testo grezzo), e un ``context`` col vicinato nel grafo
    (memorie parent/child/related). Includiamo tutto: per la *context recall* ci
    interessa sapere se il fatto atteso era da qualche parte nel recuperato.
    """
    collected: list[str] = []

    for result in res.results:
        if getattr(result, "memory", None):
            collected.append(result.memory)
        if getattr(result, "chunk", None):
            collected.append(result.chunk)
        for chunk in getattr(result, "chunks", None) or []:
            content = getattr(chunk, "content", None)
            if content:
                collected.append(content)

        graph_context = getattr(result, "context", None)
        if graph_context:
            for group_name in ("parents", "children", "related"):
                for item in getattr(graph_context, group_name, None) or []:
                    if getattr(item, "memory", None):
                        collected.append(item.memory)

    seen: set[str] = set()
    unique: list[str] = []
    for text in collected:
        stripped = text.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique.append(stripped)
    return unique
