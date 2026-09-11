"""
Adapter per R2R (SciPhi-AI/R2R), testato *as-is*.

R2R è diverso dagli altri tre nel deployment: non è una libreria Python (LightRAG,
cognee) né un binario locale con SDK (supermemory) — è un **server** (FastAPI)
appoggiato a **Postgres+pgvector obbligatorio**, che parliamo via REST (API v3).
Il server va avviato a parte, vedi ``scripts/r2r_local.sh`` e ``docs/r2r.md`` per
i dettagli / il perché.

⚠️ Niente SDK ufficiale (``r2r`` da PyPI): trascina ``fastapi<0.116`` in modo
incondizionato (non solo nell'extra ``[core]`` del server), in conflitto con
``cognee`` che vuole ``fastapi>=0.116.2`` — impossibile risolvere insieme in
questo poetry env. L'adapter parla REST diretto con ``httpx`` (già una
dipendenza transitiva via litellm): i payload/endpoint qui sotto ricalcano
esattamente quelli dell'SDK ufficiale (ispezionato per ricavarli), non sono
reverse-engineering della REST API a occhio.

R2R usa litellm nativamente per LLM ed embedding (config server-side, non env
pre-import come cognee) → **nessuno shim**: il server è puntato su Vertex tramite
``adapters/r2r_config/r2r_olivetti.toml`` (stessi modelli, stessa dim 1536 di
tutti gli altri). R2R **genera la risposta** (``retrieval/rag``), come LightRAG
e cognee.

Due bug/limiti reali di R2R 3.6.6 scoperti lanciando il server per la prima
volta (2026-09-11), entrambi aggirati qui:

1. ``automatic_extraction`` (entità/relazioni del grafo, nel toml) **non è
   implementato** con l'orchestrazione ``simple`` — quella di default nel
   deployment "light" che usiamo (``docker-compose.r2r.yml``): il server logga
   solo un warning (``Automatic extraction not yet implemented for 'simple'
   ingestion workflows.``) e ``extraction_status`` resta ``pending`` per
   sempre. L'adapter chiama quindi esplicitamente ``POST /documents/{id}/extract``
   (``run_with_orchestration=false``, sincrona) e poi ``POST /graphs/{collection_id}/pull``
   (copia entità/relazioni nel grafo della collezione **e ne calcola gli
   embedding** — senza, ``graph_search`` non trova nulla:
   ``description_embedding`` resta NULL). Vedi ``_ensure_graph_extracted``.
2. Il preset ``search_mode="advanced"`` di R2R imposta di serie
   ``search_strategy="hyde"`` (query expansion via LLM prima della ricerca).
   La chiamata HyDE di R2R manda al provider LLM un messaggio con **solo il
   ruolo "system", senza "user"**: con Gemini/Vertex (a differenza di OpenAI)
   questo produce un ``contents`` vuoto → ``400 Unable to submit request
   because at least one contents field is required``, e la *ricerca stessa*
   fallisce (non solo la generazione). L'adapter non usa mai i preset
   ``basic``/``advanced`` di R2R: costruisce sempre ``search_settings``
   espliciti con ``search_mode="custom"`` (default ``search_strategy="vanilla"``,
   niente HyDE) — vedi ``R2RConfig.search_settings``.

Ciclo di vita (vedi ``adapters.base.RAGAdapter``):
    adapter = R2RAdapter(base_url="http://localhost:7272", config=R2RConfig())
    await adapter.setup()            # apre il client HTTP, verifica che il server risponda
    await adapter.ingest(documents)  # POST /documents + polling + extract/pull espliciti
    ans = await adapter.query("...") # POST /retrieval/rag (search_mode="custom") -> risposta + citazioni
    await adapter.teardown()         # chiude il client HTTP
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import httpx

from adapters import _vertex
from adapters.base import AdapterAnswer, Document


class R2RError(RuntimeError):
    """Errore HTTP dal server R2R. ``status_code`` distingue i casi noti (409 = duplicato)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"R2R [{status_code}]: {message}")
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class R2RConfig:
    #: "advanced" = ricerca ibrida (semantica + full-text) + ricerca sul grafo
    #: (analogo di LightRAG "hybrid" / cognee GRAPH_COMPLETION).
    #: "basic" = solo ricerca semantica (il grafo resta comunque interrogato:
    #: `GraphSearchSettings.enabled` è True di default in R2R e qui non lo tocchiamo).
    search_mode: str = "advanced"
    #: Numero massimo di chunk/entità/relazioni recuperati (``search_settings.limit``).
    limit: int = 20

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def search_settings(self) -> dict[str, Any]:
        """
        Traduce ``search_mode`` in ``search_settings`` espliciti per R2R.

        Non usiamo MAI i preset ``search_mode="basic"/"advanced"`` di R2R nella
        richiesta: il preset "advanced" porta con sé ``search_strategy="hyde"``,
        che con Gemini/Vertex rompe la ricerca (vedi il bug #2 nel docstring del
        modulo). Qui replichiamo lo stesso ibrido/non-ibrido di "advanced"/"basic"
        ma lasciando ``search_strategy`` al suo default (``"vanilla"``, niente HyDE).
        """
        hybrid = self.search_mode == "advanced"
        return {
            "use_semantic_search": True,
            "use_fulltext_search": hybrid,
            "use_hybrid_search": hybrid,
            "limit": self.limit,
        }


# ---------------------------------------------------------------------------
# L'adapter
# ---------------------------------------------------------------------------
class R2RAdapter:
    """Traduttore sottile tra l'interfaccia comune e la REST API v3 di R2R."""

    name = "r2r"

    def __init__(self, base_url: str = "http://localhost:7272", config: R2RConfig | None = None) -> None:
        self.base_url = base_url
        self.config = config or R2RConfig()
        self._http: httpx.AsyncClient | None = None

    # -- setup -------------------------------------------------------------
    async def setup(self) -> None:
        if self._http is not None:
            return
        http = httpx.AsyncClient(base_url=self.base_url, timeout=300.0)
        try:
            response = await http.get("/v3/health")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            await http.aclose()
            raise RuntimeError(
                f"R2R non risponde su {self.base_url}. Avvia il server con "
                "`./scripts/r2r_local.sh` prima di lanciare il benchmark."
            ) from exc
        self._http = http

    # -- ingest --------------------------------------------------------
    async def ingest(self, documents: Sequence[Document]) -> None:
        """
        Ingesta i documenti mancanti, aspetta che siano indicizzati e (se la
        config lo richiede) estrae il grafo.

        Idempotente tra run: prima elenca i documenti già presenti sul server
        (identificati dal ``doc_id`` in ``metadata``, non dall'id interno di R2R,
        che è generato dal server) e ingesta solo quelli nuovi — come il working
        dir per-hash di LightRAG/cognee, ma qui lo store è condiviso da un server
        che gira a parte, quindi la deduplica la facciamo noi.
        """
        known = await self._known_doc_ids()

        target_ids: list[str] = []
        for doc in documents:
            if doc.doc_id in known:
                target_ids.append(known[doc.doc_id])
                continue
            target_ids.append(await self._create_document(doc, known))

        await self._wait_until_ingested(target_ids)

        if self.config.search_mode == "advanced":
            await self._ensure_graph_extracted(target_ids)

    async def _create_document(self, doc: Document, known: dict[str, str]) -> str:
        """``POST /documents``, tollerando il 409 (già ingestato da un run precedente)."""
        try:
            payload = await self._request(
                "POST",
                "documents",
                data={"raw_text": doc.text, "metadata": json.dumps({"doc_id": doc.doc_id})},
            )
            return str(payload["results"]["document_id"])
        except R2RError as exc:
            if exc.status_code != 409:
                raise
            # race / rerun: il documento è comparso tra la list() iniziale e qui.
            known.update(await self._known_doc_ids())
            return known[doc.doc_id]

    async def _known_doc_ids(self) -> dict[str, str]:
        """``doc_id`` (il nostro) -> id interno di R2R, per i documenti già sul server."""
        payload = await self._request("GET", "documents", params={"limit": 1000, "offset": 0})
        return {
            doc["metadata"]["doc_id"]: str(doc["id"])
            for doc in payload["results"]
            if doc.get("metadata", {}).get("doc_id")
        }

    async def _wait_until_ingested(
        self, document_ids: list[str], *, poll_seconds: float = 5.0, timeout_seconds: float = 1800.0
    ) -> None:
        """Poll su ``ingestion_status`` (parsing/chunking/embedding), fino a successo o errore."""
        deadline = time.monotonic() + timeout_seconds
        pending = set(document_ids)

        while pending and time.monotonic() < deadline:
            for document_id in list(pending):
                doc = (await self._request("GET", f"documents/{document_id}"))["results"]
                if doc["ingestion_status"] == "failed":
                    raise RuntimeError(f"R2R: ingestion fallita per il documento {document_id}")
                if doc["ingestion_status"] == "success":
                    pending.discard(document_id)
            if pending:
                await asyncio.sleep(poll_seconds)

        if pending:
            raise TimeoutError(f"R2R: {len(pending)} documenti non ingestati dopo {timeout_seconds:.0f}s")

    async def _ensure_graph_extracted(self, document_ids: list[str]) -> None:
        """
        Estrae entità/relazioni per i documenti che non le hanno già, poi le
        "pulla" nel grafo della collezione.

        ``automatic_extraction`` nel toml **non ha effetto** con l'orchestrazione
        ``simple`` (il nostro deployment "light": il server si limita a loggare
        un warning e ``extraction_status`` resta ``pending`` per sempre — non è
        un `pending` transitorio da aspettare). Chiamiamo quindi esplicitamente
        ``POST /documents/{id}/extract`` con ``run_with_orchestration=false``:
        sincrona, aspetta la vera fine dell'estrazione senza bisogno di poll.
        Sequenziale (non in parallelo): ogni chiamata usa già l'LLM per
        estrarre, meglio non sommarci concorrenza extra verso Vertex.

        ``extract`` da solo non basta: le entità restano con
        ``description_embedding`` NULL finché non vengono copiate nel grafo
        della collezione (``POST /graphs/{collection_id}/pull``, che calcola
        anche gli embedding) — senza, ``graph_search`` in ``query()`` non trova
        nulla. Chiamato sempre, non solo per i documenti appena estratti: è
        idempotente (aggiorna per merge) e serve anche a "riparare" entità
        estratte da un run precedente con un adapter più vecchio.
        """
        collection_ids: set[str] = set()
        for document_id in document_ids:
            doc = (await self._request("GET", f"documents/{document_id}"))["results"]
            collection_ids.update(doc.get("collection_ids") or [])
            if doc["extraction_status"] not in ("success", "enriched"):
                await self._request(
                    "POST", f"documents/{document_id}/extract", params={"run_with_orchestration": "false"}
                )
        for collection_id in collection_ids:
            await self._request("POST", f"graphs/{collection_id}/pull")

    # -- query -------------------------------------------------------
    async def query(self, question: str) -> AdapterAnswer:
        started = time.monotonic()
        payload = await self._request(
            "POST",
            "retrieval/rag",
            json={
                "query": question,
                # "custom", mai "basic"/"advanced": vedi il bug HyDE nel docstring
                # del modulo. Il comportamento ibrido/grafo lo decidiamo noi sotto.
                "search_mode": "custom",
                "search_settings": self.config.search_settings(),
                # Stesso LLM di tutti gli altri tool, esplicito (non ci affidiamo al
                # `quality_llm` di default del server, anche se coincide: vedi
                # adapters/r2r_config/r2r_olivetti.toml).
                "rag_generation_config": {"model": _vertex.llm_model(), "temperature": 0},
            },
        )
        retrieval_seconds = time.monotonic() - started

        results = payload["results"]
        search_results = results.get("search_results") or {}
        chunk_hits = search_results.get("chunk_search_results") or []
        graph_hits = search_results.get("graph_search_results") or []

        contexts = [hit["text"] for hit in chunk_hits if hit.get("text")]
        for hit in graph_hits:
            description = (hit.get("content") or {}).get("description")
            if description:
                contexts.append(description)

        return AdapterAnswer(
            answer=results.get("generated_answer") or "",
            contexts=contexts,
            raw={
                "retrieval_latency_s": round(retrieval_seconds, 3),
                "n_chunk_hits": len(chunk_hits),
                "n_graph_hits": len(graph_hits),
                "n_citations": len(results.get("citations") or []),
            },
        )

    # -- teardown ----------------------------------------------------
    async def teardown(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # -- introspezione ---------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "api_version": "v3",
            "config": self.config.as_dict(),
            "base_url": self.base_url,
            "generation_llm": _vertex.llm_model(),
            "embedding_model": _vertex.embedding_model(),
            "note": "server self-hosted (Postgres+pgvector) - avviato a parte con scripts/r2r_local.sh",
        }

    # -- interni --------------------------------------------------------
    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Chiamata REST + decodifica dell'errore nel formato ``{"message": ...}`` di R2R."""
        if self._http is None:
            raise RuntimeError("Chiama prima `await adapter.setup()`.")
        response = await self._http.request(method, f"/v3/{path}", **kwargs)
        if response.status_code >= 400:
            detail = response.text
            try:
                detail = response.json().get("message", detail)
            except ValueError:
                pass
            raise R2RError(response.status_code, detail)
        return response.json()
