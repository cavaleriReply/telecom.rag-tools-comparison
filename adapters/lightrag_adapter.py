"""
Adapter per LightRAG (HKUDS/LightRAG), testato *as-is*.

LightRAG viene usato attraverso i suoi punti di estensione ufficiali
(``llm_model_func`` / ``embedding_func``): non tocchiamo il suo codice interno,
gli passiamo solo Gemini/Vertex tramite il backend condiviso ``adapters._vertex``.

Ciclo di vita (vedi ``adapters.base.RAGAdapter``):
    adapter = LightRAGAdapter(working_dir=..., config=LightRAGConfig(query_mode="hybrid"))
    await adapter.setup()            # costruisce l'istanza LightRAG + storage
    await adapter.ingest(documents)  # chunking + estrazione entità/relazioni + grafo
    ans = await adapter.query("...") # retrieval sul grafo/vettori + risposta LLM
    await adapter.teardown()         # flush degli storage su disco
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import lightrag as _lightrag_pkg
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from adapters import _vertex
from adapters.base import AdapterAnswer, Document


# ---------------------------------------------------------------------------
# Configurazione
#
# Solo i pochi parametri "nativi" di LightRAG che ha senso variare in questa
# fase esplorativa. Ogni combinazione = una configurazione a sé, i cui risultati
# vengono salvati separatamente (vedi eval/). La configurazione viene registrata
# per intero in run.json, quindi ogni campo qui è una scelta tracciata.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LightRAGConfig:
    #: Modalità di retrieval di LightRAG: naive | local | global | hybrid | mix.
    query_mode: str = "hybrid"
    #: Entità/relazioni recuperate dal grafo per costruire il contesto.
    top_k: int = 40
    #: Chunk di testo recuperati per costruire il contesto.
    chunk_top_k: int = 20
    #: Dimensione dei chunk in token (default nativo di LightRAG).
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    #: Reranker disabilitato: non ne configuriamo uno, così il retrieval è
    #: quello "puro" di LightRAG e il confronto tra tool resta pulito.
    enable_rerank: bool = False
    #: Dimensione dei vettori di embedding. 1536 (non i 3072 nativi di
    #: gemini-embedding-001) per PARITÀ con supermemory, il cui vector store
    #: (pgvector + HNSW) è limitato a 2000 dim. gemini-embedding-001 è addestrato
    #: Matryoshka: 1536 è una taglia di prima classe, perdita minima vs 3072.
    embedding_dimensions: int = 1536

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Binding dei modelli: da Gemini/Vertex alle firme che LightRAG si aspetta
# ---------------------------------------------------------------------------
async def _llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs: Any,
) -> str:
    """
    Firma richiesta da LightRAG per ``llm_model_func``.

    LightRAG aggiunge kwargs interni (``hashing_kv``, ``_priority``, ``enable_cot``,
    ``stream``, ``token_tracker``...): li ignoriamo e inoltriamo a Vertex solo
    ``response_format``, che LightRAG usa per farsi restituire JSON valido durante
    l'estrazione delle keyword.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages or [])
    messages.append({"role": "user", "content": prompt})

    passthrough: dict[str, Any] = {}
    if kwargs.get("response_format") is not None:
        passthrough["response_format"] = kwargs["response_format"]

    return await _vertex.acompletion(messages, **passthrough)


def _build_embedding_func(embedding_dim: int) -> EmbeddingFunc:
    """
    Avvolge l'embedding condiviso nel wrapper di LightRAG, che valida la
    dimensione dei vettori restituiti contro ``embedding_dim``.

    Qui ``embedding_dim`` è sia la taglia che chiediamo a Vertex
    (``output_dimensionality``) sia quella che dichiariamo a LightRAG: coincidono
    per costruzione. Il wrapper conta i vettori con ``total_elementi // embedding_dim``,
    quindi un valore sbagliato fa fallire ogni batch con "Vector count mismatch"
    (è il bug del binding ufficiale ``lightrag.llm.gemini.gemini_embed``, che
    dichiara 1536 mentre l'API restituisce 3072). ``setup()`` verifica comunque
    la coincidenza con una chiamata reale.
    """

    async def _embed(texts: list[str], **_: Any):
        return await _vertex.aembed(texts, dimensions=embedding_dim)

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=2048,  # limite di input di gemini-embedding-001
        func=_embed,
    )


# ---------------------------------------------------------------------------
# L'adapter
# ---------------------------------------------------------------------------
class LightRAGAdapter:
    """Traduttore sottile tra l'interfaccia comune e l'API di LightRAG."""

    name = "lightrag"

    def __init__(self, working_dir: Path, config: LightRAGConfig | None = None) -> None:
        # working_dir: dove LightRAG scrive grafo, KV store e indici vettoriali.
        # Non influenza i risultati, quindi non fa parte di LightRAGConfig.
        self.working_dir = Path(working_dir)
        self.config = config or LightRAGConfig()
        self._rag: LightRAG | None = None
        self._embedding_dim: int | None = None

    # -- setup ---------------------------------------------------------------
    async def setup(self) -> None:
        """Costruisce l'istanza LightRAG e inizializza gli storage. Idempotente."""
        if self._rag is not None:
            return

        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Dimensione richiesta dalla config; verifichiamo che Vertex la onori
        # davvero (una chiamata), così non dichiariamo a LightRAG un valore
        # diverso da quello reale.
        self._embedding_dim = self.config.embedding_dimensions
        actual = await _vertex.probe_embedding_dim(dimensions=self._embedding_dim)
        if actual != self._embedding_dim:
            raise RuntimeError(
                f"embedding: chiesti {self._embedding_dim} dim, Vertex ne ha resi {actual}"
            )

        self._rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=_llm_model_func,
            llm_model_name=_vertex.llm_model(),
            embedding_func=_build_embedding_func(self._embedding_dim),
            chunk_token_size=self.config.chunk_token_size,
            chunk_overlap_token_size=self.config.chunk_overlap_token_size,
        )
        await self._rag.initialize_storages()  # inizializza anche il pipeline status

    # -- ingest -----------------------------------------------------------
    async def ingest(self, documents: Sequence[Document]) -> None:
        """
        Indicizza il corpus: chunking, estrazione entità/relazioni via LLM,
        costruzione del knowledge graph e degli indici vettoriali.

        Usiamo ``doc_id`` sia come id sia come ``file_path``, così i frammenti
        recuperati in fase di query sono riconducibili al documento di origine.
        """
        rag = self._require_rag()
        await rag.ainsert(
            input=[doc.text for doc in documents],
            ids=[doc.doc_id for doc in documents],
            file_paths=[doc.doc_id for doc in documents],
        )

    # -- query ----------------------------------------------------------
    async def query(self, question: str) -> AdapterAnswer:
        """Esegue una domanda e restituisce risposta + materiale per l'analisi."""
        rag = self._require_rag()
        param = QueryParam(
            mode=self.config.query_mode,
            top_k=self.config.top_k,
            chunk_top_k=self.config.chunk_top_k,
            enable_rerank=self.config.enable_rerank,
        )

        started = time.monotonic()
        # aquery_llm = come aquery ma restituisce ANCHE i dati recuperati
        # (entità, relazioni, chunk), senza una seconda chiamata di retrieval.
        result: dict[str, Any] = await rag.aquery_llm(question, param=param)
        elapsed = time.monotonic() - started

        answer = (result.get("llm_response") or {}).get("content") or ""
        contexts = _extract_contexts(result)

        return AdapterAnswer(
            answer=answer,
            contexts=contexts,
            raw={
                "status": result.get("status"),
                "mode": self.config.query_mode,
                "latency_s": round(elapsed, 3),
                # llm_generated=False => risposta "canned" di LightRAG (nessun
                # contesto trovato): informazione utile per l'analisi delle cause.
                "llm_generated": (result.get("llm_response") or {}).get("llm_generated"),
                "data": result.get("data"),
                "metadata": result.get("metadata"),
            },
        )

    # -- teardown ------------------------------------------------------------
    async def teardown(self) -> None:
        """Scarica gli storage su disco. Sicuro anche se ``setup`` non è stato chiamato."""
        if self._rag is not None:
            await self._rag.finalize_storages()
            self._rag = None

    # -- introspezione -----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Riepilogo per run.json: cosa è stato eseguito, con quali modelli."""
        return {
            "tool": self.name,
            "lightrag_version": _lightrag_pkg.__version__,
            "config": self.config.as_dict(),
            "llm_model": _vertex.llm_model(),
            "embedding_model": _vertex.embedding_model(),
            "embedding_dim": self._embedding_dim,
            "working_dir": str(self.working_dir),
        }

    # -- interni ----------------------------------------------------------
    def _require_rag(self) -> LightRAG:
        if self._rag is None:
            raise RuntimeError("Chiama prima `await adapter.setup()`.")
        return self._rag


def _extract_contexts(result: dict[str, Any]) -> list[str]:
    """
    Appiattisce il materiale recuperato da LightRAG in una lista di stringhe.

    Serve per la metrica di *context recall*: il fatto atteso era presente in ciò
    che il retrieval ha portato a galla? Includiamo tutte e tre le fonti che
    LightRAG usa per costruire il prompt finale.
    """
    data = result.get("data") or {}
    contexts: list[str] = []

    for chunk in data.get("chunks") or []:
        text = (chunk.get("content") or "").strip()
        if text:
            contexts.append(text)

    for entity in data.get("entities") or []:
        name = entity.get("entity_name", "")
        description = (entity.get("description") or "").strip()
        if description:
            contexts.append(f"{name} ({entity.get('entity_type', 'UNKNOWN')}): {description}")

    for rel in data.get("relationships") or []:
        description = (rel.get("description") or "").strip()
        if description:
            contexts.append(f"{rel.get('src_id', '')} -- {rel.get('tgt_id', '')}: {description}")

    return contexts
