"""
Adapter per cognee (topoteretes/cognee), testato *as-is*.

cognee è una libreria Python (come LightRAG) e usa ``litellm`` nativamente →
**nessuno shim**: la puntiamo su Vertex con ``LLM_PROVIDER=custom`` e i modelli
del ``.env`` (gli stessi di LightRAG/supermemory). cognee **genera la risposta**
(``*_COMPLETION``), quindi non passiamo da ``_rag_prompt``.

⚠️ cognee deduce il provider da ``LLM_MODEL`` **all'import** e va in errore su
``vertex_ai/...``: le variabili d'ambiente vanno impostate PRIMA di ``import cognee``.
È ciò che fa il blocco qui sotto.

Ciclo di vita (vedi ``adapters.base.RAGAdapter``):
    adapter = CogneeAdapter(data_dir=..., config=CogneeConfig(search_type="GRAPH_COMPLETION"))
    await adapter.setup()            # punta gli store di cognee alla data_dir per-config
    await adapter.ingest(documents)  # cognee.add(...) + cognee.cognify(...) = costruzione KG
    ans = await adapter.query("...") # cognee.search(...) -> risposta generata
    await adapter.teardown()         # niente: lo store è isolato per hash
"""

from __future__ import annotations

import os

# --- Config di cognee: DEVE stare prima di `import cognee` -------------------
# "custom" = endpoint litellm-routed → cognee inoltra `vertex_ai/gemini-2.5-flash`
# a Vertex col service account. La LLM_API_KEY dev'essere non-vuota (preflight di
# cognee) ma litellm non la usa per i modelli `vertex_ai/`.
os.environ.setdefault("LLM_PROVIDER", "custom")
os.environ.setdefault("LLM_API_KEY", "vertex-service-account-no-key-needed")
os.environ.setdefault("EMBEDDING_PROVIDER", "custom")
os.environ.setdefault("EMBEDDING_API_KEY", "vertex-service-account-no-key-needed")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "1536")
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")  # niente multi-tenant
os.environ.setdefault("CACHING", "false")  # niente session memory tra le query

import cognee  # noqa: E402
from cognee import SearchType  # noqa: E402

from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Sequence  # noqa: E402

from adapters.base import AdapterAnswer, Document  # noqa: E402


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CogneeConfig:
    #: Modalità di ricerca. GRAPH_COMPLETION = flagship (vettori + struttura del
    #: grafo), analogo di LightRAG "hybrid". RAG_COMPLETION = RAG classico a chunk.
    search_type: str = "GRAPH_COMPLETION"
    #: Numero di nodi/chunk recuperati per costruire il contesto.
    top_k: int = 15
    #: Dimensione dei vettori (1536 per parità con LightRAG/supermemory).
    embedding_dimensions: int = 1536

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# L'adapter
# ---------------------------------------------------------------------------
class CogneeAdapter:
    """Traduttore sottile tra l'interfaccia comune e l'API di cognee."""

    name = "cognee"
    _DATASET = "olivetti"

    def __init__(self, data_dir: Path, config: CogneeConfig | None = None) -> None:
        # data_dir: dove cognee scrive vector db (LanceDB) e grafo (ladybug).
        # Per-hash della config → run con la stessa config riusano lo store.
        self.data_dir = Path(data_dir)
        self.config = config or CogneeConfig()

    # -- setup -----------------------------------------------------------
    async def setup(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        cognee.config.system_root_directory(str(self.data_dir / "system"))
        cognee.config.data_root_directory(str(self.data_dir / "data"))
        cognee.config.set_embedding_dimensions(self.config.embedding_dimensions)

    # -- ingest --------------------------------------------------------
    async def ingest(self, documents: Sequence[Document]) -> None:
        """
        ``add`` mette i documenti nel dataset, ``cognify`` costruisce il knowledge
        graph (chunking + estrazione entità/relazioni via LLM). È lo step pesante.
        ``incremental_loading`` (default) salta ciò che è già stato processato.
        """
        for doc in documents:
            await cognee.add(doc.text, dataset_name=self._DATASET)
        await cognee.cognify(datasets=[self._DATASET])

    # -- query -------------------------------------------------------
    async def query(self, question: str) -> AdapterAnswer:
        search_type = SearchType[self.config.search_type]

        answers = await cognee.search(
            query_text=question,
            query_type=search_type,
            datasets=[self._DATASET],
            top_k=self.config.top_k,
        )
        answer = str(answers[0]) if answers else ""

        # Contesti: una search CHUNKS separata dà i chunk grezzi (dict con "text"),
        # comparabile ai contexts di LightRAG/supermemory.
        chunk_hits = await cognee.search(
            query_text=question,
            query_type=SearchType.CHUNKS,
            datasets=[self._DATASET],
            top_k=self.config.top_k,
        )
        contexts = [
            hit["text"]
            for hit in chunk_hits
            if isinstance(hit, dict) and hit.get("text")
        ]

        return AdapterAnswer(
            answer=answer,
            contexts=contexts,
            raw={
                "search_type": search_type.name,
                "n_answers": len(answers),
                "n_chunk_hits": len(chunk_hits),
            },
        )

    # -- teardown ----------------------------------------------------
    async def teardown(self) -> None:
        # Lo store è isolato per hash-config: niente da pulire.
        # (Per un reset totale: cognee.prune.prune_data() + prune_system().)
        return None

    # -- introspezione ---------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "cognee_version": getattr(cognee, "__version__", "1.5.4"),
            "config": self.config.as_dict(),
            "llm_model": os.environ.get("LLM_MODEL"),
            "embedding_model": os.environ.get("EMBEDDING_MODEL"),
            "vector_db": "lancedb",
            "graph_db": "ladybug",
            "data_dir": str(self.data_dir),
        }
