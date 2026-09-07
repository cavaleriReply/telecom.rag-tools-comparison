"""
Interfaccia standard che ogni strumento RAG deve esporre.

Obiettivo: il codice di benchmark (``eval/``) parla SOLO con ``RAGAdapter`` e non
sa nulla di LightRAG, Cognee o supermemory. Ogni differenza tra i tool
(API, ciclo di vita, modalita' di query) vive dentro il rispettivo adapter,
che resta un traduttore sottile: non modifica il codice del tool (test "as-is").

Tre concetti:
  * ``Document``      - un documento del corpus, gia' passato per l'OCR comune.
  * ``AdapterAnswer`` - la risposta a una domanda, con il materiale per l'analisi.
  * ``RAGAdapter``    - il protocollo: ``ingest(...)`` una volta, poi ``query(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable


# ---------------------------------------------------------------------------
# Dati in ingresso
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Document:
    """Un documento del corpus. ``text`` e' l'output dell'OCR condiviso."""

    doc_id: str  # identificatore stabile, tipicamente il nome file senza estensione
    text: str
    source_path: Path | None = None


def load_documents(ocr_dir: Path) -> list[Document]:
    """
    Carica il corpus comune dai ``.txt`` prodotti da ``preprocessing``.

    Tutti gli adapter partono da qui: stessa lista di ``Document``, stesso ordine,
    cosi' le differenze di performance non dipendono dai dati in ingresso.
    """
    txt_paths = sorted(ocr_dir.glob("*.txt"))
    if not txt_paths:
        raise FileNotFoundError(
            f"Nessun .txt in {ocr_dir}. Esegui prima: poetry run python -m preprocessing"
        )
    return [
        Document(doc_id=p.stem, text=p.read_text(encoding="utf-8"), source_path=p)
        for p in txt_paths
    ]


# ---------------------------------------------------------------------------
# Dati in uscita
# ---------------------------------------------------------------------------
@dataclass
class AdapterAnswer:
    """
    Risposta di un tool a una singola domanda.

    ``answer``   - testo finale, usato per le metriche (F1 / precision / recall).
    ``contexts`` - frammenti recuperati (chunk, entita', relazioni): servono per le
                   osservazioni qualitative e per capire *perche'* un tool sbaglia.
    ``raw``      - risposta grezza del tool, non interpretata: e' il punto di
                   partenza dell'analisi delle cause. Non deve mai essere "pulita".
    """

    answer: str
    contexts: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Il protocollo
# ---------------------------------------------------------------------------
@runtime_checkable
class RAGAdapter(Protocol):
    """
    Contratto comune. Async perche' i tool sottostanti (LightRAG, Cognee) lo sono.

    Ciclo di vita atteso dal benchmark:
        adapter = XxxAdapter(...)          # config nativa del tool, esplicita
        await adapter.setup()              # init storage/connessioni (idempotente)
        await adapter.ingest(documents)    # costruzione indice/grafo (una volta)
        answer = await adapter.query(q)    # una chiamata per domanda
        await adapter.teardown()           # flush/chiusura risorse
    """

    #: nome breve del tool, usato nei percorsi dei risultati (es. "lightrag").
    name: str

    async def setup(self) -> None:
        """Inizializza risorse (storage, client). Idempotente: chiamabile piu' volte."""
        ...

    async def ingest(self, documents: Sequence[Document]) -> None:
        """Indicizza il corpus. Assumiamo una sola chiamata per run di benchmark."""
        ...

    async def query(self, question: str) -> AdapterAnswer:
        """Esegue una domanda sull'indice gia' costruito e restituisce la risposta."""
        ...

    async def teardown(self) -> None:
        """Chiude/scarica le risorse. Sicuro anche se ``setup`` non e' stato chiamato."""
        ...
