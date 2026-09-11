"""
Fase 1 - Esecuzione del benchmark.

Prende un adapter + una configurazione, indicizza il corpus OCR comune, gira
tutte le domande e salva le risposte grezze. NON giudica e NON calcola metriche:
quello è compito di ``judge.py`` e ``metrics.py`` (fasi separate, così si può
ri-giudicare senza ri-eseguire i tool).

Uso:
    poetry run python -m eval.run_benchmark lightrag --query-mode hybrid
    poetry run python -m eval.run_benchmark lightrag --query-mode mix --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from adapters.base import RAGAdapter, load_documents
from eval.dataset import load_questions
from eval import results_store

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OCR_DIR = _PROJECT_ROOT / "data" / "olivettiV0" / "ocr"
_QUESTIONS_CSV = (
    _PROJECT_ROOT / "data" / "olivettiV0" / "competency_questions" / "competency_questions.csv"
)
_LIGHTRAG_WORKDIR_ROOT = _PROJECT_ROOT / "data" / "olivettiV0" / "lightrag_workdir"
_COGNEE_DATA_ROOT = _PROJECT_ROOT / "data" / "olivettiV0" / "cognee_data"


# ---------------------------------------------------------------------------
# Costruzione dell'adapter richiesto
#
# Ogni tool ha una config nativa diversa: la mappatura CLI -> adapter vive qui,
# una funzione per tool, così aggiungere Cognee/supermemory non tocca il resto.
# ---------------------------------------------------------------------------
def _build_lightrag(args: argparse.Namespace) -> RAGAdapter:
    from adapters.lightrag_adapter import LightRAGAdapter, LightRAGConfig

    config = LightRAGConfig(query_mode=args.query_mode)
    # working dir per hash della config: il grafo persiste ed è riusato dai run
    # successivi con la stessa config (ingest lungo solo la prima volta).
    workdir = _LIGHTRAG_WORKDIR_ROOT / results_store.config_hash(config.as_dict())
    return LightRAGAdapter(working_dir=workdir, config=config)


def _build_supermemory(args: argparse.Namespace) -> RAGAdapter:
    from adapters.supermemory_adapter import SupermemoryAdapter, SupermemoryConfig

    # --query-mode qui mappa su search_mode (memories | documents | hybrid).
    config = SupermemoryConfig(
        search_mode=args.query_mode,
        base_url=os.environ.get("SUPERMEMORY_BASE_URL") or None,
    )
    # container_tag lo calcola l'adapter (deve stare in <=100 caratteri: il limite
    # di supermemory). Non lo forziamo qui con lo slug lungo della results dir.
    return SupermemoryAdapter(config=config)


def _build_cognee(args: argparse.Namespace) -> RAGAdapter:
    from adapters.cognee_adapter import CogneeAdapter, CogneeConfig

    # --query-mode mappa su search_type (GRAPH_COMPLETION | RAG_COMPLETION | ...).
    st = args.query_mode.upper() if args.query_mode != "hybrid" else "GRAPH_COMPLETION"
    config = CogneeConfig(search_type=st)
    data_dir = _COGNEE_DATA_ROOT / results_store.config_hash(config.as_dict())
    return CogneeAdapter(data_dir=data_dir, config=config)


def _build_r2r(args: argparse.Namespace) -> RAGAdapter:
    from adapters.r2r_adapter import R2RAdapter, R2RConfig

    # --query-mode mappa su search_mode: "hybrid" -> "advanced" (ibrido + grafo),
    # qualsiasi altro valore -> "basic" (solo ricerca semantica, niente grafo).
    search_mode = "advanced" if args.query_mode == "hybrid" else "basic"
    config = R2RConfig(search_mode=search_mode)
    # Server a parte (scripts/r2r_local.sh): nessun working dir da gestire qui,
    # l'adapter deduplica l'ingest guardando i documenti già sul server.
    return R2RAdapter(base_url=os.environ.get("R2R_BASE_URL", "http://localhost:7272"), config=config)


_ADAPTER_BUILDERS = {
    "lightrag": _build_lightrag,
    "supermemory": _build_supermemory,
    "cognee": _build_cognee,
    "r2r": _build_r2r,
}


# ---------------------------------------------------------------------------
# Orchestrazione
# ---------------------------------------------------------------------------
async def run_benchmark(adapter: RAGAdapter, *, limit: int | None) -> Path:
    """Indicizza il corpus, gira le domande, salva ``answers.jsonl`` + ``run.json``."""
    documents = load_documents(_OCR_DIR)
    questions = load_questions(_QUESTIONS_CSV)
    if limit is not None:
        questions = questions[:limit]

    _describe = lambda: adapter.describe() if hasattr(adapter, "describe") else {"tool": adapter.name}
    run_dir = results_store.new_run_dir(adapter.name, _describe().get("config", {}))
    print(f"Run: {run_dir}")

    await adapter.setup()  # dopo il setup describe() ha anche i valori risolti (es. embedding_dim)

    print(f"Ingest di {len(documents)} documenti...")
    ingest_started = time.monotonic()
    await adapter.ingest(documents)
    ingest_seconds = round(time.monotonic() - ingest_started, 1)
    print(f"Ingest completato in {ingest_seconds}s")

    answers: list[dict] = []
    for i, gq in enumerate(questions, start=1):
        started = time.monotonic()
        result = await adapter.query(gq.question)
        answers.append(
            {
                "id": gq.id,
                "question": gq.question,
                "answer": result.answer,
                "contexts": result.contexts,
                "raw": result.raw,
                "latency_s": round(time.monotonic() - started, 3),
            }
        )
        print(f"  [{i}/{len(questions)}] {gq.id}: {len(result.answer)} char, {len(result.contexts)} contesti")

    await adapter.teardown()

    results_store.write_jsonl(run_dir / "answers.jsonl", answers)
    results_store.write_json(
        run_dir / "run.json",
        {
            **_describe(),
            "ocr_dir": str(_OCR_DIR),
            "questions_csv": str(_QUESTIONS_CSV),
            "n_documents": len(documents),
            "n_questions": len(questions),
            "ingest_seconds": ingest_seconds,
        },
    )
    print(f"Salvate {len(answers)} risposte in {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Esegue il benchmark su un tool.")
    parser.add_argument("tool", choices=sorted(_ADAPTER_BUILDERS))
    parser.add_argument("--query-mode", default="hybrid", help="Modalità di retrieval (adapter-specifica).")
    parser.add_argument("--limit", type=int, default=None, help="Usa solo le prime N domande (debug).")
    args = parser.parse_args()

    load_dotenv(_PROJECT_ROOT / ".env")
    adapter = _ADAPTER_BUILDERS[args.tool](args)
    asyncio.run(run_benchmark(adapter, limit=args.limit))


if __name__ == "__main__":
    main()
