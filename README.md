# telecom.rag-tools-comparison

Confronto oggettivo di strumenti RAG / knowledge-retrieval (LightRAG, supermemory,
cognee) sul caso della **documentazione museale Olivetti**.

L'obiettivo non è solo un ranking per F1, ma **capire le cause tecniche** delle
differenze di performance. Gli strumenti si testano **as-is**: gli adapter sono
wrapper sottili, non modificano il codice interno dei tool.

## Struttura

| Cartella | Contenuto |
|---|---|
| `data/` | Documenti di test (PDF), testo estratto (`ocr/`), golden set (`competency_questions/`) |
| `preprocessing/` | OCR: da PDF a testo grezzo — input comune per tutti gli adapter |
| `adapters/` | Un wrapper per tool, stessa interfaccia (`adapters/base.py`) |
| `eval/` | Benchmark a 3 fasi + risultati per tool/config in `eval/results/` |
| `docs/` | [`implementazione.md`](docs/implementazione.md) (dettagli tecnici) + due diligence per tool |

## Setup

Guida completa: [`docs/setup.md`](docs/setup.md). In breve:

```bash
poetry install
cp .env.example .env    # compilare credenziali Vertex + modelli
```

`.env` richiede: `VERTEXAI_PROJECT`, `VERTEXAI_LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS`,
`LLM_MODEL` (`vertex_ai/gemini-2.5-flash`), `EMBEDDING_MODEL` (`vertex_ai/gemini-embedding-001`).
Per supermemory: binario + `./scripts/supermemory_local.sh` (vedi `docs/setup.md` §6).

## Pipeline

```bash
# 1. OCR: PDF -> data/olivettiV0/ocr/*.txt  (una volta, idempotente)
poetry run python -m preprocessing

# 2. Benchmark: gira le domande su un tool+config -> eval/results/.../answers.jsonl
poetry run python -m eval.run_benchmark lightrag --query-mode hybrid

# 3. Giudizio: label per risposta (OK/PARTIAL/WRONG/DECLINED)
poetry run python -m eval.judge eval/results/lightrag/<slug>/<timestamp>

# 4. Metriche: precision/recall/F1 + safety + context recall
poetry run python -m eval.metrics eval/results/lightrag/<slug>/<timestamp>
```

## Golden set

`data/olivettiV0/competency_questions/competency_questions.csv` — colonne
opzionali annotate a mano: `answerable` (yes/no/partial), `expected_source_docs`
(doc_id separati da `;`), `expected_points` (testo libero). Vedi `eval/dataset.py`.
