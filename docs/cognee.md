# cognee — due diligence

Dati raccolti il **2026-09-08**. Repo: <https://github.com/topoteretes/cognee>

## Progetto in salute?

| Indicatore | Valore |
|---|---|
| Stelle | ~30.600 |
| Fork | ~3.000 |
| Watcher | ~111 |
| Issue aperte | ~499 |
| Creato | 2023-08-16 |
| Ultimo push | 2026-09-08 (stesso giorno) |
| Linguaggio | Python |
| Licenza | Apache-2.0 |
| Pacchetto PyPI | `cognee` (v1.5.4 — quella che usiamo) |
| Docs | <https://docs.cognee.ai> |

Progetto molto attivo e seguito. Nota: 499 issue aperte (ratio issue/stelle più alto
degli altri due) e API in forte evoluzione — la 1.5 ha introdotto una **nuova API
`remember`/`recall`/`forget`/`improve`** affiancata alla V1 `add`/`cognify`/`search`
(quella che usiamo). Migrazioni di schema automatiche al primo avvio.

## Paper

Nessun paper "fondativo" come LightRAG. C'è però un lavoro correlato del team su
arXiv: **2505.24478** — *"Optimizing the Interface Between Knowledge Graphs and LLMs
for Complex Reasoning"* (ottimizzazione iperparametri su benchmark multi-hop QA
tipo HotPotQA). Il termine "cognify" è ripreso da Kevin Kelly.

## Architettura

Pipeline **ECL — Extract, Cognify, Load** sopra il framework dati `dlt` (che porta
molti connettori di ingest).

```
add      → Extract: carica i documenti nel dataset (dlt loaders, multi-formato)
cognify  → Cognify: chunking → estrazione entità/relazioni via LLM →
                     costruzione del knowledge graph (schema = modelli Pydantic,
                     "DataPoints" / ontologia opzionale)
           Load: scrive nel vector DB + graph DB
search   → recupero ibrido (vettori + struttura del grafo) + generazione LLM
```

- **Ibrido graph + vector**, come LightRAG: il grafo è un indice sopra i chunk,
  che restano recuperabili (`SearchType.CHUNKS` → testo verbatim, italiano nel
  nostro caso).
- **Vector DB**: LanceDB (default, embedded) · anche pgvector, Qdrant, Weaviate, Milvus...
- **Graph DB**: "ladybug" / Kuzu (default, embedded) · anche Neo4j, FalkorDB...
- **Config di traduzione** integrata (`set_translation_target_language`) — cognee
  è più consapevole della lingua di supermemory.

## Compatibilità modelli

Usa **litellm nativamente** (`structured_output_framework: litellm_native`).
Provider: `openai | anthropic | azure | bedrock | gemini | mistral | ollama |
llama_cpp | custom`. **`custom` = qualsiasi endpoint litellm-routed** → così
puntiamo cognee su **Vertex** (`vertex_ai/gemini-2.5-flash`,
`vertex_ai/gemini-embedding-001`) col solo service account, **senza shim**.

## SearchType (le più rilevanti)

| tipo | ritorna |
|---|---|
| `GRAPH_COMPLETION` | **stringa generata** — flagship (vettori + struttura grafo). Analogo di LightRAG `hybrid`. |
| `RAG_COMPLETION` | **stringa generata** — RAG classico a chunk. Analogo di LightRAG `naive` / supermemory `documents`. |
| `HYBRID_COMPLETION`, `GRAPH_COMPLETION_COT`, `TRIPLET_COMPLETION`, `TEMPORAL`, `AGENTIC_COMPLETION` | stringa generata, varianti |
| `CHUNKS`, `SUMMARIES`, `CHUNKS_LEXICAL` | dict grezzi, nessuna generazione |
| `CYPHER`, `NATURAL_LANGUAGE` | righe di grafo |

cognee **genera la risposta** (`*_COMPLETION`) → non serve `_rag_prompt`.

## Feature aggiuntive

- Nuova API `remember`/`recall`/`forget`/`improve` (memoria conversazionale) +
  session memory (di default on, `CACHING=false` per spegnerla)
- Multi-tenant / access control (di default on, `ENABLE_BACKEND_ACCESS_CONTROL=false`)
- Ontologie custom, `@cognee.agent`, feedback loop (`improve`)
- Temporal knowledge graph (`temporal_cognify=True`)

## Installazione locale (sintetica)

```bash
pip install "cognee==1.5.4"
# nel codice, PRIMA di import cognee:
#   LLM_PROVIDER=custom, LLM_API_KEY=<dummy>, EMBEDDING_PROVIDER=custom,
#   EMBEDDING_API_KEY=<dummy>, EMBEDDING_DIMENSIONS=1536,
#   ENABLE_BACKEND_ACCESS_CONTROL=false, CACHING=false
# LLM_MODEL / EMBEDDING_MODEL / VERTEXAI_* / GOOGLE_APPLICATION_CREDENTIALS dal .env
```

## Come lo usiamo in questo repo

- Adapter: [`adapters/cognee_adapter.py`](../adapters/cognee_adapter.py) — libreria
  Python, **nessuno shim** (litellm nativo → Vertex).
- ⚠️ cognee deduce il provider da `LLM_MODEL` **all'import** e va in errore su
  `vertex_ai/...` → le env vanno impostate prima di `import cognee` (lo fa il
  modulo dell'adapter).
- Store per hash-config in `data/olivettiV0/cognee_data/<hash>/` (gitignored),
  come il working dir di LightRAG.
- Config: `search_type` (GRAPH_COMPLETION default), `top_k`, `embedding_dimensions=1536`.

## Limite noto

- cognee scrive nel proprio DB relazionale di bookkeeping (`pipeline_runs`) e si
  vede qualche errore SQLAlchemy non fatale (la pipeline continua). Da tenere
  d'occhio se blocca run reali.
- API in evoluzione rapida → versione pinnata.
