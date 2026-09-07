# supermemory — due diligence

Dati raccolti il **2026-09-07**. Repo: <https://github.com/supermemoryai/supermemory>

## Progetto in salute?

| Indicatore | Valore |
|---|---|
| Stelle | ~29.250 |
| Fork | ~2.560 |
| Watcher | ~108 |
| Issue aperte | ~113 |
| Creato | 2024-02-27 |
| Ultimo push | 2026-09-02 |
| Linguaggio | TypeScript |
| Licenza | MIT |
| SDK Python | `supermemory` v3.61 (sync + async, httpx) |
| Docs | <https://supermemory.ai/docs> |

Progetto attivo, molto seguito. Benchmark dichiarati: #1 su LongMemEval, LoCoMo,
ConvoMem; "95% Recall@15", "99.4% context reduction".

## Paper

Nessun paper accademico collegato (a differenza di LightRAG). C'è un blog tecnico
e i risultati sui benchmark di memoria conversazionale, ma non una pubblicazione.

## Cos'è (e cosa NON è)

È un **motore di memoria/contesto**, non una libreria RAG generativa.
**Recupera passaggi e fatti, non genera la risposta.** Ogni documento produce:
- **chunk** (testo grezzo per RAG),
- **memorie** (fatti + relazioni in un "grafo temporale vettoriale"),
- **profilo** (riassunto statico + dinamico).

Pipeline di ingest **asincrona**: `queued → extracting → chunking → embedding →
indexing → done`, poi fase "**dreaming**" che popola il grafo (`dynamic` = raggruppa
i documenti, default; `instant` = subito).

Retrieval: `search` ibrido (memorie + chunk) con `threshold`, `rerank`,
`rewrite_query`. Nessun endpoint di generazione risposta.

## Compatibilità modelli

- **Hosted** (API key): usa i modelli proprietari di supermemory. Embedding e
  chunking non configurabili.
- **Hosted** (API key): modelli proprietari, non configurabili.
- **Self-hosted** (`supermemory-server` **lite** v0.0.8):
  - **LLM provider**: solo chiavi a token — `OPENAI_API_KEY | ANTHROPIC_API_KEY |
    GEMINI_API_KEY (AI Studio) | GROQ_API_KEY`. **Vertex NON è supportato direttamente.**
  - **Embedding** "pluggable" (`SUPERMEMORY_EMBEDDING_*`): `local | openai | gemini`.
    Default locale ONNX `Xenova/bge-base-en-v1.5` (~109M param, 768d, solo inglese).
    Alternativa multilingue locale: `Xenova/bge-m3` (1024d).
  - **Vector store**: pgvector + indice HNSW → **limite rigido 2000 dimensioni**.
  - Il grafo è un DB embedded, creato al primo avvio.
- Noi aggiriamo entrambi i limiti con lo **shim** ([`scripts/vertex_openai_shim.py`](../scripts/vertex_openai_shim.py)):
  finge di essere OpenAI (`OPENAI_BASE_URL` + `provider=openai`) e inoltra chat ed
  embeddings a Vertex col service account. Vedi [`implementazione.md` §10](implementazione.md).

## Feature aggiuntive

- App web, plugin browser, MCP server (Claude Desktop / Cursor / VS Code).
- Wrapper per Vercel AI SDK, LangChain, LangGraph, OpenAI Agents.
- Connettori (Google Drive, Notion, ...).
- Profili utente e "forgetting" automatico (non rilevanti per il nostro caso).

## Installazione locale — la config che usiamo

Scelta del progetto: **self-hosted**, tutti e 3 i ruoli su Gemini/Vertex col
**solo service account** (nessuna `GEMINI_API_KEY` di AI Studio), tramite lo shim:

| Ruolo | Modello | Come |
|---|---|---|
| Embedder | `gemini-embedding-001` @ **1536d** | shim → Vertex (`_vertex.aembed`) — stesso codice e dim di LightRAG |
| Estrazione ("dreaming") | `gemini-2.5-flash` | shim → Vertex (`OPENAI_MODEL`) |
| Generazione | `gemini-2.5-flash` | fatta dall'adapter, non dal server |

`1536d` e non 3072: pgvector + HNSW di supermemory è limitato a 2000 dim.
LightRAG usa la stessa taglia per parità (`LightRAGConfig.embedding_dimensions`).

```bash
# 1. binario supermemory (userspace; ~/.local/bin è già nel PATH)
curl -fsSL https://supermemory.ai/install | bash
#    al wizard: scegliere 4 (Skip). Se parte da solo il server, Ctrl-C.
#    ⚠️ ~/.supermemory/ contiene ANCHE il binario: non cancellarla.

# 2. avvio (legge il .env del progetto per le credenziali Vertex)
./scripts/supermemory_local.sh        # shim :6799 + supermemory-server :6767
```

`scripts/supermemory.env` è opzionale (override di porte / data dir).
Store del server: `data/olivettiV0/supermemory_data/` (gitignored). Nel `.env`:
`SUPERMEMORY_BASE_URL="http://localhost:6767"`.

Alternativa fully-offline (embedder NON allineato): `SUPERMEMORY_EMBEDDING_PROVIDER=local`
+ `SUPERMEMORY_EMBEDDING_MODEL=Xenova/bge-m3` (multilingue).

## Come lo usiamo in questo repo

- Adapter: [`adapters/supermemory_adapter.py`](../adapters/supermemory_adapter.py).
- **La risposta la generiamo noi** da ciò che supermemory recupera, con il prompt
  neutro condiviso [`adapters/_rag_prompt.py`](../adapters/_rag_prompt.py) e lo
  stesso LLM degli altri tool (Gemini 2.5 Flash). Quindi confrontiamo la
  **qualità del retrieval** a generazione costante.
- `container_tag` isolato per (tool, config); `teardown` pulisce i documenti.
- Deployment via env: `SUPERMEMORY_API_KEY` (hosted) oppure `SUPERMEMORY_BASE_URL`
  (self-hosted).

## Asimmetria da tenere a mente nel confronto

| | LightRAG | supermemory |
|---|---|---|
| Embedding | `gemini-embedding-001` @ 1536d | `gemini-embedding-001` @ 1536d (via shim) — **allineato** |
| Estrazione grafo | `gemini-2.5-flash` | `gemini-2.5-flash` (via shim) — **allineato** |
| Chunking | config nostra (`chunk_token_size`) | suo, non esposto |
| Generazione risposta | interna (prompt di LightRAG) | fatta da noi (prompt neutro condiviso) |

Resta come variabile non controllata solo il **chunking** e la logica interna di
costruzione grafo / retrieval — che è esattamente ciò che il confronto vuole isolare.

→ Con supermemory misuriamo soprattutto il **retrieval**. Vedi anche la proposta
di una modalità "retrieval-only" anche per LightRAG per un confronto pulito.
