# Dettagli implementativi

Aggiornato **2026-09-11**. Questo documento descrive *come* è costruito il progetto.
Due diligence dei singoli tool: [`lightrag.md`](lightrag.md), [`supermemory.md`](supermemory.md), [`cognee.md`](cognee.md), [`r2r.md`](r2r.md).
Guida da zero: [`setup.md`](setup.md).

---

## 1. Obiettivo e vincoli

Confronto oggettivo di strumenti RAG / knowledge-retrieval (**LightRAG**,
**supermemory**, **cognee**, **R2R**) sulla documentazione museale Olivetti.
Interessa **il perché** delle differenze di performance, non solo un ranking per F1.

Vincoli che hanno guidato il design:

- **Test as-is.** Nessuna modifica al codice interno dei tool. Gli adapter usano
  solo i punti di estensione ufficiali.
- **Stesso input per tutti.** Un unico passaggio di OCR produce il testo; ogni
  tool indicizza esattamente quei file.
- **Stessi modelli.** Un solo canale verso Gemini/Vertex (`adapters/_vertex.py`),
  stesse variabili `.env`, embedding a **1536 dim** ovunque. Se un tool si comporta
  diversamente, non è perché "parla" con Vertex in modo diverso.
- **Fase esplorativa.** Evaluation snella: label discrete + P/R/F1 + poche
  metriche di supporto.

---

## 2. Struttura del repo

```
data/olivettiV0/
  docs/                  4 PDF sorgente (Lettera 22, Valentine, Programma 101, design process)
  ocr/                   testo estratto, <stem>.txt — INPUT COMUNE a tutti gli adapter
  competency_questions/  competency_questions.csv (domande + golden set)
  lightrag_workdir/<hash>/   artefatti LightRAG per config (gitignored)
  supermemory_data/         store del server supermemory self-hosted (gitignored)
  cognee_data/<hash>/        store cognee: LanceDB + grafo ladybug (gitignored)
                             (R2R non ha una cartella qui: lo store è Postgres, fuori dal repo)

preprocessing/           OCR: PDF -> testo
adapters/
  base.py                interfaccia standard (Document, AdapterAnswer, RAGAdapter)
  _vertex.py             accesso condiviso a Gemini/Vertex via litellm
  _rag_prompt.py         generazione risposta per i tool retrieval-only (supermemory)
  lightrag_adapter.py    adapter LightRAG   (libreria, litellm iniettato)
  supermemory_adapter.py adapter supermemory (binario, via shim)
  cognee_adapter.py      adapter cognee     (libreria, litellm nativo)
  r2r_adapter.py         adapter R2R        (server REST, httpx diretto, litellm nativo lato server)
  r2r_config/
    r2r_olivetti.toml    overlay di config per il server R2R (LLM/embedding -> Vertex)
eval/
  dataset.py             caricamento domande + golden set
  run_benchmark.py       fase 1: esegue le domande su un tool -> answers.jsonl
  judge.py               fase 2: giudice LLM -> judgements.jsonl
  metrics.py             fase 3: aggregazione -> metrics.json
  review_gold.py         report disaccordi gold-vs-giudice (revisione annotazioni)
  results_store.py       layout su disco + config_hash()
  results/<tool>/<config-slug>__<hash>/<timestamp>/   un run

scripts/
  vertex_openai_shim.py  shim OpenAI-compatible (chat + embeddings) verso Vertex, per supermemory
  supermemory_local.sh   avvia shim + supermemory-server self-hosted
  r2r_local.sh            wrapper su `docker compose -f docker-compose.r2r.yml up`

docker-compose.r2r.yml   Postgres+pgvector + server R2R (immagine ufficiale), tutto in Docker

docs/                    setup.md, implementazione.md, lightrag.md, supermemory.md, cognee.md, r2r.md
```

---

## 3. Il flusso

```
PDF ─(preprocessing)─► ocr/*.txt ─┬─► LightRAGAdapter ────┐   (genera la risposta)
                                  ├─► CogneeAdapter ──────┤   (genera la risposta)
                                  ├─► R2RAdapter ─────────┤   (genera la risposta, con citazioni)
                                  └─► SupermemoryAdapter ─┤   (recupera; genera _rag_prompt)
                                                          ▼
competency_questions.csv ──► run_benchmark ──► answers.jsonl
                                                  │
                                    judge ──► judgements.jsonl
                                                  │
                                   metrics ──► metrics.json   (+ review_gold sulle annotazioni)
```

Le 3 fasi di eval sono **disaccoppiate**: si può ri-giudicare o ricalcolare le
metriche senza rieseguire i tool (che costano tempo e chiamate LLM).

---

## 4. Preprocessing / OCR — `preprocessing/`

- `ocr.py` — `transcribe_pdf(path)`: manda il PDF a Gemini (`LLM_MODEL`) come data
  URI base64 via `litellm.acompletion`, con un prompt di trascrizione integrale
  (no riassunto, no markdown). Nessun I/O su disco qui.
- `run.py` — batch: per ogni PDF in `docs/` produce `<stem>.txt` in `ocr/`.
  Idempotente (salta i `.txt` già presenti), parallelo sui documenti.
- Comando: `poetry run python -m preprocessing`

Risultato: 4 file, ~1.500 righe. Testo pulito pagina per pagina.

---

## 5. Interfaccia comune — `adapters/base.py`

```python
@dataclass(frozen=True)
class Document:
    doc_id: str            # id stabile = nome file senza estensione
    text: str              # output dell'OCR
    source_path: Path | None

@dataclass
class AdapterAnswer:
    answer: str            # testo finale -> metriche
    contexts: list[str]    # frammenti recuperati -> context recall + analisi cause
    raw: dict              # risposta grezza del tool, NON pulita -> analisi cause

class RAGAdapter(Protocol):     # async, structural typing (niente ereditarietà)
    name: str
    async def setup(self) -> None          # idempotente
    async def ingest(self, documents) -> None
    async def query(self, question) -> AdapterAnswer
    async def teardown(self) -> None
```

- `load_documents(ocr_dir)` legge i `.txt` e restituisce la stessa lista di
  `Document`, stesso ordine, per tutti i tool.
- Il codice di `eval/` parla **solo** con `RAGAdapter`. Ogni differenza tra tool
  vive dentro il rispettivo adapter.
- Convenzione: `describe()` ritorna un dict con `tool`, `config`, i modelli usati,
  ecc. → finisce in `run.json` (richiamato dopo `setup()`, così ha i valori risolti).

---

## 6. Backend modelli condiviso — `adapters/_vertex.py`

Un solo punto di contatto con Gemini/Vertex, via `litellm`. Legge `LLM_MODEL` e
`EMBEDDING_MODEL` da `.env`.

| Funzione | Cosa fa |
|---|---|
| `acompletion_message(messages, **kwargs)` | completion → dict `{content, tool_calls, finish_reason}` (formato OpenAI) |
| `acompletion(messages, **kwargs)` | come sopra ma ritorna solo il testo |
| `aembed(texts, *, dimensions=None)` | embedding di una lista → `np.ndarray (n, dim)`, L2-normalizzato |
| `probe_embedding_dim(dimensions=)` | verifica con una chiamata reale che dimensione escono i vettori |
| `llm_model()` / `embedding_model()` | i nomi modello da `.env` |

Dettagli:

- **Retry** (`tenacity`) con backoff su errori transitori di Vertex
  (`RateLimitError`, `InternalServerError`, `Timeout`, ...). Errori di
  prompt/permessi NON ritentati. `litellm.suppress_debug_info = True` (i banner).
- **Embedding in parallelo** con un semaforo (`_EMBED_MAX_CONCURRENCY = 8`):
  `gemini-embedding-001` accetta una sola istanza per richiesta.
- **`tool_calls`**: `acompletion_message` normalizza le function-call di Gemini
  in forma OpenAI — serve allo shim (supermemory estrae le memorie con un agente).
- **`dimensions`** → `output_dimensionality`. Progetto = **1536** (vedi §13).
- **Normalizzazione L2**: `gemini-embedding-001` è unitario solo a 3072; sotto no
  → `aembed` normalizza sempre.

---

## 7. Generazione per i tool retrieval-only — `adapters/_rag_prompt.py`

Solo **supermemory** recupera passaggi senza generare (LightRAG, cognee e R2R
generano da sé). `answer_from_contexts(question, contexts)`:

- prompt minimale e neutro: *"rispondi solo con quello che c'è nel contesto,
  altrimenti dichiara di non sapere"*. Niente prompt engineering: quello che
  misuriamo per supermemory è il **retrieval**.
- stesso LLM di tutti (`gemini-2.5-flash` via `_vertex`).
- `contexts` vuoto → "Non ho trovato questa informazione nei documenti."

---

## 8. Adapter LightRAG — `adapters/lightrag_adapter.py`

LightRAG **v1.5.7** (pinnata; l'API cambia spesso).

### Come è usato as-is

LightRAG accetta `llm_model_func` e `embedding_func` come parametri: è il suo
punto di estensione ufficiale.

- `_llm_model_func(prompt, system_prompt, history_messages, **kwargs)` — ignora i
  kwargs interni di LightRAG, inoltra a Vertex solo `response_format`.
- `_build_embedding_func(embedding_dim)` — avvolge `_vertex.aembed` (con
  `dimensions=1536`) nel wrapper `EmbeddingFunc` di LightRAG.

### `LightRAGConfig`

`query_mode` (naive|local|global|hybrid|mix), `top_k`, `chunk_top_k`,
`chunk_token_size`, `chunk_overlap_token_size`, `enable_rerank=False`,
`embedding_dimensions=1536`.

### Ciclo di vita

- `setup()` — verifica con una chiamata che Vertex renda 1536 dim, poi costruisce
  `LightRAG(...)` e `initialize_storages()`. Working dir = `lightrag_workdir/<config-hash>/`,
  persistito e riusato dai run con la stessa config.
- `ingest(documents)` — `rag.ainsert(input=[...], ids=[doc_id], file_paths=[doc_id])`:
  chunking + estrazione entità/relazioni via LLM + costruzione grafo + indici.
- `query(question)` — `rag.aquery_llm(...)`: una chiamata restituisce risposta **e**
  dati recuperati (entità, relazioni, chunk). `_extract_contexts` li appiattisce.
- `teardown()` — `finalize_storages()`.

### Il bug dell'embedding_dim che evitiamo

Il binding ufficiale `lightrag.llm.gemini.gemini_embed` dichiara `embedding_dim=1536`
ma l'API restituisce 3072 → `ValueError: Vector count mismatch`. Noi non lo abbiamo:
la stessa `embedding_dimensions` è sia ciò che chiediamo a Vertex sia ciò che
dichiariamo a LightRAG, e `setup()` verifica la coincidenza.

---

## 9. Adapter cognee — `adapters/cognee_adapter.py`

cognee **v1.5.4** (pinnata). È una **libreria Python** e usa **litellm nativamente**
→ **nessuno shim**: la puntiamo su Vertex e cognee **genera la risposta**.

### Configurazione (env prima di `import cognee`)

cognee deduce il provider da `LLM_MODEL` **all'import** e va in errore su
`vertex_ai/...`. L'adapter imposta, in cima al modulo, prima dell'import:

```
LLM_PROVIDER=custom            # endpoint litellm-routed -> Vertex col service account
LLM_API_KEY=<dummy non vuoto>  # il preflight lo pretende; litellm non lo usa per vertex_ai/
EMBEDDING_PROVIDER=custom
EMBEDDING_API_KEY=<dummy>
EMBEDDING_DIMENSIONS=1536
ENABLE_BACKEND_ACCESS_CONTROL=false   # niente multi-tenant
CACHING=false                          # niente session memory tra le query
```

`LLM_MODEL` / `EMBEDDING_MODEL` / `VERTEXAI_*` / `GOOGLE_APPLICATION_CREDENTIALS`
li legge dal `.env` del progetto → stessi modelli di LightRAG/supermemory.

### `CogneeConfig`

`search_type` (`GRAPH_COMPLETION` default = vettori + struttura grafo, analogo di
LightRAG `hybrid`; `RAG_COMPLETION` = RAG classico a chunk), `top_k`,
`embedding_dimensions=1536`.

### Ciclo di vita

- `setup()` — `cognee.config.system_root_directory()` + `data_root_directory()`
  puntano lo store a `cognee_data/<config-hash>/` (LanceDB vettori + grafo ladybug).
  Al primo avvio cognee fa ~50 migrazioni Alembic dello schema (lente, una tantum).
- `ingest(documents)` — `cognee.add(text, dataset_name=...)` per doc, poi
  `cognee.cognify(datasets=[...])` = ECL (Extract, Cognify, Load): chunking +
  estrazione entità/relazioni + knowledge graph. `incremental_loading` salta il già fatto.
- `query(question)` — `cognee.search(query_text, query_type, datasets, top_k)` →
  lista con la stringa generata. Una seconda `search(SearchType.CHUNKS)` per i
  `contexts` (chunk grezzi, in italiano).
- `teardown()` — niente: lo store è isolato per hash.

### Limite noto

Errori SQLAlchemy non fatali sul DB di bookkeeping (`pipeline_runs`) — spariti
con lo store per-hash isolato (DB fresco → Alembic migra pulito).

---

## 10. Adapter supermemory — `adapters/supermemory_adapter.py`

supermemory è un **motore di memoria conversazionale**, non una libreria RAG
generativa. Binario self-hosted (`supermemory-server` lite, TypeScript/bun).

### 3 ruoli LLM distinti

| Ruolo | Chi | Modello |
|---|---|---|
| **Embedder** (ingest + query) | server supermemory | `gemini-embedding-001` @ 1536d, via shim (§12) |
| **Estrazione** ("dreaming": memorie → grafo temporale) | server supermemory | `gemini-2.5-flash`, via shim |
| **Generazione** risposta | **il nostro adapter** (`_rag_prompt`) | `gemini-2.5-flash` |

supermemory NON genera: `search` restituisce solo passaggi/memorie. Con
supermemory misuriamo soprattutto la **qualità del retrieval**, a generazione costante.

### `SupermemoryConfig`

`search_mode` (memories|documents|hybrid), `limit`, `rerank`, `threshold`,
`rewrite_query`, `dreaming` ("instant" = grafo subito). `base_url` e
`cleanup_on_teardown` sono **campi operativi**, esclusi da `as_dict()` (non
influenzano i risultati né lo slug/hash).

### Ciclo di vita

- `setup()` — `AsyncSupermemory(api_key=..., base_url=...)`. Key fittizia ok su localhost.
- `ingest(documents)` — `documents.add(...)` è **asincrono**: `_wait_until_ready`
  fa polling su `documents.get(id)` finché `status == "done"` (+ `dreaming_status`
  per hybrid/memories). `custom_id` accetta solo `[A-Za-z0-9_:-]` → `_safe_id()`
  slugifica il `doc_id` (che resta intero in `metadata`).
- `query(question)` — `search.memories(...)` → `_extract_contexts` raccoglie
  `memory` / `chunk` / vicinato del grafo → `answer_from_contexts` genera.
- `teardown()` — se `cleanup_on_teardown`, `documents.delete_bulk(container_tags=[tag])`.

### Isolamento

`container_tag` = `olivettiV0__<mode>__<hash8>` (≤100 char: limite di supermemory).
Lo store persiste tra run del server, ma `cleanup_on_teardown` cancella il
container → ogni run ri-ingesta da zero.

---

## 11. Adapter R2R — `adapters/r2r_adapter.py`

R2R è l'unico dei quattro che non è né libreria Python né binario con SDK: è un
**server** (FastAPI) appoggiato a **Postgres+pgvector obbligatorio**, parlato via
REST v3. Va avviato a parte, tutto in Docker via
[`docker-compose.r2r.yml`](../docker-compose.r2r.yml) (`scripts/r2r_local.sh`
è il wrapper) — vedi `docs/r2r.md` e `docs/setup.md` §8.

### Niente SDK ufficiale

Il pacchetto `r2r` da PyPI trascina `fastapi<0.116` **incondizionatamente**
(non solo nell'extra `[core]` del server), in conflitto con `cognee` che vuole
`fastapi>=0.116.2` → `poetry lock` fallisce se coesistono nello stesso ambiente.
L'adapter parla REST diretto con `httpx` (già una dipendenza transitiva di
litellm, resa esplicita in `pyproject.toml`): i payload/endpoint ricalcano
esattamente quelli dell'SDK ufficiale (letto dai sorgenti per ricavarli).

### Come è usato as-is

R2R usa **litellm nativamente** per LLM ed embedding — `provider = "litellm"`
nel toml, e il completion provider di default (`"r2r"`) fa da router per
prefisso del nome modello: tutto ciò che non è `openai/ | azure/ | deepseek/ |
ollama/ | lmstudio/ | anthropic/ | azure-foundry/` (o senza prefisso) ricade nel
fallback litellm → `vertex_ai/...` ci arriva senza shim, come cognee. La
config è server-side (`adapters/r2r_config/r2r_olivetti.toml`, deep-merge sopra
i default di R2R): `quality_llm`/`fast_llm` e `embedding.base_model` puntati su
Vertex, `base_dimension=1536`.

### `R2RConfig`

`search_mode` (`advanced` default = ricerca ibrida + grafo, analogo di LightRAG
`hybrid`/cognee `GRAPH_COMPLETION`; `basic` = solo semantica), `limit` (top_k).
Nostro concetto, **mai** i preset `basic`/`advanced` di R2R lato richiesta:
`search_settings()` li traduce in `search_settings` espliciti — vedi i bug sotto.

### Ciclo di vita

- `setup()` — apre il client `httpx`, verifica `GET /v3/health` (errore chiaro
  se il server non è su, invece dello stacktrace httpx grezzo).
- `ingest(documents)` — `POST /documents` (`raw_text` + `metadata={"doc_id":...}`)
  per i documenti non ancora presenti (dedup guardando `GET /documents`, non
  c'è un working dir locale da controllare). Poi:
  1. poll su `ingestion_status` (parsing/chunking/embedding) fino a successo;
  2. se `search_mode="advanced"`: `POST /documents/{id}/extract` esplicito per
     ogni documento non ancora estratto, poi `POST /graphs/{collection_id}/pull`
     — entrambi necessari, vedi i bug sotto (`_ensure_graph_extracted`).
- `query(question)` — `POST /retrieval/rag` con `search_mode="custom"` (mai
  `"advanced"`, vedi bug #3): risposta generata (`generated_answer`, con
  citazioni strutturate) + `search_results` (chunk grezzi + entità/relazioni
  del grafo, appiattiti in `contexts`).
- `teardown()` — chiude il client `httpx`.

### Bug trovati in R2R (server 3.6.6, deployment "light") e aggirati qui

Scoperti lanciando il server per la prima volta (2026-09-11); dettagli e
tracce complete in `docs/r2r.md`:

1. `R2R_PROJECT_NAME` con maiuscole rompe lo schema Postgres (case-folding
   incoerente tra due punti del codice di R2R) → tenuto minuscolo nel compose.
2. `automatic_extraction` **non funziona** con l'orchestrazione `simple`
   (deployment "light"): l'adapter chiama `documents/{id}/extract` a mano, e
   anche così le entità restano senza embedding finché non si chiama anche
   `graphs/{collection_id}/pull` — altrimenti `graph_search` non trova nulla.
3. Il preset `search_mode="advanced"` di R2R porta `search_strategy="hyde"`,
   che manda a Gemini/Vertex un messaggio col solo ruolo "system" (nessun
   "user") → Vertex rifiuta la richiesta (`contents` vuoto) e **la ricerca
   stessa fallisce**, non solo la generazione. L'adapter non usa mai i preset
   di R2R, solo `search_mode="custom"` con `search_settings` espliciti.

✅ **Smoke-testato end-to-end** (2026-09-11, `--limit 3`, dopo i fix sopra):
risposte corrette e ben citate (es. "3,7 kg, 85×302×324 mm" per la Lettera 22,
identico al golden set). Non ancora eseguito il run completo a 51 domande.
Osservazione da verificare su run completo: le descrizioni delle entità del
grafo escono **in inglese** anche su corpus italiano (come supermemory — vedi
§16). La sintesi per comunità (community-level GraphRAG) resta fuori scope:
solo entità/relazioni "pullate" per-documento, non le comunità
(`collections.extract` + build communities, passo manuale extra non fatto).

---

## 12. Lo shim OpenAI→Vertex — `scripts/vertex_openai_shim.py`

### Perché esiste

`supermemory-server` (lite) accetta come model provider solo chiavi a token
(`OPENAI_API_KEY | ANTHROPIC_API_KEY | GEMINI_API_KEY | GROQ_API_KEY`); **non
supporta Vertex**, e non abbiamo una `GEMINI_API_KEY` di AI Studio. Come provider
di embedding: solo `local | openai | gemini`.

(cognee, LightRAG e R2R **non** hanno bisogno dello shim: cognee e LightRAG sono
librerie Python che usano litellm direttamente; R2R è un server ma supporta
litellm nativamente lato config — vedi §11.)

### Cos'è

Mini-server HTTP (aiohttp, ~230 righe, nessuna dipendenza nuova) che **finge di
essere OpenAI** e inoltra a Vertex, riusando `adapters._vertex`:

| Endpoint | Backend | Note |
|---|---|---|
| `POST /v1/chat/completions` | `_vertex.acompletion_message` → `gemini-2.5-flash` | inoltra `tools`/`tool_choice` e ritorna `tool_calls` — l'estrazione memorie di supermemory è un **agente a function-calling** (fino a 40 step). Stream "finto" (un chunk). |
| `POST /v1/embeddings` | `_vertex.aembed` → `gemini-embedding-001` | `--embedding-dimensions 1536` come default: supermemory **non manda `dimensions`** |
| `GET /v1/models`, `GET /health` | statici | |

`SHIM_DEBUG=1` → middleware che logga metodo/path/body di ogni richiesta.

```
supermemory ──"OpenAI (chat + embeddings)"──► SHIM :6799 ──► Vertex (service account)
```

Config passata da `supermemory_local.sh`: `OPENAI_API_KEY=sk-shim-not-used`,
`OPENAI_BASE_URL`, `OPENAI_MODEL/FAST/TEXT=gemini-2.5-flash`,
`SUPERMEMORY_EMBEDDING_PROVIDER=openai` + `_BASE_URL` + `_MODEL=gemini-embedding-001`
+ `_DIMENSIONS=1536`.

---

## 13. La questione delle dimensioni degli embedding

`gemini-embedding-001` ha dimensione **nativa 3072** (L2-normalizzata solo lì).
**Il progetto usa 1536 ovunque**: il vector store di supermemory (pgvector + HNSW)
ha un limite rigido di **2000 dim** — a 3072 il server non parte. 1536 è una
taglia Matryoshka di prima classe. LightRAG e cognee usano la stessa taglia per
**parità di embedder**.

| Giunzione | Rischio | Come è chiusa |
|---|---|---|
| LightRAG ↔ nano-vectordb | dim dichiarata ≠ reale → "Vector count mismatch" | `embedding_dimensions` (1536) = ciò che si chiede a Vertex **e** ciò che si dichiara; `setup()` verifica |
| supermemory ↔ shim | supermemory non manda `dimensions` → 3072 → non entra in `vector(1536)` → chunk non indicizzati (silenzioso) | shim `--embedding-dimensions 1536` di default, `raise` se non combacia |
| supermemory ↔ pgvector | dim > 2000 → server non parte; store bloccato sulla prima dim | `SUPERMEMORY_EMBEDDING_DIMENSIONS=1536` + cartella dati vuota |
| cognee ↔ LanceDB | LanceDB non ha limiti di dim | `EMBEDDING_DIMENSIONS=1536` per parità, non per necessità |
| R2R ↔ pgvector | stesso vincolo di supermemory: `base_dimension` fissa la colonna al primo ingest | `base_dimension=1536` in `r2r_olivetti.toml`, impostato prima di ogni primo ingest |
| confronto tra i 4 | embedder diversi → spazi non confrontabili | stesso modello, stessa dim, stesso codice `_vertex.aembed` (o il suo equivalente litellm lato server per R2R), stessa normalizzazione |

**Accortezze:** mai lanciare `supermemory-server` da solo (userebbe `bge-base-en-v1.5`
768d); se si cambia dimensione, svuotare prima gli store.

---

## 14. Setup supermemory self-hosted — `scripts/supermemory_local.sh`

1. carica le credenziali Vertex dal `.env`;
2. `pkill` di shim rimasti, avvia lo shim (`:6799`), ne attende l'health;
3. configura il server: tutto verso lo shim, dim 1536;
4. `exec supermemory-server` (`:6767`); `trap` uccide lo shim all'uscita.

Prerequisito: `curl -fsSL https://supermemory.ai/install | bash` (userspace, no
sudo, no Node). Al wizard: **4 (Skip)**. ⚠️ `~/.supermemory/` contiene **anche il
binario** — non cancellarla. Nel `.env`: `SUPERMEMORY_BASE_URL="http://localhost:6767"`.

---

## 15. Evaluation — `eval/`

### Golden set — `dataset.py`

`competency_questions.csv` con colonne (compilate a mano, tutte opzionali):

| colonna | valori | per |
|---|---|---|
| `answerable` | `yes` / `no` / `partial` | recall, hallucination, astensione |
| `expected_points` | testo libero | precision/recall di contenuto |
| `expected_source_docs` | doc_id separati da `;` | localizzare i fallimenti di retrieval |
| `category` | `meccanica` / `storia` / `design` / `marketing` / `cultura` | metriche per categoria |

Domande senza `answerable` → `skipped_no_gold`, fuori dalle metriche.
Stato: 51/51 `answerable` (28 yes / 14 no / 9 partial), 37/51 `expected_points`, 51/51 `category`.

### Layout risultati — `results_store.py`

```
eval/results/<tool>/<config-slug>__<hash8>/<timestamp>Z/
    run.json          describe() + n_documenti, n_domande, ingest_seconds
    answers.jsonl     per domanda: answer, contexts, raw, latency_s   (gitignored: grosso)
    judgements.jsonl  per domanda: label, rationale, missing_points, context_supports
    metrics.json      aggregati
```

`config_hash()` = SHA1[:8] della config → **stabile a prescindere dal formato
dello slug leggibile**; usato per il working dir persistente (LightRAG, cognee).

### Fase 1 — `run_benchmark.py`

`poetry run python -m eval.run_benchmark <lightrag|supermemory|cognee> --query-mode <mode> [--limit N]`

Una funzione `_build_<tool>` per tool. `--query-mode` mappa sulla config nativa
(LightRAG: `query_mode`; supermemory: `search_mode`; cognee: `hybrid`→`GRAPH_COMPLETION`,
`rag_completion`→`RAG_COMPLETION`).

### Fase 2 — `judge.py`

`poetry run python -m eval.judge <run_dir>`

Giudice LLM (`_vertex.acompletion`, `response_format=json_object`, **`temperature=0`**)
confronta la risposta con: corpus OCR (indipendente dal tool) + `expected_points` +
i `contexts` recuperati. Label, in quest'ordine di priorità:

| label | quando |
|---|---|
| `DECLINED` | la risposta dichiara di non sapere — **anche se l'info era nel corpus** (fallimento di copertura ≠ allucinazione) |
| `WRONG` | afferma fatti non presenti nel corpus (allucinazione vera) |
| `PARTIAL` | fatti corretti e supportati, ma incompleti/imprecisi |
| `OK` | corretta e con pieno riscontro (copre i punti attesi) |

`context_supports` (`true/false`) separa **retrieval** da **generazione**.

### Fase 3 — `metrics.py`

`poetry run python -m eval.metrics <run_dir>`

Helper `_tally()` calcolato sul totale **e per `category`** (`by_category`).

| gruppo | metrica | definizione |
|---|---|---|
| Contenuto (domande `answerable` yes/partial) | `precision` | `OK / (OK + PARTIAL + WRONG)` |
| | `recall` | `OK / (# rispondibili)` — `DECLINED` = miss |
| | `f1` | media armonica |
| Sicurezza | `hallucination_rate` | `WRONG / (# domande giudicate)` — su qualsiasi domanda |
| | `abstention_rate` | `DECLINED / (# domande answerable=no)` |
| | `answered_anyway_rate` | `non-DECLINED / (# answerable=no)` — informativo (o gold troppo stretto, o over-answering) |
| Retrieval | `context_recall` | `context_supports==true / (# con il flag)` |

### `review_gold.py`

`poetry run python -m eval.review_gold <run_dir>` — elenca i disaccordi
gold-vs-giudice (es. `answerable=no` ma il giudice dice supportato) con risposta +
rationale, per correggere le annotazioni.

---

## 16. Asimmetrie del confronto

| | LightRAG | cognee | supermemory | R2R |
|---|---|---|---|---|
| Tipo | libreria Python | libreria Python | binario self-hosted (via shim) | server self-hosted (Postgres, REST) |
| Embedder | `gemini-embedding-001` @ 1536 | idem | idem (via shim) — **allineati** | idem (litellm nativo) — **allineati** |
| LLM estrazione | `gemini-2.5-flash` | idem | idem (via shim) — **allineati** | idem (litellm nativo) — **allineati** |
| Chunking | config nostra (`chunk_token_size`) | di cognee, non esposto | di supermemory, non esposto | di R2R, non esposto (`ingestion.chunk_size` di default, non toccato) |
| Vector / graph store | nano-vectordb / GraphML | LanceDB / ladybug | pgvector / rivet | Postgres + pgvector (obbligatorio, unico store non intercambiabile) |
| Generazione risposta | **interna** (prompt suo, "comprehensive") | **interna** (prompt suo, "be brief") | **fatta da noi** (`_rag_prompt`, neutro) | **interna** (prompt suo, neutro + citazioni obbligatorie) |
| Prompt di estrazione | tipi per misure/procedure | grafo "stile Wikipedia" + date | lista da assistente conversazionale | grafo con XML tags, entità pesate 0-10, "ogni entità deve avere almeno una relazione" |

→ Variabili non controllate residue: **chunking** e la **logica interna** di
costruzione grafo / retrieval / prompt di generazione — che è esattamente ciò che
il confronto vuole isolare.

**Prompt di estrazione e generazione a confronto** (estratti reali):
- LightRAG estrazione: entity types con `Data: ...measurements` e `Method: Procedures`;
  merge = "integrate ALL key information, do NOT omit any detail".
- cognee estrazione: "knowledge graph, nodes akin to Wikipedia nodes"; sezione
  numeri = **solo date**. Generazione (`answer_simple_question.txt`): tutto il
  prompt è *"Answer the question using the provided context. Be as brief as possible."*
- supermemory estrazione: agente a tool, *"Extract every atomic fact... more
  memories is better than fewer"*, esempi tutti *"user preferences / user facts /
  events with dates"* → conversazionale, niente categoria per specifiche tecniche.
- LightRAG generazione (`rag_response`): *"comprehensive, well-structured... ALL
  pieces of information... Multiple Paragraphs"*.
- R2R estrazione (`graph_extraction.yaml`): output in tag XML, richiede un
  `relationship_weight` 0-10 e impone *"each entity must have at least one
  relationship... create intermediate entities if needed"* — spinge verso un
  grafo denso per costruzione, non per esaustività dei dettagli come LightRAG.
  Generazione (`rag.yaml`): *"Answer the query given ... Use line item
  references like [c910e2e] to refer to provided search results"* — minimale e
  neutro nello stile (vicino a `_rag_prompt`), ma orientato alla citazione
  verificabile più che LightRAG/cognee.

È da qui che nasce il divario OK 25 / 20 / 9 (LightRAG / cognee / supermemory).
**Confermato dal run completo di R2R** (51 domande, 2026-09-11, vedi §17): 24
OK, secondo per F1 dietro LightRAG — la regola di copertura strutturale
("ogni entità deve avere una relazione") produce un grafo denso ma, a
differenza del bucket misure/procedure di LightRAG, non spinge verso il
dettaglio quantitativo; il prompt di generazione minimale e orientato alle
citazioni resta comunque efficace quanto quello di LightRAG sulla sicurezza
(0% allucinazioni per entrambi).

---

## 17. Stato attuale (2026-09-11)

| Pezzo | Stato |
|---|---|
| OCR | ✅ 4 `.txt` |
| Interfaccia comune + `_vertex` + `_rag_prompt` | ✅ |
| Adapter LightRAG | ✅ smoke + run completo (51 domande) |
| Adapter supermemory | ✅ smoke + run completo. Server via `supermemory_local.sh` |
| Adapter cognee | ✅ smoke + run completo |
| Adapter R2R | ✅ smoke + run completo (51 domande). REST diretto, no SDK (vedi §11). 3 bug di R2R trovati e aggirati |
| Shim OpenAI→Vertex (chat con tool-calling + embeddings) | ✅ |
| Golden set (answerable / expected_points / category) | ✅ prima versione |
| `run_benchmark` / `judge` / `metrics` / `review_gold` | ✅ girati su run reali |
| **Confronto a 4** (hybrid) | ✅ Content F1: LightRAG 0.676 · R2R 0.649 · cognee 0.541 · supermemory 0.286 |

Dettaglio del confronto a 4 (51 domande, 37 rispondibili):

| | LightRAG | R2R | cognee | supermemory |
|---|---|---|---|---|
| Content F1 | **0.676** | 0.649 | 0.541 | 0.286 |
| OK / PARTIAL / WRONG (su 37) | 25 / 11 / 1 | 24 / 13 / 0 | 20 / 15 / 2 | 9 / 17 / 0 (+11 DECLINED) |
| hallucination_rate | 0.020 | **0.0** | 0.059 | **0.0** |
| abstention su `no` (14) | 35.7% | 35.7% | 42.9% | 85.7% |
| context_recall | 0.980 | 0.961 | 0.961 | 0.804 |
| ingest (1ª volta) | 442s | 298s | 265s | 228s |
| latenza mediana | 8.9s | 7.1s | 5.3s | 3.4s |
| contesti/domanda (media) | 221 | 56 | 6 | 20 |

R2R secondo per F1, appaiato a LightRAG su zero allucinazioni, retrieval quasi
alla pari con cognee (96.1% recall). Nessuna categoria in cui sia nettamente
il peggiore. Report visivo completo (grafici, confronto prompt, i 3 bug di
R2R): artifact ["Olivetti RAG Bench"](https://claude.ai/code/artifact/4e59a447-6799-4dcd-b2bf-639ff460f940).

### Prossimi passi

1. Run di parità sul top_k / modalità "RAG classico" per tutti e quattro
   (LightRAG `naive`, cognee `rag_completion`, supermemory `documents`, R2R
   `basic`) — isolare quanto del divario è retrieval vs generazione.
2. Rivedere le ~6 annotazioni golden `answerable=no` ancora in disaccordo col giudice.
3. Eventuale invocazione della community-level GraphRAG di R2R (`collections.extract`
   + build communities, non fatta finora) per un confronto più fedele al
   "graph mode" di LightRAG/cognee.

---

## 18. Comandi rapidi

```bash
poetry install
# .env: credenziali Vertex + LLM_MODEL + EMBEDDING_MODEL + SUPERMEMORY_BASE_URL + R2R_BASE_URL

poetry run python -m preprocessing                                   # OCR

./scripts/supermemory_local.sh                                       # (altro terminale) solo per supermemory
./scripts/r2r_local.sh                                                # (altro terminale) solo per R2R — serve Docker

poetry run python -m eval.run_benchmark lightrag    --query-mode hybrid
poetry run python -m eval.run_benchmark cognee      --query-mode hybrid
poetry run python -m eval.run_benchmark supermemory --query-mode hybrid
poetry run python -m eval.run_benchmark r2r         --query-mode hybrid
poetry run python -m eval.judge    eval/results/<tool>/<slug>/<ts>
poetry run python -m eval.metrics  eval/results/<tool>/<slug>/<ts>
poetry run python -m eval.review_gold eval/results/<tool>/<slug>/<ts>
```
