# Dettagli implementativi

Stato al **2026-09-07**. Questo documento descrive *come* è costruito il progetto.
Per la due diligence dei singoli tool: [`lightrag.md`](lightrag.md), [`supermemory.md`](supermemory.md).

---

## 1. Obiettivo e vincoli

Confronto oggettivo di strumenti RAG / knowledge-retrieval sulla documentazione
museale Olivetti. Interessa **il perché** delle differenze di performance, non solo
un ranking per F1.

Vincoli che hanno guidato il design:

- **Test as-is.** Nessuna modifica al codice interno dei tool. Gli adapter usano
  solo i punti di estensione ufficiali.
- **Stesso input per tutti.** Un unico passaggio di OCR produce il testo; ogni
  tool indicizza esattamente quei file.
- **Stessi modelli dove possibile.** Un solo canale verso Gemini/Vertex
  (`adapters/_vertex.py`), stesse variabili `.env`. Se un tool si comporta
  diversamente, non è perché "parla" con Vertex in modo diverso.
- **Fase esplorativa.** Evaluation snella: label discrete + P/R/F1 + poche
  metriche di supporto. Niente scoring continuo.

---

## 2. Struttura del repo

```
data/olivettiV0/
  docs/                  4 PDF sorgente (Lettera 22, Valentine, Programma 101, design process)
  ocr/                   testo estratto, <stem>.txt — INPUT COMUNE a tutti gli adapter
  competency_questions/  competency_questions.csv (domande + golden set opzionale)
  lightrag_workdir/      artefatti LightRAG per config (gitignored)
  supermemory_data/      store del server supermemory self-hosted (gitignored)

preprocessing/           OCR: PDF -> testo
adapters/
  base.py                interfaccia standard (Document, AdapterAnswer, RAGAdapter)
  _vertex.py             accesso condiviso a Gemini/Vertex via litellm
  _rag_prompt.py         generazione risposta per i tool retrieval-only
  lightrag_adapter.py    adapter LightRAG
  supermemory_adapter.py adapter supermemory
eval/
  dataset.py             caricamento domande + golden set
  run_benchmark.py       fase 1: esegue le domande su un tool -> answers.jsonl
  judge.py               fase 2: giudice LLM -> judgements.jsonl
  metrics.py             fase 3: aggregazione -> metrics.json
  results_store.py       layout su disco dei risultati
  results/<tool>/<config-slug>__<hash>/<timestamp>/   un run

scripts/
  vertex_openai_shim.py       shim OpenAI-compatible (chat + embeddings) verso Vertex, per supermemory
  supermemory_local.sh        avvia shim + supermemory-server self-hosted
```

---

## 3. Il flusso

```
PDF ──(preprocessing)──► ocr/*.txt ──┬─► LightRAGAdapter ──┐
                                     └─► SupermemoryAdapter ─┤
                                                            ▼
competency_questions.csv ──► run_benchmark ──► answers.jsonl
                                                  │
                                    judge ──► judgements.jsonl
                                                  │
                                   metrics ──► metrics.json
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

Risultato attuale: 4 file, ~1.500 righe. Testo pulito pagina per pagina.

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

class RAGAdapter(Protocol):     # async
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
- Convenzione (non nel Protocol ma rispettata da entrambi): `describe()` ritorna
  un dict con `tool`, `config`, i modelli usati, ecc. → finisce in `run.json`.

---

## 6. Backend modelli condiviso — `adapters/_vertex.py`

Un solo punto di contatto con Gemini/Vertex, via `litellm`. Legge `LLM_MODEL` e
`EMBEDDING_MODEL` da `.env`.

| Funzione | Cosa fa |
|---|---|
| `acompletion(messages, *, model=None, **kwargs)` | completion, ritorna solo il testo |
| `aembed(texts, *, model=None, dimensions=None)` | embedding di una lista → `np.ndarray (n, dim)` |
| `probe_embedding_dim(dimensions=)` | verifica con una chiamata reale che dimensione escono i vettori |
| `llm_model()` / `embedding_model()` | i nomi modello da `.env` |

Dettagli:

- **Retry** (`tenacity`) con backoff esponenziale su errori transitori di Vertex
  (`RateLimitError`, `InternalServerError`, `Timeout`, ...). Gli errori di
  prompt/permessi NON vengono ritentati.
- **Embedding in parallelo** con un semaforo (`_EMBED_MAX_CONCURRENCY = 8`):
  `gemini-embedding-001` accetta una sola istanza per richiesta, quindi la "batch"
  la facciamo noi. In sequenza l'ingest di LightRAG (centinaia di embedding di
  entità/relazioni) sarebbe lentissimo.
- **`dimensions`**: passato a Vertex come `output_dimensionality`. Il progetto usa
  **1536** ovunque (vedi §11). `None` = dimensione nativa (3072).
- **Normalizzazione L2**: `gemini-embedding-001` restituisce vettori unitari solo
  a 3072 dim; a taglie ridotte no. `aembed` L2-normalizza sempre, così ogni vector
  store si comporta uguale a prescindere dalla metrica.

---

## 7. Generazione per i tool retrieval-only — `adapters/_rag_prompt.py`

Alcuni tool (supermemory) recuperano passaggi ma **non generano** la risposta.
Per non introdurre differenze dovute a prompt diversi, tutti questi tool usano lo
stesso `answer_from_contexts(question, contexts)`:

- prompt minimale, volutamente "povero": *"rispondi solo con quello che c'è nel
  contesto, altrimenti dichiara di non sapere"*. Niente prompt engineering: quello
  che misuriamo è il **retrieval**, non il prompt.
- stesso LLM di tutti (`gemini-2.5-flash` via `_vertex.acompletion`).
- se `contexts` è vuoto → risposta fissa "Non ho trovato questa informazione nei documenti."

---

## 8. Adapter LightRAG — `adapters/lightrag_adapter.py`

LightRAG **v1.5.7** (pinnata in `pyproject.toml`; l'API cambia spesso).

### Come è usato as-is

LightRAG accetta `llm_model_func` e `embedding_func` come parametri: è il suo
punto di estensione ufficiale. Non tocchiamo il suo codice.

- `_llm_model_func(prompt, system_prompt, history_messages, **kwargs)` — LightRAG
  aggiunge kwargs interni (`hashing_kv`, `_priority`, `enable_cot`, `stream`, ...):
  li ignoriamo, inoltriamo a Vertex solo `response_format` (che LightRAG usa per
  farsi dare JSON valido durante l'estrazione keyword).
- `_build_embedding_func(embedding_dim)` — avvolge `_vertex.aembed` nel wrapper
  `EmbeddingFunc` di LightRAG.

### `LightRAGConfig` (parametri nativi esposti)

`query_mode` (naive|local|global|hybrid|mix), `top_k`, `chunk_top_k`,
`chunk_token_size`, `chunk_overlap_token_size`, `enable_rerank=False` (nessun
reranker configurato → retrieval "puro"), `embedding_dimensions=1536` (§11).

### Ciclo di vita

- `setup()` — prende `embedding_dimensions` dalla config e **verifica** con una
  chiamata reale che Vertex la onori, poi costruisce `LightRAG(...)` e
  `initialize_storages()`.
- `ingest(documents)` — `rag.ainsert(input=[...], ids=[doc_id], file_paths=[doc_id])`.
  `doc_id` come `file_path` → i frammenti recuperati sono riconducibili al documento.
  Fa chunking + estrazione entità/relazioni via LLM + costruzione grafo + indici.
- `query(question)` — usa `rag.aquery_llm(...)`: una sola chiamata restituisce
  risposta **e** dati recuperati (entità, relazioni, chunk). `_extract_contexts`
  li appiattisce in `contexts`. `raw` porta `status`, `llm_generated`
  (`False` = risposta "canned" di LightRAG, nessun contesto trovato), `data`, `metadata`.
- `teardown()` — `finalize_storages()` (flush su disco).

### Il bug dell'embedding_dim che evitiamo

Il binding ufficiale `lightrag.llm.gemini.gemini_embed` dichiara `embedding_dim=1536`
ma l'API restituisce 3072. Il wrapper `EmbeddingFunc.__call__` conta i vettori con
`total_elementi // embedding_dim` usando il valore **dichiarato** → ogni batch
risulta 2× → `ValueError: Vector count mismatch`.

**Noi non lo abbiamo** perché la stessa `embedding_dimensions` è sia ciò che
chiediamo a Vertex (`output_dimensionality`) sia ciò che dichiariamo a LightRAG,
e `setup()` verifica la coincidenza con una chiamata reale. Dichiarato == reale
per costruzione.

### Smoke test (2026-09-07)

Ingest 4 doc OK; 3 query corrette (dimensioni+peso Lettera 22, designer,
fondazione Olivetti 1908). `contexts` ~220-300 elementi per query (tutto il
retrieved di `hybrid`, `top_k=40`). Ingest lento (~minuti: estrazione entità +
gleaning su 16 chunk).

---

## 9. Adapter supermemory — `adapters/supermemory_adapter.py`

supermemory è un **motore di memoria/contesto**, non una libreria RAG generativa.

### Differenza architetturale chiave: 3 ruoli LLM distinti

| Ruolo | Chi lo fa | Modello |
|---|---|---|
| **Embedder** | server supermemory (in ingest e query) | `gemini-embedding-001` @ 1536d, via shim (§10) |
| **Estrazione** ("dreaming": fatti/relazioni → grafo) | server supermemory (in ingest) | `gemini-2.5-flash`, via shim |
| **Generazione** risposta | **il nostro adapter** (`_rag_prompt`) | `gemini-2.5-flash` |

Tutti e tre i ruoli passano da Vertex, con lo stesso service account e (embedder
+ generazione) gli stessi identici modelli di LightRAG.

supermemory NON genera la risposta: `search` restituisce solo passaggi/memorie.
Quindi con supermemory misuriamo soprattutto la **qualità del retrieval**, a
generazione costante.

### `SupermemoryConfig`

`search_mode` (memories|documents|hybrid), `limit`, `rerank`, `threshold`,
`rewrite_query`, `dreaming` ("instant" = grafo subito, predicibile per un
benchmark), `base_url` (None = hosted, valorizzato = self-hosted),
`cleanup_on_teardown`.

### Ciclo di vita

- `setup()` — crea `AsyncSupermemory(api_key=..., base_url=...)`.
- `ingest(documents)` — `documents.add(content, container_tag, custom_id, metadata,
  dreaming)` è **asincrono**: ritorna subito un id. `_wait_until_ready` fa polling
  su `documents.get(id)` finché `status == "done"` (e, per hybrid/memories,
  `dreaming_status == "done"`). Timeout 30 min, poll 5 s.
  Nota: `custom_id` accetta solo `[A-Za-z0-9_:-]` → `_safe_id()` slugifica il
  `doc_id` (che resta intero in `metadata["doc_id"]`).
- `query(question)` — `search.memories(q, container_tag, search_mode, limit,
  rerank, threshold, rewrite_query)` → `_extract_contexts` raccoglie `memory`,
  `chunk`, `chunks[].content`, e il vicinato nel grafo (`context.parents/children/
  related`). Poi `answer_from_contexts` genera la risposta.
- `teardown()` — se `cleanup_on_teardown`, `documents.delete_bulk(container_tags=[tag])`,
  poi `client.close()`.

### Isolamento dei run

`container_tag` = namespace per (tool, config): `olivettiV0__<config-slug>`. Due
run non si contaminano; il cleanup cancella solo il proprio container.

### Stato

Adapter completo, `describe()` verificato. **Non ancora smoke-tested end-to-end**:
serve il binario supermemory installato e in esecuzione.

---

## 10. Lo shim OpenAI→Vertex — `scripts/vertex_openai_shim.py`

### Perché esiste

`supermemory-server` (edizione **lite**) accetta come "model provider" solo chiavi
a token: `OPENAI_API_KEY | ANTHROPIC_API_KEY | GEMINI_API_KEY | GROQ_API_KEY`.
Non supporta Vertex direttamente, e la `GEMINI_API_KEY` è di Google AI Studio, che
non abbiamo (abbiamo solo il service account Vertex). Come provider di embedding
accetta solo `local | openai | gemini`.

### Cos'è

Un mini-server HTTP (aiohttp, ~200 righe, nessuna dipendenza nuova) che **finge di
essere OpenAI** e inoltra tutto a Vertex, riusando `adapters._vertex`:

| Endpoint | Backend | Note |
|---|---|---|
| `POST /v1/chat/completions` | `_vertex.acompletion_message` → `gemini-2.5-flash` | inoltra `tools`/`tool_choice` e ritorna `tool_calls` — l'estrazione memorie di supermemory è un **agente a function-calling**. Stream "finto" (un chunk). |
| `POST /v1/embeddings` | `_vertex.aembed` → `gemini-embedding-001` | default `--embedding-dimensions 1536`: supermemory **non manda `dimensions`** nella richiesta pur avendo il piano a 1536 |
| `GET /v1/models`, `GET /health` | statici | |

`SHIM_DEBUG=1` attiva un middleware che logga metodo/path/body di ogni richiesta.

```
supermemory ──"OpenAI (chat + embeddings)"──► SHIM :6799 ──► Vertex (service account)
            ◄──"risposte formato OpenAI"─────  SHIM       ◄──
```

supermemory viene configurato (da `supermemory_local.sh`) con:
```
OPENAI_API_KEY=sk-shim-not-used            # deve solo ESISTERE (lo shim ignora l'auth)
OPENAI_BASE_URL=http://127.0.0.1:6799/v1
OPENAI_MODEL / OPENAI_FAST_MODEL / OPENAI_TEXT_MODEL = gemini-2.5-flash
SUPERMEMORY_EMBEDDING_PROVIDER=openai
SUPERMEMORY_EMBEDDING_BASE_URL=http://127.0.0.1:6799/v1
SUPERMEMORY_EMBEDDING_MODEL=gemini-embedding-001
SUPERMEMORY_EMBEDDING_DIMENSIONS=1536
```

### Vantaggi

- Non tocchiamo né supermemory né Vertex (vincolo as-is).
- **Embedder e LLM identici a LightRAG**: stessa funzione, stesso modello, stessa
  normalizzazione → stesso spazio di embedding, confronto sul retrieval pulito.
- Nessuna `GEMINI_API_KEY`: usa il service account Vertex.
- **Anche il modello di estrazione è pinnato** esatto a `gemini-2.5-flash`.

### Comportamento su `dimensions`

Se la richiesta include `dimensions`, lo shim lo gira a Vertex come
`output_dimensionality` e **fa `raise` se il vettore che torna non è di quella
taglia** — meglio un errore rumoroso che vettori sbagliati nel DB di supermemory.

---

## 11. La questione delle dimensioni degli embedding

`gemini-embedding-001` ha dimensione **nativa 3072** (dove i vettori sono anche
L2-normalizzati), ma supporta tagli inferiori (128–3072) via `output_dimensionality`
(dove **non** sono normalizzati → lo fa `_vertex.aembed`).

**Il progetto usa 1536 dim ovunque.** Motivo: il vector store di supermemory
(**pgvector + indice HNSW**) ha un limite rigido di **2000 dimensioni**. A 3072
`supermemory-server` rifiuta di partire. 1536 è una taglia di prima classe per
`gemini-embedding-001` (addestramento Matryoshka), perdita di qualità minima.
LightRAG usa la stessa taglia (`LightRAGConfig.embedding_dimensions = 1536`) per
**parità di embedder** nel confronto.

Ogni giunzione dove una dimensione può sballare, e come la chiudiamo:

| Giunzione | Rischio | Come è garantita la compatibilità |
|---|---|---|
| LightRAG ↔ nano-vectordb | dichiarare una dim ≠ reale → "Vector count mismatch" | `embedding_dimensions` (1536) è sia ciò che si chiede a Vertex sia ciò che si dichiara a LightRAG; `setup()` verifica con una chiamata reale |
| supermemory ↔ shim | supermemory **non manda `dimensions`** → lo shim darebbe i 3072 nativi → non entrano in `vector(1536)` → chunk non indicizzati (silenziosamente) | lo shim ha `--embedding-dimensions 1536` come default, e fa `raise` se la taglia ottenuta non combacia |
| supermemory ↔ pgvector | dim > 2000 → il server non parte; il DB si "blocca" sulla prima dim vista | `SUPERMEMORY_EMBEDDING_DIMENSIONS=1536` + cartella dati creata vuota con questa config |
| LightRAG ↔ supermemory (il confronto) | embedder diversi → spazi non confrontabili | stesso modello, stessa dim (1536), stesso codice (`_vertex.aembed`), stessa normalizzazione |

### Accortezze operative

- La cartella `data/olivettiV0/supermemory_data/` va creata **vuota** con questa
  config.
- **Mai lanciare `supermemory-server` da solo**: userebbe il default
  `Xenova/bge-base-en-v1.5` a 768d e bloccherebbe lo store a 768. Sempre via
  `./scripts/supermemory_local.sh`. (Se il binario lo lancia da sé dopo l'install,
  fermarlo con Ctrl-C.)
- Se un domani si cambia la dimensione: prima svuotare la cartella. (supermemory
  comunque blocca il mix di dimensioni con un errore, non corrompe in silenzio.)

---

## 12. Setup supermemory self-hosted — `scripts/supermemory_local.sh`

Lo script:

1. carica le credenziali Vertex dal `.env` del progetto (`VERTEXAI_*`,
   `GOOGLE_APPLICATION_CREDENTIALS`, `LLM_MODEL`, `EMBEDDING_MODEL`);
2. avvia lo shim (`:6799`) in background e ne attende l'health;
3. configura il server: **tutto** verso lo shim via `OPENAI_BASE_URL` +
   `OPENAI_MODEL=gemini-2.5-flash` e `SUPERMEMORY_EMBEDDING_BASE_URL` +
   `SUPERMEMORY_EMBEDDING_DIMENSIONS=1536`;
4. `exec supermemory-server` (`:6767`); un `trap` uccide lo shim all'uscita.

Prerequisito: binario installato con `curl -fsSL https://supermemory.ai/install | bash`
(userspace, `~/.supermemory/bin` + `~/.local/bin` — quest'ultimo è già nel PATH —,
no sudo, no Node, verifica SHA256). Al wizard: scegliere **4 (Skip)** — la config
la passa lo script via env.
⚠️ `~/.supermemory/` contiene **anche il binario**, non solo dati: non cancellarla.

`scripts/supermemory.env` (gitignored) è opzionale: solo override di porte /
data dir. Nel `.env` del progetto: `SUPERMEMORY_BASE_URL="http://localhost:6767"`.

---

## 13. Evaluation — `eval/`

### Golden set — `dataset.py`

Estende `competency_questions.csv` con colonne **opzionali** (compilate a mano):

| colonna | valori | significato |
|---|---|---|
| `answerable` | `yes` / `no` / `partial` | rispondibile dai 4 documenti? |
| `expected_source_docs` | doc_id separati da `;` | documenti attesi |
| `expected_points` | testo libero | punti chiave di una buona risposta |

Le colonne possono mancare (`answerable == "unknown"`): quelle domande finiscono
in `skipped_no_gold` e non entrano nelle metriche.

### Layout risultati — `results_store.py`

```
eval/results/<tool>/<config-slug>__<hash8>/<timestamp>Z/
    run.json          describe() dell'adapter + n_documenti, n_domande, ingest_seconds, ...
    answers.jsonl     per domanda: id, question, answer, contexts, raw, latency_s
    judgements.jsonl  per domanda: label, rationale, missing_points, context_supports
    metrics.json      aggregati
```

`<config-slug>` = coppie `chiave=valore` della config, leggibili; `<hash8>` =
SHA1 della config completa (due config diverse non collidono mai). Lo stesso
(tool, config) rieseguito aggiunge un nuovo `<timestamp>` → storico nel tempo.

### Fase 1 — `run_benchmark.py`

`poetry run python -m eval.run_benchmark <tool> --query-mode <mode> [--limit N]`

Costruisce l'adapter (una funzione `_build_<tool>` per tool), carica corpus e
domande, `setup → ingest → loop query → teardown`, scrive `answers.jsonl` +
`run.json`.

### Fase 2 — `judge.py`

`poetry run python -m eval.judge <run_dir>`

Per ogni risposta, un LLM giudice (`_vertex.acompletion` con
`response_format=json_object`, `temp` di default) confronta la risposta con:

- il **corpus OCR** (indipendente dal tool valutato — è lo stesso testo per tutti);
- gli `expected_points` del golden set (se presenti);
- i `contexts` recuperati dal tool.

Output per domanda:

| campo | valori |
|---|---|
| `label` | `OK` / `PARTIAL` / `WRONG` (allucinazione) / `DECLINED` (dichiara di non sapere) |
| `rationale` | una frase |
| `missing_points` | punti attesi non coperti |
| `context_supports` | `true/false` — i frammenti recuperati bastavano a rispondere? |

`context_supports` separa **retrieval** da **generazione**: distingue "non ha
recuperato il frammento giusto" da "l'ha recuperato ma non l'ha usato". È il
segnale principale per l'analisi delle cause.

Parsing tollerante dell'output (JSON puro o primo blocco `{...}`); label non
valida → `WRONG`.

### Fase 3 — `metrics.py`

`poetry run python -m eval.metrics <run_dir>`

| Gruppo | Metrica | Definizione |
|---|---|---|
| Contenuto (domande `answerable` in yes/partial) | `precision` | `OK / (OK + PARTIAL + WRONG)` |
| | `recall` | `OK / (# domande rispondibili)` — `DECLINED` conta come miss |
| | `f1` | media armonica |
| Sicurezza (domande `answerable == no`) | `hallucination_rate` | risposte sostanziali `/ (# non rispondibili)` |
| | `correct_abstention_rate` | `DECLINED / (# non rispondibili)` |
| Retrieval (tutti i giudizi) | `context_recall` | `context_supports==true / (# con il flag)` |

Più `label_distribution` e `skipped_no_gold`.

---

## 14. Asimmetrie del confronto e come le gestiamo

| | LightRAG | supermemory |
|---|---|---|
| Embedding | `gemini-embedding-001` @ 1536d (iniettato) | `gemini-embedding-001` @ 1536d (via shim) — **allineato** |
| Chunking | config nostra (`chunk_token_size`) | di supermemory, non esposto |
| Estrazione grafo | `gemini-2.5-flash` (via `llm_model_func`) | `gemini-2.5-flash` (via shim) — **allineato** |
| Generazione risposta | interna a LightRAG (prompt suo) | fatta da noi (`_rag_prompt`, prompt neutro) |

→ Con supermemory misuriamo soprattutto il **retrieval**. Idea aperta (non ancora
implementata): una modalità "retrieval-only + generazione nostra" anche per
LightRAG (`QueryParam(only_need_context=True)`), per un confronto a generazione
costante su tutti i tool.

---

## 15. Stato attuale

| Pezzo | Stato |
|---|---|
| OCR | ✅ fatto, 4 `.txt` prodotti |
| Interfaccia comune + `_vertex` + `_rag_prompt` | ✅ |
| Adapter LightRAG | ✅ smoke-tested end-to-end (a 3072d; da rifare a 1536d) |
| Adapter supermemory | ✅ scritto, `describe()` ok, **non smoke-tested** |
| Shim OpenAI→Vertex (chat + embeddings) | ✅ testato isolato |
| Binario supermemory installato | ✅ v0.0.8 |
| `supermemory_local.sh` fa partire il server | ⏳ ultimo errore risolto (dim 1536), da riprovare |
| `run_benchmark` / `judge` / `metrics` | ✅ scheletro funzionante, non ancora girati su un run reale |
| Golden set (`answerable` ecc.) | ⏳ lo compila l'utente |
| Adapter Cognee | ❌ non iniziato |

### Prossimi passi

1. Utente: compila almeno `answerable` nel CSV.
2. Riavviare `scripts/supermemory_local.sh` (fix dim 1536), smoke-test
   dell'adapter supermemory.
3. Run completo (51 domande) su LightRAG + supermemory → judge → metrics.
4. Analisi delle cause sui casi di divergenza.
5. Adapter Cognee.

---

## 16. Comandi rapidi

```bash
poetry install
# .env: credenziali Vertex + LLM_MODEL + EMBEDDING_MODEL + SUPERMEMORY_BASE_URL

poetry run python -m preprocessing                              # OCR

./scripts/supermemory_local.sh                                  # (in un altro terminale) server supermemory

poetry run python -m eval.run_benchmark lightrag --query-mode hybrid
poetry run python -m eval.run_benchmark supermemory --query-mode hybrid
poetry run python -m eval.judge   eval/results/<tool>/<slug>/<ts>
poetry run python -m eval.metrics eval/results/<tool>/<slug>/<ts>
```
