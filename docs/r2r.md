# R2R — due diligence

Dati raccolti il **2026-09-11**. Repo: <https://github.com/SciPhi-AI/R2R>

## Progetto in salute?

| Indicatore | Valore |
|---|---|
| Stelle | ~8.000 |
| Fork | ~648 |
| Watcher | ~44 |
| Issue aperte | ~126 |
| Creato | 2024-02-12 |
| Ultimo push | 2025-11-07 — **~10 mesi fa** |
| Ultima release | v3.6.5 (2025-06-06) / PyPI 3.6.6 |
| Linguaggio | Python |
| Licenza | MIT |
| Pacchetto PyPI | `r2r` (client) / `r2r[core]` (server) |
| Docs | <https://r2r-docs.sciphi.ai> |

⚠️ A differenza di cognee (push lo stesso giorno della rilevazione) e di LightRAG,
R2R non ha commit da ~10 mesi pur non essendo archiviato: **segnale di salute più
debole** degli altri due, da tenere presente nel leggere i risultati — non è detto
che bug o limiti notati qui siano ancora attuali o vengano corretti.

## Architettura

R2R non è una libreria Python (come LightRAG/cognee) né un binario locale con
SDK sottile (come supermemory): è un **server FastAPI** appoggiato a
**Postgres+pgvector obbligatorio**, esposto via REST (API v3) con SDK Python
(`R2RClient`/`R2RAsyncClient`, sincrono/asincrono) e JS.

```
documents.create(raw_text=...) → Parsing → Chunking → Embedding → (Extracting →
                                   entità/relazioni, se automatic_extraction=true)
                                 → Storing (Postgres+pgvector) → [Enriching]
retrieval.rag(query=...)        → ricerca (vettoriale/full-text/grafo, a seconda
                                   di search_mode) + generazione LLM con citazioni
```

- **Vector + full-text + graph**, tutto su Postgres: pgvector per i vettori,
  tabelle relazionali per entità/relazioni/comunità. Niente store embedded
  intercambiabile come LightRAG (nano-vectordb) o cognee (LanceDB/Kuzu) — è
  sempre Postgres, il che rende R2R il deployment più pesante dei quattro.
- **Due deployment**: "light" (`pip install r2r[core]; python -m r2r.serve`,
  Postgres va comunque fornito a parte) o "full" via Docker Compose (aggiunge
  Hatchet per l'orchestrazione, un servizio di graph clustering e una
  dashboard). Non serve la modalità "full" per questo confronto.
- **Estrazione del grafo automatica per documento** (stage `EXTRACTING` nel
  ciclo di vita di ogni documento, `automatic_extraction=true` di default) —
  analogo di LightRAG/cognee. La sintesi per **comunità** (community summaries,
  vero "GraphRAG" alla Microsoft) è invece un passo *manuale* separato
  (`collections.extract` + build communities): non lo invochiamo, per
  restare coerenti con "nessuna pipeline extra oltre i punti di ingresso
  ufficiali" — la ricerca sul grafo (`search_mode="advanced"`) usa comunque
  le entità/relazioni per-documento, già presenti.
- **Ricerca**: `search_mode` — `basic` (solo semantica), `advanced` (ibrida:
  semantica + full-text + grafo), `custom` (tutti i flag manuali via
  `search_settings`). Generazione con **citazioni strutturate** (span nel testo
  + payload della fonte).

## Compatibilità modelli

Il completion provider di default (`"r2r"`) fa da router per prefisso del nome
modello: `anthropic/` → provider Anthropic dedicato, `azure-foundry/` →
dedicato, `openai/ | azure/ | deepseek/ | ollama/ | lmstudio/` (o nessun
prefisso) → provider OpenAI-compatible, **qualsiasi altro prefisso con "/" →
LiteLLM** (verificato leggendo `core/providers/llm/r2r_llm.py`). `vertex_ai/...`
ricade quindi nel fallback litellm — **nessuno shim**, come cognee. Stesso
discorso per l'embedding: `provider = "litellm"` nel toml → litellm diretto.

## Configurazione (`r2r.toml`, deep-merge sopra i default)

R2R carica il suo `r2r.toml` di default e fa un merge di un file passato via
`R2R_CONFIG_PATH` (o `R2R_CONFIG_NAME` per i profili predefiniti in
`core/configs/`, es. `gemini.toml`, `ollama.toml`). Sezioni rilevanti:

- `[app]` — `quality_llm` (generazione RAG), `fast_llm` (operazioni interne).
- `[embedding]` / `[completion_embedding]` — `provider`, `base_model`,
  `base_dimension` (va fissata **prima del primo ingest**: crea la colonna
  pgvector a quella dimensione, stesso vincolo di supermemory §12 in
  `implementazione.md`).
- `[database]` — Postgres: host/porta/credenziali via `R2R_POSTGRES_*` (env,
  non toml).
- `[auth]` — `require_authentication` (default `false`: un solo utente
  implicito, comodo per un confronto a singolo corpus come questo).

## Ciclo di vita di un documento

`POST /documents` (`raw_text=...` + `metadata`) risponde subito con un
`document_id` e passa per una pipeline **asincrona**: `pending → parsing →
chunking → embedding → [extracting → knowledge graph] → storing → success`
(o `failed`). L'estrazione del grafo ha uno stato **separato**
(`extraction_status`: `pending/processing/success/enriched/failed`) da quello
di ingestion, come supermemory ha `status` e `dreaming_status` distinti — **ma**
nel deployment "light" (orchestrazione `simple`) l'estrazione automatica non
parte da sola (bug #2 sotto): va richiamata esplicitamente, e poi "pullata" nel
grafo della collezione perché le entità abbiano un embedding cercabile (bug #2).

## SearchType (le più rilevanti)

| `search_mode` | comportamento |
|---|---|
| `basic` | solo ricerca semantica (vettoriale) — ma il grafo resta comunque interrogato: `GraphSearchSettings.enabled=True` è il default globale, `basic` non lo tocca |
| `advanced` | ibrida (semantica + full-text) + ricerca sul grafo — analogo di LightRAG `hybrid` / cognee `GRAPH_COMPLETION`. ⚠️ **Rotto con Gemini/Vertex** (bug #3): porta con sé `search_strategy="hyde"` |
| `custom` | tutti i flag a mano (`use_hybrid_search`, `use_semantic_search`, `use_fulltext_search`, `search_strategy`, `graph_search_settings.enabled`, ...) — **quello che usiamo sempre**, per evitare il preset "advanced" |

`retrieval.rag(...)` genera sempre la risposta (con citazioni); `retrieval.search(...)`
solo retrieval, nessuna generazione (non lo usiamo: R2R genera come LightRAG/cognee).

## Feature aggiuntive (non usate qui)

- **Deep Research Agent** (`retrieval.agent`): reasoning multi-step con tool
  (ricerca web, python executor) — fuori scope per un confronto RAG "as-is".
- Multimodale (PDF/immagini/audio con OCR/VLM/trascrizione), gestione utenti e
  collezioni, quantizzazione dei vettori, reranking.
- Community summaries (vero GraphRAG alla Microsoft) — passo manuale, non
  invocato (vedi sopra).

## Installazione locale (sintetica)

Tutto in Docker, via [`docker-compose.r2r.yml`](../docker-compose.r2r.yml) — non
c'è modalità "solo libreria" come LightRAG/cognee, Postgres+pgvector è sempre
richiesto:

```bash
docker compose -f docker-compose.r2r.yml up   # Postgres+pgvector + immagine ufficiale sciphiai/r2r
```

`docker compose` legge `.env` dalla directory da cui viene lanciato per le
credenziali Vertex (nessun venv Python, nessun `pip install`) — vedi
`docs/setup.md` §8 per la guida passo passo e `./scripts/r2r_local.sh` come
wrapper già pronto.

## Come lo usiamo in questo repo

- Adapter: [`adapters/r2r_adapter.py`](../adapters/r2r_adapter.py) — **REST
  diretto con `httpx`**, non l'SDK ufficiale: il pacchetto `r2r` da PyPI
  trascina `fastapi<0.116` in modo incondizionato (non solo nell'extra
  `[core]`), in conflitto con `cognee` che vuole `fastapi>=0.116.2` —
  irrisolvibile nello stesso poetry env (verificato con `poetry lock`). I
  payload/endpoint dell'adapter ricalcano esattamente quelli dell'SDK ufficiale
  (letto dai sorgenti per ricavarli), non sono una REST API dedotta a occhio.
- Server: [`docker-compose.r2r.yml`](../docker-compose.r2r.yml) — Postgres+pgvector
  (volume Docker persistente, come il working dir di LightRAG) + l'immagine
  ufficiale `sciphiai/r2r:3.6.6` (pinnata, come `lightrag-hku`/`cognee` in
  `pyproject.toml`), entrambi in Docker: nessuna dipendenza Python del server
  (FastAPI, boto3, hatchet, anthropic SDK, ...) nel poetry env del progetto.
  [`scripts/r2r_local.sh`](../scripts/r2r_local.sh) è un wrapper sottile che
  verifica il `.env` e lancia `docker compose up`.
- Config del server: [`adapters/r2r_config/r2r_olivetti.toml`](../adapters/r2r_config/r2r_olivetti.toml)
  — overlay minimale sopra i default di R2R (montato in sola lettura nel
  container): `quality_llm`/`fast_llm` e `embedding.base_model` puntati su
  `vertex_ai/...`, `base_dimension=1536`.
- `search_mode` (`advanced` default = ibrido + grafo, `basic` = solo semantica)
  — nostro concetto, tradotto in `search_settings` espliciti (mai i preset di
  R2R: vedi bug #3 sotto).

## Bug trovati (server R2R 3.6.6, deployment "light")

Tre bug reali, tutti scoperti lanciando il server per la prima volta
(2026-09-11) e aggirati nell'adapter/nel compose — nessuno è un problema del
nostro codice:

1. **`R2R_PROJECT_NAME` case-sensitive.** Diventa il nome dello schema
   Postgres. `postgres.py` lo crea **quotato** (`CREATE SCHEMA IF NOT EXISTS
   "olivettiV0"` — case preservato), ma `prompts_handler.py` (trigger della
   tabella `prompts`, `CREATE OR REPLACE FUNCTION
   {project_name}.update_updated_at_column()`) lo referenzia **senza quote** →
   Postgres lo minuscolizza → `schema "olivettiv0" does not exist` allo
   startup. Aggirato tenendo `R2R_PROJECT_NAME` **tutto minuscolo**
   (`olivettiv0` in `docker-compose.r2r.yml`).
2. **Niente estrazione/embedding del grafo senza passi manuali.**
   `automatic_extraction=true` non ha alcun effetto con l'orchestrazione
   `simple` (warning silenzioso, `extraction_status` resta `pending` per
   sempre). E anche chiamando `POST /documents/{id}/extract` a mano, le
   entità restano con `description_embedding` NULL — invisibili a
   `graph_search` — finché non si chiama anche `POST
   /graphs/{collection_id}/pull` (copia entità/relazioni nel grafo della
   collezione e ne calcola gli embedding). L'adapter fa entrambe le chiamate
   (`_ensure_graph_extracted`).
3. **Il preset `search_mode="advanced"` rompe la ricerca con Gemini/Vertex.**
   Porta con sé `search_strategy="hyde"` (query expansion via LLM prima della
   ricerca vera e propria). La chiamata HyDE di R2R manda al provider LLM un
   messaggio con **solo il ruolo "system", senza "user"** — funziona con
   OpenAI/Anthropic ma non con Gemini: l'API Vertex separa `system_instruction`
   da `contents` e rifiuta un `contents` vuoto (`400 Unable to submit request
   because at least one contents field is required`). Risultato: non solo la
   generazione, **la ricerca stessa fallisce** (anche `retrieval.search`, non
   solo `retrieval.rag`). L'adapter non usa mai i preset `basic`/`advanced` di
   R2R: manda sempre `search_mode="custom"` con `search_settings` espliciti
   (`search_strategy` resta al default `"vanilla"`, niente HyDE).

## Limiti noti / attenzioni

- ✅ **Smoke-testato end-to-end** (2026-09-11, `--limit 3`, `search_mode`
  ibrido+grafo): risposte corrette e ben citate, in linea con LightRAG/cognee
  su questo stesso corpus (es. "3,7 kg, 85×302×324 mm" per la Lettera 22,
  identico al golden set). Vedi i bug sopra per cosa c'è voluto per arrivarci.
- **Entità del grafo in inglese anche su corpus italiano** (osservato nelle
  descrizioni estratte, es. "Lettera 22" descritta in inglese) — stesso
  pattern di traduzione automatica di supermemory (vedi
  `docs/implementazione.md` §16, confronto prompt); LightRAG e cognee restano
  in italiano. Impatto da verificare su un run completo.
- Deployment più pesante degli altri tre: richiede Docker + Postgres, non gira
  "solo con poetry install" come LightRAG/cognee.
- Ultimo push ~10 mesi fa (vedi sopra): progetto meno attivo di LightRAG/cognee
  di recente — coerente con l'aver trovato 3 bug non banali al primo utilizzo.
- La community-level GraphRAG (sintesi per comunità) resta fuori scope: solo
  entità/relazioni per-documento "pullate" nel grafo, non le comunità
  (`collections.extract` + build communities) — passo aggiuntivo volutamente
  non fatto (vedi "Architettura" sopra).
