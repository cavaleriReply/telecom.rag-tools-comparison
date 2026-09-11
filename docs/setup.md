# Setup — da zero a benchmark

Guida lineare. Per il "perché" dei pezzi vedi [`implementazione.md`](implementazione.md).

---

## 0. Prerequisiti

- Python 3.12, [Poetry](https://python-poetry.org/) 2.x
- Un service account GCP con accesso a **Vertex AI** (file JSON delle credenziali)
- `curl`, `bash` (per supermemory)

---

## 1. Dipendenze

```bash
cd telecom.rag-tools-comparison
poetry install
```

Crea `.venv/` nel progetto (config in `poetry.toml`).

---

## 2. `.env`

Copia il template e compila:

```bash
cp .env.example .env
```

```dotenv
VERTEXAI_PROJECT="prj-..."
VERTEXAI_LOCATION="europe-west1"
GOOGLE_APPLICATION_CREDENTIALS=/percorso/assoluto/service-account.json

LLM_MODEL="vertex_ai/gemini-2.5-flash"
EMBEDDING_MODEL="vertex_ai/gemini-embedding-001"

SUPERMEMORY_BASE_URL="http://localhost:6767"   # solo se usi supermemory
R2R_BASE_URL="http://localhost:7272"           # solo se usi R2R
```

Verifica l'accesso a Vertex:

```bash
poetry run python -c "
import asyncio; from dotenv import load_dotenv; load_dotenv('.env')
from adapters import _vertex
print(asyncio.run(_vertex.probe_embedding_dim(dimensions=1536)), 'dim ok')
"
```

---

## 3. OCR (una volta)

```bash
poetry run python -m preprocessing
```

Produce `data/olivettiV0/ocr/*.txt` — l'input comune per tutti i tool.
Idempotente: rilanciare non ri-trascrive (usa `--overwrite` per forzare).

---

## 4. Golden set (manuale, opzionale ma serve per le metriche)

Apri `data/olivettiV0/competency_questions/competency_questions.csv` e aggiungi
le colonne (tutte opzionali):

| colonna | valori | serve per |
|---|---|---|
| `answerable` | `yes` / `no` / `partial` | recall, hallucination rate, corretta astensione |
| `expected_points` | testo libero, i punti chiave attesi | precision/recall di contenuto (il giudice) |
| `expected_source_docs` | doc_id (stem del file) separati da `;` | localizzare i fallimenti di retrieval |
| `category` | etichetta libera (`meccanica`, `storia`, ...) | metriche per categoria (`by_category`) |

Senza queste colonne il benchmark gira lo stesso, ma `metrics.py` salta le
domande non annotate (`skipped_no_gold`).

Prima implementazione: 51/51 `answerable` (28 yes / 14 no / 9 partial), 37/51
`expected_points`, tutte con `category`.

---

## 5. LightRAG

Nessun setup extra: la libreria è già in `poetry install`. I modelli arrivano
da `.env` via `adapters/_vertex.py`.

```bash
poetry run python -m eval.run_benchmark lightrag --query-mode hybrid --limit 3   # prova rapida
```

Working dir e grafo finiscono in `data/olivettiV0/lightrag_workdir/<config-hash>/`
(gitignored). L'ingest richiede qualche minuto (estrazione entità/relazioni) **la
prima volta**: il grafo persiste e i run successivi con la stessa config lo
riusano (`ingest_seconds` in `run.json` lo mostra). Per rifare da zero: cancella
la cartella dell'hash.

---

## 6. supermemory (self-hosted)

### 6a. Installa il binario (una volta)

```bash
curl -fsSL https://supermemory.ai/install | bash
```

- installa in `~/.supermemory/bin` + `~/.local/bin` (già nel PATH), **no sudo, no Node**
- al wizard "Pick a provider": scegli **4 (Skip)** — la config la passa lo script
- se parte il server da solo e fallisce ("No model provider API key"): **Ctrl-C**, è normale
- ⚠️ `~/.supermemory/` contiene **anche il binario**: non cancellarla

### 6b. Avvia (ogni sessione)

In un **terminale dedicato** (resta occupato):

```bash
./scripts/supermemory_local.sh
```

Lo script:
- uccide shim rimasti da run precedenti, poi avvia `scripts/vertex_openai_shim.py`
  su `:6799` (shim OpenAI→Vertex per chat + embeddings, dim 1536)
- avvia `supermemory-server` su `:6767` puntato allo shim
- alla chiusura (Ctrl-C) uccide lo shim

Atteso a schermo:
```
shim pronto.
supermemory-server :6767
  llm      = openai-compat -> shim :6799 (gemini-2.5-flash)
  embedder = openai-compat -> shim :6799 (gemini-embedding-001, 1536d)
...
→ supermemory ready
```

### 6c. Reset dello store

Se cambi dimensione di embedding, o l'ingest si è sporcato:

```bash
# ferma il server (Ctrl-C), poi:
rm -rf data/olivettiV0/supermemory_data && mkdir data/olivettiV0/supermemory_data
```

### 6d. Esegui

```bash
poetry run python -m eval.run_benchmark supermemory --query-mode hybrid --limit 3
```

---

## 7. cognee

Nessun setup extra: libreria Python (in `poetry install`), usa litellm → Vertex
direttamente (nessuno shim). L'adapter imposta da sé le env che cognee richiede
(`LLM_PROVIDER=custom`, ecc.) prima di importare cognee.

```bash
poetry run python -m eval.run_benchmark cognee --query-mode hybrid           # -> GRAPH_COMPLETION
poetry run python -m eval.run_benchmark cognee --query-mode rag_completion   # -> RAG_COMPLETION
```

Store per hash-config in `data/olivettiV0/cognee_data/<hash>/` (gitignored),
persistito e riusato dai run con la stessa config. Al primo avvio di ogni store
cognee fa ~50 migrazioni Alembic (lente, una tantum).

---

## 8. R2R (self-hosted, Docker Compose)

A differenza degli altri tre, R2R è un **server** appoggiato a Postgres+pgvector
(obbligatorio): non basta `poetry install`. Tutto l'ambiente (Postgres + server
R2R, immagine ufficiale) è in [`docker-compose.r2r.yml`](../docker-compose.r2r.yml)
— niente pip/venv a parte. Vedi [`docs/r2r.md`](r2r.md) per il perché delle scelte.

### 8a. Prerequisiti (una volta)

- **Docker** in esecuzione (Docker Compose v2, cioè il comando `docker compose`
  senza trattino — incluso in Docker Desktop e nelle install recenti di Docker Engine).
- `.env` compilato con `VERTEXAI_PROJECT` / `VERTEXAI_LOCATION` /
  `GOOGLE_APPLICATION_CREDENTIALS` (§2) — `docker compose` li legge da lì.

### 8b. Avvia (ogni sessione)

In un **terminale dedicato** (resta occupato):

```bash
./scripts/r2r_local.sh
```

È un wrapper sottile su `docker compose -f docker-compose.r2r.yml up`: tutta
la configurazione vive nel compose file. Avvia:
- **Postgres+pgvector** (volume Docker `olivetti_r2r_pgdata`, persistente tra
  un run e l'altro, come il working dir di LightRAG — anche porta `:5433` sull'host,
  comoda per ispezionare con `psql` se ti serve, R2R ci arriva comunque via rete Docker interna)
- **server R2R** (immagine ufficiale `sciphiai/r2r:3.6.6`, pinnata) su `:7272`,
  puntato su Vertex via litellm nativo (`adapters/r2r_config/r2r_olivetti.toml`,
  montato in sola lettura nel container: `vertex_ai/gemini-2.5-flash` +
  `vertex_ai/gemini-embedding-001` @ 1536d) — **nessuno shim**, a differenza di supermemory.
  Le credenziali Vertex (JSON del service account) sono montate anch'esse in
  sola lettura, non copiate nell'immagine.

Ctrl-C ferma i container ma **non** cancella il volume Postgres: i dati restano.
Per girare in background: `docker compose -f docker-compose.r2r.yml up -d`
(poi `docker compose -f docker-compose.r2r.yml logs -f r2r` per i log).

### 8c. Reset dello store

Se cambi `base_dimension` nel toml, o l'ingest si è sporcato:

```bash
docker compose -f docker-compose.r2r.yml down -v   # ferma E cancella il volume
```

Al prossimo `./scripts/r2r_local.sh` riparte da un Postgres vuoto (nuove
migrazioni al primo avvio, qualche secondo).

### 8d. Esegui

```bash
poetry run python -m eval.run_benchmark r2r --query-mode hybrid --limit 3
```

✅ Smoke-testato end-to-end (2026-09-11): con `search_mode` ibrido+grafo,
l'ingest fa anche estrazione entità/relazioni e "pull" nel grafo della
collezione (§8b, dettagli in `docs/r2r.md`) — la primissima volta su un
corpus può richiedere qualche minuto in più delle volte successive (chiamate
LLM per l'estrazione, come l'ingest degli altri tool). La prima chiamata a
`/v3/health` dopo `up` può metterci qualche secondo (migrazioni Postgres del
server al primo avvio).

---

## 9. Benchmark completo + valutazione

```bash
# 1. esecuzione (un comando per tool)
poetry run python -m eval.run_benchmark lightrag    --query-mode hybrid
poetry run python -m eval.run_benchmark cognee      --query-mode hybrid
poetry run python -m eval.run_benchmark supermemory --query-mode hybrid   # serve ./scripts/supermemory_local.sh attivo
poetry run python -m eval.run_benchmark r2r         --query-mode hybrid   # serve ./scripts/r2r_local.sh attivo

# 2. giudizio (per ogni run dir stampata sopra)
poetry run python -m eval.judge   eval/results/<tool>/<slug>/<timestamp>

# 3. metriche
poetry run python -m eval.metrics eval/results/<tool>/<slug>/<timestamp>

# 4. (opzionale) disaccordi gold vs giudice — per rivedere le annotazioni answerable
poetry run python -m eval.review_gold eval/results/<tool>/<slug>/<timestamp>
```

I risultati restano in `eval/results/<tool>/<config-slug>__<hash>/<timestamp>/`
(`answers.jsonl` è gitignored — grosso e rigenerabile).

---

## Troubleshooting

| Sintomo | Causa / fix |
|---|---|
| `supermemory-server` non parte, "No model provider API key" | Stai lanciando il binario **da solo**. Usa `./scripts/supermemory_local.sh` (imposta `OPENAI_API_KEY` fittizio + base_url). |
| `exceeds the pgvector HNSW limit of 2000` | Embedding a >2000 dim. Il progetto usa 1536 ovunque; se hai cambiato config, reset dello store (6c). |
| supermemory: ingest "done" ma ricerca sempre vuota (0 risultati) | Lo shim restituisce una dimensione diversa da quella dello store. Verifica che lo script passi `--embedding-dimensions 1536` e fai reset dello store. |
| supermemory: 0 memorie estratte | Lo shim non inoltra le tool-call. Assicurati di avere l'ultima versione di `scripts/vertex_openai_shim.py`. |
| supermemory: `400 ... containerTag Must be 100 characters or less` | Risolto: l'adapter genera un tag corto (`olivettiV0__<mode>__<hash>`). Se rieseguito con codice vecchio, aggiorna. |
| `address already in use :6799` | Shim di un run precedente ancora vivo. Lo script ora fa `pkill` all'avvio; altrimenti: `ps aux | grep '[v]ertex_openai_shim' ` e killa il PID. |
| LightRAG: `Vector count mismatch` | Non dovrebbe capitare (dimensione verificata in `setup()`). Se capita, la dim richiesta ≠ quella resa da Vertex. |
| cognee: `ProviderNotDeducibleError` all'import | L'adapter imposta `LLM_PROVIDER=custom` prima di `import cognee`; capita solo importando cognee a mano senza quella env. |
| cognee: errori SQLAlchemy su `pipeline_runs` | Non fatali (la pipeline continua). Con lo store per-hash isolato spariscono; se persistono, `rm -rf data/olivettiV0/cognee_data/<hash>`. |
| cognee: primo run lentissimo | ~50 migrazioni Alembic sullo store fresco, una tantum per hash-config. |
| 429 da Vertex | `adapters/_vertex.py` ritenta con backoff; se persiste, abbassa la concorrenza (`_EMBED_MAX_CONCURRENCY`, `--concurrency` nel judge). |
| R2R: `R2R non risponde su http://localhost:7272` | Il server non è su, o è ancora in avvio (migrazioni al primo boot). Avvia `./scripts/r2r_local.sh` in un terminale dedicato e attendi che l'healthcheck sia `healthy` (`docker compose -f docker-compose.r2r.yml ps`) prima del benchmark. |
| R2R: `port is already allocated` (Docker) | Porta `:7272` o `:5433` già occupata (un run precedente, o un altro servizio). `docker compose -f docker-compose.r2r.yml down` per fermare i container di questo progetto; altrimenti trova cosa occupa la porta (`lsof -i :7272`) e chiudilo. |
| R2R: `manca VERTEXAI_PROJECT nel .env` (o simili) all'avvio del compose | `docker compose` legge `.env` dalla directory da cui viene lanciato: `scripts/r2r_local.sh` si mette da solo nella root del progetto; se lanci `docker compose` a mano, fallo dalla root (dove sta il `.env`), non da `scripts/`. |
| R2R: `Unable to submit request because at least one contents field is required` (400 da Vertex) | Bug di R2R: il preset `search_mode="basic"/"advanced"` porta `search_strategy="hyde"`, che con Gemini manda un messaggio senza turno "user" e rompe anche la ricerca. L'adapter non usa mai quei preset (sempre `search_mode="custom"`) — se vedi questo errore, qualcosa sta chiamando l'API con un preset invece che con `search_settings` espliciti. Vedi `docs/r2r.md` bug #3. |
| R2R: la ricerca col grafo torna vuota, entità con `description_embedding` NULL | `automatic_extraction` non funziona con l'orchestrazione `simple` (deployment light) e l'estrazione manuale da sola non basta: serve anche `POST /graphs/{collection_id}/pull`. L'adapter fa entrambe le chiamate (`_ensure_graph_extracted`); se hai ingestato con una versione più vecchia dell'adapter, il prossimo `ingest()` ripara da solo (richiama `pull`, idempotente). Vedi `docs/r2r.md` bug #2. |
| R2R: cambio `base_dimension` nel toml e l'ingest fallisce | Come supermemory: la dimensione va fissata **prima** del primo ingest (crea la colonna pgvector). Reset dello store (8c: `down -v`). |
| R2R: `asyncpg.exceptions.InvalidSchemaNameError: schema "..." does not exist` all'avvio | Bug di R2R: `R2R_PROJECT_NAME` con maiuscole crea uno schema Postgres case-preservato ma un trigger interno lo referenzia senza quote (minuscolizzato) → mismatch. Il compose usa già `olivettiv0` (tutto minuscolo) per aggirarlo — se l'hai cambiato, rimettilo minuscolo e resetta lo store (8c). Vedi `docs/r2r.md`. |
