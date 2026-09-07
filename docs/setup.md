# Setup — da zero a benchmark

Guida lineare. Per il "perché" dei pezzi vedi [`implementazione.md`](implementazione.md).

---

## 0. Prerequisiti

- Python 3.12, [Poetry](https://python-poetry.org/) 2.x
- Un service account GCP con accesso a **Vertex AI** (file JSON delle credenziali)
- `curl`, `bash` (per supermemory)
- ~1.5 GB RAM liberi durante l'ingest di supermemory

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
le colonne:

| colonna | valori |
|---|---|
| `answerable` | `yes` / `no` / `partial` |
| `expected_source_docs` | doc_id separati da `;` |
| `expected_points` | testo libero, i punti chiave attesi |

Senza queste colonne il benchmark gira lo stesso, ma `metrics.py` salta le
domande non annotate.

---

## 5. LightRAG

Nessun setup extra: la libreria è già in `poetry install`. I modelli arrivano
da `.env` via `adapters/_vertex.py`.

```bash
poetry run python -m eval.run_benchmark lightrag --query-mode hybrid --limit 3   # prova rapida
```

Working dir e grafo finiscono in `data/olivettiV0/lightrag_workdir/<config>/`
(gitignored). L'ingest richiede qualche minuto (estrazione entità/relazioni).

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

## 7. Benchmark completo + valutazione

```bash
# 1. esecuzione (un comando per tool)
poetry run python -m eval.run_benchmark lightrag    --query-mode hybrid
poetry run python -m eval.run_benchmark supermemory --query-mode hybrid

# 2. giudizio (per ogni run dir stampata sopra)
poetry run python -m eval.judge   eval/results/<tool>/<slug>/<timestamp>

# 3. metriche
poetry run python -m eval.metrics eval/results/<tool>/<slug>/<timestamp>
```

I risultati restano in `eval/results/<tool>/<config-slug>__<hash>/<timestamp>/`.

---

## Troubleshooting

| Sintomo | Causa / fix |
|---|---|
| `supermemory-server` non parte, "No model provider API key" | Stai lanciando il binario **da solo**. Usa `./scripts/supermemory_local.sh` (imposta `OPENAI_API_KEY` fittizio + base_url). |
| `exceeds the pgvector HNSW limit of 2000` | Embedding a >2000 dim. Il progetto usa 1536 ovunque; se hai cambiato config, reset dello store (6c). |
| supermemory: ingest "done" ma ricerca sempre vuota (0 risultati) | Lo shim restituisce una dimensione diversa da quella dello store. Verifica che lo script passi `--embedding-dimensions 1536` e fai reset dello store. |
| supermemory: 0 memorie estratte | Lo shim non inoltra le tool-call. Assicurati di avere l'ultima versione di `scripts/vertex_openai_shim.py`. |
| `address already in use :6799` | Shim di un run precedente ancora vivo. Lo script ora fa `pkill` all'avvio; altrimenti: `ps aux | grep '[v]ertex_openai_shim' ` e killa il PID. |
| LightRAG: `Vector count mismatch` | Non dovrebbe capitare (dimensione verificata in `setup()`). Se capita, la dim richiesta ≠ quella resa da Vertex. |
| 429 da Vertex | `adapters/_vertex.py` ritenta con backoff; se persiste, abbassa la concorrenza (`_EMBED_MAX_CONCURRENCY`, `--concurrency` nel judge). |
