# LightRAG — due diligence

Dati raccolti il **2026-09-07**. Repo: <https://github.com/HKUDS/LightRAG>

## Progetto in salute?

| Indicatore | Valore |
|---|---|
| Stelle | ~39.450 |
| Fork | ~5.550 |
| Watcher | ~302 |
| Issue aperte | ~226 |
| Contributori | ~337 (incl. anonimi) |
| Creato | 2024-10-02 |
| Ultimo commit | 2026-09-07 (stesso giorno) |
| Attività recenti | >100 commit / 30 giorni, ~326 PR merge / 60 giorni |
| Licenza | MIT |
| Ultima release | `v1.5.7` (2026-09-02) — **è la versione che usiamo** |
| Pacchetto PyPI | `lightrag-hku` (server + WebUI: `lightrag-hku[api]`) |

Progetto molto attivo. Nota: molte PR recenti hanno branch `claude/issue-*` →
manutenzione fortemente assistita da agenti AI. Il ritmo di rilascio è alto
(1.4.x → 1.5.x in poche settimane), quindi l'**API cambia spesso**: per il
confronto "as-is" teniamo la versione pinnata in `pyproject.toml`.

## Paper

"LightRAG: Simple and Fast Retrieval-Augmented Generation" — Guo et al., 2024.
arXiv: **2410.05779**. Accettato a **EMNLP 2025**.

## Compatibilità modelli

Aperti e chiusi entrambi supportati: OpenAI, Azure OpenAI, AWS Bedrock, Gemini,
Ollama, HuggingFace, vLLM e qualunque endpoint OpenAI-compatible. L'LLM e
l'embedding si iniettano come funzioni (`llm_model_func` / `embedding_func`):
è il punto di estensione ufficiale, non una modifica al codice.

Per BAAI/bge-m3 è indicato come embedding locale di riferimento.

## Storage

Default: 4 store JSON in-memory (ok solo per test — è il nostro caso, 4 documenti).
Produzione: PostgreSQL, MongoDB, Neo4j, Milvus, Qdrant, OpenSearch, Memgraph.

## Feature aggiuntive

- **WebUI** (`/webui`): gestione documenti + **esplorazione visuale del knowledge
  graph**. `/workspace`: interfaccia di sola query.
- **REST API server** completo (`lightrag-server`).
- Modalità di retrieval: `naive`, `local`, `global`, `hybrid`, `mix`.

## Installazione locale (sintetica)

```bash
# solo libreria (quello che serve a questo repo)
pip install "lightrag-hku==1.5.7"

# oppure con server + WebUI + knowledge-graph explorer
pip install "lightrag-hku[api]==1.5.7"
lightrag-server            # espone REST API + WebUI su http://localhost:9621
```

## Issue rilevanti per il nostro caso

Da verificare incrociando con i nostri risultati (analisi delle cause):

- Granularità dell'estrazione entità/relazioni e comportamento di
  `force_llm_summary_on_merge`: quando LightRAG fonde entità duplicate ne
  **riassume** la descrizione via LLM → possibile perdita di dettagli specifici
  (es. misure, codici). È l'analogo del problema visto su Cognee con le "dimensioni".
- Nessuna issue aperta trovata che segnali *esattamente* questo sul nostro tipo
  di documento; da ricontrollare quando avremo le risposte sbagliate concrete.

## Come lo usiamo in questo repo

- Adapter: [`adapters/lightrag_adapter.py`](../adapters/lightrag_adapter.py) — wrapper sottile, LightRAG `as-is`.
- Modelli: Gemini 2.5 Flash + `gemini-embedding-001` **@ 1536 dim** su Vertex AI,
  via `litellm` ([`adapters/_vertex.py`](../adapters/_vertex.py)), come tutti gli
  altri tool. 1536 (non i 3072 nativi) per parità con supermemory — vedi
  [`implementazione.md` §11](implementazione.md).
- Storage: JSON default, working dir sotto `data/olivettiV0/lightrag_workdir/<config>/`.
- Config esposte: `query_mode`, `top_k`, `chunk_top_k`, `chunk_token_size`,
  `embedding_dimensions`, reranker disattivato. Ogni combinazione = risultati
  separati in `eval/results/`.
