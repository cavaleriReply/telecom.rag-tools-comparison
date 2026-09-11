#!/usr/bin/env bash
#
# Avvia l'ambiente R2R self-hosted: Postgres+pgvector + server R2R, entrambi
# in Docker via docker-compose.r2r.yml (vedi quel file e docs/r2r.md per i
# dettagli). Wrapper sottile: tutta la configurazione vive nel compose file,
# qui c'è solo il controllo "il .env esiste ed è compilato" prima di partire,
# per un errore leggibile invece dello `${VAR:?...}` grezzo di Docker Compose.
#
# Prerequisiti:
#   1. Docker in esecuzione.
#   2. .env del progetto con VERTEXAI_PROJECT / VERTEXAI_LOCATION /
#      GOOGLE_APPLICATION_CREDENTIALS.
#
# Uso:  ./scripts/r2r_local.sh   (tienilo in un terminale dedicato: Ctrl-C
#       ferma i container, i dati restano nel volume Docker)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"   # docker compose legge .env dalla working dir: deve essere la root del progetto

[[ -f ".env" ]] || { echo "manca .env nella root del progetto (copia .env.example)"; exit 1; }
set -a
# shellcheck disable=SC1091
source .env
set +a
: "${VERTEXAI_PROJECT:?manca VERTEXAI_PROJECT nel .env}"
: "${VERTEXAI_LOCATION:?manca VERTEXAI_LOCATION nel .env}"
: "${GOOGLE_APPLICATION_CREDENTIALS:?manca GOOGLE_APPLICATION_CREDENTIALS nel .env}"
[[ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]] || { echo "GOOGLE_APPLICATION_CREDENTIALS non punta a un file: $GOOGLE_APPLICATION_CREDENTIALS"; exit 1; }

echo "R2R su :7272 (Postgres+pgvector su :5433, solo per ispezione da host)"
echo "  llm      = litellm nativo -> vertex_ai/gemini-2.5-flash (nessuno shim)"
echo "  embedder = litellm nativo -> vertex_ai/gemini-embedding-001, 1536d"
echo "  config   = adapters/r2r_config/r2r_olivetti.toml"
echo

exec docker compose -f docker-compose.r2r.yml up
