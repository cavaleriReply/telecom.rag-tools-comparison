#!/usr/bin/env bash
#
# Avvia supermemory-server (self-hosted) instradando TUTTO su Gemini/Vertex
# tramite lo shim OpenAI-compatible (scripts/vertex_openai_shim.py).
#
#   embedder    : gemini-embedding-001 (1536d)  -> shim  (identico a LightRAG)
#   estrazione  : gemini-2.5-flash               -> shim
#   generazione : la fa l'adapter, non il server -> gemini-2.5-flash
#
# supermemory-server (lite) non supporta Vertex direttamente: accetta solo
# provider a chiave (OpenAI/Anthropic/Gemini-AIStudio/Groq). Lo shim si fa
# passare per "OpenAI" e inoltra a Vertex col service account.
#
# Prerequisiti:
#   1. binario:  curl -fsSL https://supermemory.ai/install | bash
#   2. .env del progetto con VERTEXAI_PROJECT / VERTEXAI_LOCATION /
#      GOOGLE_APPLICATION_CREDENTIALS / LLM_MODEL / EMBEDDING_MODEL
#
# Uso:  ./scripts/supermemory_local.sh   (tienilo in un terminale dedicato)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- credenziali Vertex dal .env del progetto (le usa lo shim) ----------
set -a
# shellcheck disable=SC1091
source "$PROJECT_ROOT/.env"
[[ -f "$PROJECT_ROOT/scripts/supermemory.env" ]] && source "$PROJECT_ROOT/scripts/supermemory.env"
set +a

: "${VERTEXAI_PROJECT:?manca VERTEXAI_PROJECT nel .env}"
: "${GOOGLE_APPLICATION_CREDENTIALS:?manca GOOGLE_APPLICATION_CREDENTIALS nel .env}"

SHIM_PORT="${SHIM_PORT:-6799}"
SHIM_URL="http://127.0.0.1:$SHIM_PORT/v1"
export SUPERMEMORY_PORT="${SUPERMEMORY_PORT:-6767}"
export SUPERMEMORY_DATA_DIR="${SUPERMEMORY_DATA_DIR:-$PROJECT_ROOT/data/olivettiV0/supermemory_data}"

# --- 1. shim OpenAI -> Vertex (chat + embeddings) -----------------------
# Uccidi eventuali shim rimasti da run precedenti (porta occupata -> il nuovo
# shim non parte e il server parlerebbe con una versione vecchia).
pkill -f "scripts/vertex_openai_shim.py" 2>/dev/null || true
sleep 1

echo "avvio shim OpenAI->Vertex su :$SHIM_PORT ..."
poetry run python "$PROJECT_ROOT/scripts/vertex_openai_shim.py" \
    --port "$SHIM_PORT" --embedding-dimensions 1536 &
SHIM_PID=$!
trap 'kill $SHIM_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
    curl -sf "http://127.0.0.1:$SHIM_PORT/health" >/dev/null && break
    sleep 1
done
curl -sf "http://127.0.0.1:$SHIM_PORT/health" >/dev/null || { echo "shim non risponde su :$SHIM_PORT"; exit 1; }
echo "shim pronto."

# --- 2. configurazione del server: tutto verso lo shim -----------------
# LLM di estrazione/summary -> shim, modello pinnato a gemini-2.5-flash
export OPENAI_API_KEY="sk-shim-not-used"      # deve solo ESISTERE per il check di avvio
export OPENAI_BASE_URL="$SHIM_URL"
export OPENAI_MODEL="gemini-2.5-flash"
export OPENAI_FAST_MODEL="gemini-2.5-flash"
export OPENAI_TEXT_MODEL="gemini-2.5-flash"

# Embedder -> shim, gemini-embedding-001. Dim 1536 (non 3072): il vector store
# di supermemory (pgvector + HNSW) è limitato a 2000 dim. LightRAG usa la stessa
# taglia per parità (vedi LightRAGConfig.embedding_dimensions).
export SUPERMEMORY_EMBEDDING_PROVIDER="openai"
export SUPERMEMORY_EMBEDDING_BASE_URL="$SHIM_URL"
export SUPERMEMORY_EMBEDDING_MODEL="gemini-embedding-001"
export SUPERMEMORY_EMBEDDING_DIMENSIONS="1536"

echo "supermemory-server :$SUPERMEMORY_PORT"
echo "  llm      = openai-compat -> shim :$SHIM_PORT (gemini-2.5-flash)"
echo "  embedder = openai-compat -> shim :$SHIM_PORT (gemini-embedding-001, 1536d)"
echo "  data dir = $SUPERMEMORY_DATA_DIR"
cd "$PROJECT_ROOT"   # il data dir "segue" la working dir se SUPERMEMORY_DATA_DIR non bastasse
exec supermemory-server
