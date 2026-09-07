"""
Generazione della risposta per i tool *retrieval-only* (es. supermemory).

Alcuni strumenti recuperano i passaggi ma non producono una risposta. Per non
introdurre differenze dovute a prompt diversi, tutti questi tool usano lo stesso
prompt minimale e lo stesso LLM (Gemini via ``_vertex``).

Il prompt è volutamente povero: nessuna istruzione "furba", solo "rispondi con
quello che c'è nel contesto, altrimenti dichiara di non saperlo". Così quello
che misuriamo è la qualità del *retrieval*, non del prompt engineering.
"""

from __future__ import annotations

from adapters import _vertex

_SYSTEM_PROMPT = (
    "Rispondi alla domanda usando esclusivamente il CONTESTO fornito. "
    "Se il contesto non contiene l'informazione, rispondi esattamente: "
    "\"Non ho trovato questa informazione nei documenti.\" "
    "Non usare conoscenza esterna. Rispondi in italiano, in modo conciso."
)


async def answer_from_contexts(question: str, contexts: list[str]) -> str:
    """Genera una risposta a partire dai passaggi recuperati da un tool retrieval-only."""
    if not contexts:
        return "Non ho trovato questa informazione nei documenti."

    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    return await _vertex.acompletion(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTESTO:\n{numbered}\n\nDOMANDA: {question}"},
        ]
    )
