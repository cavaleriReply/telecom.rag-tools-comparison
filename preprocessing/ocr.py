"""
OCR / estrazione testo da un singolo documento.

Scopo unico di questo modulo: trasformare un PDF nel suo testo integrale usando
lo stesso LLM multimodale (Gemini su Vertex AI) configurato in ``.env``.
Qui non c'e' I/O su disco ne' conoscenza della struttura del progetto:
l'orchestrazione batch sta in ``preprocessing/run.py``.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import litellm

# Prompt di trascrizione: vogliamo il testo cosi' com'e', senza sintesi ne'
# formattazione, per non introdurre bias prima che i framework RAG lo indicizzino.
_OCR_PROMPT = (
    "Trascrivi INTEGRALMENTE tutto il testo presente in questo documento PDF, "
    "pagina per pagina, nell'ordine di lettura naturale. Non riassumere, non "
    "commentare, non aggiungere markdown - restituisci solo il testo trascritto, "
    "cosi' com'e' nel documento."
)


def _pdf_to_data_uri(pdf_path: Path) -> str:
    """Codifica il PDF come data URI base64: il formato che litellm passa a Gemini."""
    encoded = base64.b64encode(pdf_path.read_bytes()).decode()
    return f"data:application/pdf;base64,{encoded}"


async def transcribe_pdf(pdf_path: Path, *, model: str | None = None) -> str:
    """
    Restituisce il testo integrale di ``pdf_path``.

    ``model`` default = variabile d'ambiente ``LLM_MODEL``
    (es. ``vertex_ai/gemini-2.5-flash``). Le credenziali Vertex vengono lette da
    litellm dalle env ``VERTEXAI_*`` / ``GOOGLE_APPLICATION_CREDENTIALS``.
    """
    model = model or os.environ["LLM_MODEL"]
    response = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": _pdf_to_data_uri(pdf_path)}},
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""
