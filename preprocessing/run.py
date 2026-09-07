"""
OCR batch: da una cartella di PDF a una cartella di ``.txt``.

Il testo prodotto qui e' l'UNICO input comune per tutti gli adapter RAG: ogni
framework deve indicizzare esattamente questi file, cosi' le differenze di
performance dipendono dal framework e non da come sono stati letti i documenti.

Uso:
    poetry run python -m preprocessing                 # dataset olivettiV0, default
    poetry run python -m preprocessing --overwrite     # ritrascrive tutto
    poetry run python -m preprocessing --input-dir ... --output-dir ...
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from preprocessing.ocr import transcribe_pdf

# ---------------------------------------------------------------------------
# Percorsi di default per il dataset corrente (Olivetti V0).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INPUT_DIR = _PROJECT_ROOT / "data" / "olivettiV0" / "docs"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "olivettiV0" / "ocr"


# ---------------------------------------------------------------------------
# Trascrizione di un singolo file
# ---------------------------------------------------------------------------
async def _transcribe_one(
    pdf_path: Path, output_dir: Path, *, overwrite: bool
) -> tuple[Path, str]:
    """Trascrive un PDF in ``<stem>.txt``. Ritorna (path_txt, stato) per il report."""
    txt_path = output_dir / f"{pdf_path.stem}.txt"
    if txt_path.exists() and not overwrite:
        return txt_path, "skip (gia' presente)"
    text = await transcribe_pdf(pdf_path)
    txt_path.write_text(text, encoding="utf-8")
    return txt_path, f"ok ({len(text)} caratteri)"


# ---------------------------------------------------------------------------
# Orchestrazione batch
# ---------------------------------------------------------------------------
async def run(input_dir: Path, output_dir: Path, *, overwrite: bool = False) -> list[Path]:
    """Trascrive tutti i PDF di ``input_dir`` in ``output_dir``. Idempotente di default."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"Nessun PDF trovato in {input_dir}")

    # I documenti sono pochi e indipendenti: li trascriviamo in parallelo.
    results = await asyncio.gather(
        *(_transcribe_one(p, output_dir, overwrite=overwrite) for p in pdf_paths)
    )
    for pdf_path, (txt_path, status) in zip(pdf_paths, results):
        print(f"  {pdf_path.name}  ->  {txt_path.name}  [{status}]")
    return [txt_path for txt_path, _ in results]


# ---------------------------------------------------------------------------
# Entry-point CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR batch dei documenti sorgente (PDF -> TXT)."
    )
    parser.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ritrascrive anche i .txt gia' presenti.",
    )
    args = parser.parse_args()

    # Carica LLM_MODEL e le credenziali Vertex dal .env del progetto.
    load_dotenv(_PROJECT_ROOT / ".env")

    print(f"OCR: {args.input_dir}  ->  {args.output_dir}")
    asyncio.run(run(args.input_dir, args.output_dir, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
