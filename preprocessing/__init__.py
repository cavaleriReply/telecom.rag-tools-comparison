"""Preprocessing comune: OCR dei PDF sorgente in testo grezzo per tutti gli adapter."""

from preprocessing.ocr import transcribe_pdf
from preprocessing.run import run

__all__ = ["transcribe_pdf", "run"]
