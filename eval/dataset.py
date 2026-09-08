"""
Caricamento del set di domande + golden set.

Il golden set vive nello stesso CSV delle domande
(``data/olivettiV0/competency_questions/competency_questions.csv``), con colonne
opzionali aggiunte a mano:

    id, question,
    answerable            -> yes | no | partial   (rispondibile dai 4 documenti?)
    expected_points       -> testo libero: i punti chiave di una buona risposta
    expected_source_docs  -> doc_id separati da ";"
    category              -> etichetta libera (meccanica | storia | design | ...)

Le colonne golden possono mancare (fase esplorativa): in quel caso i campi
restano vuoti e le metriche che li richiedono vengono semplicemente saltate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Answerable = Literal["yes", "no", "partial", "unknown"]


@dataclass(frozen=True)
class GoldQuestion:
    id: str
    question: str
    answerable: Answerable = "unknown"
    expected_source_docs: tuple[str, ...] = ()
    expected_points: str = ""
    category: str = ""

    @property
    def has_gold(self) -> bool:
        """True se qualcuno ha annotato questa domanda (serve per le metriche)."""
        return self.answerable != "unknown" or bool(self.expected_points)


def _parse_answerable(value: str) -> Answerable:
    normalized = (value or "").strip().lower()
    return normalized if normalized in ("yes", "no", "partial") else "unknown"


def _parse_docs(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(";") if part.strip())


def load_questions(csv_path: Path) -> list[GoldQuestion]:
    """Legge il CSV e restituisce le domande con le annotazioni golden disponibili."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    questions = [
        GoldQuestion(
            id=row["id"].strip(),
            question=row["question"].strip(),
            answerable=_parse_answerable(row.get("answerable", "")),
            expected_source_docs=_parse_docs(row.get("expected_source_docs", "")),
            expected_points=(row.get("expected_points", "") or "").strip(),
            category=(row.get("category", "") or "").strip().lower(),
        )
        for row in rows
        if row.get("question", "").strip()
    ]
    if not questions:
        raise ValueError(f"Nessuna domanda trovata in {csv_path}")
    return questions
