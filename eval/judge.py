"""
Fase 2 - Il giudice.

Per ogni risposta di un run, un LLM giudice assegna UNA label discreta
confrontandola con il testo sorgente OCR (indipendente dal tool valutato) e,
se disponibili, con i ``expected_points`` del golden set.

Label:
    OK        risposta corretta e supportata dal corpus
    PARTIAL   parzialmente corretta / incompleta rispetto ai punti attesi
    WRONG     afferma cose non supportate dal corpus (allucinazione)
    DECLINED  il tool dichiara di non sapere / non risponde

In più, un flag sul retrieval (indipendente dalla generazione):
    context_supports  i frammenti recuperati bastavano a rispondere?
Serve a distinguere "non ha recuperato" da "ha recuperato ma non ha usato".

Uso:
    poetry run python -m eval.judge eval/results/lightrag/<slug>/<timestamp>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from adapters import _vertex
from eval import results_store
from eval.dataset import GoldQuestion, load_questions

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OCR_DIR = _PROJECT_ROOT / "data" / "olivettiV0" / "ocr"
_QUESTIONS_CSV = (
    _PROJECT_ROOT / "data" / "olivettiV0" / "competency_questions" / "competency_questions.csv"
)

_LABELS = ("OK", "PARTIAL", "WRONG", "DECLINED")

_SYSTEM_PROMPT = """Sei un fact-checker rigoroso. Valuti se la RISPOSTA di un sistema è supportata
dal CORPUS fornito (il testo sorgente), non se è vera nel mondo reale.

Rispondi SOLO con un oggetto JSON:
{"label": "<OK|PARTIAL|WRONG|DECLINED>",
 "rationale": "<una frase>",
 "missing_points": ["<punto atteso non coperto>", ...],
 "context_supports": <true|false>}

Scegli la label in quest'ordine:
1. DECLINED: la RISPOSTA dichiara di non sapere / non aver trovato l'informazione
   (es. "Non ho trovato...", "non è disponibile", "il contesto non specifica").
   Vale ANCHE se l'informazione era in realtà nel corpus: in quel caso è un
   fallimento di copertura, NON un'allucinazione — resta DECLINED.
2. WRONG: la risposta AFFERMA fatti non presenti nel corpus (allucinazione vera).
3. PARTIAL: afferma fatti corretti e supportati, ma incompleti o con imprecisioni.
4. OK: corretta e con pieno riscontro nel corpus (copre i punti attesi, se indicati).

- context_supports: true se i FRAMMENTI RECUPERATI, da soli, contengono abbastanza per rispondere."""


@dataclass
class Judgement:
    id: str
    label: str
    rationale: str
    missing_points: list[str]
    context_supports: bool | None


def _load_corpus_text() -> str:
    """Concatena i .txt OCR: è il riferimento indipendente dal tool valutato."""
    parts = [
        f"=== {p.name} ===\n{p.read_text(encoding='utf-8')}"
        for p in sorted(_OCR_DIR.glob("*.txt"))
    ]
    return "\n\n".join(parts)


def _parse_judge_json(raw: str) -> dict:
    """Parsing tollerante: JSON puro, oppure il primo blocco {...} nel testo."""
    for candidate in (raw.strip(), *(re.findall(r"\{.*\}", raw, re.DOTALL))):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


async def _judge_one(
    answer_row: dict, gold: GoldQuestion | None, corpus_text: str
) -> Judgement:
    expected = gold.expected_points if gold else ""
    user_prompt = (
        f"CORPUS:\n{corpus_text}\n\n"
        f"DOMANDA: {answer_row['question']}\n"
        f"PUNTI ATTESI (dal golden set, può essere vuoto): {expected or '(non forniti)'}\n\n"
        f"FRAMMENTI RECUPERATI DAL SISTEMA:\n{chr(10).join(answer_row.get('contexts') or []) or '(nessuno)'}\n\n"
        f"RISPOSTA DA VALUTARE:\n{answer_row['answer'] or '(vuota)'}"
    )
    raw = await _vertex.acompletion(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,  # il giudice dev'essere il più deterministico possibile
    )
    data = _parse_judge_json(raw)
    label = str(data.get("label", "")).strip().upper()
    return Judgement(
        id=answer_row["id"],
        label=label if label in _LABELS else "WRONG",
        rationale=str(data.get("rationale", "")),
        missing_points=list(data.get("missing_points") or []),
        context_supports=data.get("context_supports"),
    )


async def judge_run(run_dir: Path, *, concurrency: int = 3) -> Path:
    """Giudica ``answers.jsonl`` del run e scrive ``judgements.jsonl``."""
    answers = results_store.read_jsonl(run_dir / "answers.jsonl")
    gold_by_id = {q.id: q for q in load_questions(_QUESTIONS_CSV)}
    corpus_text = _load_corpus_text()

    semaphore = asyncio.Semaphore(concurrency)

    async def _guarded(row: dict) -> Judgement:
        async with semaphore:
            judgement = await _judge_one(row, gold_by_id.get(row["id"]), corpus_text)
        print(f"  {judgement.id}: {judgement.label}")
        return judgement

    judgements = await asyncio.gather(*(_guarded(row) for row in answers))
    results_store.write_jsonl(
        run_dir / "judgements.jsonl", [vars(j) for j in judgements]
    )
    print(f"Scritti {len(judgements)} giudizi in {run_dir / 'judgements.jsonl'}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Giudica le risposte di un run.")
    parser.add_argument("run_dir", type=Path, help="Cartella del run (contiene answers.jsonl).")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(_PROJECT_ROOT / ".env")
    asyncio.run(judge_run(args.run_dir, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
