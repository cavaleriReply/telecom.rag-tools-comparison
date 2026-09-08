"""
Report gold-vs-judge — aiuta a rivedere le annotazioni ``answerable``.

Il giudice controlla la groundedness in modo indipendente dal golden set. Quando
i due non concordano, o l'annotazione è da correggere o c'è un caso limite: qui
li elenchiamo con risposta + rationale, così la revisione è veloce.

Casi segnalati (per un run già giudicato):
  gold=no        & label OK/PARTIAL  -> "non rispondibile" ma il sistema ha dato
                                        una risposta che il giudice ritiene supportata
  gold=yes/part. & label DECLINED    -> "rispondibile" ma il sistema si è tirato indietro
  gold=yes       & label WRONG       -> allucinazione su una domanda rispondibile

Uso:
    poetry run python -m eval.review_gold eval/results/<tool>/<slug>/<timestamp>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval import results_store
from eval.dataset import load_questions

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_QUESTIONS_CSV = (
    _PROJECT_ROOT / "data" / "olivettiV0" / "competency_questions" / "competency_questions.csv"
)


def _flag(answerable: str, label: str) -> str | None:
    if answerable == "no" and label in ("OK", "PARTIAL"):
        return "gold=no ma risposta supportata"
    if answerable in ("yes", "partial") and label == "DECLINED":
        return "gold=answerable ma il sistema declina"
    if answerable == "yes" and label == "WRONG":
        return "allucinazione su domanda rispondibile"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Disaccordi gold vs giudice per un run.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    gold_by_id = {q.id: q for q in load_questions(_QUESTIONS_CSV)}
    judgements = {j["id"]: j for j in results_store.read_jsonl(args.run_dir / "judgements.jsonl")}
    answers = {a["id"]: a for a in results_store.read_jsonl(args.run_dir / "answers.jsonl")}

    disagreements = 0
    for qid, judgement in judgements.items():
        gold = gold_by_id.get(qid)
        if gold is None:
            continue
        reason = _flag(gold.answerable, judgement["label"])
        if reason is None:
            continue
        disagreements += 1
        print(f"\n[{qid}] {gold.question}")
        print(f"  ⚠  {reason}  (gold={gold.answerable}, giudice={judgement['label']})")
        print(f"  risposta:  {(answers.get(qid, {}).get('answer', '') or '')[:220]}")
        print(f"  rationale: {judgement.get('rationale', '')[:220]}")

    total = len(judgements)
    print(f"\n{disagreements}/{total} disaccordi da rivedere.")


if __name__ == "__main__":
    main()
