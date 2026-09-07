"""
Fase 3 - Metriche.

Aggrega i giudizi di un run in numeri confrontabili tra tool e configurazioni.
Impianto volutamente snello (fase esplorativa).

Due gruppi di metriche:

1. Accuratezza sui contenuti (domande con ``answerable`` in {yes, partial}):
     precision = OK / (OK + PARTIAL + WRONG)      # delle risposte date, quante giuste
     recall    = OK / (# domande rispondibili)     # delle rispondibili, quante risolte
     f1        = media armonica
   ``DECLINED`` su una domanda rispondibile conta come miss (nel denominatore di recall).

2. Sicurezza / comportamento (domande con ``answerable`` == no):
     hallucination_rate    = risposte sostanziali (OK|PARTIAL|WRONG) / (# non rispondibili)
     correct_abstention    = DECLINED / (# non rispondibili)

Più un segnale sul retrieval, su tutte le domande giudicate:
     context_recall = context_supports==true / (# giudizi con il flag valorizzato)

Le domande senza annotazione golden (``answerable`` == unknown) sono contate a
parte in ``skipped_no_gold`` e non entrano in nessuna metrica.
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


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if not precision or not recall:
        return None
    return 2 * precision * recall / (precision + recall)


def compute_metrics(run_dir: Path) -> dict:
    judgements = {j["id"]: j for j in results_store.read_jsonl(run_dir / "judgements.jsonl")}
    gold_by_id = {q.id: q for q in load_questions(_QUESTIONS_CSV)}

    ok = partial = wrong = declined = 0          # su domande rispondibili
    answerable_total = 0
    hallucinated = abstained = 0                 # su domande NON rispondibili
    not_answerable_total = 0
    ctx_supported = ctx_total = 0
    skipped_no_gold = 0
    label_distribution: dict[str, int] = {}

    for qid, judgement in judgements.items():
        label = judgement["label"]
        label_distribution[label] = label_distribution.get(label, 0) + 1

        if judgement.get("context_supports") is not None:
            ctx_total += 1
            ctx_supported += int(bool(judgement["context_supports"]))

        gold = gold_by_id.get(qid)
        if gold is None or gold.answerable == "unknown":
            skipped_no_gold += 1
            continue

        if gold.answerable in ("yes", "partial"):
            answerable_total += 1
            ok += label == "OK"
            partial += label == "PARTIAL"
            wrong += label == "WRONG"
            declined += label == "DECLINED"
        elif gold.answerable == "no":
            not_answerable_total += 1
            if label == "DECLINED":
                abstained += 1
            else:
                hallucinated += 1

    precision = _ratio(ok, ok + partial + wrong)
    recall = _ratio(ok, answerable_total)

    return {
        "run_dir": str(run_dir),
        "n_judged": len(judgements),
        "skipped_no_gold": skipped_no_gold,
        "label_distribution": label_distribution,
        "content": {
            "answerable_questions": answerable_total,
            "ok": ok,
            "partial": partial,
            "wrong": wrong,
            "declined": declined,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        },
        "safety": {
            "not_answerable_questions": not_answerable_total,
            "hallucination_rate": _ratio(hallucinated, not_answerable_total),
            "correct_abstention_rate": _ratio(abstained, not_answerable_total),
        },
        "retrieval": {
            "judged_with_flag": ctx_total,
            "context_recall": _ratio(ctx_supported, ctx_total),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcola le metriche di un run giudicato.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    metrics = compute_metrics(args.run_dir)
    results_store.write_json(args.run_dir / "metrics.json", metrics)

    import json

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
