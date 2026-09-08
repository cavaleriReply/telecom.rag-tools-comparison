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

2. Sicurezza / comportamento:
     hallucination_rate    = WRONG / (# domande giudicate)   [il giudice dice
                             "non supportata dal corpus" — su qualsiasi domanda]
     abstention_rate       = DECLINED / (# domande answerable=no)
     answered_anyway_rate  = non-DECLINED / (# domande answerable=no)   [informativo:
                             o il sistema ha detto "non nel corpus", o l'annotazione
                             golden è troppo stretta → report gold-vs-judge]

Più un segnale sul retrieval, su tutte le domande giudicate:
     context_recall = context_supports==true / (# giudizi con il flag valorizzato)

Le domande senza annotazione golden (``answerable`` == unknown) sono contate a
parte in ``skipped_no_gold`` e non entrano in nessuna metrica.

Gli stessi numeri sono ricalcolati per ``category`` (``by_category`` nell'output):
è il taglio dove emergono i punti di forza/debolezza di ciascun tool.
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


def _tally(pairs: list[tuple]) -> dict:
    """
    Calcola content / safety / retrieval per una lista di (gold, judgement).

    Usato sia sull'insieme completo sia per singola categoria: la logica di
    conteggio vive qui una volta sola.
    """
    ok = partial = wrong = declined = answerable_total = 0
    n_wrong_total = n_total = 0
    abstained = answered_anyway = not_answerable_total = 0
    ctx_supported = ctx_total = 0

    for gold, judgement in pairs:
        label = judgement["label"]
        n_total += 1
        n_wrong_total += label == "WRONG"

        if judgement.get("context_supports") is not None:
            ctx_total += 1
            ctx_supported += int(bool(judgement["context_supports"]))

        if gold.answerable in ("yes", "partial"):
            answerable_total += 1
            ok += label == "OK"
            partial += label == "PARTIAL"
            wrong += label == "WRONG"
            declined += label == "DECLINED"
        elif gold.answerable == "no":
            not_answerable_total += 1
            abstained += label == "DECLINED"
            # non-DECLINED su una domanda "non rispondibile": il sistema ha
            # risposto comunque. WRONG = allucinazione vera; OK/PARTIAL = o il
            # sistema ha correttamente detto "non nel corpus", o l'annotazione
            # golden era troppo stretta -> vedi il report gold-vs-judge.
            answered_anyway += label != "DECLINED"

    precision = _ratio(ok, ok + partial + wrong)
    recall = _ratio(ok, answerable_total)

    return {
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
            # allucinazione = il giudice dice "non supportata dal corpus" (WRONG),
            # su QUALSIASI domanda. È il numero di sicurezza più importante.
            "hallucination_rate": _ratio(n_wrong_total, n_total),
            "not_answerable_questions": not_answerable_total,
            "abstention_rate": _ratio(abstained, not_answerable_total),
            "answered_anyway_rate": _ratio(answered_anyway, not_answerable_total),
        },
        "retrieval": {
            "judged_with_flag": ctx_total,
            "context_recall": _ratio(ctx_supported, ctx_total),
        },
    }


def compute_metrics(run_dir: Path) -> dict:
    judgements = {j["id"]: j for j in results_store.read_jsonl(run_dir / "judgements.jsonl")}
    gold_by_id = {q.id: q for q in load_questions(_QUESTIONS_CSV)}

    label_distribution: dict[str, int] = {}
    scored: list[tuple] = []        # (gold, judgement) con annotazione golden
    skipped_no_gold = 0

    for qid, judgement in judgements.items():
        label = judgement["label"]
        label_distribution[label] = label_distribution.get(label, 0) + 1
        gold = gold_by_id.get(qid)
        if gold is None or gold.answerable == "unknown":
            skipped_no_gold += 1
        else:
            scored.append((gold, judgement))

    by_category: dict[str, dict] = {}
    categories = sorted({g.category for g, _ in scored if g.category})
    for cat in categories:
        by_category[cat] = _tally([(g, j) for g, j in scored if g.category == cat])

    return {
        "run_dir": str(run_dir),
        "n_judged": len(judgements),
        "skipped_no_gold": skipped_no_gold,
        "label_distribution": label_distribution,
        **_tally(scored),                 # content / safety / retrieval sul totale
        "by_category": by_category,
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
