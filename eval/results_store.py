"""
Layout su disco dei risultati, uno per (tool, configurazione, run nel tempo).

    eval/results/<tool>/<config-slug>__<hash>/<timestamp>/
        run.json          metadati del run (tool, config, modelli, corpus, tempi)
        answers.jsonl      una riga per domanda: risposta + contesti + raw
        judgements.jsonl   una riga per domanda: label del giudice + motivazione
        metrics.json       aggregati

Lo stesso (tool, config) rieseguito nel tempo aggiunge un nuovo timestamp:
così si vede se le performance cambiano (aggiornamenti del tool, del modello...).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = _PROJECT_ROOT / "eval" / "results"


def _config_slug(config: dict[str, Any]) -> str:
    """
    Nome cartella leggibile + hash corto della config completa.

    La parte leggibile aiuta a orientarsi tra le cartelle; l'hash garantisce
    che due configurazioni diverse non collidano mai anche se lo slug è uguale.
    """
    readable = "-".join(f"{key}={config[key]}" for key in sorted(config))
    readable = readable.replace("/", "_").replace(" ", "")[:80]
    fingerprint = hashlib.sha1(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return f"{readable}__{fingerprint}"


def new_run_dir(tool: str, config: dict[str, Any]) -> Path:
    """Crea e restituisce la cartella per un nuovo run, marcata col timestamp UTC."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = RESULTS_ROOT / tool / _config_slug(config) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
