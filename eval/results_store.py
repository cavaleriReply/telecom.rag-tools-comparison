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


def config_hash(config: dict[str, Any]) -> str:
    """
    Hash corto e stabile della config: identifica una configurazione a prescindere
    dal formato dello slug leggibile. Usato per gli artefatti che devono
    persistere tra run (working dir di LightRAG) e come suffisso dello slug.
    """
    return hashlib.sha1(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]


def _config_slug(config: dict[str, Any]) -> str:
    """
    Nome cartella: parte leggibile (per orientarsi) + ``config_hash`` (per
    l'unicità, anche se lo slug leggibile venisse troncato o cambiasse formato).
    """
    tokens = [
        f"{key}={config[key]}".replace("/", "_").replace(" ", "") for key in sorted(config)
    ]
    # Token interi finché stanno nel budget (niente tagli a metà valore). È un
    # nome di cartella, quindi possiamo essere generosi.
    readable, budget = "", 120
    for token in tokens:
        if len(readable) + len(token) + 1 > budget:
            break
        readable = f"{readable}-{token}" if readable else token
    return f"{readable}__{config_hash(config)}"


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
