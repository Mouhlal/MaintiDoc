"""
prepare_data.py — Préparation du dataset de fine-tuning du reranker
=====================================================================

Construit des triplets (query, positive_passage, hard_negative_passage)
et sépare strictement train / validation / test par famille d'équipements
pour éviter toute fuite de données (data leakage).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SYNTHETIC_TRAINING_EXAMPLES: list[dict[str, Any]] = [
    {
        "equipment_type": "pompe_centrifuge",
        "split": "train",
        "query": "Comment purger la pompe centrifuge avant démarrage ?",
        "positive": "Ouvrir la vanne de purge d'air située en partie haute du corps de pompe jusqu'à écoulement continu de liquide sans bulles.",
        "hard_negative": "Vérifier le niveau d'huile du compresseur à vis avant chaque démarrage hebdomadaire.",
    },
    {
        "equipment_type": "pompe_centrifuge",
        "split": "train",
        "query": "Causes de cavitation sur pompe P-101",
        "positive": "La cavitation apparaît lorsque la pression à l'aspiration descend sous la pression de vapeur du fluide, provoquant bruit et vibrations.",
        "hard_negative": "La pression nominale de refoulement est mesurée en sortie de pompe par le manomètre M-12.",
    },
    {
        "equipment_type": "compresseur_air",
        "split": "val",
        "query": "Fréquence de remplacement du filtre à air compresseur GA15",
        "positive": "Remplacer la cartouche du filtre d'aspiration toutes les 2000 heures de fonctionnement ou annuellement en environnement poussiéreux.",
        "hard_negative": "Purger les condensats du réservoir d'air comprimé quotidiennement via la vanne manuelle.",
    },
    {
        "equipment_type": "moteur_electrique",
        "split": "test",
        "query": "Température maximale admissible des roulements moteur ABB",
        "positive": "La température de palier ne doit pas dépasser 80°C en fonctionnement continu ; déclencher une inspection au-delà.",
        "hard_negative": "Mesurer la résistance d'isolement des bobinages à 500V DC avant remise sous tension.",
    },
]


def build_splits(output_dir: str | Path) -> dict[str, Path]:
    """Sauvegarde les fichiers train.jsonl, val.jsonl et test.jsonl sans fuite."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for ex in SYNTHETIC_TRAINING_EXAMPLES:
        buckets[ex["split"]].append(ex)

    paths: dict[str, Path] = {}
    for split_name, rows in buckets.items():
        file_path = out / f"{split_name}.jsonl"
        with file_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[split_name] = file_path

    return paths


if __name__ == "__main__":
    created = build_splits(Path(__file__).parent / "data")
    for split, p in created.items():
        print(f"{split}: {p}")
