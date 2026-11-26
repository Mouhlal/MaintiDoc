"""
train_reranker.py — Adaptation ciblée d'un CrossEncoder multilingue
===================================================================

Compare les métriques avant et après fine-tuning du reranker :
  1. Sans reranker (ordre Qdrant brut)
  2. Avec reranker initial (ms-marco-MiniLM-L-6-v2)
  3. Avec reranker adapté sur le domaine maintenance industrielle

Usage :
    python fine_tuning/train_reranker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from torch.utils.data import DataLoader

from backend.config import settings
from fine_tuning.prepare_data import SYNTHETIC_TRAINING_EXAMPLES, build_splits


def train(output_dir: str | Path | None = None, epochs: int = 1) -> Path:
    """Entraîne légèrement le CrossEncoder sur les paires positives et négatifs difficiles."""
    from sentence_transformers import CrossEncoder, InputExample

    data_dir = Path(__file__).parent / "data"
    build_splits(data_dir)

    save_path = Path(output_dir or (Path(__file__).parent / "checkpoints" / "maintidoc-reranker"))
    save_path.mkdir(parents=True, exist_ok=True)

    train_samples: list[InputExample] = []
    for row in SYNTHETIC_TRAINING_EXAMPLES:
        if row["split"] == "train":
            train_samples.append(InputExample(texts=[row["query"], row["positive"]], label=1.0))
            train_samples.append(InputExample(texts=[row["query"], row["hard_negative"]], label=0.0))

    logger.info(f"Chargement du modèle de base : {settings.reranker_model}")
    model = CrossEncoder(settings.reranker_model, num_labels=1, device="cpu")
    dataloader = DataLoader(train_samples, shuffle=True, batch_size=2)

    logger.info(f"Début du fine-tuning ({len(train_samples)} exemples, {epochs} epoch(s))...")
    model.fit(
        train_dataloader=dataloader,
        epochs=epochs,
        warmup_steps=1,
        output_path=str(save_path),
        show_progress_bar=False,
    )
    model.save(str(save_path))
    logger.info(f"Reranker adapté sauvegardé dans : {save_path}")
    return save_path


if __name__ == "__main__":
    train()
