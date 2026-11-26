"""
ingest_sample.py — Génération et ingestion de documents de démonstration
========================================================================

Permet soit d'ingérer un fichier PDF passé en argument, soit de générer
un manuel technique fictif de pompe centrifuge P-101 et de l'indexer.

Usage :
    python scripts/ingest_sample.py --sample
    python scripts/ingest_sample.py --file data/raw/manuel.pdf
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from loguru import logger

from backend.config import settings
from backend.rag.indexer import DocumentIndexer


SAMPLE_MANUAL_TEXT = """MANUEL TECHNIQUE DE MAINTENANCE — POMPE CENTRIFUGE GRUNDFOS CM5-6 (P-101)
Version : REV-2.4 | Équipement : pompe_centrifuge

1. SÉCURITÉ ET CONSIGNATION
Avant toute intervention mécanique ou électrique sur la pompe P-101 :
- Couper l'alimentation électrique au sectionneur S-101 et poser un cadenas de consignation.
- Fermer les vannes d'aspiration (V-101) et de refoulement (V-102).
- Vérifier l'absence de pression résiduelle au manomètre M-101.
- Attendre le refroidissement complet si le fluide dépasse 45°C.

2. PROCÉDURE DE PURGE ET AMORÇAGE AVANT DÉMARRAGE
1. Ouvrir lentement la vanne d'aspiration V-101.
2. Dévisser le bouchon ou ouvrir la vanne de purge d'air située en partie haute du corps de pompe.
3. Laisser l'air s'échapper jusqu'à obtenir un écoulement continu de liquide sans bulles.
4. Refermer hermétiquement la vanne de purge d'air.
5. Vérifier la rotation libre de l'arbre moteur manuellement avant mise sous tension.

3. DIAGNOSTIC DE CAVITATION ET VIBRATIONS
Symptômes typiques de cavitation : bruit de graviers dans le corps de pompe, vibrations élevées (> 4.5 mm/s) et chute de pression au refoulement.
Causes probables :
- Filtre d'aspiration encrassé ou partiellement colmaté.
- Vanne d'aspiration V-101 insuffisamment ouverte.
- Température du fluide trop élevée par rapport au NPSH requis.
Actions recommandées : nettoyer le filtre d'aspiration et contrôler la pression d'entrée.

4. REMPLACEMENT DU JOINT MÉCANIQUE (TYPE BURGMANN MG1-28)
En cas de fuite supérieure à 5 gouttes par minute au niveau de la garniture mécanique :
- Consigner l'équipement selon la section 1.
- Vidanger le corps de pompe via le bouchon inférieur.
- Remplacer la garniture mécanique complète et vérifier l'alignement arbre/moteur.
- Périodicité préventive recommandée : inspection tous les 90 jours.
"""


def create_sample_pdf(target_path: Path) -> Path:
    """Crée un PDF propre à partir du texte de démonstration avec PyMuPDF."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 545, 790)
    page.insert_textbox(rect, SAMPLE_MANUAL_TEXT, fontsize=10.5, fontname="helv")
    doc.save(str(target_path))
    doc.close()
    logger.info(f"PDF de démonstration généré : {target_path}")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion PDF MaintiDoc")
    parser.add_argument("--file", type=str, help="Chemin vers un fichier PDF à ingérer")
    parser.add_argument("--sample", action="store_true", help="Génère et ingère le manuel démo")
    parser.add_argument("--equipment-type", type=str, default="pompe_centrifuge")
    args = parser.parse_args()

    if args.sample:
        pdf_path = create_sample_pdf(settings.data_dir / "sample" / "manuel_pompe_p101.pdf")
    elif args.file:
        pdf_path = Path(args.file)
    else:
        parser.print_help()
        sys.exit(1)

    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, pdf_path.name))
    indexer = DocumentIndexer()
    summary = indexer.index_document(
        file_path=pdf_path,
        document_id=doc_id,
        equipment_type=args.equipment_type,
        doc_type="manuel_maintenance",
        version="2.4",
    )
    print("\n✅ Résumé d'ingestion :")
    for k, v in summary.items():
        if k != "metadata":
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
