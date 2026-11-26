"""
main.py — Point d'entrée du serveur MCP autonome MaintiDoc
===========================================================

Usage :
    python mcp_server/main.py
    # ou
    make mcp
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from backend.mcp.server import mcp


def main() -> None:
    logger.info("Démarrage du serveur MCP MaintiDoc...")
    mcp.run()


if __name__ == "__main__":
    main()
