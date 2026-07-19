"""Permite `python -m docs2llm <subcomando>` (ver cli.py para los subcomandos)."""

import sys

from docs2llm.cli import main

if __name__ == "__main__":
    sys.exit(main())
