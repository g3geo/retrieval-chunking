"""Enter questions in the terminal, or use --query for a single search."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chunkeval.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["search", *sys.argv[1:]]))

