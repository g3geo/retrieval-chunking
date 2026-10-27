"""Run only the chunk stage; use --help for available arguments."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chunkeval.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["chunk", *sys.argv[1:]]))

