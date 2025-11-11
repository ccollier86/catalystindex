import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Ensure local packages (fastapi, pydantic) are discoverable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
