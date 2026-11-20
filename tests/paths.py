"""Put the step folders on sys.path so tests import their modules by name, as the scripts do."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ('Pipeline', 'PriceAdjustment', 'DataCheck', 'DataDownload'):
    path = str(ROOT/folder)
    if path not in sys.path:
        sys.path.insert(0, path)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
