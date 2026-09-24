"""Put the project root on sys.path so tests import ``models`` and ``scripts``
the same way the pipeline scripts do, regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
