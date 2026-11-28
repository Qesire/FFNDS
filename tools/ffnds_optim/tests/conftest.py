"""Use one canonical package name when testing from any repository directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
