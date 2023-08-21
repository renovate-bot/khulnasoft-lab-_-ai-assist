from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent
TPL_DIR = ASSETS_DIR / "tpl"

__all__ = [
    "ASSETS_DIR",
    "TPL_DIR",
]
