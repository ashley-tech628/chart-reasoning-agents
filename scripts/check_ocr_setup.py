"""Check whether the local Tesseract OCR executable is visible to pytesseract.

This project uses `pytesseract` as a Python wrapper, but the wrapper alone is
not enough.  The Tesseract executable must also be installed and discoverable
from PATH, or assigned to `pytesseract.pytesseract.tesseract_cmd`.
"""

from __future__ import annotations

try:
    import pytesseract
except Exception as exc:
    print(f"pytesseract import failed: {exc}")
    raise SystemExit(1)

print(f"pytesseract module: {getattr(pytesseract, '__file__', 'unknown')}")
try:
    print(f"configured executable: {pytesseract.pytesseract.tesseract_cmd}")
    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    print("OCR setup looks OK.")
except Exception as exc:
    print("Tesseract executable is not callable from Python.")
    print(f"Error: {exc}")
    print()
    print("Fix on Windows: install the Tesseract OCR application, then either add")
    print("its install directory to PATH or set this near the top of offline_vision.py:")
    print("pytesseract.pytesseract.tesseract_cmd = r'C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe'")
    raise SystemExit(1)
