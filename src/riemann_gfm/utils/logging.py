"""Minimal experiment logger. Writes a CSV row per epoch/step."""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Dict, Optional


class CSVLogger:
    def __init__(self, path: str, fieldnames: Optional[list[str]] = None) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fieldnames = fieldnames
        self._file = None
        self._writer = None
        self._start = time.time()

    def _ensure_writer(self, fieldnames: list[str]) -> None:
        if self._writer is not None:
            return
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, row: Dict) -> None:
        row = {**row, "wall_s": round(time.time() - self._start, 3)}
        if self._writer is None:
            self._ensure_writer(list(row.keys()))
        self._writer.writerow(row)
        self._file.flush()
        # Also echo to stdout so notebooks show progress.
        summary = " ".join(f"{k}={v}" for k, v in row.items())
        print(summary, file=sys.stdout)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
