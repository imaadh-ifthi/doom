"""CSV logging and metric telemetry utilities for DOOM SNN runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class CSVLogger:
    """Lightweight CSV logger that writes metric rows to disk incrementally."""

    def __init__(
        self,
        filepath: Union[str, Path],
        fieldnames: Optional[List[str]] = None,
    ) -> None:
        """Initialize the CSV logger.

        Args:
            filepath: Path to the target CSV file.
            fieldnames: Optional initial list of CSV column header names.
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames) if fieldnames else None
        self._file = None
        self._writer = None

        if self.fieldnames is not None and not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """Write a dictionary of metric values to the CSV file.

        Args:
            row: Dictionary mapping column names to scalar values.
        """
        file_exists = self.filepath.exists()
        current_keys = list(row.keys())

        if self.fieldnames is None:
            self.fieldnames = current_keys
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        # Check if new keys need to be handled
        new_keys = [k for k in current_keys if k not in self.fieldnames]
        if new_keys:
            # Expand existing fieldnames and rewrite header if necessary
            self.fieldnames.extend(new_keys)

        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow(row)

    def close(self) -> None:
        """Close logger resources."""
        pass
