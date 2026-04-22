#!/usr/bin/env python3
"""Utility script to prepare MUSDB18 metadata for training.

This script is intentionally lightweight for publication. It documents the
expected folder layout and can be expanded with the user's full preprocessing
logic from the notebook.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path("data/raw")
    root.mkdir(parents=True, exist_ok=True)
    print("Place the MUSDB18 dataset under data/raw/ and generate metadata CSV files:")
    print("  - musdb_train.csv")
    print("  - musdb_valid.csv")
    print("  - musdb_test.csv")
    print("You can port the preprocessing cells from notebooks/Project.ipynb into this script.")


if __name__ == "__main__":
    main()
