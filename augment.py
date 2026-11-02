"""Root entrypoint for augmentation pipeline."""
import os
import sys

# Ensure src/ is importable
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline import main, run_augmentation

if __name__ == "__main__":
    main()
