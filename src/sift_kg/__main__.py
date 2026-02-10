"""Entry point for python -m sift_kg execution.

This module enables running sift-kg as a module:
    python -m sift_kg --help
    python -m sift_kg extract ./docs/
"""

from sift_kg.cli import app

if __name__ == "__main__":
    app()
