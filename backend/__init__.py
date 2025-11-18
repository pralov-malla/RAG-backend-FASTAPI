"""Backend package initializer.

This file makes `backend` a Python package so relative imports work when
running the app with ``uvicorn backend.main:app``.
"""

__all__ = [
    "main",
    "file_utils",
    "rag_chain",
]
