"""Entry point for Docker / HuggingFace Spaces:  uvicorn app:app --port 7860"""

from server.app import app  # noqa: F401
