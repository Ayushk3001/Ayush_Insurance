"""App alias module to support 'uvicorn app.main:app' command from submission guidelines."""
from src.main import app

__all__ = ["app"]
