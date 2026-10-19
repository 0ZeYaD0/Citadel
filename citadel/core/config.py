"""Application configuration for Citadel."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from citadel.core.exceptions import ConfigurationError


load_dotenv()


@dataclass(frozen=True)
class Settings:
	"""Runtime settings loaded from environment variables."""

	GEMINI_API_KEY: str
	TEXTBOOKS_DIR: Path = Path("data/textbooks")
	CHROMA_PERSIST_DIR: Path = Path("chroma_db")
	CHUNK_SIZE: int = 1000
	CHUNK_OVERLAP: int = 200


def _read_int(name: str, default: int) -> int:
	"""Read a positive integer environment variable."""
	import os

	raw_value = os.getenv(name)
	if raw_value is None or not raw_value.strip():
		return default

	try:
		value = int(raw_value)
	except ValueError as exc:
		raise ConfigurationError(f"{name} must be an integer.") from exc

	if value <= 0:
		raise ConfigurationError(f"{name} must be greater than zero.")
	return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
	"""Return the cached application settings.

	Raises:
		ConfigurationError: If the required Gemini API key is missing or a
			numeric setting is invalid.
	"""
	import os

	api_key = os.getenv("GEMINI_API_KEY", "").strip()
	if not api_key or api_key == "your_gemini_api_key_here":
		raise ConfigurationError(
			"GEMINI_API_KEY must be set to a valid Gemini API key."
		)

	chunk_size = _read_int("CHUNK_SIZE", 1000)
	chunk_overlap = _read_int("CHUNK_OVERLAP", 200)
	if chunk_overlap >= chunk_size:
		raise ConfigurationError("CHUNK_OVERLAP must be less than CHUNK_SIZE.")

	return Settings(
		GEMINI_API_KEY=api_key,
		TEXTBOOKS_DIR=Path(os.getenv("TEXTBOOKS_DIR", "data/textbooks")),
		CHROMA_PERSIST_DIR=Path(
			os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
		),
		CHUNK_SIZE=chunk_size,
		CHUNK_OVERLAP=chunk_overlap,
	)
