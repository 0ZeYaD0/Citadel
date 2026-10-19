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

	gemini_api_key: str = ""
	textbooks_dir: Path = Path("data/textbooks")
	chroma_persist_dir: Path = Path("chroma_db")
	chroma_collection_name: str = "citadel_docs"
	embedding_model: str = "models/gemini-embedding-001"
	chat_model: str = "gemini-3.8-flash"
	chunk_size: int = 1000
	chunk_overlap: int = 200

	@property
	def GEMINI_API_KEY(self) -> str:
		return self.gemini_api_key

	@property
	def TEXTBOOKS_DIR(self) -> Path:
		return self.textbooks_dir

	@property
	def CHROMA_PERSIST_DIR(self) -> Path:
		return self.chroma_persist_dir

	@property
	def CHUNK_SIZE(self) -> int:
		return self.chunk_size

	@property
	def CHUNK_OVERLAP(self) -> int:
		return self.chunk_overlap


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

	chunk_size = _read_int("CHUNK_SIZE", 1000)
	chunk_overlap = _read_int("CHUNK_OVERLAP", 200)
	if chunk_overlap >= chunk_size:
		raise ConfigurationError("CHUNK_OVERLAP must be less than CHUNK_SIZE.")

	return Settings(
		gemini_api_key=api_key,
		textbooks_dir=Path(os.getenv("TEXTBOOKS_DIR", "data/textbooks")),
		chroma_persist_dir=Path(
			os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
		),
		chroma_collection_name=os.getenv(
			"CHROMA_COLLECTION_NAME", "citadel_docs"
		),
		embedding_model=os.getenv(
			"EMBEDDING_MODEL", "models/gemini-embedding-001"
		),
		chat_model=os.getenv("CHAT_MODEL", "gemini-2.5-flash"),
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)


settings = get_settings()
