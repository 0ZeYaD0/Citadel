"""Application configuration for Citadel."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from citadel.core.exceptions import ConfigurationError


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(value: str, default: str) -> Path:
	"""Resolve configured relative paths from the project root."""
	path = Path(value or default)
	return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
	"""Runtime settings loaded from environment variables."""

	gemini_api_key: str = ""
	groq_api_key: str = ""
	groq_model: str = "qwen/qwen3.8-27b"
	textbooks_dir: Path = Path("data/textbooks")
	chroma_persist_dir: Path = Path("chroma_db")
	chroma_collection_name: str = "citadel_local_minilm"
	embedding_model: str = "models/gemini-embedding-001"
	local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
	chat_model: str = "gemini-3.5-flash-lite"
	gemini_llm_model: str = "gemini-3.5-flash-lite"
	llm_temperature: float = 0.1
	llm_max_output_tokens: int = 2048
	retrieval_top_k: int = 5
	chunk_size: int = 1500
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

	@property
	def GEMINI_LLM_MODEL(self) -> str:
		return self.gemini_llm_model

	@property
	def LOCAL_EMBEDDING_MODEL(self) -> str:
		return self.local_embedding_model

	@property
	def LLM_TEMPERATURE(self) -> float:
		return self.llm_temperature

	@property
	def LLM_MAX_OUTPUT_TOKENS(self) -> int:
		return self.llm_max_output_tokens

	@property
	def GROQ_API_KEY(self) -> str:
		return self.groq_api_key

	@property
	def GROQ_MODEL(self) -> str:
		return self.groq_model


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


def _read_float(name: str, default: float) -> float:
	"""Read a non-negative floating-point environment variable."""
	import os

	raw_value = os.getenv(name)
	if raw_value is None or not raw_value.strip():
		return default

	try:
		value = float(raw_value)
	except ValueError as exc:
		raise ConfigurationError(f"{name} must be a number.") from exc

	if value < 0:
		raise ConfigurationError(f"{name} cannot be negative.")
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
	groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

	chunk_size = _read_int("CHUNK_SIZE", 1500)
	chunk_overlap = _read_int("CHUNK_OVERLAP", 200)
	llm_temperature = _read_float("LLM_TEMPERATURE", 0.1)
	llm_max_output_tokens = _read_int("LLM_MAX_OUTPUT_TOKENS", 2048)
	retrieval_top_k = _read_int("RETRIEVAL_TOP_K", 5)
	if chunk_overlap >= chunk_size:
		raise ConfigurationError("CHUNK_OVERLAP must be less than CHUNK_SIZE.")

	return Settings(
		gemini_api_key=api_key,
		groq_api_key=groq_api_key,
		groq_model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
		textbooks_dir=_project_path(
			os.getenv("TEXTBOOKS_DIR", "data/textbooks"),
			"data/textbooks",
		),
		chroma_persist_dir=_project_path(
			os.getenv("CHROMA_PERSIST_DIR", "chroma_db"),
			"chroma_db",
		),
		chroma_collection_name=os.getenv(
			"CHROMA_COLLECTION_NAME", "citadel_local_minilm"
		),
		embedding_model=os.getenv(
			"EMBEDDING_MODEL", "models/gemini-embedding-001"
		),
		local_embedding_model=os.getenv(
			"LOCAL_EMBEDDING_MODEL",
			"sentence-transformers/all-MiniLM-L6-v2",
		),
		chat_model=os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite"),
		gemini_llm_model=os.getenv(
			"GEMINI_LLM_MODEL",
			os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite"),
		),
		llm_temperature=llm_temperature,
		llm_max_output_tokens=llm_max_output_tokens,
		retrieval_top_k=retrieval_top_k,
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)


settings = get_settings()
