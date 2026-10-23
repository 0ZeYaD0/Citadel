"""Application configuration for Citadel."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from citadel.core.exceptions import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _project_path(value: str | None, default: str) -> Path:
	"""Resolve relative runtime paths from the project root."""
	path = Path(value or default)
	return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
	"""Runtime settings loaded from environment variables."""

	groq_api_key: str = ""
	hf_token: str = ""
	hf_home: Path | None = None
	groq_model: str = "qwen/qwen3.8-27b"
	api_title: str = "Citadel RAG API"
	api_description: str = "Study assistant backed by a local RAG pipeline."
	api_version: str = "2.0.0"
	api_host: str = "0.0.0.0"
	api_port: int = 8000
	database_path: Path = Path("citadel.db")
	static_dir: Path = Path("static")
	study_prompts_path: Path = Path("prompts/study_tools.txt")
	chroma_persist_dir: Path = Path("chroma_db")
	chroma_collection_name: str = "citadel_local_minilm"
	local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
	llm_temperature: float = 0.1
	llm_max_output_tokens: int = 2048
	retrieval_top_k: int = 5
	min_similarity: float = 0.25
	llm_max_retries: int = 2
	ingestion_batch_size: int = 50
	ingestion_max_retries: int = 8
	ingestion_min_retry_wait_seconds: float = 60.0
	file_io_workers: int = 4
	sqlite_busy_timeout_ms: int = 5000
	sqlite_lock_retries: int = 3
	sqlite_retry_delay_seconds: float = 0.05
	query_cache_size: int = 128
	load_pipeline_in_background: bool = True
	chunk_size: int = 1500
	chunk_overlap: int = 200


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


def _read_bool(name: str, default: bool) -> bool:
	"""Read a boolean environment variable."""
	import os

	raw_value = os.getenv(name)
	if raw_value is None or not raw_value.strip():
		return default
	value = raw_value.strip().lower()
	if value in {"1", "true", "yes", "on"}:
		return True
	if value in {"0", "false", "no", "off"}:
		return False
	raise ConfigurationError(f"{name} must be a boolean.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
	import os

	groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
	hf_token = os.getenv("HF_TOKEN", "").strip()

	chunk_size = _read_int("CHUNK_SIZE", 1500)
	chunk_overlap = _read_int("CHUNK_OVERLAP", 200)
	llm_temperature = _read_float("LLM_TEMPERATURE", 0.1)
	llm_max_output_tokens = _read_int("LLM_MAX_OUTPUT_TOKENS", 2048)
	retrieval_top_k = _read_int("RETRIEVAL_TOP_K", 5)
	api_port = _read_int("API_PORT", 8000)
	llm_max_retries = _read_int("LLM_MAX_RETRIES", 2)
	ingestion_batch_size = _read_int("INGESTION_BATCH_SIZE", 50)
	ingestion_max_retries = _read_int("INGESTION_MAX_RETRIES", 8)
	ingestion_min_retry_wait_seconds = _read_float(
		"INGESTION_MIN_RETRY_WAIT_SECONDS", 60.0
	)
	min_similarity = _read_float("MIN_SIMILARITY", 0.25)
	file_io_workers = _read_int("FILE_IO_WORKERS", 4)
	sqlite_busy_timeout_ms = _read_int("SQLITE_BUSY_TIMEOUT_MS", 5000)
	sqlite_lock_retries = _read_int("SQLITE_LOCK_RETRIES", 3)
	sqlite_retry_delay_seconds = _read_float("SQLITE_RETRY_DELAY_SECONDS", 0.05)
	query_cache_size = _read_int("QUERY_CACHE_SIZE", 128)
	load_pipeline_in_background = _read_bool("LOAD_PIPELINE_IN_BACKGROUND", True)
	if chunk_overlap >= chunk_size:
		raise ConfigurationError("CHUNK_OVERLAP must be less than CHUNK_SIZE.")

	return Settings(
		groq_api_key=groq_api_key,
		hf_token=hf_token,
		hf_home=(
			_project_path(os.getenv("HF_HOME"), ".hf_cache")
			if os.getenv("HF_HOME")
			else None
		),
		groq_model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
		api_title=os.getenv("API_TITLE", "Citadel RAG API"),
		api_description=os.getenv(
			"API_DESCRIPTION", "Study assistant backed by a local RAG pipeline."
		),
		api_version=os.getenv("API_VERSION", "2.0.0"),
		api_host=os.getenv("API_HOST", "0.0.0.0"),
		api_port=api_port,
		database_path=_project_path(os.getenv("DATABASE_PATH"), "citadel.db"),
		static_dir=_project_path(os.getenv("STATIC_DIR"), "static"),
		study_prompts_path=_project_path(
			os.getenv("STUDY_PROMPTS_PATH"), "prompts/study_tools.txt"
		),
		chroma_persist_dir=_project_path(
			os.getenv("CHROMA_PERSIST_DIR"), "chroma_db"
		),
		chroma_collection_name=os.getenv(
			"CHROMA_COLLECTION_NAME", "citadel_local_minilm"
		),
		local_embedding_model=os.getenv(
			"LOCAL_EMBEDDING_MODEL",
			"sentence-transformers/all-MiniLM-L6-v2",
		),
		llm_temperature=llm_temperature,
		llm_max_output_tokens=llm_max_output_tokens,
		retrieval_top_k=retrieval_top_k,
		min_similarity=min_similarity,
		llm_max_retries=llm_max_retries,
		ingestion_batch_size=ingestion_batch_size,
		ingestion_max_retries=ingestion_max_retries,
		ingestion_min_retry_wait_seconds=ingestion_min_retry_wait_seconds,
		file_io_workers=file_io_workers,
		sqlite_busy_timeout_ms=sqlite_busy_timeout_ms,
		sqlite_lock_retries=sqlite_lock_retries,
		sqlite_retry_delay_seconds=sqlite_retry_delay_seconds,
		query_cache_size=query_cache_size,
		load_pipeline_in_background=load_pipeline_in_background,
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)


settings = get_settings()
