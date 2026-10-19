"""LangChain-backed textbook loading and semantic chunking."""

import hashlib
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from citadel.core.config import get_settings
from citadel.core.exceptions import DocumentIngestionError, EmptyDatasetError
from citadel.core.logger import get_logger
from citadel.domain.interfaces import BaseDocumentProcessor


class TextbookDocumentProcessor(BaseDocumentProcessor):
	"""Load PDFs and text files into deterministic LangChain documents."""

	_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}

	def __init__(
		self,
		chunk_size: int | None = None,
		chunk_overlap: int | None = None,
	) -> None:
		"""Initialize the processor with validated splitter settings."""
		settings = get_settings()
		self._chunk_size = (
			chunk_size if chunk_size is not None else settings.CHUNK_SIZE
		)
		self._chunk_overlap = (
			chunk_overlap
			if chunk_overlap is not None
			else settings.CHUNK_OVERLAP
		)
		if self._chunk_size <= 0:
			raise ValueError("chunk_size must be greater than zero.")
		if self._chunk_overlap < 0:
			raise ValueError("chunk_overlap cannot be negative.")
		if self._chunk_overlap >= self._chunk_size:
			raise ValueError("chunk_overlap must be less than chunk_size.")

		self._logger = get_logger(__name__)
		self._splitter = RecursiveCharacterTextSplitter(
			chunk_size=self._chunk_size,
			chunk_overlap=self._chunk_overlap,
			separators=["\n\n", "\n", ". ", " ", ""],
			add_start_index=True,
		)

	def load_and_split(self, directory_path: Path) -> List[Document]:
		"""Load all supported files in a directory and return their chunks.

		Args:
			directory_path: Directory containing PDF, Markdown, or text files.

		Raises:
			DocumentIngestionError: If the directory does not exist or a file
				cannot be processed.
			EmptyDatasetError: If the directory contains no supported files or
				all supported files produce no content.
		"""
		if not directory_path.exists() or not directory_path.is_dir():
			raise DocumentIngestionError(
				f"Textbook directory does not exist: {directory_path}"
			)

		files = sorted(
			path
			for path in directory_path.iterdir()
			if path.is_file() and path.suffix.lower() in self._SUPPORTED_SUFFIXES
		)
		self._logger.info("Scanned %d textbook files in %s", len(files), directory_path)
		if not files:
			raise EmptyDatasetError(
				f"No supported textbook files found in {directory_path}."
			)

		chunks: List[Document] = []
		for file_path in files:
			chunks.extend(self.process_single_file(file_path))

		if not chunks:
			raise EmptyDatasetError(
				f"Supported textbook files in {directory_path} contained no text."
			)

		self._logger.info("Produced %d total textbook chunks", len(chunks))
		return chunks

	def process_single_file(self, file_path: Path) -> List[Document]:
		"""Load and split one PDF, Markdown, or text file."""
		if not file_path.is_file() or file_path.suffix.lower() not in self._SUPPORTED_SUFFIXES:
			raise DocumentIngestionError(
				f"Unsupported textbook file: {file_path}"
			)

		try:
			documents = self._load_documents(file_path)
			normalized_documents = [
				self._normalize_document(document) for document in documents
			]
			normalized_documents = [
				document
				for document in normalized_documents
				if document.page_content.strip()
			]
			split_documents = self._splitter.split_documents(normalized_documents)
			chunks = self._to_langchain_chunks(file_path, split_documents)
			self._logger.info(
				"Processed %s: %d source pages/documents, %d chunks",
				file_path.name,
				len(documents),
				len(chunks),
			)
			return chunks
		except DocumentIngestionError:
			raise
		except Exception as exc:
			raise DocumentIngestionError(
				f"Failed to process textbook file {file_path}: {exc}"
			) from exc

	def _load_documents(self, file_path: Path) -> List[Document]:
		"""Load LangChain documents using the loader for a file type."""
		if file_path.suffix.lower() == ".pdf":
			return PyPDFLoader(str(file_path)).load()
		return TextLoader(str(file_path), encoding="utf-8").load()

	@staticmethod
	def _normalize_document(document: Document) -> Document:
		"""Normalize extraction artifacts while preserving document metadata."""
		content = document.page_content.replace("\x00", "")
		content = content.replace("\r\n", "\n").replace("\r", "\n")
		content = re.sub(r"\n{3,}", "\n\n", content)
		return Document(page_content=content.strip(), metadata=dict(document.metadata))

	def _to_langchain_chunks(
		self,
		file_path: Path,
		documents: List[Document],
	) -> List[Document]:
		"""Add deterministic IDs and domain metadata to LangChain documents."""
		chunks: List[Document] = []
		source_file = file_path.name
		for chunk_index, document in enumerate(documents):
			content = document.page_content.strip()
			if not content:
				continue
			raw_page = document.metadata.get("page")
			page_number = raw_page + 1 if isinstance(raw_page, int) else None
			chunk_id = self._build_chunk_id(
				source_file, page_number, chunk_index, content
			)
			metadata = dict(document.metadata)
			metadata.update(
				{
					"chunk_id": chunk_id,
					"source_file": source_file,
					"page_number": page_number,
					"chunk_index": chunk_index,
					"total_char_length": len(content),
				}
			)
			chunks.append(Document(page_content=content, metadata=metadata))
		return chunks

	@staticmethod
	def _build_chunk_id(
		source_file: str,
		page_number: int | None,
		chunk_index: int,
		content: str,
	) -> str:
		"""Build an idempotent SHA-256 identifier for a chunk."""
		identity = f"{source_file}\0{page_number}\0{chunk_index}\0{content}"
		return hashlib.sha256(identity.encode("utf-8")).hexdigest()
