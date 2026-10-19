"""Abstractions owned by the document-ingestion domain."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain_core.documents import Document


class BaseDocumentProcessor(ABC):
	"""Contract for loading source files into domain document chunks."""

	@abstractmethod
	def load_and_split(self, directory_path: Path) -> List[Document]:
		"""Load supported files into LangChain document chunks."""
		raise NotImplementedError

	@abstractmethod
	def process_single_file(self, file_path: Path) -> List[Document]:
		"""Load one supported file into LangChain document chunks."""
		raise NotImplementedError
