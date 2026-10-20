"""Custom exceptions used by the Citadel application."""


class CitadelBaseError(Exception):
	"""Base class for expected Citadel application errors."""


class ConfigurationError(CitadelBaseError):
	"""Raised when application configuration is missing or invalid."""


class DocumentIngestionError(CitadelBaseError):
	"""Raised when a document cannot be loaded or processed."""


class EmptyDatasetError(DocumentIngestionError):
	"""Raised when no supported, non-empty documents are available."""


class GenerationError(CitadelBaseError):
	"""Raised when a grounded answer cannot be generated."""


class EmptyQuestionError(GenerationError):
	"""Raised when a student submits a blank question."""
