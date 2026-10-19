"""Logging utilities for Citadel."""

import logging


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
	"""Return a consistently configured logger.

	Args:
		name: Name to use for the logger, usually ``__name__``.

	Returns:
		A logger configured with the Citadel console format.
	"""
	logger = logging.getLogger(name)
	if not logger.handlers:
		handler = logging.StreamHandler()
		handler.setFormatter(logging.Formatter(_LOG_FORMAT))
		logger.addHandler(handler)
		logger.setLevel(logging.INFO)
		logger.propagate = False
	return logger
