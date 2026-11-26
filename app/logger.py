"""
app/logger.py

Structured JSON logger for MP076 prediction service.
Outputs one JSON line per prediction to stdout (never to a file).
Does not log any sensitive customer personal details.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Include extra attributes if passed in record.__dict__
        if hasattr(record, "prediction_data") and isinstance(record.prediction_data, dict):
            log_entry.update(record.prediction_data)

        return json.dumps(log_entry)


def setup_logger(name: str = "atm_prediction_service") -> logging.Logger:
    """Configure and return a structured JSON logger writing strictly to stdout."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-imported
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# Global logger instance
service_logger = setup_logger()
