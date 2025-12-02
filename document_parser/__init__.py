"""Document parser library with OCR support."""

from document_parser.config import ExtractorConfig, OCRConfig
from document_parser.detector import DocumentDescriptor, DocumentDetector
from document_parser.exceptions import (
    DecodingError,
    DocumentParserError,
    ExtractionError,
    InvalidBase64Error,
    UnsupportedTypeError,
)
from document_parser.extractor import PyMuPDFExtractor
from document_parser.handler import DocumentHandler
from document_parser.models import DocumentExtractionResult
from document_parser.parser import parse_document

__version__ = "0.1.0"

__all__ = [
    # High-level API
    "parse_document",
    # Core classes
    "DocumentHandler",
    "DocumentDetector",
    "PyMuPDFExtractor",
    # Data models
    "DocumentExtractionResult",
    "DocumentDescriptor",
    # Configuration
    "OCRConfig",
    "ExtractorConfig",
    # Exceptions
    "DocumentParserError",
    "InvalidBase64Error",
    "UnsupportedTypeError",
    "ExtractionError",
    "DecodingError",
]
